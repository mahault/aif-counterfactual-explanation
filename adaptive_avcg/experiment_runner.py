"""
Full temporal experiment orchestrator.

For each (dataset, drift_type, seed):
  1. Load data, train initial base model + Rashomon set + CVAE, generate initial CEs
  2. For t = 1..T:
     - Apply drift(t) to training data
     - Retrain base model
     - Create new Rashomon set
     - Measure D_t via DriftDetector
     - Measure actual ECI of cached CEs
     - Run each strategy (decision policy) over its own cached CVAE
     - Log per-timestep metrics

The same temporal scaffold drives two experiment modes:
  - run_single / run_sweep: the four headline strategies
    (NEVER_RETRAIN, ALWAYS_RETRAIN, ADAPTIVE, FIXED_INTERVAL)
  - run_ablation_single / run_ablation_sweep: the ADAPTIVE policy under the
    valence/anxiety term ablation (ADAPT_BASE/VALENCE/ANXIETY/FULL), with
    NEVER/ALWAYS retained as reference bounds.

A strategy is described by a *spec* dict: {'kind': ..., 'config': ExperimentConfig}.
'kind' is one of {'never', 'always', 'fixed', 'adaptive'}; 'config' supplies the
policy hyperparameters (including the use_valence / use_anxiety ablation flags).
"""

import os
import json
import copy
import time
from dataclasses import replace
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from adaptive_avcg.avcg_core import (
    DEVICE, set_seed, get_dataset,
    MCDropoutMLP, MCDropoutRashomonSet, TabularCVAE, TabularDataset,
    train_mc_dropout, train_cvae_rashomon,
    generate_cf_amortized, validity, rashomon_validity_ratio
)
from adaptive_avcg.drift_simulator import DriftSimulator
from adaptive_avcg.drift_detector import DriftDetector
from adaptive_avcg.recomputation_policy import RecomputationPolicy
from adaptive_avcg.config import ExperimentConfig


class ExperimentRunner:
    """Orchestrates the full temporal drift experiment."""

    def __init__(self, config: ExperimentConfig = None, results_dir: str = './results'):
        self.config = config or ExperimentConfig()
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Strategy specs
    # ------------------------------------------------------------------

    def _standard_specs(self) -> dict:
        """The four headline strategies."""
        return {
            'NEVER_RETRAIN':  {'kind': 'never',    'config': self.config},
            'ALWAYS_RETRAIN': {'kind': 'always',   'config': self.config},
            'ADAPTIVE':       {'kind': 'adaptive', 'config': self.config},
            'FIXED_INTERVAL': {'kind': 'fixed',    'config': self.config},
        }

    def _ablation_specs(self) -> dict:
        """ADAPTIVE under the valence/anxiety term ablation, with NEVER/ALWAYS bounds.

        Each ADAPT_* variant gets a config copy with the affective-term switches
        set; everything else (drift signal, staleness, validity penalty, warm-up)
        is identical, so differences isolate the contribution of valence/anxiety.
        """
        def variant(use_valence, use_anxiety):
            return {'kind': 'adaptive',
                    'config': replace(self.config,
                                      use_valence=use_valence,
                                      use_anxiety=use_anxiety)}
        return {
            'NEVER_RETRAIN':  {'kind': 'never',  'config': self.config},
            'ALWAYS_RETRAIN': {'kind': 'always', 'config': self.config},
            'ADAPT_BASE':    variant(False, False),
            'ADAPT_VALENCE': variant(True,  False),
            'ADAPT_ANXIETY': variant(False, True),
            'ADAPT_FULL':    variant(True,  True),
        }

    def _labelfree_specs(self) -> dict:
        """Label-free F-policy vs an oracle (label-every-step) reference.

        ORACLE_BASE decides on measured validity every step (needs T labels).
        F_POLICY calibrates F->invalidity on the first `calibration_steps` labelled
        steps, then decides from free energy alone (label-free thereafter).
        The comparison: does F_POLICY match ORACLE_BASE validity at a fraction of
        the labelling cost?
        """
        base = replace(self.config, use_valence=False, use_anxiety=False)
        return {
            'NEVER_RETRAIN':  {'kind': 'never',  'config': self.config},
            'ALWAYS_RETRAIN': {'kind': 'always', 'config': self.config},
            'ORACLE_BASE':    {'kind': 'adaptive',    'config': base},
            'F_POLICY':       {'kind': 'adaptive_f',  'config': base},
            'FDT_POLICY':     {'kind': 'adaptive_fd', 'config': base},
        }

    def _calfix_specs(self) -> dict:
        """Two fixes for the dead-calibration-window failure, vs oracle.

        F_RECAL  : sparse re-calibration — refit F->invalidity every recal_interval
                   steps (so it learns the slope once invalidity appears). Uses a few
                   labels.
        F_TRIGGER: label-free change-detector — retrain when F rises trigger_k std devs
                   above its post-retrain baseline (std estimated from the initial
                   window; uses NO validity labels).
        """
        base = replace(self.config, use_valence=False, use_anxiety=False)
        return {
            'NEVER_RETRAIN':  {'kind': 'never',  'config': self.config},
            'ALWAYS_RETRAIN': {'kind': 'always', 'config': self.config},
            'ORACLE_BASE':    {'kind': 'adaptive',         'config': base},
            'F_RECAL':        {'kind': 'adaptive_f_recal', 'config': base},
            'F_TRIGGER':      {'kind': 'adaptive_f_trigger','config': base},
        }

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_calfix_single(self, dataset_name: str, drift_type: str,
                          seed: int, results_dir: str = None) -> dict:
        """Run a single calibration-fix comparison experiment."""
        results_dir = results_dir or './results_calfix'
        os.makedirs(results_dir, exist_ok=True)
        return self._run_temporal(dataset_name, drift_type, seed,
                                  self._calfix_specs(),
                                  results_dir=results_dir, file_suffix="_calfix",
                                  skip_existing=True)

    def run_labelfree_single(self, dataset_name: str, drift_type: str,
                             seed: int, results_dir: str = None) -> dict:
        """Run a single label-free F-policy comparison experiment."""
        results_dir = results_dir or './results_labelfree'
        os.makedirs(results_dir, exist_ok=True)
        return self._run_temporal(dataset_name, drift_type, seed,
                                  self._labelfree_specs(),
                                  results_dir=results_dir, file_suffix="_labelfree",
                                  skip_existing=True)

    def run_single(self, dataset_name: str, drift_type: str, seed: int) -> dict:
        """Run a single (dataset, drift_type, seed) experiment for the 4 strategies."""
        return self._run_temporal(dataset_name, drift_type, seed,
                                   self._standard_specs(),
                                   results_dir=self.results_dir, file_suffix="")

    def run_ablation_single(self, dataset_name: str, drift_type: str,
                            seed: int, results_dir: str = None) -> dict:
        """Run a single valence/anxiety ablation experiment."""
        results_dir = results_dir or './results_ablation'
        os.makedirs(results_dir, exist_ok=True)
        return self._run_temporal(dataset_name, drift_type, seed,
                                  self._ablation_specs(),
                                  results_dir=results_dir, file_suffix="_ablation",
                                  skip_existing=True)

    # ------------------------------------------------------------------
    # Core temporal loop (shared by standard + ablation runs)
    # ------------------------------------------------------------------

    def _run_temporal(self, dataset_name: str, drift_type: str, seed: int,
                      strategy_specs: dict, results_dir: str,
                      file_suffix: str = "", skip_existing: bool = False) -> dict:
        """Run the temporal drift loop for the given set of strategy specs.

        Returns dict with per-timestep metrics for every strategy in the spec.
        If skip_existing and the output file already exists and loads, it is
        returned unchanged (makes sweeps resumable after an interruption).
        """
        fname = f"{dataset_name}_{drift_type}_seed{seed}{file_suffix}.json"
        fpath = os.path.join(results_dir, fname)
        if skip_existing and os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    existing = json.load(f)
                print(f"[skip] {fname} already exists — reusing")
                return existing
            except (json.JSONDecodeError, OSError):
                print(f"[skip] {fname} exists but is unreadable — recomputing")

        set_seed(seed)
        print(f"\n{'='*60}")
        print(f"Dataset={dataset_name}  Drift={drift_type}  Seed={seed}")
        print(f"Strategies={list(strategy_specs)}")
        print(f"{'='*60}")

        cfg = self.config

        # ------------------------------------------------------------------
        # Step 1: Load data, train initial models
        # ------------------------------------------------------------------
        (train_ds, val_ds, test_ds,
         scaler, feat_names, class_names,
         lower, upper) = get_dataset(dataset_name)

        input_dim = train_ds.features.shape[1]
        num_classes = len(class_names)
        latent_dim = max(5, input_dim // 4)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        # Separate batch-1 VALIDATION loader for decision-time CF validity. Decisions
        # are made on this split; validity is REPORTED on the test split — so the
        # policy never sees the metric it is scored on (no evaluation leakage).
        val_eval_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # Train initial base model
        print("\n[Init] Training base MC Dropout model...")
        base_model = MCDropoutMLP(input_dim, num_classes).to(DEVICE)
        train_mc_dropout(base_model, train_loader, epochs=cfg.base_model_epochs)

        # Create initial Rashomon set
        print("[Init] Creating Rashomon set...")
        ensemble = MCDropoutRashomonSet(base_model, n_samples=cfg.n_ensemble)
        ensemble.evaluate_all(val_loader)
        print(f"  Rashomon set size (eps={cfg.epsilon}): "
              f"{ensemble.rashomon_size(cfg.epsilon)}")

        # Train initial CVAE
        print("[Init] Training Rashomon CVAE...")
        cvae_init = TabularCVAE(input_dim, num_classes, latent_dim).to(DEVICE)
        train_cvae_rashomon(cvae_init, train_loader, ensemble, cfg.epsilon,
                            epochs=cfg.cvae_epochs, name="Initial CVAE")

        # Initialize drift detector (tracks absolute drift from t=0)
        detector = DriftDetector(ensemble, cfg.epsilon, val_loader,
                                 n_samples=cfg.val_n_samples)

        # Shared free-energy policy (for the shared affective trace)
        policy = RecomputationPolicy(cfg)

        # Original training data (numpy) for drift simulation
        X_train_np = train_ds.features.numpy()
        y_train_np = train_ds.labels.numpy()
        rng = np.random.RandomState(seed)
        simulator = DriftSimulator(X_train_np, y_train_np, cfg, rng=rng)

        # ------------------------------------------------------------------
        # Strategy state: each maintains its own CVAE, detector, and policy
        # ------------------------------------------------------------------
        strategies = {}
        for name, spec in strategy_specs.items():
            scfg = spec.get('config') or cfg
            strategies[name] = {
                'kind': spec['kind'],
                'config': scfg,
                'cvae': copy.deepcopy(cvae_init),
                'detector': DriftDetector(ensemble, cfg.epsilon, val_loader,
                                          n_samples=cfg.val_n_samples),
                'policy': RecomputationPolicy(scfg),
                'retrain_count': 0,
                'last_retrain_t': -10**9,   # for the adaptive retrain cooldown
                'label_evals': 0,           # decision-time validity measurements used
                'metrics': defaultdict(list),
            }

        # Measure initial validity (t=0)
        init_validity = self._measure_validity(
            cvae_init, test_loader, base_model, lower, upper, cfg.n_test_samples)
        init_rash_val = self._measure_rashomon_validity(
            cvae_init, test_loader, base_model, ensemble, lower, upper,
            cfg.epsilon, cfg.n_test_samples)

        for s in strategies.values():
            s['metrics']['validity'].append(init_validity)
            s['metrics']['rashomon_validity'].append(init_rash_val)
            s['metrics']['D_t'].append(0.0)
            s['metrics']['eci'].append(0.0)
            s['metrics']['retrained'].append(False)
            s['metrics']['cumulative_retrains'].append(0)

        # Shared metrics (same for all strategies)
        shared_metrics = defaultdict(list)
        shared_metrics['D_t'].append(0.0)
        shared_metrics['eci_ground_truth'].append(0.0)

        # ------------------------------------------------------------------
        # Step 2: Temporal loop
        # ------------------------------------------------------------------
        F_prev = None

        for t in range(1, cfg.T + 1):
            t_start = time.time()
            print(f"\n--- t={t}/{cfg.T} ---")

            # Apply drift
            X_shifted, y_shifted = simulator.apply(drift_type, t)
            shifted_ds = TabularDataset(X_shifted, y_shifted)
            shifted_loader = DataLoader(shifted_ds, batch_size=64, shuffle=True)

            # Retrain base model on shifted data
            print(f"  Retraining base model on shifted data...")
            new_base = MCDropoutMLP(input_dim, num_classes).to(DEVICE)
            train_mc_dropout(new_base, shifted_loader,
                             epochs=cfg.base_model_epochs, name="Base (shifted)")

            # Create new Rashomon set
            new_ensemble = MCDropoutRashomonSet(new_base, n_samples=cfg.n_ensemble)
            new_ensemble.evaluate_all(val_loader)

            # Measure drift (absolute, from t=0)
            drift_result = detector.compute_drift(new_ensemble)
            D_t = drift_result['mean_kl']
            print(f"  D_t (mean KL) = {D_t:.6f}")

            # ECI of the STALE cached generator under the new posterior (drift
            # degradation diagnostic; not fed to any decision).
            eci_gt = RecomputationPolicy.compute_eci(
                cvae_init, new_ensemble, test_loader, lower, upper,
                cfg.epsilon, new_base, n_eval=cfg.n_test_samples)
            print(f"  ECI (stale, ground truth) = {eci_gt:.4f}")

            shared_metrics['D_t'].append(D_t)
            shared_metrics['eci_ground_truth'].append(eci_gt)
            shared_metrics['drift_result'].append(drift_result)

            # Compute free energy for the shared affective trace
            F_t = policy.compute_free_energy(
                cvae_init, new_ensemble, val_loader, lower, upper, cfg.epsilon)

            # ------------------------------------------------------------------
            # Run each strategy
            # ------------------------------------------------------------------
            for sname, sstate in strategies.items():
                retrained = self._strategy_step(
                    sname, sstate, t, new_ensemble, new_base, val_loader,
                    val_eval_loader, lower, upper, F_t)

                if retrained:
                    sstate['last_retrain_t'] = t

                # Retrain CVAE if the strategy decided to
                if retrained:
                    print(f"  [{sname}] RETRAINING CVAE at t={t}")
                    new_cvae = TabularCVAE(input_dim, num_classes, latent_dim).to(DEVICE)
                    train_cvae_rashomon(
                        new_cvae, shifted_loader, new_ensemble, cfg.epsilon,
                        epochs=cfg.retrain_epochs, name=f"CVAE ({sname})")
                    sstate['cvae'] = new_cvae
                    sstate['retrain_count'] += 1
                    # Update detector reference for this strategy
                    sstate['detector'].update_reference(new_ensemble)

                # Measure validity of this strategy's CVAE under new base model
                v = self._measure_validity(
                    sstate['cvae'], test_loader, new_base, lower, upper,
                    cfg.n_test_samples)
                rv = self._measure_rashomon_validity(
                    sstate['cvae'], test_loader, new_base, new_ensemble,
                    lower, upper, cfg.epsilon, cfg.n_test_samples)

                sstate['metrics']['validity'].append(v)
                sstate['metrics']['rashomon_validity'].append(rv)
                sstate['metrics']['D_t'].append(D_t)
                sstate['metrics']['eci'].append(eci_gt)
                sstate['metrics']['retrained'].append(retrained)
                sstate['metrics']['cumulative_retrains'].append(sstate['retrain_count'])

                print(f"  [{sname}] validity={v:.3f}  rash_val={rv:.3f}  "
                      f"retrained={retrained}  total_retrains={sstate['retrain_count']}")

            # Data-grounded Theorem 5.1 quantities, estimated on a FRESH generator
            # (the always-retrained CVAE) against the current posterior — the
            # well-specified case the bound is about. Evaluated on the validation
            # split, never the reported test split.
            fresh = strategies.get('ALWAYS_RETRAIN')
            fresh_cvae = fresh['cvae'] if fresh else cvae_init
            bound = RecomputationPolicy.estimate_theorem_bound(
                fresh_cvae, new_ensemble, val_eval_loader, lower, upper,
                cfg.epsilon, new_base, n_eval=cfg.n_test_samples,
                p_target=cfg.p_target)
            shared_metrics['sigma_sq'].append(bound['sigma_sq'])
            shared_metrics['beta'].append(bound['beta'])
            shared_metrics['tau_data'].append(bound['tau'])
            print(f"  [Thm5.1, fresh gen] sigma^2={bound['sigma_sq']:.4f}  "
                  f"beta={bound['beta']:.3f}  tau_data={bound['tau']:.4f}")

            F_prev = F_t
            # Do NOT update the main detector reference — we want cumulative drift from t=0
            elapsed = time.time() - t_start
            print(f"  Step time: {elapsed:.1f}s")

        # ------------------------------------------------------------------
        # Compile results
        # ------------------------------------------------------------------
        result = {
            'dataset': dataset_name,
            'drift_type': drift_type,
            'seed': seed,
            'config': {
                'T': cfg.T,
                'epsilon': cfg.epsilon,
                'n_ensemble': cfg.n_ensemble,
                'n_test_samples': cfg.n_test_samples,
                'p_target': cfg.p_target,
                'theoretical_threshold': cfg.theoretical_threshold,
            },
            'shared': {k: [_serialize(v) for v in vs]
                       for k, vs in shared_metrics.items()},
            'strategies': {},
        }

        for sname, sstate in strategies.items():
            result['strategies'][sname] = {
                'kind': sstate['kind'],
                'use_valence': sstate['config'].use_valence,
                'use_anxiety': sstate['config'].use_anxiety,
                'retrain_count': sstate['retrain_count'],
                'label_evals': sstate['label_evals'],
                'metrics': {k: [_serialize(v) for v in vs]
                            for k, vs in sstate['metrics'].items()},
            }

        # Save to disk
        fname = f"{dataset_name}_{drift_type}_seed{seed}{file_suffix}.json"
        fpath = os.path.join(results_dir, fname)
        with open(fpath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {fpath}")

        return result

    def _strategy_step(self, sname, sstate, t, new_ensemble, new_base,
                       val_loader, val_eval_loader, lower, upper, F_t) -> bool:
        """Execute one timestep of a single strategy; return whether it retrains."""
        kind = sstate['kind']
        scfg = sstate['config']

        if kind == 'never':
            return False

        if kind == 'always':
            return True

        if kind == 'fixed':
            return (t % scfg.fixed_interval_K == 0)

        if kind == 'adaptive':
            # Measure drift from this strategy's own reference (last retrain)
            adaptive_drift = sstate['detector'].compute_drift(new_ensemble)
            adaptive_D_t = adaptive_drift['mean_kl']

            # CURRENT-step validity of this strategy's cached CFs under the current
            # model, on the VALIDATION split. This is the un-lagged myopic signal:
            # at deployment you can run cached CFs through the just-updated model
            # right now. (Reported validity is measured separately, on test.)
            val_now = self._measure_validity(
                sstate['cvae'], val_eval_loader, new_base, lower, upper,
                self.config.n_test_samples)
            sstate['metrics']['val_validity'].append(val_now)

            # Free energy for the affective trace (uses this strategy's CVAE)
            adaptive_F_t = sstate['policy'].compute_free_energy(
                sstate['cvae'], new_ensemble, val_loader,
                lower, upper, scfg.epsilon)
            adaptive_F_prev = sstate.get('_F_prev', None)
            sstate['_F_prev'] = adaptive_F_t

            # Decide on the CURRENT validation-validity trend (no one-step lag)
            vt = sstate['metrics']['val_validity']
            decision = sstate['policy'].decide(
                adaptive_D_t, F_t=adaptive_F_t, F_prev=adaptive_F_prev,
                validity_trend=vt)

            # Log affective trace
            sstate['metrics']['G_reuse'].append(decision['G_reuse'])
            sstate['metrics']['G_retrain'].append(decision['G_retrain'])
            sstate['metrics']['valence'].append(decision['valence'])
            sstate['metrics']['anxiety'].append(decision['anxiety'])
            sstate['metrics']['adaptive_D_t'].append(adaptive_D_t)

            # Calibrate D_t -> invalidity link on validation invalidity (leakage-free)
            if len(sstate['metrics']['adaptive_D_t']) >= 3:
                Ds = sstate['metrics']['adaptive_D_t']
                Vs = [1.0 - v for v in sstate['metrics']['val_validity']]
                min_len = min(len(Ds), len(Vs))
                sstate['policy'].calibrate(Ds[:min_len], Vs[:min_len])

            # ORACLE reference: measured current validity is a label evaluation.
            sstate['label_evals'] += 1
            retrain = (decision['decision'] == 'RETRAIN')
            return self._apply_cooldown(sstate, t, scfg, retrain)

        if kind == 'adaptive_f':
            # Label-free: decide from free energy F (no validity labels), after a
            # short labelled window used only to calibrate F -> invalidity.
            F_t = sstate['policy'].compute_free_energy(
                sstate['cvae'], new_ensemble, val_loader, lower, upper, scfg.epsilon)
            tol = 1.0 - scfg.p_target
            in_calibration = t <= scfg.calibration_steps

            if in_calibration:
                # During calibration we DO spend labels (to fit F -> invalidity) and
                # decide on the measured invalidity, exactly like the oracle.
                val_now = self._measure_validity(
                    sstate['cvae'], val_eval_loader, new_base, lower, upper,
                    self.config.n_test_samples)
                sstate['label_evals'] += 1
                inv_now = 1.0 - val_now
                sstate['metrics']['F_cal_F'].append(F_t)
                sstate['metrics']['F_cal_inv'].append(inv_now)
                sstate['policy'].calibrate_F(sstate['metrics']['F_cal_F'],
                                             sstate['metrics']['F_cal_inv'])
                pred_inv = inv_now
            else:
                # Label-free decision: predict invalidity from F alone.
                pred_inv = sstate['policy'].predict_invalidity_from_F(F_t)
                if pred_inv is None:        # calibration failed -> conservative reuse
                    pred_inv = 0.0

            sstate['metrics']['F_t'].append(F_t)
            sstate['metrics']['pred_inv_F'].append(pred_inv)
            retrain = pred_inv > tol
            return self._apply_cooldown(sstate, t, scfg, retrain)

        if kind == 'adaptive_fd':
            # Label-free, BOTH signals: free energy F (good on abrupt drift) and
            # predictive KL drift D_t (good on gradual). Calibrate both F->inv and
            # D_t->inv on the initial labelled window, then decide on the worst case.
            F_t = sstate['policy'].compute_free_energy(
                sstate['cvae'], new_ensemble, val_loader, lower, upper, scfg.epsilon)
            D_t = sstate['detector'].compute_drift(new_ensemble)['mean_kl']
            tol = 1.0 - scfg.p_target

            if t <= scfg.calibration_steps:
                val_now = self._measure_validity(
                    sstate['cvae'], val_eval_loader, new_base, lower, upper,
                    self.config.n_test_samples)
                sstate['label_evals'] += 1
                inv_now = 1.0 - val_now
                sstate['metrics']['F_cal_F'].append(F_t)
                sstate['metrics']['F_cal_D'].append(D_t)
                sstate['metrics']['F_cal_inv'].append(inv_now)
                sstate['policy'].calibrate_F(sstate['metrics']['F_cal_F'],
                                             sstate['metrics']['F_cal_inv'])
                sstate['policy'].calibrate(sstate['metrics']['F_cal_D'],
                                           sstate['metrics']['F_cal_inv'])
                pred_inv = inv_now
            else:
                est = [e for e in (sstate['policy'].predict_invalidity_from_F(F_t),
                                   sstate['policy']._predict_invalidity_from_drift(D_t))
                       if e is not None]
                pred_inv = max(est) if est else 0.0

            sstate['metrics']['F_t'].append(F_t)
            sstate['metrics']['adaptive_D_t'].append(D_t)
            sstate['metrics']['pred_inv_F'].append(pred_inv)
            retrain = pred_inv > tol
            return self._apply_cooldown(sstate, t, scfg, retrain)

        if kind == 'adaptive_f_recal':
            # Sparse re-calibration: refit F->invalidity on the initial window AND
            # every recal_interval steps, so the fit updates once invalidity appears.
            F_t = sstate['policy'].compute_free_energy(
                sstate['cvae'], new_ensemble, val_loader, lower, upper, scfg.epsilon)
            tol = 1.0 - scfg.p_target
            is_label_step = (t <= scfg.calibration_steps) or (t % scfg.recal_interval == 0)
            if is_label_step:
                val_now = self._measure_validity(
                    sstate['cvae'], val_eval_loader, new_base, lower, upper,
                    self.config.n_test_samples)
                sstate['label_evals'] += 1
                inv_now = 1.0 - val_now
                sstate['metrics']['F_cal_F'].append(F_t)
                sstate['metrics']['F_cal_inv'].append(inv_now)
                sstate['policy'].calibrate_F(sstate['metrics']['F_cal_F'],
                                             sstate['metrics']['F_cal_inv'])
                pred_inv = inv_now
            else:
                pred_inv = sstate['policy'].predict_invalidity_from_F(F_t)
                if pred_inv is None:
                    pred_inv = 0.0
            sstate['metrics']['F_t'].append(F_t)
            sstate['metrics']['pred_inv_F'].append(pred_inv)
            return self._apply_cooldown(sstate, t, scfg, pred_inv > tol)

        if kind == 'adaptive_f_trigger':
            # Fully label-free change-detector on free energy. No validity labels:
            # estimate F's noise std over the initial window, then retrain when F
            # exceeds its post-retrain baseline by trigger_k std devs.
            F_t = sstate['policy'].compute_free_energy(
                sstate['cvae'], new_ensemble, val_loader, lower, upper, scfg.epsilon)
            sstate['metrics']['F_t'].append(F_t)
            win = sstate['metrics']['F_t']

            # Reset baseline to the fresh F right after a retrain.
            if t - 1 == sstate['last_retrain_t']:
                sstate['F_baseline'] = F_t

            if t <= scfg.calibration_steps:
                if t == scfg.calibration_steps:    # end of window: set noise + baseline
                    sstate['F_sigma'] = float(np.std(win)) or 1e-6
                    sstate.setdefault('F_baseline', float(np.mean(win)))
                return False                        # observe only during window
            thresh = sstate['F_baseline'] + scfg.trigger_k * sstate.get('F_sigma', 1e-6)
            sstate['metrics']['pred_inv_F'].append(F_t - sstate['F_baseline'])
            return self._apply_cooldown(sstate, t, scfg, F_t > thresh)

        raise ValueError(f"Unknown strategy kind: {kind}")

    @staticmethod
    def _apply_cooldown(sstate, t, scfg, retrain):
        """Deny a retrain that fires inside the refractory cooldown window."""
        cd = scfg.retrain_cooldown
        if retrain and cd > 0 and (t - sstate['last_retrain_t']) <= cd:
            sstate['metrics']['denied_by_cooldown'].append(t)
            return False
        return retrain

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    def run_sweep(self, datasets=None, drift_types=None, seeds=None) -> list:
        """Run full sweep: datasets x drift_types x seeds (standard strategies)."""
        return self._run_sweep(self.run_single, 'all_results.json',
                               datasets, drift_types, seeds, self.results_dir)

    def run_ablation_sweep(self, datasets=None, drift_types=None, seeds=None,
                           results_dir: str = './results_ablation') -> list:
        """Run full valence/anxiety ablation sweep."""
        os.makedirs(results_dir, exist_ok=True)
        runner = lambda ds, dt, seed: self.run_ablation_single(ds, dt, seed,
                                                               results_dir=results_dir)
        return self._run_sweep(runner, 'all_ablation_results.json',
                               datasets, drift_types, seeds, results_dir)

    def _run_sweep(self, run_fn, combined_name, datasets, drift_types,
                   seeds, results_dir) -> list:
        cfg = self.config
        datasets = datasets or cfg.datasets
        drift_types = drift_types or cfg.drift_types
        seeds = seeds or cfg.seeds

        all_results = []
        total = len(datasets) * len(drift_types) * len(seeds)
        idx = 0

        for ds in datasets:
            for dt in drift_types:
                for seed in seeds:
                    idx += 1
                    print(f"\n{'#'*60}")
                    print(f"# Experiment {idx}/{total}")
                    print(f"{'#'*60}")
                    all_results.append(run_fn(ds, dt, seed))

        combined_path = os.path.join(results_dir, combined_name)
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {combined_path}")

        return all_results

    # ------------------------------------------------------------------
    # Validity measurement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_validity(cvae, test_loader, base_model, lower, upper,
                          n_eval: int) -> float:
        """Fraction of CFs that are valid under base model."""
        cvae.eval()
        valid = 0
        total = 0

        for x, y in test_loader:
            if total >= n_eval:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                base_model.train()
                outs = torch.stack([base_model(x) for _ in range(20)]).mean(0)
                pred = outs.max(1)[1].item()
                if pred != y.item():
                    continue

                target_cf = 1 - pred
                cf, _ = generate_cf_amortized(cvae, x, target_cf, lower, upper)
                if cf is None or not torch.all(torch.isfinite(cf)):
                    continue

                valid += validity(cf, target_cf, base_model)
                total += 1

        return valid / max(total, 1)

    @staticmethod
    def _measure_rashomon_validity(cvae, test_loader, base_model, ensemble,
                                    lower, upper, epsilon: float,
                                    n_eval: int) -> float:
        """Mean Rashomon validity ratio across test samples."""
        cvae.eval()
        ratios = []
        total = 0

        for x, y in test_loader:
            if total >= n_eval:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                base_model.train()
                outs = torch.stack([base_model(x) for _ in range(20)]).mean(0)
                pred = outs.max(1)[1].item()
                if pred != y.item():
                    continue

                target_cf = 1 - pred
                cf, _ = generate_cf_amortized(cvae, x, target_cf, lower, upper)
                if cf is None or not torch.all(torch.isfinite(cf)):
                    continue

                r = rashomon_validity_ratio(cf, target_cf, ensemble, epsilon)
                ratios.append(r)
                total += 1

        return float(np.mean(ratios)) if ratios else 0.0


def _serialize(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _serialize(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize(vv) for vv in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str, type(None))):
        return v
    return str(v)
