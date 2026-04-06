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
     - Run 4 strategies: NEVER_RETRAIN, ALWAYS_RETRAIN, ADAPTIVE, FIXED_INTERVAL
     - Log per-timestep metrics
"""

import os
import json
import time
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

    def run_single(self, dataset_name: str, drift_type: str, seed: int) -> dict:
        """Run a single (dataset, drift_type, seed) experiment.

        Returns dict with per-timestep metrics for all 4 strategies.
        """
        set_seed(seed)
        print(f"\n{'='*60}")
        print(f"Dataset={dataset_name}  Drift={drift_type}  Seed={seed}")
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

        # Initialize drift detector
        detector = DriftDetector(ensemble, cfg.epsilon, val_loader,
                                 n_samples=cfg.val_n_samples)

        # Initialize recomputation policy
        policy = RecomputationPolicy(cfg)

        # Original training data (numpy) for drift simulation
        X_train_np = train_ds.features.numpy()
        y_train_np = train_ds.labels.numpy()
        rng = np.random.RandomState(seed)
        simulator = DriftSimulator(X_train_np, y_train_np, cfg, rng=rng)

        # ------------------------------------------------------------------
        # Strategy state: each maintains its own CVAE (and detector for ADAPTIVE)
        # ------------------------------------------------------------------
        import copy

        strategies = {}
        for name in cfg.strategies:
            strategies[name] = {
                'cvae': copy.deepcopy(cvae_init),
                'detector': DriftDetector(ensemble, cfg.epsilon, val_loader,
                                          n_samples=cfg.val_n_samples),
                'policy': RecomputationPolicy(cfg),
                'retrain_count': 0,
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

            # Measure drift
            drift_result = detector.compute_drift(new_ensemble)
            D_t = drift_result['mean_kl']
            print(f"  D_t (mean KL) = {D_t:.6f}")

            # Measure actual ECI (ground truth — uses original CVAE)
            eci_gt = RecomputationPolicy.compute_eci(
                cvae_init, new_ensemble, test_loader, lower, upper,
                cfg.epsilon, new_base, n_eval=cfg.n_test_samples)
            print(f"  ECI (ground truth) = {eci_gt:.4f}")

            shared_metrics['D_t'].append(D_t)
            shared_metrics['eci_ground_truth'].append(eci_gt)
            shared_metrics['drift_result'].append(drift_result)

            # Compute free energy for affective trace
            F_t = policy.compute_free_energy(
                cvae_init, new_ensemble, val_loader, lower, upper, cfg.epsilon)

            # ------------------------------------------------------------------
            # Run each strategy
            # ------------------------------------------------------------------
            for sname, sstate in strategies.items():
                retrained = False

                if sname == 'NEVER_RETRAIN':
                    pass  # never retrain

                elif sname == 'ALWAYS_RETRAIN':
                    retrained = True

                elif sname == 'ADAPTIVE':
                    # Measure drift from ADAPTIVE's own reference (last retrain)
                    adaptive_drift = sstate['detector'].compute_drift(new_ensemble)
                    adaptive_D_t = adaptive_drift['mean_kl']

                    # Compute F_t using ADAPTIVE's current CVAE (not initial)
                    adaptive_F_t = sstate['policy'].compute_free_energy(
                        sstate['cvae'], new_ensemble, val_loader,
                        lower, upper, cfg.epsilon)
                    adaptive_F_prev = sstate.get('_F_prev', None)
                    sstate['_F_prev'] = adaptive_F_t

                    # Use the policy to decide
                    vt = sstate['metrics']['validity']
                    decision = sstate['policy'].decide(
                        adaptive_D_t, F_t=adaptive_F_t, F_prev=adaptive_F_prev,
                        validity_trend=vt)
                    retrained = (decision['decision'] == 'RETRAIN')

                    # Log affective trace
                    sstate['metrics']['G_reuse'].append(decision['G_reuse'])
                    sstate['metrics']['G_retrain'].append(decision['G_retrain'])
                    sstate['metrics']['valence'].append(decision['valence'])
                    sstate['metrics']['anxiety'].append(decision['anxiety'])
                    sstate['metrics']['adaptive_D_t'].append(adaptive_D_t)

                    # Calibrate policy with accumulated data
                    if len(sstate['metrics']['adaptive_D_t']) >= 3:
                        Ds = sstate['metrics']['adaptive_D_t']
                        Vs = [1.0 - v for v in sstate['metrics']['validity'][1:]]
                        min_len = min(len(Ds), len(Vs))
                        sstate['policy'].calibrate(Ds[:min_len], Vs[:min_len])

                elif sname == 'FIXED_INTERVAL':
                    retrained = (t % cfg.fixed_interval_K == 0)

                # Retrain CVAE if needed
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

            F_prev = F_t
            # Update main detector reference (tracks absolute drift)
            # Do NOT update — we want cumulative drift from t=0
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
            },
            'shared': {k: [_serialize(v) for v in vs]
                       for k, vs in shared_metrics.items()},
            'strategies': {},
        }

        for sname, sstate in strategies.items():
            result['strategies'][sname] = {
                'retrain_count': sstate['retrain_count'],
                'metrics': {k: [_serialize(v) for v in vs]
                            for k, vs in sstate['metrics'].items()},
            }

        # Save to disk
        fname = f"{dataset_name}_{drift_type}_seed{seed}.json"
        fpath = os.path.join(self.results_dir, fname)
        with open(fpath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {fpath}")

        return result

    def run_sweep(self, datasets=None, drift_types=None, seeds=None) -> list:
        """Run full sweep: datasets x drift_types x seeds.

        Returns list of all result dicts.
        """
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
                    result = self.run_single(ds, dt, seed)
                    all_results.append(result)

        # Save combined results
        combined_path = os.path.join(self.results_dir, 'all_results.json')
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
