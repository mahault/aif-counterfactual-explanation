"""
Principled REUSE/RETRAIN policy for adaptive counterfactual recomputation.

Decision rule (decision-theoretic, no tuned cost weights):

    RETRAIN  iff  predicted_invalidity > (1 - p_target)

i.e. recompute exactly when the cached generator is predicted to exceed the
invalidity we are willing to tolerate. `p_target` is a stated requirement, not a
fitted hyperparameter — it is the *only* knob, and the threshold (1 - p_target) is
its direct, interpretable consequence.

`predicted_invalidity` is the worst case over up to three signals, all expressed in
the same invalidity units in [0, 1], so the affective-term ablation is a fair
comparison at a SHARED threshold (differences reflect signal quality, not a
different appetite for compute):

  inv_now      Myopic core. Invalidity already realised by the cached CFs under the
               current model. Computable at deployment by running cached CFs through
               the current model — no ground-truth labels required.
  inv_kl       "Anxiety": drift-based anticipation. Calibrated linear map from the
               predictive KL drift D_t to invalidity, fit online from observed
               (D_t, invalidity) pairs. No minimum-slope floor: if drift carries no
               signal (e.g. saturated/abrupt drift) the fit is flat and this term
               simply never fires. Returns None until calibrated (>=3 points with
               drift variation) so the policy never acts on an un-calibrated signal.
  inv_forecast "Valence": trend-based anticipation. One-step extrapolation of the
               realised invalidity using its rate of change — the discrete analogue
               of -dF (Joffily & Coricelli, 2013).

Ablation flags (config.use_valence / use_anxiety) toggle the two anticipatory
signals on top of the always-on myopic core.

This module also estimates the Theorem 5.1 quantities (sigma^2, beta) from the
Rashomon predictive spread so the validity bound / threshold is data-grounded
rather than set from placeholder constants.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_avcg.avcg_core import (
    DEVICE, generate_cf_amortized, validity, rashomon_validity_ratio
)
from adaptive_avcg.config import ExperimentConfig


class RecomputationPolicy:
    """Decision-theoretic recomputation rule with data-calibrated drift link."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._F_history = []
        self._D_history = []
        self._cal_slope = None       # calibrated D_t -> invalidity slope
        self._cal_intercept = 0.0
        self._F_cal_slope = None     # calibrated F (free energy) -> invalidity
        self._F_cal_intercept = 0.0
        self._steps_since_retrain = 0
        self._total_steps = 0

    # ------------------------------------------------------------------
    # Free energy (used for the affective trace; not a tuned decision cost)
    # ------------------------------------------------------------------

    def compute_free_energy(self, cached_cvae, new_ensemble, val_loader,
                            lower, upper, epsilon: float,
                            n_eval: int = 30) -> float:
        """Re-evaluate the AVCG free energy of the cached generator under the new
        Rashomon posterior: F_t = -E[expected_log_prob_rashomon] + prox term."""
        cached_cvae.eval()
        total_loss = 0.0
        count = 0
        for x, y in val_loader:
            if count >= n_eval:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                y_t = 1 - y
                mu, logvar = cached_cvae.encode(x, y_t)
                z = cached_cvae.reparameterize(mu, logvar)
                x_prime = cached_cvae.decode(z, y_t)
                exp_lp = new_ensemble.expected_log_prob_rashomon(x_prime, y_t, epsilon)
                proximity = F.mse_loss(x_prime, x, reduction='mean')
                total_loss += -exp_lp.item() + 0.1 * proximity.item()
                count += len(y)
        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Calibrated drift -> invalidity link (honest least-squares, no floor)
    # ------------------------------------------------------------------

    def calibrate(self, drift_values: list, invalidity_values: list):
        """Fit invalidity ~= a + b * D_t from observed pairs. No minimum-slope
        floor — a flat or negative fit (drift carries no invalidity signal) is left
        as-is, so the drift term will not spuriously trigger recomputation."""
        if len(drift_values) < 3:
            return
        D = np.asarray(drift_values, dtype=float)
        V = np.asarray(invalidity_values, dtype=float)
        if np.std(D) < 1e-10:
            return  # no drift variation yet -> cannot fit; stay un-calibrated
        b, a = np.polyfit(D, V, 1)   # V ~= a + b*D
        self._cal_slope = float(b)
        self._cal_intercept = float(a)

    def _predict_invalidity_from_drift(self, D_t: float):
        """Calibrated drift-based invalidity estimate, or None if un-calibrated."""
        if self._cal_slope is None:
            return None
        return float(np.clip(self._cal_intercept + self._cal_slope * D_t, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Label-free free-energy -> invalidity calibration (F is computable
    # without validity labels; diagnosed r~0.8-0.9, AUC~0.95 vs invalidity)
    # ------------------------------------------------------------------

    def calibrate_F(self, F_values: list, invalidity_values: list):
        """Fit invalidity ~= a + b*F from a short labelled calibration window.
        After this, decisions use only F (label-free)."""
        if len(F_values) < 3:
            return
        F = np.asarray(F_values, dtype=float)
        V = np.asarray(invalidity_values, dtype=float)
        if np.std(F) < 1e-9:
            return
        b, a = np.polyfit(F, V, 1)
        self._F_cal_slope = float(b)
        self._F_cal_intercept = float(a)

    def predict_invalidity_from_F(self, F_t: float):
        """Label-free invalidity estimate from free energy, or None if un-calibrated."""
        if self._F_cal_slope is None:
            return None
        return float(np.clip(self._F_cal_intercept + self._F_cal_slope * F_t, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, D_t: float, F_t: float = None, F_prev: float = None,
               validity_trend: list = None) -> dict:
        """RETRAIN iff predicted invalidity exceeds tolerance (1 - p_target)."""
        tol = 1.0 - self.config.p_target

        # Myopic core: invalidity already realised by the cached generator.
        if validity_trend:
            inv_now = 1.0 - validity_trend[-1]
            inv_prev = (1.0 - validity_trend[-2]) if len(validity_trend) >= 2 else inv_now
        else:
            inv_now = inv_prev = 0.0

        # Anxiety: drift-based anticipation (calibrated KL link).
        inv_kl = self._predict_invalidity_from_drift(D_t)

        # Valence: trend-based anticipation (one-step -dF extrapolation).
        rate = inv_now - inv_prev                          # >0 => worsening
        inv_forecast = float(np.clip(inv_now + max(0.0, rate), 0.0, 1.0))
        valence = (-(F_t - F_prev)
                   if (F_t is not None and F_prev is not None) else None)

        # Assemble predicted invalidity under continued REUSE.
        signals = [inv_now]
        if self.config.use_anxiety and inv_kl is not None:
            signals.append(inv_kl)
        if self.config.use_valence:
            signals.append(inv_forecast)
        predicted_inv = max(signals)

        decision = 'RETRAIN' if predicted_inv > tol else 'REUSE'

        self._total_steps += 1
        if decision == 'RETRAIN':
            self._steps_since_retrain = 0
        else:
            self._steps_since_retrain += 1
        self._D_history.append(D_t)
        if F_t is not None:
            self._F_history.append(F_t)

        return {
            'decision': decision,
            'predicted_invalidity': predicted_inv,
            'tolerance': tol,
            'inv_now': inv_now,
            'inv_kl': inv_kl,
            'inv_forecast': inv_forecast,
            # Trace keys (kept for plotting/back-compat); semantics are invalidity
            # units now: G_reuse = predicted invalidity, G_retrain = tolerance.
            'G_reuse': predicted_inv,
            'G_retrain': tol,
            'valence': valence,
            'anxiety': (inv_kl if inv_kl is not None else 0.0),
            'F_t': F_t,
            'D_t': D_t,
            'threshold_empirical': predicted_inv > tol,
            'use_valence': self.config.use_valence,
            'use_anxiety': self.config.use_anxiety,
        }

    # ------------------------------------------------------------------
    # Ground-truth invalidity + data-grounded Theorem 5.1 quantities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_eci(cached_cvae, new_ensemble, eval_loader, lower, upper,
                    epsilon: float, base_model, n_eval: int = 60) -> float:
        """Expected counterfactual invalidity of CACHED (stale) CEs under the new
        posterior: fraction of cached CFs that are INVALID (1 - validity). Used as a
        drift-degradation signal; not the Theorem 5.1 quantity (see below)."""
        cached_cvae.eval()
        valid_count = 0
        total = 0
        for x, y in eval_loader:
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
                cf, _ = generate_cf_amortized(cached_cvae, x, target_cf, lower, upper)
                if cf is None or not torch.all(torch.isfinite(cf)):
                    continue
                v_base = validity(cf, target_cf, base_model)
                v_rash = rashomon_validity_ratio(cf, target_cf, new_ensemble, epsilon)
                valid_count += int(v_base == 1 and v_rash >= 0.5)
                total += 1
        if total == 0:
            return 0.0
        return 1.0 - valid_count / total

    @staticmethod
    def estimate_theorem_bound(cvae, ensemble, eval_loader, lower, upper,
                               epsilon: float, base_model, n_eval: int = 60,
                               p_target: float = 0.90) -> dict:
        """Data-grounded Theorem 5.1 / Corollary 3.5 quantities.

        Estimated for CFs generated by `cvae` against the SAME `ensemble` they are
        evaluated under — i.e. the well-specified (non-stale) case the theorem is
        about. Pass a FRESH generator (e.g. the always-retrained CVAE); estimating
        these from a stale generator under drifted posterior measures degradation,
        not the bound, and gives a meaningless (often negative) margin.

            sigma_sq : mean over CFs of Var_{theta in Rashomon}[ P_theta(y'|cf) ]
            beta     : mean over CFs of the margin  E_R[P_theta(y'|cf)] - 0.5
            tau      : first-order admissible drift. Cor 3.5 gives validity >~
                       1 - sigma^2/beta^2; solving for p_target:
                       tau = beta^2 * (1 - p_target) - sigma^2.
                       tau <= 0 => bound is vacuous (cannot certify p_target even at
                       zero drift) — reported honestly. beta <= 0 => CFs not even
                       valid in-sample, theorem inapplicable.
            n        : number of CFs evaluated
        """
        cvae.eval()
        members = ensemble.get_rashomon_models(epsilon)
        sigmas, betas = [], []
        total = 0
        for x, y in eval_loader:
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
                probs = np.asarray(
                    [torch.exp(m(cf))[0, target_cf].item() for m in members],
                    dtype=float)
                sigmas.append(float(probs.var()))
                betas.append(float(probs.mean() - 0.5))
                total += 1
        if total == 0:
            return {'sigma_sq': 0.0, 'beta': 0.0, 'tau': 0.0, 'n': 0}
        sigma_sq = float(np.mean(sigmas))
        beta = float(np.mean(betas))
        tau = beta ** 2 * (1.0 - p_target) - sigma_sq
        return {'sigma_sq': sigma_sq, 'beta': beta, 'tau': float(tau), 'n': total}
