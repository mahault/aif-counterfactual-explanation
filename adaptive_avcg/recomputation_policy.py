"""
EFE-based REUSE/RETRAIN policy selection grounded in the Free Energy Principle.

Two-policy active inference: pi in {REUSE, RETRAIN}.

The decision rule IS active inference policy selection (Section 4 of paper scaffold):
- Free energy F_t = AVCG objective G(x) evaluated under current Rashomon posterior P_R^t.
  When the posterior drifts, F_t increases (cached CVAE is suboptimal for new posterior).
- Valence = -dF/dt (Joffily & Coricelli, 2013). Negative valence = system is deteriorating.
- Anxiety = anticipated future F increase under REUSE. When anxiety exceeds threshold,
  the system triggers RETRAIN.
- Expected Free Energy for each policy:
    G(REUSE) = lambda_inv * ECI_t (estimated from D_t)
    G(RETRAIN) = C_retrain (fixed computational cost)
- Decision: RETRAIN when G(REUSE) > G(RETRAIN)

Produces a full affective inference trace: F_t, valence, anxiety, G(REUSE), G(RETRAIN), decision.
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
    """Active inference recomputation decision rule."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._F_history = []          # free energy trace
        self._D_history = []          # drift (mean_kl) trace
        self._validity_history = []   # actual validity trace
        self._empirical_slope = None  # calibrated from (D_t, validity_drop) pairs
        self._steps_since_retrain = 0  # steps since last retrain decision
        self._total_steps = 0          # total decide() calls (for warm-up)

    # ------------------------------------------------------------------
    # Free Energy computation
    # ------------------------------------------------------------------

    def compute_free_energy(self, cached_cvae, new_ensemble, val_loader,
                            lower, upper, epsilon: float,
                            n_eval: int = 30) -> float:
        """Re-evaluate AVCG loss under new Rashomon posterior.

        F_t = -E[expected_log_prob_rashomon(cf)] + lambda_prox * proximity
        averaged over a batch of val samples.
        """
        cached_cvae.eval()
        total_loss = 0.0
        count = 0

        for x, y in val_loader:
            if count >= n_eval:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            target_cf = 1 - y

            with torch.no_grad():
                y_t = target_cf
                mu, logvar = cached_cvae.encode(x, y_t)
                z = cached_cvae.reparameterize(mu, logvar)
                x_prime = cached_cvae.decode(z, y_t)

                # Rashomon expected log prob under NEW ensemble
                exp_lp = new_ensemble.expected_log_prob_rashomon(
                    x_prime, y_t, epsilon)
                proximity = F.mse_loss(x_prime, x, reduction='mean')

                # AVCG loss (higher = worse)
                loss = -exp_lp.item() + 0.1 * proximity.item()
                total_loss += loss
                count += len(y)

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Affective quantities
    # ------------------------------------------------------------------

    def compute_valence(self, F_t: float, F_prev: float) -> float:
        """Valence = -dF. Negative valence means system is deteriorating."""
        return -(F_t - F_prev)

    def compute_anxiety(self, D_t: float, validity_trend: list) -> float:
        """Projected future invalidity if REUSE continues.

        anxiety_t = E[F_{t+1} | pi=REUSE] - F_t
        Approximated as: current drift rate * projected validity loss.
        """
        if len(validity_trend) < 2:
            return D_t * self.config.invalidity_cost_weight

        # Linear extrapolation of validity decline
        recent = validity_trend[-3:]  # last 3 steps
        if len(recent) >= 2:
            slopes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            avg_slope = np.mean(slopes)
            # Anxiety = anticipated validity drop scaled by cost weight
            projected_drop = max(0.0, -avg_slope)  # positive when validity declining
            return (D_t + projected_drop) * self.config.invalidity_cost_weight
        return D_t * self.config.invalidity_cost_weight

    # ------------------------------------------------------------------
    # Expected Free Energy for each policy
    # ------------------------------------------------------------------

    def compute_efe(self, policy: str, D_t: float, retrain_cost: float = None) -> float:
        """G(pi) for each policy.

        G(REUSE) = lambda_inv * ECI_t, where ECI estimated from D_t
        G(RETRAIN) = C_retrain (fixed)
        """
        if retrain_cost is None:
            retrain_cost = self.config.retrain_cost

        if policy == 'REUSE':
            # ECI estimated from drift via calibrated or linear relationship
            eci_estimate = self._estimate_eci_from_drift(D_t)
            return self.config.invalidity_cost_weight * eci_estimate
        elif policy == 'RETRAIN':
            return retrain_cost
        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _estimate_eci_from_drift(self, D_t: float) -> float:
        """Map D_t to expected counterfactual invalidity.

        If calibrated: use empirical slope from (D_t, validity_drop) pairs.
        Otherwise: linear approximation ECI ~ D_t.
        Scales with steps since last retrain to capture cumulative staleness.
        """
        # Base ECI from drift signal
        if self._empirical_slope is not None:
            base = self._empirical_slope * D_t
        else:
            base = D_t

        # Scale by staleness: more steps since retrain = higher expected invalidity
        staleness = 1.0 + 0.1 * self._steps_since_retrain
        return base * staleness

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, drift_values: list, validity_drops: list):
        """Calibrate ECI estimate from (D_t, validity_drop) pairs via linear regression.

        validity_drop = 1 - actual_validity (higher = more invalid).
        """
        if len(drift_values) < 3:
            return

        D = np.array(drift_values)
        V = np.array(validity_drops)

        # Linear regression: V = slope * D + intercept
        if np.std(D) > 1e-10:
            slope = np.corrcoef(D, V)[0, 1] * np.std(V) / np.std(D)
            # Enforce minimum slope: D_t always has a noise floor,
            # so a near-zero slope means calibration hasn't seen enough stale data yet
            self._empirical_slope = max(slope, 0.2)
        else:
            self._empirical_slope = 1.0

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, D_t: float, F_t: float = None, F_prev: float = None,
               validity_trend: list = None) -> dict:
        """Make REUSE/RETRAIN decision with full affective trace.

        Returns dict with:
            decision: 'REUSE' or 'RETRAIN'
            G_reuse: EFE of REUSE
            G_retrain: EFE of RETRAIN
            valence: -dF (if F_t, F_prev provided)
            anxiety: projected future cost
            F_t: free energy (if provided)
            D_t: drift value
            threshold_empirical: G(REUSE) > G(RETRAIN)
            threshold_theoretical: D_t > tau from Theorem 5.1
        """
        G_reuse = self.compute_efe('REUSE', D_t)
        G_retrain = self.compute_efe('RETRAIN', D_t)

        # Compute affect
        valence = None
        if F_t is not None and F_prev is not None:
            valence = self.compute_valence(F_t, F_prev)

        anxiety = 0.0
        if validity_trend is not None:
            anxiety = self.compute_anxiety(D_t, validity_trend)

        # Incorporate anxiety into G(REUSE): anticipated future cost of continued reuse
        G_reuse += anxiety

        # Valence signal: if free energy is increasing (negative valence), system is
        # deteriorating — add the magnitude as urgency signal to G(REUSE)
        if valence is not None and valence < 0:
            G_reuse += abs(valence) * self.config.invalidity_cost_weight

        # Validity-based penalty: if last observed validity is below target,
        # the current CVAE is already failing — strong signal to retrain
        if validity_trend and len(validity_trend) >= 2:
            last_val = validity_trend[-1]
            if last_val < self.config.p_target:
                G_reuse += (self.config.p_target - last_val) * self.config.invalidity_cost_weight

        # Warm-up: force REUSE for first K steps to observe drift–validity relationship
        in_warmup = self._total_steps < self.config.warmup_steps

        # Decision: RETRAIN when anticipated invalidity exceeds retraining cost
        if in_warmup:
            decision = 'REUSE'  # observe only during warm-up
        else:
            decision = 'RETRAIN' if G_reuse > G_retrain else 'REUSE'

        # Update counters
        self._total_steps += 1
        if decision == 'RETRAIN':
            self._steps_since_retrain = 0
        else:
            self._steps_since_retrain += 1

        # Track history
        self._D_history.append(D_t)
        if F_t is not None:
            self._F_history.append(F_t)

        return {
            'decision': decision,
            'G_reuse': G_reuse,
            'G_retrain': G_retrain,
            'valence': valence,
            'anxiety': anxiety,
            'F_t': F_t,
            'D_t': D_t,
            'threshold_empirical': G_reuse > G_retrain,
            'threshold_theoretical': D_t > self.config.theoretical_threshold,
        }

    # ------------------------------------------------------------------
    # Expected Counterfactual Invalidity (ground truth)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_eci(cached_cvae, new_ensemble, test_loader, lower, upper,
                    epsilon: float, base_model, n_eval: int = 60) -> float:
        """Expected counterfactual invalidity of cached CEs under new posterior.

        Ground truth signal: generate CFs with cached CVAE, check validity
        against NEW Rashomon set and base model.

        Returns: fraction of CFs that are INVALID (1 - validity).
        """
        cached_cvae.eval()
        valid_count = 0
        total = 0

        for x, y in test_loader:
            if total >= n_eval:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                # Check prediction is correct
                base_model.train()
                outs = torch.stack([base_model(x) for _ in range(20)]).mean(0)
                pred = outs.max(1)[1].item()
                if pred != y.item():
                    continue

                target_cf = 1 - pred
                cf, _ = generate_cf_amortized(cached_cvae, x, target_cf, lower, upper)
                if cf is None or not torch.all(torch.isfinite(cf)):
                    continue

                # Validity under base model
                v_base = validity(cf, target_cf, base_model)

                # Rashomon validity under new ensemble
                v_rash = rashomon_validity_ratio(cf, target_cf, new_ensemble, epsilon)

                # Combined: valid if both base model AND majority of Rashomon set agree
                valid_count += int(v_base == 1 and v_rash >= 0.5)
                total += 1

        if total == 0:
            return 0.0
        return 1.0 - valid_count / total  # invalidity = 1 - validity
