"""
All experiment hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExperimentConfig:
    # Datasets (4 total, matching the AVCG experimental suite)
    datasets: List[str] = field(
        default_factory=lambda: ['credit', 'adult_income', 'spambase', 'pneumonia'])

    # Temporal parameters
    T: int = 20                          # number of timesteps
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55, 66, 77, 88, 99, 101)  # 10 seeds

    # AVCG parameters
    epsilon: float = 0.2
    n_ensemble: int = 50
    n_test_samples: int = 60
    cvae_epochs: int = 50
    base_model_epochs: int = 50
    retrain_epochs: int = 50             # epochs when retraining CVAE

    # Drift parameters — calibrated so NEVER_RETRAIN drops to ~50-60% validity by T=20
    # Covariate shift: Gaussian noise with sigma = sigma_base * t on ALL features
    covariate_sigma_base: float = 0.25   # 5x stronger: sigma=5.0 at t=20
    covariate_feature_frac: float = 1.0  # all features, not 50%

    # Label noise: flip alpha_base * t percent of labels (capped at alpha_cap)
    label_alpha_base: float = 0.025      # 2.5% per step → 50% at T=20
    label_alpha_cap: float = 0.45        # cap at 45% (near random)

    # Subpopulation shift: undersample class 0 by sub_rate * t (floor at sub_floor)
    sub_rate: float = 0.05               # 5% per step
    sub_floor: float = 0.15              # more aggressive floor (15%)

    # Rotation drift: rotate first 2 PCA components by rotation_deg * t degrees
    rotation_deg: float = 5.0            # 100° total at T=20

    # Recomputation policy.
    # The principled rule retrains iff predicted invalidity > (1 - p_target); its only
    # knob is p_target. The three constants below are DEPRECATED — they belonged to the
    # old tuned EFE rule (retrain_cost / invalidity_cost_weight / staleness / warmup) and
    # no longer affect the decision. Kept only so old result files still deserialize.
    retrain_cost: float = 0.5            # DEPRECATED (old EFE cost)
    invalidity_cost_weight: float = 5.0  # DEPRECATED (old EFE lambda_inv)
    warmup_steps: int = 3                # DEPRECATED (old forced-REUSE warm-up)
    fixed_interval_K: int = 5            # retrain every K steps for FIXED_INTERVAL strategy

    # Retrain budget: minimum steps between retrains for the ADAPTIVE variants
    # (refractory cooldown). 0 = unconstrained (retrain whenever the rule fires).
    # >0 imposes a compute budget: at most ~T/(cooldown+1) retrains, so *when* you
    # spend a retrain matters — this is the regime where anticipation can pay off.
    # Sweeping this traces the validity-vs-compute Pareto frontier.
    retrain_cooldown: int = 0

    # Label-free F-policy: number of initial steps with validity labels used to
    # calibrate the free-energy -> invalidity map, after which decisions use only F.
    calibration_steps: int = 5
    # Calibration fixes for the gradual-drift dead-window failure:
    recal_interval: int = 3       # F_RECAL: spend a label every this many steps to refit
    trigger_k: float = 3.0        # F_TRIGGER: retrain when F exceeds baseline + k*std(F)

    # Affective-term ablation switches (Joffily & Coricelli, 2013).
    # When both are True (default) the ADAPTIVE policy uses the full affective
    # decision signal. Toggle individually to ablate the contribution of each
    # term to G(REUSE) — see RecomputationPolicy.decide and scripts/run_ablation.py.
    use_valence: bool = True             # add -dF urgency signal to G(REUSE)
    use_anxiety: bool = True             # add anticipated-invalidity signal to G(REUSE)

    # Theoretical threshold from Theorem 5.1.
    # NOTE: beta_sq / sigma_sq / sigma_sq_max are arbitrary placeholders, not yet
    # estimated from data. With these values tau ~= 4.45 while observed D_t <= ~0.95,
    # so `theoretical_threshold` is NEVER reached and does not drive decisions — the
    # cost-benefit rule G(REUSE) > G(RETRAIN) does. See RESULTS.md "Theorem 5.1
    # threshold: currently not operative". Estimate these from the Rashomon predictive
    # spread before claiming the threshold governs recomputation.
    p_target: float = 0.90              # target validity
    beta_sq: float = 1.0                # posterior concentration parameter squared
    sigma_sq: float = 0.01             # noise variance
    sigma_sq_max: float = 0.1          # max noise variance

    # Validation set fraction for drift detection
    val_n_samples: int = 200             # number of val samples for predictive KL

    @property
    def theoretical_threshold(self) -> float:
        """Theorem 5.1: tau = (p_target * beta^2 - sigma^2) / (2 * sigma^2_max)"""
        return (self.p_target * self.beta_sq - self.sigma_sq) / (2.0 * self.sigma_sq_max)

    @property
    def drift_types(self) -> List[str]:
        return ['covariate', 'label_noise', 'subpopulation', 'rotation']

    @property
    def strategies(self) -> List[str]:
        return ['NEVER_RETRAIN', 'ALWAYS_RETRAIN', 'ADAPTIVE', 'FIXED_INTERVAL']

    @property
    def ablation_strategies(self) -> List[str]:
        """ADAPTIVE term-ablation variants (plus NEVER/ALWAYS reference bounds).

        ADAPT_BASE = drift/staleness/validity signals only (no affective terms);
        the remaining variants switch the valence and anxiety terms on/off.
        """
        return ['NEVER_RETRAIN', 'ALWAYS_RETRAIN',
                'ADAPT_BASE', 'ADAPT_VALENCE', 'ADAPT_ANXIETY', 'ADAPT_FULL']
