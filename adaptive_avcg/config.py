"""
All experiment hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExperimentConfig:
    # Datasets
    datasets: List[str] = field(default_factory=lambda: ['credit', 'adult_income', 'spambase'])

    # Temporal parameters
    T: int = 20                          # number of timesteps
    seeds: Tuple[int, ...] = (11, 22, 33)

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

    # Recomputation policy
    retrain_cost: float = 0.5            # normalized computational cost of retraining
    invalidity_cost_weight: float = 5.0  # lambda_inv: weight of invalidity vs compute cost
    fixed_interval_K: int = 5            # retrain every K steps for FIXED_INTERVAL strategy
    warmup_steps: int = 3                # ADAPTIVE observes (forced REUSE) before deciding

    # Theoretical threshold from Theorem 5.1
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
