"""
Adaptive AVCG: Drift detection and recomputation for
amortized counterfactual explanations under distribution shift.

Part of "Adaptive Amortized Counterfactual Inference via Free Energy Minimization"
"""

from adaptive_avcg.config import ExperimentConfig

# Lazy imports — these require torch, numpy, etc.
# Import explicitly when needed:
#   from adaptive_avcg.drift_detector import DriftDetector
#   from adaptive_avcg.recomputation_policy import RecomputationPolicy
#   from adaptive_avcg.drift_simulator import DriftSimulator
#   from adaptive_avcg.experiment_runner import ExperimentRunner

__all__ = [
    'ExperimentConfig',
    'DriftDetector',
    'DriftSimulator',
    'RecomputationPolicy',
    'ExperimentRunner',
]


def __getattr__(name):
    if name == 'DriftDetector':
        from adaptive_avcg.drift_detector import DriftDetector
        return DriftDetector
    elif name == 'RecomputationPolicy':
        from adaptive_avcg.recomputation_policy import RecomputationPolicy
        return RecomputationPolicy
    elif name == 'DriftSimulator':
        from adaptive_avcg.drift_simulator import DriftSimulator
        return DriftSimulator
    elif name == 'ExperimentRunner':
        from adaptive_avcg.experiment_runner import ExperimentRunner
        return ExperimentRunner
    raise AttributeError(f"module 'adaptive_avcg' has no attribute {name!r}")
