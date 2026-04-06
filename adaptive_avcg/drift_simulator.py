"""
Data drift simulation: four types of distribution shift applied to training data.

Each drift type is a function that takes the original (clean) training data
and a timestep t, returning shifted (X_train, y_train) numpy arrays.
"""

import numpy as np
from sklearn.decomposition import PCA
from adaptive_avcg.config import ExperimentConfig


class DriftSimulator:
    """Applies progressive distribution shift to training data."""

    def __init__(self, X_train_clean: np.ndarray, y_train_clean: np.ndarray,
                 config: ExperimentConfig, rng: np.random.RandomState = None):
        """
        Args:
            X_train_clean: Original clean training features (n_samples, n_features)
            y_train_clean: Original clean training labels (n_samples,)
            config: Experiment configuration with drift parameters
            rng: Random state for reproducibility
        """
        self.X_clean = X_train_clean.copy()
        self.y_clean = y_train_clean.copy()
        self.config = config
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Pre-compute PCA for rotation drift
        self.n_features = X_train_clean.shape[1]
        if self.n_features >= 2:
            self._pca = PCA(n_components=min(self.n_features, 2))
            self._pca.fit(X_train_clean)
        else:
            self._pca = None

        # Pre-select which features get covariate noise (fixed across timesteps)
        n_noisy = max(1, int(self.n_features * config.covariate_feature_frac))
        self._noisy_features = self.rng.choice(
            self.n_features, size=n_noisy, replace=False)

    def apply(self, drift_type: str, t: int) -> tuple:
        """Apply drift of given type at timestep t.

        Returns:
            (X_shifted, y_shifted) numpy arrays
        """
        if drift_type == 'covariate':
            return self._covariate_shift(t)
        elif drift_type == 'label_noise':
            return self._label_noise(t)
        elif drift_type == 'subpopulation':
            return self._subpopulation_shift(t)
        elif drift_type == 'rotation':
            return self._rotation_drift(t)
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")

    def _covariate_shift(self, t: int) -> tuple:
        """Gaussian noise on selected features, sigma grows with t."""
        sigma = self.config.covariate_sigma_base * t
        X = self.X_clean.copy()
        noise = self.rng.randn(X.shape[0], len(self._noisy_features)) * sigma
        X[:, self._noisy_features] += noise
        return X, self.y_clean.copy()

    def _label_noise(self, t: int) -> tuple:
        """Flip a growing fraction of labels."""
        alpha = min(self.config.label_alpha_base * t, self.config.label_alpha_cap)
        y = self.y_clean.copy()
        n_flip = int(len(y) * alpha)
        if n_flip > 0:
            flip_idx = self.rng.choice(len(y), size=n_flip, replace=False)
            y[flip_idx] = 1 - y[flip_idx]
        return self.X_clean.copy(), y

    def _subpopulation_shift(self, t: int) -> tuple:
        """Progressively undersample class 0."""
        retain_frac = max(self.config.sub_floor, 1.0 - self.config.sub_rate * t)
        class0_idx = np.where(self.y_clean == 0)[0]
        class1_idx = np.where(self.y_clean == 1)[0]

        n_keep = max(1, int(len(class0_idx) * retain_frac))
        kept_class0 = self.rng.choice(class0_idx, size=n_keep, replace=False)

        all_idx = np.concatenate([kept_class0, class1_idx])
        self.rng.shuffle(all_idx)
        return self.X_clean[all_idx].copy(), self.y_clean[all_idx].copy()

    def _rotation_drift(self, t: int) -> tuple:
        """Rotate first 2 PCA components by a growing angle."""
        if self._pca is None or self.n_features < 2:
            return self.X_clean.copy(), self.y_clean.copy()

        angle_rad = np.radians(self.config.rotation_deg * t)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])

        X = self.X_clean.copy()
        # Project to PCA space, rotate first 2 components, project back
        Z = self._pca.transform(X)  # (n_samples, 2)
        Z_rot = Z @ R.T
        # Reconstruct: X_rot = Z_rot @ components + mean
        X_rot = self._pca.inverse_transform(Z_rot)
        return X_rot, self.y_clean.copy()
