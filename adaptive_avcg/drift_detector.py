"""
Predictive KL divergence tracker between successive Rashomon-restricted posteriors.

Key insight: Raw *parametric* KL between discrete uniform posteriors over mask sets
is degenerate (the support is a finite set of masks, so the parameter-space KL is
ill-defined or jumps to infinity under disjoint supports). Instead we measure drift
in **probability (function) space** via the predictive KL divergence:
    D_t = E_{x~val}[ KL( P_R^{t+1}(y|x) || P_R^t(y|x) ) ]
where P_R^t(y|x) = mean softmax over Rashomon members at time t.
For binary classification this is KL between two Bernoullis — well-defined and bounded.

Justification: by the data-processing inequality, the function-space (predictive) KL
lower-bounds the parameter-space KL, so it is a conservative, well-behaved drift signal
even when the parametric posteriors do not admit a tractable KL. See the function-space
VI / KL-DPI discussion in https://iclr-blogposts.github.io/2024/blog/dpi-fsvi/.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_avcg.avcg_core import DEVICE


def _kl_bernoulli(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """KL(Bern(p) || Bern(q)) element-wise."""
    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def _jsd_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence between Bernoullis."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_bernoulli(p, m) + 0.5 * _kl_bernoulli(q, m)


class DriftDetector:
    """Tracks predictive KL divergence between successive Rashomon posteriors."""

    def __init__(self, reference_ensemble, epsilon: float, val_loader: DataLoader,
                 n_samples: int = 200):
        """
        Args:
            reference_ensemble: MCDropoutRashomonSet at t=0
            epsilon: Rashomon threshold
            val_loader: validation DataLoader for computing predictive distributions
            n_samples: max number of validation samples to use
        """
        self.epsilon = epsilon
        self.val_loader = val_loader
        self.n_samples = n_samples

        # Cache reference predictions
        self._ref_probs = self._compute_predictive_probs(reference_ensemble)
        self._ref_acc = self._compute_accuracy(reference_ensemble)
        self._ref_nll = self._compute_nll(reference_ensemble)
        self._ref_set_size = reference_ensemble.rashomon_size(epsilon)

    def _get_val_batch(self):
        """Collect up to n_samples (x, y) pairs from val_loader."""
        xs, ys = [], []
        count = 0
        for x, y in self.val_loader:
            xs.append(x)
            ys.append(y)
            count += len(y)
            if count >= self.n_samples:
                break
        return torch.cat(xs, dim=0)[:self.n_samples].to(DEVICE), \
               torch.cat(ys, dim=0)[:self.n_samples].to(DEVICE)

    def _compute_predictive_probs(self, ensemble) -> np.ndarray:
        """Compute P_R(y=1|x) for each val sample. Returns (n_samples,) array."""
        members = ensemble.get_rashomon_models(self.epsilon)
        x_val, _ = self._get_val_batch()

        # Mean softmax probability of class 1 across Rashomon members
        prob_sum = torch.zeros(len(x_val), device=DEVICE)
        for m in members:
            m.eval()
            with torch.no_grad():
                logits = m(x_val)  # (n, 2) log-softmax
                prob_sum += torch.exp(logits[:, 1])
        probs = (prob_sum / len(members)).cpu().numpy()
        # Guard against a member emitting non-finite logits (can happen when a base
        # model diverges under heavy label noise). Replace with 0.5 (no-information),
        # which contributes ~0 KL rather than propagating NaN through the drift signal.
        n_bad = int(np.sum(~np.isfinite(probs)))
        if n_bad:
            print(f"  [DriftDetector] WARNING: {n_bad} non-finite predictive probs "
                  f"replaced with 0.5")
            probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        return probs

    def _compute_accuracy(self, ensemble) -> float:
        """Ensemble accuracy on validation set."""
        members = ensemble.get_rashomon_models(self.epsilon)
        x_val, y_val = self._get_val_batch()

        pred_sum = torch.zeros(len(x_val), 2, device=DEVICE)
        for m in members:
            m.eval()
            with torch.no_grad():
                pred_sum += torch.exp(m(x_val))
        preds = pred_sum.argmax(dim=1)
        return (preds == y_val).float().mean().item()

    def _compute_nll(self, ensemble) -> float:
        """Mean NLL across Rashomon members on val set."""
        members = ensemble.get_rashomon_models(self.epsilon)
        x_val, y_val = self._get_val_batch()

        nll_sum = 0.0
        for m in members:
            m.eval()
            with torch.no_grad():
                logits = m(x_val)
                nll_sum += F.nll_loss(logits, y_val).item()
        return nll_sum / len(members)

    def compute_drift(self, new_ensemble) -> dict:
        """Compute drift between reference and new ensemble.

        Returns dict with:
            mean_kl: mean predictive KL over val samples
            max_kl: max predictive KL
            std_kl: std of predictive KL
            mean_jsd: mean Jensen-Shannon divergence (ablation)
            accuracy_delta: change in ensemble accuracy (ablation)
            nll_delta: change in mean NLL (ablation)
            set_size_ratio: new_size / ref_size (ablation)
        """
        new_probs = self._compute_predictive_probs(new_ensemble)

        # Predictive KL: KL(P_R^new || P_R^ref)
        kl_per_sample = _kl_bernoulli(new_probs, self._ref_probs)

        # Ablation measures
        jsd_per_sample = _jsd_bernoulli(new_probs, self._ref_probs)
        new_acc = self._compute_accuracy(new_ensemble)
        new_nll = self._compute_nll(new_ensemble)
        new_size = new_ensemble.rashomon_size(self.epsilon)

        # Defensive: ignore any residual non-finite entries so the drift signal
        # is always a real number the policy can act on.
        kl_per_sample = kl_per_sample[np.isfinite(kl_per_sample)]
        jsd_per_sample = jsd_per_sample[np.isfinite(jsd_per_sample)]
        if kl_per_sample.size == 0:
            kl_per_sample = np.zeros(1)
        if jsd_per_sample.size == 0:
            jsd_per_sample = np.zeros(1)

        return {
            'mean_kl': float(np.mean(kl_per_sample)),
            'max_kl': float(np.max(kl_per_sample)),
            'std_kl': float(np.std(kl_per_sample)),
            'mean_jsd': float(np.mean(jsd_per_sample)),
            'accuracy_delta': float(new_acc - self._ref_acc),
            'nll_delta': float(new_nll - self._ref_nll),
            'set_size_ratio': float(new_size / max(self._ref_set_size, 1)),
        }

    def update_reference(self, new_ensemble):
        """Reset reference after a RETRAIN decision."""
        self._ref_probs = self._compute_predictive_probs(new_ensemble)
        self._ref_acc = self._compute_accuracy(new_ensemble)
        self._ref_nll = self._compute_nll(new_ensemble)
        self._ref_set_size = new_ensemble.rashomon_size(self.epsilon)
