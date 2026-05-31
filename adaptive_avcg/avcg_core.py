"""
AVCG Core — Jamie's classes extracted from Playground_code_Share.ipynb (Cell 0)
for programmatic import. Code is verbatim with minimal changes:
  - DEVICE is configurable at module level
  - set_seed is importable

Attribution: Original code by Jamie (Playground_code_Share.ipynb).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import random
import time
import copy
from collections import defaultdict
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurable device — override before calling any functions if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
    def __len__(self):          return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


def get_dataset(name='credit'):
    print(f"\n--- Loading '{name}' ---")
    if name == 'adult_income':
        ds = fetch_ucirepo(id=2)
        X  = ds.data.features
        y  = ds.data.targets['income'].str.strip().replace({'<=50K': 0, '>50K': 1})
        class_names = ['<=50K', '>50K']
    elif name == 'credit':
        ds = fetch_ucirepo(id=144)
        X  = ds.data.features
        y  = ds.data.targets.iloc[:, 0].map({1: 0, 2: 1})
        class_names = ['Good Credit', 'Bad Credit']
    elif name == 'spambase':
        ds = fetch_ucirepo(id=94)
        X  = ds.data.features
        y  = ds.data.targets.iloc[:, 0]
        class_names = ['Not Spam', 'Spam']
    elif name == 'pneumonia':
        # PneumoniaMNIST: 28x28 grayscale chest X-rays, binary (normal/pneumonia).
        # Flattened to a 784-dim vector so it flows through the same flat-vector
        # MLP/CVAE pipeline as the tabular datasets. medmnist's curated splits are
        # concatenated and re-split below for uniform treatment with the others.
        from medmnist import PneumoniaMNIST
        imgs, labels = [], []
        for split in ('train', 'val', 'test'):
            d = PneumoniaMNIST(split=split, download=True)
            imgs.append(d.imgs.reshape(len(d.imgs), -1).astype('float32') / 255.0)
            labels.append(d.labels.reshape(-1).astype('int64'))
        X = pd.DataFrame(np.concatenate(imgs, axis=0),
                         columns=[f'px{i}' for i in range(imgs[0].shape[1])])
        y = pd.Series(np.concatenate(labels, axis=0))
        class_names = ['Normal', 'Pneumonia']
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    lower = torch.tensor(X_train.min(axis=0), dtype=torch.float32).to(DEVICE)
    upper = torch.tensor(X_train.max(axis=0), dtype=torch.float32).to(DEVICE)

    return (TabularDataset(X_train, y_train),
            TabularDataset(X_val,   y_val),
            TabularDataset(X_test,  y_test),
            scaler, X.columns.tolist(), class_names, lower, upper)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MCDropoutMLP(nn.Module):
    """MC Dropout model used as base model and for NERCE training."""
    def __init__(self, input_dim, num_classes, hidden_dim1=64,
                 hidden_dim2=32, dropout_rate=0.2):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim1)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.fc2   = nn.Linear(hidden_dim1, hidden_dim2)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3   = nn.Linear(hidden_dim2, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x));  x = self.drop1(x)
        x = F.relu(self.fc2(x));  x = self.drop2(x)
        return F.log_softmax(self.fc3(x), dim=-1)


class FrozenMaskMLP(nn.Module):
    """A deterministic sub-network sampled from the MC Dropout posterior."""
    def __init__(self, base_model, p=0.2):
        super().__init__()
        self.base = copy.deepcopy(base_model)
        self.base.eval()
        self.mask1 = (torch.rand(self.base.fc1.out_features) > p).float() / (1.0 - p)
        self.mask2 = (torch.rand(self.base.fc2.out_features) > p).float() / (1.0 - p)

    def forward(self, x):
        x = F.relu(self.base.fc1(x))
        x = x * self.mask1.to(x.device)
        x = F.relu(self.base.fc2(x))
        x = x * self.mask2.to(x.device)
        return F.log_softmax(self.base.fc3(x), dim=-1)


class MCDropoutRashomonSet:
    """Samples N models from MC Dropout and filters them via the epsilon threshold."""
    def __init__(self, base_model, n_samples=50):
        self.models = [FrozenMaskMLP(base_model).to(DEVICE) for _ in range(n_samples)]
        self.val_losses = np.full(n_samples, np.inf)

    def evaluate_all(self, val_loader):
        print(f"  Evaluating {len(self.models)} MC Dropout masks on validation set...")
        ce = nn.NLLLoss()
        for i, model in enumerate(self.models):
            total, n = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y  = x.to(DEVICE), y.to(DEVICE)
                    total += ce(model(x), y).item() * len(y)
                    n     += len(y)
            self.val_losses[i] = total / n
        print(f"  Sampled val losses  min={self.val_losses.min():.4f}  "
              f"max={self.val_losses.max():.4f}  mean={self.val_losses.mean():.4f}")

    def get_rashomon_models(self, epsilon):
        L_star = self.val_losses.min()
        in_set = [m for m, l in zip(self.models, self.val_losses)
                  if l <= L_star + epsilon]
        return in_set if in_set else [self.models[int(np.argmin(self.val_losses))]]

    def rashomon_size(self, epsilon):
        return int((self.val_losses <= self.val_losses.min() + epsilon).sum())

    def expected_log_prob_rashomon(self, x_prime, target_cf, epsilon):
        """Approximates E_{theta~P_R}[log P_theta] using ONLY accepted masks."""
        members      = self.get_rashomon_models(epsilon)
        log_prob_sum = 0.0
        for m in members:
            lp = m(x_prime).gather(1, target_cf.view(-1, 1)).squeeze()
            log_prob_sum = log_prob_sum + lp
        return (log_prob_sum / len(members)).mean()


class TabularCVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=10, hidden_dim=64):
        super().__init__()
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.latent_dim  = latent_dim
        self.label_emb   = nn.Embedding(num_classes, num_classes)
        self.enc_fc1  = nn.Linear(input_dim + num_classes, hidden_dim)
        self.enc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.enc_logv = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc1  = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.dec_fc2  = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        h = F.relu(self.enc_fc1(torch.cat([x, self.label_emb(y)], dim=1)))
        return self.enc_mu(h), self.enc_logv(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z, y):
        h = F.relu(self.dec_fc1(torch.cat([z, self.label_emb(y)], dim=1)))
        return self.dec_fc2(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_mc_dropout(model, train_loader, epochs, name="MC Dropout"):
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    ce  = nn.NLLLoss()
    print(f"  Training {name}...")
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); ce(model(x), y).backward(); opt.step()


def train_cvae_rashomon(cvae, train_loader, ensemble, epsilon, epochs=50,
                        lambda_class=1.0, lambda_prox=0.1, lambda_kl=0.1,
                        name="CVAE"):
    """Trains CVAE using the Rashomon-restricted ELBO-like objective."""
    opt = optim.Adam(cvae.parameters(), lr=0.001, weight_decay=1e-4)
    print(f"  Training {name} (eps={epsilon})...")
    for _ in range(epochs):
        cvae.train()
        for x, y in train_loader:
            x, y       = x.to(DEVICE), y.to(DEVICE)
            target_cf  = 1 - y
            mu, logvar = cvae.encode(x, target_cf)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            z          = cvae.reparameterize(mu, logvar)
            x_prime    = cvae.decode(z, target_cf)
            exp_lp     = ensemble.expected_log_prob_rashomon(x_prime, target_cf, epsilon)
            proximity  = F.mse_loss(x_prime, x, reduction='mean')
            loss       = -(lambda_class * exp_lp) + (lambda_kl * kl_div) + (lambda_prox * proximity)
            opt.zero_grad(); loss.backward(); opt.step()


# ---------------------------------------------------------------------------
# Counterfactual generation
# ---------------------------------------------------------------------------

def generate_cf_amortized(cvae, x_original, target_class, lower, upper, n_samples=5):
    """Single forward pass through trained CVAE; pick lowest-proximity sample."""
    cvae.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_t        = torch.tensor([target_class], device=DEVICE)
        mu, logvar = cvae.encode(x_original, y_t)
        candidates = [cvae.decode(cvae.reparameterize(mu, logvar), y_t).clamp(lower, upper)
                      for _ in range(n_samples)]
        dists   = [torch.linalg.norm(c - x_original).item() for c in candidates]
        x_prime = candidates[int(np.argmin(dists))]
    return x_prime, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def validity(cf, target_class, model, n_mc=20):
    model.train()
    with torch.no_grad():
        outs = torch.stack([model(cf) for _ in range(n_mc)]).mean(0)
    return int(outs.max(1)[1].item() == target_class)


def validity_after_training(cf, target_class, base_model, n_masks=10):
    """
    Simulates model shift by drawing n_masks fresh FrozenMaskMLPs from the
    base MC Dropout model - independent of the Rashomon set used for training.
    Returns the fraction of fresh masks that predict y' for this CF.
    """
    votes = 0
    for _ in range(n_masks):
        fresh_mask = FrozenMaskMLP(base_model).to(cf.device)
        with torch.no_grad():
            votes += int(fresh_mask(cf).max(1)[1].item() == target_class)
    return votes / n_masks


def rashomon_validity_ratio(cf, target_class, ensemble, epsilon):
    """Fraction of Rashomon-set members that predict y' for this CF."""
    members = ensemble.get_rashomon_models(epsilon)
    votes = 0
    for m in members:
        m.eval()
        with torch.no_grad():
            votes += int(m(cf).max(1)[1].item() == target_class)
    return votes / len(members)
