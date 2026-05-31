"""Run a single (cooldown, drift, seed) ablation. Thread-capped for parallel use.

Usage: python scripts/run_one.py <cooldown> <drift> <seed> <results_dir>
"""
import os
import sys

# Cap threads BEFORE importing torch so many of these can run concurrently
# without oversubscribing the cores.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_num_threads(2)

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner

cooldown = int(sys.argv[1])
drift = sys.argv[2]
seed = int(sys.argv[3])
results_dir = sys.argv[4]

cfg = ExperimentConfig()
cfg.retrain_cooldown = cooldown
runner = ExperimentRunner(config=cfg, results_dir='./results')
runner.run_ablation_single('credit', drift, seed, results_dir=results_dir)
