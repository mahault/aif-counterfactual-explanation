"""Run a single label-free F-policy comparison. Thread-capped for parallel use.

Usage: python scripts/run_one_lf.py <drift> <seed> <results_dir>
"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_num_threads(2)

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner

drift = sys.argv[1]
seed = int(sys.argv[2])
results_dir = sys.argv[3]

cfg = ExperimentConfig()          # cooldown 0 (free retraining)
ExperimentRunner(cfg, results_dir='./results').run_labelfree_single(
    'credit', drift, seed, results_dir=results_dir)
