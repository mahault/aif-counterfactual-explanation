"""Run a single calibration-fix comparison. Thread-capped for parallel use.
Usage: python scripts/run_one_calfix.py <drift> <seed> <results_dir>
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(2)
from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner
drift, seed, rdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ExperimentRunner(ExperimentConfig(), results_dir='./results').run_calfix_single('credit', drift, seed, results_dir=rdir)
