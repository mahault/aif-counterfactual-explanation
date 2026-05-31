"""
Entry point for drift detection experiments.

Usage:
    # Full sweep (3 datasets x 4 drift types x 3 seeds) — ~24h GPU, ~72h CPU
    python scripts/run_experiments.py

    # Single experiment for development / smoke testing
    python scripts/run_experiments.py --dataset credit --drift covariate --seed 11 --T 5

    # Generate figures from existing results
    python scripts/run_experiments.py --plot-only --results-dir results/
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner
from adaptive_avcg.plotting import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive AVCG drift detection experiments')

    parser.add_argument('--dataset', type=str, default=None,
                        choices=['credit', 'adult_income', 'spambase', 'pneumonia'],
                        help='Single dataset to run (default: all)')
    parser.add_argument('--drift', type=str, default=None,
                        choices=['covariate', 'label_noise', 'subpopulation', 'rotation'],
                        help='Single drift type (default: all)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed (default: all from config)')
    parser.add_argument('--T', type=int, default=None,
                        help='Number of timesteps (default: 20)')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory for results (default: ./results)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate figures from existing results')
    parser.add_argument('--n-test', type=int, default=None,
                        help='Number of test samples per timestep (default: 60)')

    args = parser.parse_args()

    # Configure
    config = ExperimentConfig()
    if args.T is not None:
        config.T = args.T
    if args.n_test is not None:
        config.n_test_samples = args.n_test

    results_dir = args.results_dir

    if args.plot_only:
        print("Generating figures from existing results...")
        generate_all_figures(results_dir)
        return

    runner = ExperimentRunner(config=config, results_dir=results_dir)

    if args.dataset and args.drift and args.seed:
        # Single experiment
        runner.run_single(args.dataset, args.drift, args.seed)
    else:
        # Sweep with optional filters
        datasets = [args.dataset] if args.dataset else None
        drift_types = [args.drift] if args.drift else None
        seeds = (args.seed,) if args.seed else None
        runner.run_sweep(datasets=datasets, drift_types=drift_types, seeds=seeds)

    # Generate figures
    print("\n" + "="*60)
    print("Generating publication figures...")
    print("="*60)
    generate_all_figures(results_dir)


if __name__ == '__main__':
    main()
