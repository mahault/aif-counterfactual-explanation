"""
Valence/anxiety term ablation for the ADAPTIVE recomputation policy.

Runs the ADAPTIVE policy under four affective-term conditions —
  ADAPT_BASE     (no valence, no anxiety)
  ADAPT_VALENCE  (valence only)
  ADAPT_ANXIETY  (anxiety only)
  ADAPT_FULL     (both)
— with NEVER_RETRAIN / ALWAYS_RETRAIN kept as reference bounds. All variants share
the same per-timestep base-model/Rashomon recomputation, so differences in validity
isolate the contribution of the valence and anxiety terms (Joffily & Coricelli, 2013).

Usage:
    # Single ablation experiment (smoke test):
    python scripts/run_ablation.py --dataset credit --drift rotation --seed 11 --T 5

    # Full ablation sweep (datasets x drift types x seeds):
    python scripts/run_ablation.py

    # Regenerate the ablation figure from existing results:
    python scripts/run_ablation.py --plot-only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner
from adaptive_avcg.plotting import generate_ablation_figures


def main():
    parser = argparse.ArgumentParser(
        description='Valence/anxiety term ablation for adaptive AVCG')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['credit', 'adult_income', 'spambase', 'pneumonia'])
    parser.add_argument('--drift', type=str, default=None,
                        choices=['covariate', 'label_noise', 'subpopulation', 'rotation'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--n-test', type=int, default=None)
    parser.add_argument('--results-dir', type=str, default='./results_ablation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate the ablation figure from existing results')
    args = parser.parse_args()

    if args.plot_only:
        print("Generating ablation figure from existing results...")
        generate_ablation_figures(args.results_dir)
        return

    config = ExperimentConfig()
    if args.T is not None:
        config.T = args.T
    if args.n_test is not None:
        config.n_test_samples = args.n_test

    runner = ExperimentRunner(config=config, results_dir='./results')

    if args.dataset and args.drift and args.seed is not None:
        runner.run_ablation_single(args.dataset, args.drift, args.seed,
                                   results_dir=args.results_dir)
    else:
        datasets = [args.dataset] if args.dataset else None
        drift_types = [args.drift] if args.drift else None
        seeds = (args.seed,) if args.seed is not None else None
        runner.run_ablation_sweep(datasets=datasets, drift_types=drift_types,
                                  seeds=seeds, results_dir=args.results_dir)

    print("\n" + "=" * 60)
    print("Generating ablation figure...")
    print("=" * 60)
    generate_ablation_figures(args.results_dir)


if __name__ == '__main__':
    main()
