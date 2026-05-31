"""
Retrain-budget (cooldown) sweep — the regime where smart timing can pay off.

For each cooldown c, the ADAPTIVE variants may retrain at most once every c+1 steps,
so they must *time* a limited number of retrains well. Sweeping c traces a
validity-vs-compute Pareto frontier per variant. If drift/trend anticipation genuinely
helps, +FULL dominates ADAPT_BASE on that frontier (more validity at equal/fewer
retrains). Cooldown 0 (unconstrained) is the existing run in ./results_ablation.

Resumable: each (cooldown, drift, seed) writes its own file and is skipped if present.

Usage:  python scripts/run_cooldown_sweep.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner

COOLDOWNS = [2, 5]                                   # c=0 already in results_ablation
DRIFTS = ['covariate', 'label_noise', 'rotation']    # subpopulation never degrades
SEEDS = (11, 22, 33, 44, 55)

for c in COOLDOWNS:
    cfg = ExperimentConfig()
    cfg.retrain_cooldown = c
    cfg.seeds = SEEDS
    rdir = f'./results_cooldown/c{c}'
    print(f"\n{'#'*60}\n# Cooldown c={c}  -> {rdir}\n{'#'*60}")
    runner = ExperimentRunner(config=cfg, results_dir='./results')
    runner.run_ablation_sweep(datasets=['credit'], drift_types=DRIFTS,
                              seeds=SEEDS, results_dir=rdir)

print("\nCooldown sweep complete.")
