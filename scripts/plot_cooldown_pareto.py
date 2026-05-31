"""
Validity-vs-compute Pareto from the cooldown sweep.

For each drift type, plots mean validity against mean retrains, one connected line per
ADAPTIVE variant across cooldown levels {0, 2, 5}. The question: does +FULL sit
ABOVE/LEFT of ADAPT_BASE (more validity for the same or fewer retrains)? If the lines
overlap, anticipation adds nothing even under a retrain budget.

c=0 is read from ./results_ablation (unconstrained); c>0 from ./results_cooldown/c{c}.
Restricted to the seeds present in every cooldown level for a fair comparison.

Usage:  python scripts/plot_cooldown_pareto.py
"""
import sys
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOLDOWNS = [0, 2, 5]
DRIFTS = ['covariate', 'label_noise', 'rotation']
VARIANTS = {
    'ADAPT_BASE':    ('#95a5a6', 'Base (no affect)'),
    'ADAPT_VALENCE': ('#3498db', '+ Valence'),
    'ADAPT_ANXIETY': ('#e67e22', '+ Anxiety'),
    'ADAPT_FULL':    ('#9b59b6', '+ Full'),
}


def dir_for(c):
    return (os.path.join(ROOT, 'results_ablation') if c == 0
            else os.path.join(ROOT, 'results_cooldown', f'c{c}'))


def load(c, drift):
    """Return {variant: [(retrains, mean_val), ...per seed]} for a cooldown/drift."""
    out = {}
    for fp in glob.glob(os.path.join(dir_for(c), f'credit_{drift}_seed*_ablation.json')):
        d = json.load(open(fp))
        seed = d['seed']
        for v in VARIANTS:
            s = d['strategies'].get(v)
            if s:
                out.setdefault(v, {})[seed] = (
                    s['retrain_count'], float(np.mean(s['metrics']['validity'][1:])))
    return out


def main():
    fig, axes = plt.subplots(1, len(DRIFTS), figsize=(6 * len(DRIFTS), 5))
    if len(DRIFTS) == 1:
        axes = [axes]

    print(f"{'drift':12s} {'cooldown':>8s} {'variant':14s} {'retrains':>9s} {'mean_val':>9s}")
    for ax, drift in zip(axes, DRIFTS):
        # seeds common to all cooldown levels (fair comparison)
        per_c = {c: load(c, drift) for c in COOLDOWNS}
        common = None
        for c in COOLDOWNS:
            base = per_c[c].get('ADAPT_BASE', {})
            common = set(base) if common is None else (common & set(base))
        common = sorted(common or [])

        for v, (color, label) in VARIANTS.items():
            xs, ys = [], []
            for c in COOLDOWNS:
                pts = per_c[c].get(v, {})
                vals = [pts[s] for s in common if s in pts]
                if not vals:
                    continue
                r = np.mean([p[0] for p in vals])
                mv = np.mean([p[1] for p in vals])
                xs.append(r); ys.append(mv)
                print(f"{drift:12s} {c:>8d} {v:14s} {r:9.1f} {mv:9.3f}")
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]
            ax.plot(xs, ys, '-o', color=color, label=label, markersize=6,
                    linewidth=1.5)
        ax.set_title(f"{drift.replace('_',' ').title()}  (n={len(common)} seeds)",
                     fontsize=12)
        ax.set_xlabel('Mean Retrains (compute budget)', fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Mean Validity', fontsize=11)
    axes[-1].legend(fontsize=9, loc='lower right')
    fig.suptitle('Validity vs. Retrain Budget — cooldown sweep {0, 2, 5}\n'
                 '(does +Full sit above/left of Base?)', fontsize=13, y=1.04)
    plt.tight_layout()
    out = os.path.join(ROOT, 'results_cooldown', 'fig9_cooldown_pareto')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved {out}.png")


if __name__ == '__main__':
    main()
