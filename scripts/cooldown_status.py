"""Interim status + full metrics for the cooldown (retrain-budget) sweep."""
import os, glob, json, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOLDOWNS = [0, 2, 5]
DRIFTS = ['covariate', 'label_noise', 'rotation']
VARIANTS = ['ADAPT_BASE', 'ADAPT_VALENCE', 'ADAPT_ANXIETY', 'ADAPT_FULL']


def dir_for(c):
    return (os.path.join(ROOT, 'results_ablation') if c == 0
            else os.path.join(ROOT, 'results_cooldown', f'c{c}'))


def runs(c, drift):
    return [json.load(open(f)) for f in
            glob.glob(os.path.join(dir_for(c), f'credit_{drift}_seed*_ablation.json'))]


n2 = len(glob.glob(os.path.join(ROOT, 'results_cooldown', 'c2', '*_ablation.json')))
n5 = len(glob.glob(os.path.join(ROOT, 'results_cooldown', 'c5', '*_ablation.json')))
print(f"PROGRESS: c2={n2}/15  c5={n5}/15  total={n2+n5}/30")

for drift in DRIFTS:
    print(f"\n=== {drift} ===")
    print(f"  {'cooldown':>8s}  {'variant':14s}  {'n':>2s}  {'retrains':>14s}  {'mean_val':>14s}")
    for c in COOLDOWNS:
        rs = runs(c, drift)
        if not rs:
            print(f"  {c:>8d}  (no runs yet)")
            continue
        for v in VARIANTS:
            mv, rc = [], []
            for r in rs:
                s = r['strategies'].get(v)
                if s:
                    mv.append(np.mean(s['metrics']['validity'][1:]))
                    rc.append(s['retrain_count'])
            if mv:
                print(f"  {c:>8d}  {v:14s}  {len(mv):>2d}  "
                      f"{np.mean(rc):5.1f}+/-{np.std(rc):.1f}     "
                      f"{np.mean(mv):.3f}+/-{np.std(mv):.3f}")
    # quick FULL-vs-BASE at the most-constrained level present
    for c in [5, 2]:
        rs = runs(c, drift)
        if not rs:
            continue
        seeds = {r['seed'] for r in rs}
        pairs = []
        for r in rs:
            b = r['strategies'].get('ADAPT_BASE'); f = r['strategies'].get('ADAPT_FULL')
            if b and f:
                pairs.append((np.mean(f['metrics']['validity'][1:]),
                              np.mean(b['metrics']['validity'][1:])))
        if len(pairs) >= 2:
            d = np.array([a - bb for a, bb in pairs])
            print(f"    [c={c}] FULL-BASE mean_val = {d.mean():+.3f} +/- {d.std():.3f} (n={len(d)})")
        break
