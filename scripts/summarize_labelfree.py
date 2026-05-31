"""Summary for the label-free F-policy comparison: does F_POLICY match ORACLE_BASE
validity at a fraction of the labelling cost?"""
import os, sys, glob, json, numpy as np
from scipy import stats

RDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_labelfree')
ORDER = ['NEVER_RETRAIN', 'ORACLE_BASE', 'F_POLICY', 'FDT_POLICY', 'ALWAYS_RETRAIN']

by = {}
for fp in glob.glob(os.path.join(RDIR, '*_labelfree.json')):
    d = json.load(open(fp)); by.setdefault(d['drift_type'], []).append(d)

for drift, runs in sorted(by.items()):
    print(f"\n=== {drift}  (n={len(runs)}) ===")
    print(f"  {'strategy':15s} {'mean_val':>14s} {'retrains':>10s} {'label_evals':>12s}")
    for v in ORDER:
        mv, rc, le = [], [], []
        for r in runs:
            s = r['strategies'].get(v)
            if not s:
                continue
            mv.append(np.mean(s['metrics']['validity'][1:]))
            rc.append(s['retrain_count']); le.append(s.get('label_evals', 0))
        if mv:
            print(f"  {v:15s} {np.mean(mv):.3f}+/-{np.std(mv):.3f} "
                  f"{np.mean(rc):8.1f}   {np.mean(le):10.1f}")
    # Label-free policies vs ORACLE_BASE: validity gap + labelling savings
    for pol in ['F_POLICY', 'FDT_POLICY']:
        pairs = [(np.mean(r['strategies'][pol]['metrics']['validity'][1:]),
                  np.mean(r['strategies']['ORACLE_BASE']['metrics']['validity'][1:]))
                 for r in runs if pol in r['strategies'] and 'ORACLE_BASE' in r['strategies']]
        if len(pairs) >= 2:
            f = np.array([p[0] for p in pairs]); o = np.array([p[1] for p in pairs])
            t, p = stats.ttest_rel(f, o)
            le_f = np.mean([r['strategies'][pol]['label_evals'] for r in runs])
            le_o = np.mean([r['strategies']['ORACLE_BASE']['label_evals'] for r in runs])
            print(f"  {pol:11s} - ORACLE validity = {(f-o).mean():+.3f} +/- {(f-o).std():.3f} "
                  f"[p={p:.3f}]  | labels {le_f:.0f} vs {le_o:.0f} "
                  f"({100*(1-le_f/le_o):.0f}% fewer)")
