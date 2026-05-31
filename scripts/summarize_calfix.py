"""Summary for the calibration-fix comparison."""
import os, sys, glob, json, numpy as np
from scipy import stats
RDIR=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results_calfix')
ORDER=['NEVER_RETRAIN','ORACLE_BASE','F_RECAL','F_TRIGGER','ALWAYS_RETRAIN']
by={}
for fp in glob.glob(os.path.join(RDIR,'*_calfix.json')):
    d=json.load(open(fp)); by.setdefault(d['drift_type'],[]).append(d)
for drift,runs in sorted(by.items()):
    print(f"\n=== {drift} (n={len(runs)}) ===")
    print(f"  {'strategy':15s} {'mean_val':>14s} {'retrains':>9s} {'labels':>7s}")
    for v in ORDER:
        mv,rc,le=[],[],[]
        for r in runs:
            s=r['strategies'].get(v)
            if not s: continue
            mv.append(np.mean(s['metrics']['validity'][1:])); rc.append(s['retrain_count']); le.append(s.get('label_evals',0))
        if mv: print(f"  {v:15s} {np.mean(mv):.3f}+/-{np.std(mv):.3f} {np.mean(rc):7.1f} {np.mean(le):7.1f}")
    for pol in ['F_RECAL','F_TRIGGER']:
        pr=[(np.mean(r['strategies'][pol]['metrics']['validity'][1:]),np.mean(r['strategies']['ORACLE_BASE']['metrics']['validity'][1:])) for r in runs if pol in r['strategies'] and 'ORACLE_BASE' in r['strategies']]
        if len(pr)>=2:
            f=np.array([p[0] for p in pr]); o=np.array([p[1] for p in pr]); t,p=stats.ttest_rel(f,o)
            le=np.mean([r['strategies'][pol]['label_evals'] for r in runs])
            print(f"  {pol:10s} - ORACLE = {(f-o).mean():+.3f} +/- {(f-o).std():.3f} [p={p:.3f}] | labels {le:.0f} vs 20")
