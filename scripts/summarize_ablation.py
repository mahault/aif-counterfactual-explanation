"""Summary table for the valence/anxiety ablation results."""
import json
import os
import sys
import glob
import numpy as np

results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_ablation')

VARIANTS = ['NEVER_RETRAIN', 'ADAPT_BASE', 'ADAPT_VALENCE',
            'ADAPT_ANXIETY', 'ADAPT_FULL', 'ALWAYS_RETRAIN']

files = sorted(glob.glob(os.path.join(results_dir, '*_ablation.json')))
if not files:
    print(f"No ablation results in {results_dir}")
    sys.exit(0)

# Group by (dataset, drift)
by_group = {}
for fp in files:
    with open(fp) as f:
        data = json.load(f)
    key = (data['dataset'], data['drift_type'])
    by_group.setdefault(key, []).append(data)

for (dataset, drift), runs in sorted(by_group.items()):
    print(f"\n{'='*84}")
    print(f"  {dataset}  /  {drift}   (n_seeds={len(runs)})")
    print(f"{'='*84}")
    print(f"  {'variant':16s}  {'mean_val':>14s}  {'final_val':>14s}  "
          f"{'retrains':>11s}  {'val/retrain':>11s}")
    retrain_means = {}
    for variant in VARIANTS:
        means, finals, retrains = [], [], []
        for data in runs:
            s = data['strategies'].get(variant)
            if s is None:
                continue
            v = s['metrics']['validity']
            means.append(np.mean(v[1:]))
            finals.append(v[-1])
            retrains.append(s['retrain_count'])
        if not means:
            continue
        mr = np.mean(retrains)
        retrain_means[variant] = mr
        eff = (np.mean(means) / mr) if mr > 0 else float('nan')
        print(f"  {variant:16s}  "
              f"{np.mean(means):.3f}+/-{np.std(means):.3f}  "
              f"{np.mean(finals):.3f}+/-{np.std(finals):.3f}  "
              f"{mr:5.1f}+/-{np.std(retrains):.1f}  "
              f"{eff:>11.4f}")
    # Data-grounded Theorem 5.1 quantities (averaged over time and seeds).
    sig, bet, tau, maxD = [], [], [], []
    for data in runs:
        sh = data['shared']
        if 'sigma_sq' in sh:
            sig += [v for v in sh['sigma_sq'] if v == v]
            bet += [v for v in sh['beta'] if v == v]
            tau += [v for v in sh['tau_data'] if v == v]
        maxD.append(max(v for v in sh['D_t'] if v == v))
    if tau:
        import numpy as _np
        frac_bind = _np.mean([t > 0 for t in tau])
        print("  [theorem] data-grounded  sigma^2=%.4f  beta=%.3f  tau=%.4f  "
              "(tau>0 in %.0f%% of steps; max D_t=%.3f)" %
              (_np.mean(sig), _np.mean(bet), _np.mean(tau), 100 * frac_bind, max(maxD)))

    # Paired significance test (across seeds) for the headline claim: does the FULL
    # affective policy beat the no-affect BASE on mean validity?
    paired = []
    for data in runs:
        b = data['strategies'].get('ADAPT_BASE'); f = data['strategies'].get('ADAPT_FULL')
        if b and f:
            paired.append((np.mean(f['metrics']['validity'][1:]),
                           np.mean(b['metrics']['validity'][1:])))
    if len(paired) >= 2:
        diffs = np.array([a - c for a, c in paired])
        md, sd = diffs.mean(), diffs.std(ddof=1)
        se = sd / np.sqrt(len(diffs)) if sd > 0 else 0.0
        t = md / se if se > 0 else float('inf')
        try:
            from scipy import stats
            p = 2 * stats.t.sf(abs(t), df=len(diffs) - 1)
            ptxt = "p=%.3f" % p
        except Exception:
            ptxt = "t=%.2f (scipy not installed)" % t
        sig_txt = "SIGNIFICANT" if (se > 0 and abs(t) > 2.0) else "not significant"
        print("  [FULL vs BASE] mean_val diff = %+.3f +/- %.3f (n=%d)  %s  -> %s" %
              (md, sd, len(diffs), ptxt, sig_txt))

    # Honest compute-confound flag.
    adapt = {k: retrain_means[k] for k in
             ('ADAPT_BASE', 'ADAPT_VALENCE', 'ADAPT_ANXIETY', 'ADAPT_FULL')
             if k in retrain_means}
    if adapt and (max(adapt.values()) - min(adapt.values()) < 0.5):
        print("  [compute-matched] adaptive variants retrain ~equally here -> "
              "validity differences are a fair timing comparison")
    elif adapt:
        print("  [confound] adaptive variants differ in retrain count -> compare "
              "val/retrain, not mean_val (higher mean_val may be just more compute)")
