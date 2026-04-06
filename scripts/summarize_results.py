"""Quick summary of all experiment results."""
import json, glob, os, numpy as np

results_dir = r'C:\Users\mahau\OneDrive\Desktop\projects\aif-counterfactual-explanation\results'

for drift in ['label_noise', 'covariate', 'subpopulation', 'rotation']:
    print(f"\n{'='*70}")
    print(f"  DRIFT: {drift}")
    print(f"{'='*70}")

    for strategy in ['NEVER_RETRAIN', 'ALWAYS_RETRAIN', 'ADAPTIVE', 'FIXED_INTERVAL']:
        vals_final = []
        vals_mean = []
        retrains = []

        for seed in [11, 22, 33]:
            fpath = os.path.join(results_dir, f'credit_{drift}_seed{seed}.json')
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                data = json.load(f)

            m = data['strategies'][strategy]['metrics']
            v = m['validity']
            vals_final.append(v[-1])
            vals_mean.append(np.mean(v[1:]))  # skip t=0
            retrains.append(data['strategies'][strategy]['retrain_count'])

        if vals_final:
            print(f"  {strategy:20s}  final_val={np.mean(vals_final):.3f}±{np.std(vals_final):.3f}  "
                  f"mean_val={np.mean(vals_mean):.3f}±{np.std(vals_mean):.3f}  "
                  f"retrains={np.mean(retrains):.1f}±{np.std(retrains):.1f}")
