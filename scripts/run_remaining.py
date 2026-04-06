"""Run the 3 remaining drift types (covariate, subpopulation, rotation) on credit, 3 seeds each."""
import sys
sys.path.insert(0, r'C:\Users\mahau\OneDrive\Desktop\projects\aif-counterfactual-explanation')

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.experiment_runner import ExperimentRunner

cfg = ExperimentConfig()
runner = ExperimentRunner(cfg, results_dir='./results')

for drift in ['covariate', 'subpopulation', 'rotation']:
    for seed in [11, 22, 33]:
        print(f"\n{'#'*60}")
        print(f"# Running credit / {drift} / seed={seed}")
        print(f"{'#'*60}")
        runner.run_single('credit', drift, seed)

print("\nAll remaining experiments complete!")
