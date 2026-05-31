"""
Fast, data-free unit check for the PRINCIPLED recomputation rule and its
valence/anxiety ablation.

Rule under test:  RETRAIN iff predicted_invalidity > (1 - p_target), where
    predicted_invalidity = max( inv_now,
                                inv_kl       if use_anxiety (and calibrated),
                                inv_forecast if use_valence )

The four variants share the SAME threshold, so the ablation is compute-fair by
construction; the flags only add anticipatory signals on top of the myopic core.
"""
import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_avcg.config import ExperimentConfig
from adaptive_avcg.recomputation_policy import RecomputationPolicy

# p_target=0.90 -> tolerance = 0.10
# validity_trend = [0.95, 0.93]: inv_now=0.07 (<tol), rate=0.02 -> inv_forecast=0.09 (<tol)
# calibrated drift link gives inv_kl=0.20 (>tol) at D_t=0.5
VTREND = [0.95, 0.93]
D_T = 0.5
CAL_D = [0.0, 0.5, 1.0]
CAL_INV = [0.0, 0.2, 0.4]          # slope 0.4, intercept 0 -> inv_kl(0.5)=0.20


def decide_variant(use_valence, use_anxiety, calibrate=True):
    cfg = replace(ExperimentConfig(), use_valence=use_valence, use_anxiety=use_anxiety)
    pol = RecomputationPolicy(cfg)
    if calibrate:
        pol.calibrate(CAL_D, CAL_INV)
    return pol.decide(D_T, F_t=1.0, F_prev=0.9, validity_trend=VTREND)


base = decide_variant(False, False)
val = decide_variant(True, False)
anx = decide_variant(False, True)
full = decide_variant(True, True)

tol = base['tolerance']
fails = []


def approx(a, b, t=1e-9):
    return abs(a - b) <= t


# Component values are identical across variants (only their inclusion differs).
inv_now = base['inv_now']
inv_kl = base['inv_kl']            # computed even when anxiety off (for the trace)
inv_forecast = base['inv_forecast']

if not approx(inv_now, 0.07):
    fails.append(f"inv_now {inv_now} != 0.07")
if inv_kl is None or not approx(inv_kl, 0.20):
    fails.append(f"inv_kl {inv_kl} != 0.20")
if not approx(inv_forecast, 0.09):
    fails.append(f"inv_forecast {inv_forecast} != 0.09")

# predicted invalidity = max over active signals
if not approx(base['predicted_invalidity'], inv_now):
    fails.append("BASE should use inv_now only")
if not approx(val['predicted_invalidity'], max(inv_now, inv_forecast)):
    fails.append("VALENCE should add inv_forecast")
if not approx(anx['predicted_invalidity'], max(inv_now, inv_kl)):
    fails.append("ANXIETY should add inv_kl")
if not approx(full['predicted_invalidity'], max(inv_now, inv_kl, inv_forecast)):
    fails.append("FULL should use all signals")

# Decisions: only the drift (anxiety) signal crosses tolerance here.
expected = {'BASE': 'REUSE', 'VALENCE': 'REUSE', 'ANXIETY': 'RETRAIN', 'FULL': 'RETRAIN'}
got = {'BASE': base['decision'], 'VALENCE': val['decision'],
       'ANXIETY': anx['decision'], 'FULL': full['decision']}
if got != expected:
    fails.append(f"decisions {got} != {expected}")

# An un-calibrated policy must NOT act on the drift signal (inv_kl is None).
anx_uncal = decide_variant(False, True, calibrate=False)
if anx_uncal['inv_kl'] is not None:
    fails.append("un-calibrated inv_kl should be None")
if anx_uncal['decision'] != 'REUSE':
    fails.append("un-calibrated anxiety must fall back to myopic (REUSE here)")

# Flags propagate to the trace.
for name, d, exp in [('val', val, (True, False)), ('anx', anx, (False, True)),
                     ('full', full, (True, True))]:
    if (d['use_valence'], d['use_anxiety']) != exp:
        fails.append(f"trace flags wrong for {name}")

print(f"tolerance (1-p_target) = {tol:.2f}")
print(f"inv_now={inv_now:.3f}  inv_kl={inv_kl:.3f}  inv_forecast={inv_forecast:.3f}")
print(f"predicted: BASE={base['predicted_invalidity']:.3f} VAL={val['predicted_invalidity']:.3f} "
      f"ANX={anx['predicted_invalidity']:.3f} FULL={full['predicted_invalidity']:.3f}")
print(f"decisions: {got}")

if fails:
    print("\nFAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("\nAll principled-rule + ablation checks passed.")
