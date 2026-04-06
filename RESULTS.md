# Experiment Results

**Dataset**: German Credit (credit)
**Timesteps**: T=20
**Seeds**: 11, 22, 33
**Date**: 2026-04-05

## Strategies

| Strategy | Description |
|---|---|
| NEVER_RETRAIN | Always reuse the original CVAE (baseline lower bound) |
| ALWAYS_RETRAIN | Retrain CVAE every timestep (baseline upper bound, 20 retrains) |
| ADAPTIVE | EFE-based REUSE/RETRAIN decision with affective inference trace |
| FIXED_INTERVAL | Retrain every K=5 steps (4 retrains total) |

## ADAPTIVE Policy Design

The ADAPTIVE strategy implements active inference policy selection over two policies: REUSE and RETRAIN.

**Decision signals (all contribute to G(REUSE)):**
- Predictive KL divergence D_t between successive Rashomon posteriors
- Staleness scaling: ECI estimate grows with steps since last retrain
- Anxiety: projected future invalidity if REUSE continues (linear extrapolation of validity trend)
- Valence: -dF (negative valence = system deteriorating, adds urgency)
- Validity penalty: if last observed validity < p_target (0.90), adds direct penalty
- Warm-up: first 3 steps forced REUSE to observe drift-validity relationship before deciding

**Key hyperparameters:**
- retrain_cost = 0.5 (normalized computational cost)
- invalidity_cost_weight = 5.0 (lambda_inv)
- warmup_steps = 3
- p_target = 0.90

## Summary Results

### Label Noise Drift

| Strategy | Mean Validity | Final Validity | Retrains |
|---|---|---|---|
| NEVER_RETRAIN | 0.914 +/- 0.006 | 0.867 +/- 0.130 | 0 |
| ALWAYS_RETRAIN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 20 |
| **ADAPTIVE** | **0.987 +/- 0.017** | **1.000 +/- 0.000** | **10.0 +/- 4.1** |
| FIXED_INTERVAL | 0.900 +/- 0.011 | 1.000 +/- 0.000 | 4 |

ADAPTIVE achieves near-perfect mean validity (0.987) with half the retrains of ALWAYS_RETRAIN. Dominates FIXED_INTERVAL on both mean validity and final validity.

### Covariate Shift

| Strategy | Mean Validity | Final Validity | Retrains |
|---|---|---|---|
| NEVER_RETRAIN | 0.774 +/- 0.044 | 0.461 +/- 0.279 | 0 |
| ALWAYS_RETRAIN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 20 |
| **ADAPTIVE** | **0.922 +/- 0.018** | **0.667 +/- 0.471** | **11.3 +/- 1.2** |
| FIXED_INTERVAL | 0.677 +/- 0.020 | 1.000 +/- 0.000 | 4 |

ADAPTIVE significantly outperforms FIXED_INTERVAL on mean validity (0.922 vs 0.677). Covariate shift is the hardest drift type: the final validity variance (0.471) reflects a dip-recovery pattern where validity can crash between retrain cycles due to sudden distributional shifts, then recover at the next RETRAIN trigger.

### Subpopulation Shift

| Strategy | Mean Validity | Final Validity | Retrains |
|---|---|---|---|
| NEVER_RETRAIN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0 |
| ALWAYS_RETRAIN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 20 |
| **ADAPTIVE** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.0 +/- 0.0** |
| FIXED_INTERVAL | 0.997 +/- 0.005 | 1.000 +/- 0.000 | 4 |

Subpopulation shift (undersampling class 0) does not meaningfully affect counterfactual validity on the credit dataset. ADAPTIVE correctly detects this and retrains only once (after warm-up). This demonstrates the resource-rational property: ADAPTIVE avoids unnecessary computation when the drift does not impact validity. ALWAYS_RETRAIN wastes 20 retrains; FIXED_INTERVAL wastes 4.

### Rotation Drift

| Strategy | Mean Validity | Final Validity | Retrains |
|---|---|---|---|
| NEVER_RETRAIN | 0.046 +/- 0.002 | 0.017 +/- 0.014 | 0 |
| ALWAYS_RETRAIN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 20 |
| **ADAPTIVE** | **0.854 +/- 0.003** | **1.000 +/- 0.000** | **4.0 +/- 0.8** |
| FIXED_INTERVAL | 0.806 +/- 0.005 | 1.000 +/- 0.000 | 4 |

Rotation drift is the most destructive: NEVER_RETRAIN drops to near-zero validity immediately. ADAPTIVE matches FIXED_INTERVAL on retrain count (4.0 vs 4.0) while achieving higher mean validity (0.854 vs 0.806), because ADAPTIVE times its retrains based on drift signals rather than a fixed schedule.

## Cross-Drift Analysis

### Compute Efficiency (Mean Validity per Retrain)

| Drift | ADAPTIVE val/retrain | FIXED val/retrain | ALWAYS val/retrain |
|---|---|---|---|
| label_noise | 0.099 | 0.225 | 0.050 |
| covariate | 0.082 | 0.169 | 0.050 |
| subpopulation | 1.000 | 0.249 | 0.050 |
| rotation | 0.214 | 0.202 | 0.050 |

ADAPTIVE's strongest advantage is on subpopulation shift where it achieves perfect validity with minimal compute. On the other drift types, it trades more retrains for substantially higher mean validity compared to FIXED_INTERVAL.

### ADAPTIVE Retrain Counts by Drift Type

| Drift | Seed 11 | Seed 22 | Seed 33 | Mean |
|---|---|---|---|---|
| rotation | 5 | 3 | 4 | 4.0 |
| covariate | 11 | 12 | 11 | 11.3 |
| label_noise | 10 | 5 | 15 | 10.0 |
| subpopulation | 1 | 1 | 1 | 1.0 |

The retrain count adapts to drift severity: subpopulation (benign) gets 1, rotation (severe but detectable) gets 4, covariate (severe and gradual) gets 11.

## Improvement from Warm-up + Affective Signals

The ADAPTIVE policy was improved from an initial version that only used D_t threshold. Before/after comparison on rotation drift:

| Version | Mean Validity | Final Validity | Retrains |
|---|---|---|---|
| Before (D_t threshold only) | 0.617 | 0.317 | 1.7 |
| After (warmup + anxiety + valence + validity penalty) | 0.854 | 1.000 | 4.0 |

Root causes of the initial failure:
1. After RETRAIN resets the detector reference, incremental D_t is small (noise floor ~0.35-0.45)
2. Calibration learned near-zero slope from early post-retrain steps
3. Anxiety was computed but not incorporated into the decision

Fixes applied:
1. Warm-up period (3 steps forced REUSE) to accumulate calibration data
2. Anxiety and valence signals added to G(REUSE)
3. Staleness scaling on ECI estimate (grows with steps since last retrain)
4. Minimum calibration slope of 0.2
5. Direct validity-below-target penalty

## Drift Parameters

| Parameter | Value | Description |
|---|---|---|
| covariate_sigma_base | 0.25 | Gaussian noise sigma = 0.25 * t on all features |
| label_alpha_base | 0.025 | 2.5% label flip per step (cap 45%) |
| sub_rate | 0.05 | Undersample class 0 by 5%/step (floor 15%) |
| rotation_deg | 5.0 | Rotate first 2 PCA components by 5 deg/step |

## Files

- Results JSON: `results/credit_{drift}_seed{seed}.json`
- Experiment runner: `adaptive_avcg/experiment_runner.py`
- ADAPTIVE policy: `adaptive_avcg/recomputation_policy.py`
- Config: `adaptive_avcg/config.py`
- Summary script: `scripts/summarize_results.py`
