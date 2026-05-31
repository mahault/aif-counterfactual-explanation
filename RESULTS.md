# Experiment Results

**Dataset**: German Credit (credit)
**Timesteps**: T=20
**Seeds**: up to 10 (11–101)

## Summary of findings (current)

After an audit found the original policy and evaluation were confounded, the policy was
rebuilt to be principled and the evaluation made leakage-free. Net conclusions (credit;
see sections below for methods, numbers, and significance tests):

1. **Affective terms do not robustly improve validity.** With a fair, un-lagged baseline,
   the valence/anxiety terms give no significant validity gain — in the free-retraining
   regime (10 seeds) *or* under a retrain-budget cooldown sweep (2/27 paired tests reach
   p<0.05, both <0.01 and inconsistent = chance). The original "combination improves
   validity" was an artifact of a lagged baseline and a compute-confounded metric.
2. **The decision rule is now principled**, not tuned: `RETRAIN iff predicted invalidity
   > (1 − p_target)`, single knob, no fitted cost weights.
3. **Theorem 5.1 is non-vacuous** when its σ²/β² are estimated on a *fresh* generator
   (β≈0.43>0, τ>0); the earlier "vacuous" reading was a stale-generator measurement
   artifact.
4. **The defensible positive result — free energy as a label-efficient recomputation
   signal.** `F` tracks invalidity within-run (AUC≈0.95). A label-free F-policy with
   **sparse re-calibration (F_RECAL)** matches per-step-oracle validity on every drift
   type (no significant gap, n=5) at **~50% of the labelling cost**. A fully label-free
   change-detector (F_TRIGGER) is too fragile (fails on abrupt drift).

Caveats: credit only, small n — needs more seeds / equivalence tests and confirmation on
adult_income / spambase / pneumonia before publication.

**Date**: original 2026-04-05; principled rebuild + label-free study 2026-05-30.

> ⚠️ **SUPERSEDED (v1).** The sections from here through "Improvement from Warm-up +
> Affective Signals" describe the original **tuned** ADAPTIVE policy (EFE cost weights,
> warm-up, validity penalty, `min_slope`) and the 3-seed results in `results/`. An audit
> found this policy and its evaluation were confounded (tuned-to-eval, compute-confounded
> metric, lagged baseline, eval leakage, inert theorem threshold). It has been replaced
> by a principled single-knob rule with a leakage-free protocol — see
> **"Adaptive Recomputation Policy & Affective-Term Ablation"** below for the current
> design, the audit history, and the (in-progress) corrected results. The `ADAPTIVE`
> numbers in this upper section have not been regenerated under the new policy.

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

## Adaptive Recomputation Policy & Affective-Term Ablation

> **Status (2026-05-29):** the policy and evaluation were rebuilt to be principled
> after an audit found the v1 results were confounded. The full 10-seed credit sweep
> with the corrected protocol is **running**; the table below reports the first
> completed run only and is marked PRELIMINARY. v1 (tuned-policy) numbers have been
> **withdrawn** — see "Audit history".

### Decision rule (principled)

Recompute exactly when the cached generator is predicted to exceed the invalidity we
are willing to tolerate:

> **RETRAIN  iff  predicted_invalidity > (1 − p_target)**

`p_target` is a stated requirement and the *only* knob; the threshold is its direct
consequence. There are no fitted cost weights. `predicted_invalidity` is the worst
case over up to three signals, all in invalidity units in [0, 1] (so the four
variants share one threshold and the ablation is compute-fair by construction):

| Signal | Meaning | Ablation flag |
|---|---|---|
| `inv_now` | invalidity already realised by cached CFs under the current model (myopic core; measured at decision time, no labels) | always on |
| `inv_kl` | "anxiety": drift-based anticipation via an honestly-calibrated `D_t→invalidity` link (no slope floor; `None` until calibrated) | `use_anxiety` |
| `inv_forecast` | "valence": one-step trend extrapolation (discrete −dF analogue) | `use_valence` |

Variants: `ADAPT_BASE` (neither flag) / `+VALENCE` / `+ANXIETY` / `+FULL`;
`NEVER_RETRAIN` / `ALWAYS_RETRAIN` are reference bounds.

### Evaluation protocol (leakage-free, un-lagged)

- **Decisions** use the cached generator's **current-step** validity on a held-out
  **validation** split (`val_validity`). Reported **validity** is measured on the
  **test** split. The policy therefore never sees the metric it is scored on.
- **Un-lagged BASE:** because `inv_now` is the *current* (not previous) step's
  measured invalidity, the myopic baseline is a fair, strong reactive controller —
  not a one-step-lagged strawman.
- **Theorem 5.1 quantities** (`σ²`, `β`, `τ`) are estimated on a **fresh**
  (always-retrained) generator against the current posterior — the well-specified
  case the bound is about — never on a stale generator under drift.

**Run / verify:**
```
python scripts/run_ablation.py --dataset credit          # full sweep + fig8
python scripts/summarize_ablation.py results_ablation     # table + significance test
python scripts/test_ablation_logic.py                     # fast, data-free rule check
```

### Audit history — what was wrong in v1 and how it was fixed

| v1 problem | Why it biased the result | Fix |
|---|---|---|
| **Tuned policy** (`retrain_cost`, `invalidity_cost_weight`, `staleness`, `min_slope=0.2`, `warmup`, validity penalty); iterated against the eval | "works because fitted to the test" | Replaced by the single-knob `predicted_inv > 1−p_target` rule; old constants deprecated |
| **Reported mean validity alone** | rises just by retraining more — conflates timing with compute | Report validity **with** retrains (`val/retrain`, Pareto `fig8`); shared threshold |
| **Lagged BASE** (`inv_now` used previous step) | handicaps the baseline, inflates the "anticipation helps" effect | `inv_now` = current-step validation validity |
| **Eval leakage** (decisions used test-set validity) | policy driven by the scored metric | decide on validation, report on test |
| **Theorem `τ` inert / vacuous** | `σ²,β` were placeholders / measured on stale CFs (β<0) | `σ²,β` estimated from the fresh-generator Rashomon spread |

### Premise: does predictive KL drift predict invalidity?

Correlation of `D_t` (absolute predictive KL from t=0) with NEVER_RETRAIN validity
drop, pooled over seeds (policy-independent; from the v1 sweep, still informative):

| Drift | R² |
|---|---|
| label_noise | 0.41 (moderate) |
| covariate | 0.27 (weak–moderate) |
| subpopulation | 0.05 (validity never drops) |
| **rotation** | **0.02, r=−0.15 (no relationship)** |

Predictive KL is a moderate invalidity predictor for *gradual* drift and carries
essentially **no** linear signal for *abrupt* (rotation) drift (validity saturates
near 0 while `D_t` keeps rising). The "KL predicts invalidity" premise is
drift-type-dependent, not universal.

### Theorem 5.1 (data-grounded): non-vacuous on a fresh generator

Estimated correctly (fresh generator vs current posterior), the bound has teeth.
First completed run (covariate, seed 11, T=20):

- `β` (margin) **positive at 100% of steps**, mean ≈ 0.43.
- `τ` (admissible drift, `β²(1−p) − σ²`) **positive at 100% of steps**, 0.024 → 0.007,
  shrinking sensibly as `σ²` grows with drift.

The earlier "vacuous / β<0" reading was a **measurement artifact** of estimating the
bound on a *stale* generator under drift, not a property of the theorem.

### Results — PRELIMINARY (1 of 40 runs; full 10-seed sweep running)

German Credit, covariate, seed 11, T=20. Reported (test) validity:

| Variant | mean validity | retrains |
|---|---|---|
| NEVER_RETRAIN | 0.760 | 0 |
| ADAPT_BASE (fair, no affect) | 0.995 | 10 |
| +VALENCE | 0.996 | 11 |
| +ANXIETY | 0.994 | 12 |
| +FULL | 0.995 | 12 |
| ALWAYS_RETRAIN | 1.000 | 20 |

**Preliminary finding:** with a *fair* (un-lagged) baseline, the four variants are
indistinguishable (~0.995) and the affective ones retrain **more** (12 vs 10) for **no**
validity gain. The v1 "combination improves validity" was an artifact of the lagged
baseline. The principled reactive core ("retrain when current measured invalidity
exceeds tolerance") already suffices on this data.

If this holds across the full sweep, the honest message is roughly the *opposite* of
the optimistic framing: a simple principled reactive rule is enough; valence/anxiety
anticipation adds no measurable validity at matched-ish compute here. This does not
undermine the framework (adaptive recomputation, drift detection, the now-non-vacuous
validity bound all stand) — it scopes the *affective* contribution. Anticipation may
still pay off where current validity is unmeasurable or drift is abrupt/costly enough
that a single invalid step matters — to be checked per drift type in the full sweep.

### Retrain-budget Pareto (cooldown sweep) — the decisive test

The free-retraining regime above has a ceiling (BASE ~0.99, near-zero invalid steps),
so it cannot reveal whether *timing* helps. We re-ran credit under a retrain **cooldown**
c ∈ {0, 2, 5} (adaptive variants may retrain at most once per c+1 steps), tracing a
validity-vs-compute Pareto (5 seeds; covariate/label_noise/rotation).
Figure: `results_cooldown/fig9_cooldown_pareto.png`.

The cooldown breaks the ceiling (e.g. covariate c=5 drops to ~0.75), so there is real
headroom. **Paired test of each affective term vs BASE at matched budget** (mean
validity diff [p-value], n=5):

| drift | c | +VALENCE | +ANXIETY | +FULL |
|---|---|---|---|---|
| covariate | 0 | +0.002 [.19] | +0.002 [.15] | +0.004 [.11] |
| covariate | 2 | +0.020 [.50] | −0.019 [.69] | −0.001 [.94] |
| covariate | 5 | −0.032 [.12] | +0.005 [.85] | −0.022 [.52] |
| label_noise | 0 | +0.007 [.01]* | +0.004 [.10] | +0.005 [.10] |
| label_noise | 2 | −0.034 [.12] | +0.008 [.04]* | +0.004 [.83] |
| label_noise | 5 | +0.000 [.99] | +0.012 [.52] | +0.008 [.73] |
| rotation | 0 | +0.001 [.26] | −0.000 [.88] | +0.002 [.10] |
| rotation | 2 | +0.004 [.06] | +0.001 [.68] | +0.005 [.07] |
| rotation | 5 | −0.003 [.49] | −0.013 [.37] | −0.004 [.81] |

**Verdict: no robust benefit from the affective terms, even under a retrain budget.**
Only 2 of 27 comparisons reach p<0.05 — both <0.01 in magnitude, both inconsistent
across budgets (VALENCE is +0.007 at c=0 but −0.034 at c=2), and ~1–2 false positives
are expected from 27 uncorrected tests. Several effects are *negative*. The principled
reactive BASE — `RETRAIN iff current measured invalidity > (1 − p_target)` — is a
strong, simple, single-knob rule that valence/anxiety anticipation does not reliably
beat, in any drift type or budget tested on credit.

This was tested in the regime most favorable to anticipation (constrained compute,
real headroom). The affective elaboration may still help where current validity is
*unmeasurable* at decision time, under a different cost model, or on harder datasets —
none of which is shown here. As it stands, the defensible contribution is the
principled adaptive-recomputation framework + the (non-vacuous) validity bound + the
drift detector, **not** an affective-term validity improvement.

### Label-free recomputation via free energy (credit, c=0, 5 seeds)

Motivated by the amortized setting (avoid re-checking validity every step), we test
whether the free energy `F` — computable WITHOUT validity labels — can drive
recomputation. `F_POLICY` calibrates `F → invalidity` on the first 5 labelled steps,
then decides label-free (`RETRAIN iff F-predicted invalidity > 1 − p_target`).
`ORACLE_BASE` measures validity every step (20 labels). Diagnostic first established
that `F` tracks current invalidity strongly within-run (r≈0.78–0.90, AUC≈0.91–0.96,
uniformly across drift/budget) — but it is the free-energy *level*, not valence
(the `−dF` rate, r≈0.04–0.47), that carries the signal.

Label-free policies (calibrate on first 5 steps, then decide label-free) vs the
per-step ORACLE_BASE (20 labels). `FDT_POLICY` uses max(inv_F, inv_Dt):

| Drift | ORACLE | F_POLICY | FDT_POLICY | FDT gap [p] | FDT retrains | labels |
|---|---|---|---|---|---|---|
| rotation | 0.999 | 1.000 | 0.999 | +0.000 [.78] | 5.0 | 5 vs 20 (75% fewer) |
| label_noise | 0.993 | 0.886 | 0.888 | −0.105 [.011] | 0.0 | 5 vs 20 |
| covariate | 0.990 | 0.807 | 0.820 | −0.171 [.001] | 0.4 | 5 vs 20 |

**Finding: label-free works for abrupt drift, fails for gradual — and the cause is the
calibration protocol, not the signal.** On rotation, F (and FDT) match the oracle at
75% fewer labels. On gradual drifts both collapse to ~NEVER (0 retrains). Adding D_t
(FDT) does **not** help — confirming the blocker isn't signal choice.

**Confirmed mechanism:** on gradual drift the 5-step calibration window contains
invalidity `[0,0,0,0,0]` (degradation hasn't started), so the F→inv and D_t→inv fits
are trained on zero-variance targets and predict 0.0 invalidity forever, even as actual
invalidity later spikes to 0.5–0.8. The signal is real (F tracks invalidity r≈0.8); the
calibration simply never sees invalidity to learn from.

**Implication:** the fix is the calibration protocol, not more signals — either
**sparse periodic re-calibration** (spend a label every M steps so the fit updates once
invalidity appears) or a **calibration-free relative trigger** (retrain when F rises a
set amount above its post-retrain baseline; a change-detector needing no invalidity
labels). The complementarity that *does* hold: `D_t` predicts gradual-drift invalidity
(R²≈0.3–0.4) and `F` predicts abrupt — useful once calibration is fixed.

### Fixing the calibration: sparse re-calibration works (credit, c=0, 5 seeds)

The dead-window failure is a calibration-protocol problem, so we tested two fixes vs
the per-step ORACLE_BASE (20 labels): **F_RECAL** (refit F→invalidity every 3 steps;
~10 labels) and **F_TRIGGER** (fully label-free 3σ change-detector on F; 0 labels).

| Drift | ORACLE | F_RECAL | gap [p] | F_TRIGGER | gap [p] |
|---|---|---|---|---|---|
| covariate | 0.990 | 0.951 | −0.039 [.32 n.s.] | 0.793 | −0.197 [.019] |
| label_noise | 0.995 | 0.978 | −0.017 [.27 n.s.] | 0.899 | −0.096 [.015] |
| rotation | 0.994 | 0.998 | +0.004 [.40 n.s.] | 0.071 | −0.923 [.000] |

(F_RECAL: 10 labels vs ORACLE's 20; F_TRIGGER: 0 labels.)

**Result — the headline, honest positive finding: F_RECAL matches the oracle on every
drift type** (no significant difference at n=5; point gaps −0.04 to +0.004) **at half
the labelling cost.** Sparse re-calibration fixes the gradual-drift collapse exactly as
the mechanism predicted: once a re-cal step observes nonzero invalidity, the F→invalidity
fit learns the slope and the policy retrains correctly. So **free energy is a
label-efficient recomputation signal — with light periodic re-calibration it achieves
oracle-level counterfactual validity at ~50% of the labels, across drift types.**

**F_TRIGGER (zero-label) fails**, catastrophically on rotation (0.2 retrains → 0.071
validity). Its σ is estimated from the initial window, but rotation drifts immediately,
so the window already has large F-variance and the 3σ threshold becomes unreachable.
Fully label-free recomputation via a naive change-detector is too fragile; the few
re-calibration labels are what make it robust.

Caveat: n=5; "matches oracle" is non-significance, not proven equivalence — confirm with
more seeds / an equivalence test, and the 50% label figure depends on `recal_interval`.
This is the defensible affective/free-energy contribution; the original
"valence-rate improves validity" claim does not hold (see above).

### Known limitations / open

- **Full 10-seed × 4-drift sweep pending** (this section will be completed with the
  aggregate table, paired FULL-vs-BASE significance test, and data-grounded `τ`
  summary on completion). Current numbers = **one run**.
- **Single dataset** (credit). `adult_income` / `spambase` / `pneumonia` not yet run.
- Watch **rotation** specifically: abrupt drift is where current-step validity could
  lag the drift enough that anticipation might genuinely help — the one regime that
  could still support the affective terms.

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
