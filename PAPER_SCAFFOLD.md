# Paper Scaffold: Adaptive Amortized Counterfactual Inference via Free Energy Minimization

**Authors**: Jamie Duell, Mahault Albarracin, Alejandro Jimenez Rodriguez
**Target venue**: JMLR / UAI / AISTATS (full paper, ~8-10 pages)
**Status**: Scaffold — awaiting Jamie's input

---

## Title Options

1. "Adaptive Amortized Counterfactual Inference via Free Energy Minimization"
2. "When to Recompute: Resource-Rational Counterfactual Explanations under Model Drift"
3. "Variational Free Energy for Adaptive Counterfactual Generation under Model Multiplicity"

---

## Abstract (draft)

Amortized counterfactual explanation methods trade upfront training cost for fast
inference, but assume a fixed model posterior. In practice, deployed models drift
over time through retraining, data shifts, or architectural updates, silently
invalidating cached explanations. We propose an adaptive framework that unifies
amortized variational counterfactual generation with a free-energy-based decision
rule for when to recompute explanations. Our method extends the Amortized
Variational Counterfactual Generator (AVCG) with a meta-inference layer that
tracks posterior drift via KL divergence between successive Rashomon-restricted
posteriors. Recomputation is triggered only when the expected counterfactual
invalidity — formalized as expected free energy over the explanation space —
exceeds a threshold. We prove that this rule provides probabilistic guarantees on
counterfactual validity under bounded model drift, and show empirically that it
reduces computation by X% while maintaining robustness parity with full
recomputation. The framework provides a principled connection between
counterfactual explainability, amortized inference, and active inference.

---

## 1. Introduction

### Motivation
- CEs are the dominant post-hoc XAI method for algorithmic recourse
  [Karimi et al. 2022, Wachter et al. 2017]
- Robustness problem: CEs are fragile to model changes [Pawelczyk et al. 2022],
  model multiplicity [Black et al. 2022], noisy execution, and input perturbations
  [Jiang et al. 2024 survey]
- Amortization solves the per-instance cost problem [Duell & Fan, AVCG] but
  introduces a NEW problem: when is the amortized model stale?
- No existing work addresses this temporal validity question with a principled
  decision rule

### Contribution (3 claims)
1. **Formal connection**: Jamie's AVCG objective G(x) IS variational free energy.
   We make this explicit and show the structural identity to the Free Energy
   Principle (Section 3).
2. **Adaptive recomputation rule**: We introduce a KL-divergence-based drift
   detector over the Rashomon-restricted posterior, with a decision rule derived
   from expected free energy (Section 4). This determines WHEN to re-amortize.
3. **Probabilistic validity guarantees under drift**: We extend Jamie's Corollary
   3.5 to the temporal setting, bounding the probability of counterfactual
   invalidity as a function of posterior drift magnitude (Section 5).

### Positioning
- Builds directly on AVCG [Duell, SIAM 2026] and PSCE [Duell & Fan, arXiv 2026]
- Connects to resource-rational inference [Lieder & Griffiths 2020]
- First paper to apply active inference / FEP formalism to counterfactual
  explainability

---

## 2. Background

### 2.1 Counterfactual Explanations
- Standard formulation: Eq (1) from survey [Jiang et al. 2024]
- Robustness taxonomy: MC, MM, NE, IC [Jiang et al. 2024 survey]
- Key gap: no temporal/adaptive dimension

### 2.2 Amortized Variational Inference
- VAE framework [Kingma & Welling 2014]
- Amortization gap [Cremer et al. 2018]
- Semi-amortized inference [Kim et al. 2018]
- Key idea: learn q_φ(x'|x) instead of solving optimization per instance

### 2.3 The AVCG Framework (Jamie's work)
- Bayesian predictor: p(y|x,D) = E_{θ~p(θ|D)}[f_θ(x)]
- Rashomon set: Θ_R = {θ : L(θ) ≤ L* + ε}
- Restricted posterior: P_R(θ) = P(θ|D)·1[θ ∈ Θ_R] / Z
- AVCG objective:

  G(x) = -R_ε(x') + D_KL(q_ψ(z|y',x) || p(z)) + λ·E_{q_ψ}[d(g_φ(x,y',z), x)]

  where R_ε(x') = E_{q_ψ}[E_{θ~P_R}[log P_θ(y'|g_φ(x,y',z))]]

- Probabilistic guarantees: Propositions 3.3-3.4, Corollary 3.5

### 2.4 Variational Free Energy and Active Inference
- Free Energy Principle: F = E_q[-log p(o,s)] - H[q(s)] [Friston]
- Expected Free Energy: G(π) = E_q(o,s|π)[-log p(o|C) - log q(s|o) + log q(s)]
- Emotional valence as ΔF [Joffily & Coricelli 2013]
- Key insight: AVCG's G(x) has identical structure to variational free energy

---

## 3. AVCG as Variational Free Energy (Contribution 1)

### 3.1 Structural Identity

Show that AVCG's objective:

  G(x) = -R_ε(x') + D_KL(q_ψ || p(z)) + λ·E[d(x', x)]

maps term-by-term onto variational free energy:

| AVCG term | VFE interpretation |
|---|---|
| -R_ε(x') = -E_q[E_{P_R}[log P_θ(y'\|g_φ(...))]] | Negative expected log-likelihood (accuracy/energy) |
| D_KL(q_ψ(z\|y',x) \|\| p(z)) | Complexity cost (deviation from prior) |
| λ·E[d(g_φ(...), x)] | Proximity cost (analogous to action cost in AIF) |

### 3.2 Interpretation

- q_ψ(z|y',x) = approximate posterior over counterfactual explanations
- g_φ(x, y', z) = generative model (decoder)
- P_R(θ) = model uncertainty (Rashomon-restricted)
- The generator learns an approximate posterior over the space of valid
  counterfactuals, exactly as in amortized variational inference

### 3.3 What This Buys Us

- Makes the connection to FEP/AIF literature precise (not just analogy)
- Opens the door to importing AIF tools: expected free energy, precision
  weighting, policy selection
- Specifically: enables the adaptive recomputation rule (Section 4)

---

## 4. Adaptive Recomputation via Expected Free Energy (Contribution 2)

### 4.1 The Problem: When Is the Amortized Generator Stale?

- AVCG trains q_ψ, g_φ against a FIXED Rashomon-restricted posterior P_R^t(θ)
- Over time, the model is retrained: P_R^t(θ) → P_R^{t+1}(θ)
- When drift is small, cached counterfactuals remain valid
- When drift is large, they become invalid
- Currently: no principled mechanism to decide

### 4.2 Posterior Drift Measure

Define the drift between successive Rashomon-restricted posteriors:

  D_t = D_KL(P_R^{t+1}(θ) || P_R^t(θ))

This measures how much the model space has shifted. In practice, approximated
via MC dropout samples from both time steps.

### 4.3 Expected Counterfactual Invalidity (ECI)

Define the expected counterfactual invalidity under drift:

  ECI_t = E_{x~p(x)} [ E_{θ~P_R^{t+1}} [ 1[M_θ(x') = M_θ(x)] ] ]

i.e., the probability that a cached counterfactual x' is no longer valid under
the new posterior.

### 4.4 The Decision Rule

**Proposition (Adaptive Recomputation Rule)**:

Given:
- Cached generator (q_ψ^t, g_φ^t) trained at time t
- Current posterior P_R^{t+1}(θ)
- Posterior drift D_t = D_KL(P_R^{t+1} || P_R^t)

Then:
- If D_t < τ: REUSE cached generator (no recomputation)
- If D_t ≥ τ: RETRAIN generator on P_R^{t+1}

Where τ is derived from the validity guarantee:

  τ = f(σ², β, target_validity)

connecting to Jamie's Corollary 3.5.

### 4.5 Free Energy Interpretation

The decision rule is itself a policy selection problem:

  π* = argmin_π G(π)

where:
- π ∈ {REUSE, RETRAIN}
- G(REUSE) = expected invalidity cost (low compute, potentially invalid CEs)
- G(RETRAIN) = retraining cost (high compute, fresh CEs)

This is exactly active inference policy selection: minimize expected free energy
over computational policies.

### 4.6 "Anxiety" as Expected Invalidity

Following Joffily & Coricelli (2013):
- Valence ∝ -dF/dt (rate of change of free energy)
- Anxiety = anticipation of increasing F when no policy reduces it

In our framework:
- High D_t + low confidence in REUSE = "anxiety" about cached explanations
- This triggers RETRAIN
- Low D_t = low anxiety = safe to REUSE

This is not metaphorical — it is a formal mapping from the decision rule to
affective inference quantities.

---

## 5. Probabilistic Validity Guarantees under Drift (Contribution 3)

### 5.1 Extending Corollary 3.5 to the Temporal Setting

Jamie's Corollary 3.5 gives:

  P_{θ~P_R}(f_{θ*}(x') > E_{P_R}[f_θ(x')] - β) ≥ 1 - σ²/β²

This holds at a FIXED time. We extend to:

**Theorem 5.1 (Validity under Bounded Drift)**:

Let D_t = D_KL(P_R^{t+1} || P_R^t) ≤ δ. Then:

  P_{θ~P_R^{t+1}}(f_θ(x') > E_{P_R^t}[f_θ(x')] - β - g(δ)) ≥ 1 - (σ² + h(δ))/β²

where g(δ), h(δ) are functions of the drift magnitude that bound the shift in
mean and variance of the predictive distribution.

(Proof sketch: Pinsker's inequality + bounded variance assumption + triangle
inequality on expectations.)

### 5.2 Deriving the Threshold τ

From Theorem 5.1, we can solve for the maximum drift δ that maintains a target
validity probability p_target:

  τ = max{δ : 1 - (σ² + h(δ))/β² ≥ p_target}

This gives a principled, non-arbitrary threshold for the decision rule.

### 5.3 Corollary: Compute Savings Bound

Under stationary model drift with rate μ_drift, the expected number of
retraining events in T time steps is bounded by:

  E[N_retrain] ≤ T · μ_drift / τ

vs. N_retrain = T for naive recomputation at every step.

---

## 6. Experiments

### 6.1 Setup
- Same datasets as AVCG: Adult Income, Credit, Spambase, PneumoniaMNIST
- Simulate model drift: retrain BNN with shifted data at each time step
- Compare: AVCG (retrain every step) vs Adaptive-AVCG (retrain when D_t > τ)

### 6.2 Metrics
- Validity (standard)
- Cross-Model Validity (from AVCG)
- Rashomon Validity Ratio (from AVCG)
- **NEW: Temporal Validity** — validity of cached CEs at time t+k
- **NEW: Computation Savings** — ratio of retraining events to total time steps
- **NEW: Drift-Validity Correlation** — does D_t predict invalidity?

### 6.3 Experiments to Run
1. **Drift magnitude sweep**: vary data shift size, measure validity degradation
2. **Threshold sensitivity**: vary τ, plot Pareto frontier of compute vs validity
3. **Comparison with heuristics**: fixed-interval retraining vs adaptive rule
4. **Ablation**: KL drift vs simpler drift measures (parameter distance, accuracy
   change)

### 6.4 Expected Results
- Adaptive-AVCG maintains >95% of full-retraining validity
- Saves 40-70% of retraining events under typical drift regimes
- KL drift is a better predictor of invalidity than parameter distance

---

## 7. Related Work

### Robust Counterfactual Explanations
- Jiang et al. 2024 survey (4 robustness types: MC, MM, NE, IC)
- Pawelczyk et al. 2020 (CEs under predictive multiplicity)
- Leofante et al. 2023 (MIP for multi-model validity)
- Hamman et al. 2023 (probabilistic guarantees for NNs)
- **Gap**: none address temporal/adaptive dimension

### Amortized Counterfactual Generation
- Duell & Fan 2026 (PSCE — model changes, not amortized)
- Duell 2026 (AVCG — amortized, not adaptive)
- **Gap**: amortization is static; no drift detection

### Active Inference and Free Energy
- Friston et al. (FEP, expected free energy)
- Joffily & Coricelli 2013 (emotional valence)
- Da Costa et al. (active inference formalization)
- **Gap**: never applied to XAI/counterfactual explanation

### Resource-Rational Computation
- Lieder & Griffiths 2020 (resource rationality)
- Gershman & Goodman 2014 (amortized inference in cognition)
- **Connection**: our adaptive rule IS resource-rational — invest compute only
  when expected payoff exceeds cost

### Model Drift Detection
- Gama et al. 2014 (concept drift survey)
- Rabanser et al. 2019 (failing loudly — detecting dataset shift)
- **Connection**: we use KL over restricted posteriors rather than raw data

---

## 8. Discussion

### 8.1 Limitations
- Amortization gap persists (inherent to variational methods)
- MC dropout is a crude posterior approximation
- Drift detection adds overhead (though small vs full retraining)
- Guarantees depend on bounded drift assumption (Assumption 3.1)

### 8.2 Future Directions
- Semi-amortized refinement [Kim et al. 2018] for high-stakes queries
- Path-based counterfactuals (sequential decisions, not just point CEs)
- Multi-agent counterfactuals (what if multiple agents change?)
- Integration with causal models for causal counterfactuals under drift

### 8.3 Broader Impact
- Practical: reduces compute cost of maintaining valid CEs in production
- Theoretical: first formal bridge between FEP/AIF and XAI
- Societal: more reliable algorithmic recourse for individuals affected by
  ML decisions

---

## 9. Conclusion

We have shown that amortized variational counterfactual generation is formally
equivalent to variational free energy minimization, and leveraged this connection
to derive a principled adaptive recomputation rule. The framework provides
probabilistic validity guarantees under model drift while substantially reducing
computational cost. This work establishes a bridge between explainability,
Bayesian inference, and active inference that opens new directions for both
fields.

---

## Appendix

### A. Proof of Theorem 5.1
### B. Implementation Details
### C. Additional Experimental Results
### D. Connection to Joffily-Coricelli Emotional Valence Framework
