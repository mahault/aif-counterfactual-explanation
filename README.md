# Adaptive Amortized Counterfactual Inference via Free Energy Minimization

Joint project: Jamie Duell, Mahault Albarracin, Alejandro Jimenez Rodriguez

## Overview

This repository contains the research materials for a paper extending the
Amortized Variational Counterfactual Generator (AVCG) with an adaptive
recomputation layer grounded in variational free energy minimization.

**Core question**: When should an amortized counterfactual generator be retrained,
and when can cached explanations be safely reused?

## Contributions

1. **AVCG = Variational Free Energy**: We show that the AVCG objective is
   structurally identical to variational free energy, establishing a formal
   bridge between counterfactual explainability and the Free Energy Principle.

2. **Adaptive Recomputation Rule**: A KL-divergence-based drift detector over
   the Rashomon-restricted posterior, with a decision threshold derived from
   expected free energy. Determines *when* to re-amortize.

3. **Temporal Validity Guarantees**: Extension of AVCG's probabilistic validity
   guarantees (Corollary 3.5) to the temporal setting under bounded model drift.

## Background

This work builds on:

- **AVCG** (Duell, SIAM 2026) -- Amortized variational counterfactual generation
  over the Rashomon set
- **PSCE** (Duell & Fan, arXiv:2601.16659) -- Probabilistically safe
  counterfactual explanations under model changes
- **Robust CE Survey** (Jiang et al., IJCAI 2024) -- Taxonomy of CE robustness:
  Model Changes, Model Multiplicity, Noisy Execution, Input Changes

## Repository Structure

```
PAPER_SCAFFOLD.md     # Full paper outline with section-by-section plan
LITERATURE_REVIEW.md  # Comprehensive lit review (45+ references)
```

## Target Venues

UAI / AISTATS / JMLR

## Key References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Friston et al. (2017). Active Inference: A Process Theory. Neural Computation.
- Joffily & Coricelli (2013). Emotional Valence and the Free-Energy Principle. PLoS Comp Bio.
- Jiang et al. (2024). Robust Counterfactual Explanations in ML: A Survey. IJCAI.
- Duell & Fan (2026). Provably Robust Bayesian CEs under Model Changes. arXiv:2601.16659.
- Lieder & Griffiths (2020). Resource-Rational Analysis. BBS.
- Albarracin et al. (2023). Designing Explainable AI with Active Inference. arXiv:2306.04025.
- Pattisapu et al. (2024). Free Energy in a Circumplex Model of Emotion. IWAI.
