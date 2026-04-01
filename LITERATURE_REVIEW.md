# Literature Review
## Adaptive Amortized Counterfactual Inference via Free Energy Minimization

**Prepared by**: Mahault Albarracin
**Date**: April 2026
**For**: Joint paper with Jamie Duell and Alejandro Jimenez Rodriguez

---

## 1. Counterfactual Explanations: Foundations and Robustness

### 1.1 Foundational CE Methods

**Wachter, S., Mittelstadt, B. & Russell, C. (2017).** "Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR." *Harvard Journal of Law & Technology*, 31, 841--887.
Foundational paper for CEs in ML. Proposes finding the minimal input change x' that flips a model's decision, formulated as: argmin_{x'} loss(f(x'), y*) + lambda * d(x, x'). Establishes the optimization-based paradigm that all subsequent CE methods build upon. The amortized approach (AVCG) can be seen as learning to solve this optimization in a single forward pass. [arXiv:1711.00399]

**Tolomei, G., Silvestri, F., Haines, A. & Lalmas, M. (2017).** "Interpretable Predictions of Tree-Based Ensembles via Actionable Feature Tweaking." *KDD 2017*.
Early work on actionable counterfactual explanations for tree ensembles, introducing the idea that explanations should suggest feasible changes. Along with Wachter et al., one of the two seminal 2017 papers that launched the CE field.

**Karimi, A.-H., Barthe, G., Scholkopf, B. & Valera, I. (2022).** "A Survey of Algorithmic Recourse: Contrastive Explanations and Consequential Recommendations." *ACM Computing Surveys*, 55(5), 1--29.
Comprehensive survey distinguishing counterfactual *explanations* (what to change) from algorithmic *recourse* (how to change it). Covers properties: validity, actionability, causality, diversity, plausibility. Our work inherits validity and proximity by default and adds temporal validity as a new property.

**Mothilal, R.K., Sharma, A. & Tan, C. (2020).** "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations." *FAccT 2020*.
Introduces DiCE: diverse counterfactual explanations via determinantal point processes. Emphasizes that users benefit from multiple actionable alternatives. Relevant because the amortized generator naturally produces diverse CEs by sampling from the learned posterior q_psi(z|y',x).

### 1.2 Robustness of Counterfactual Explanations

**Jiang, J., Leofante, F., Rago, A. & Toni, F. (2024).** "Robust Counterfactual Explanations in Machine Learning: A Survey." *IJCAI-24 Survey Track*, 8086--8094.
First comprehensive survey on robust CEs. Categorizes robustness into four types: **Model Changes (MC)** -- validity after retraining; **Model Multiplicity (MM)** -- validity across Rashomon set; **Noisy Execution (NE)** -- validity under imprecise user implementation; **Input Changes (IC)** -- consistency for similar inputs. Reviews 30+ methods across all four categories. **Key gap identified by our work**: none of the surveyed methods address the *temporal/adaptive* dimension -- when to recompute CEs as models evolve. All methods are static: they generate robust CEs at a point in time but provide no mechanism for detecting when those CEs become stale.

**Pawelczyk, M., Agarwal, C., Joshi, S., Upadhyay, S. & Lakkaraju, H. (2022).** "Exploring Counterfactual Explanations Through the Lens of Adversarial Examples: A Theoretical and Empirical Analysis." *AISTATS 2022*.
Demonstrates that popular CE methods generate explanations indistinguishable from adversarial examples -- extremely susceptible to small perturbations. This motivates the entire robustness agenda and directly motivates the need for monitoring CE validity over time.

### 1.3 Robustness against Model Changes (MC)

**Upadhyay, S., Joshi, S. & Lakkaraju, H. (2021).** "Towards Robust and Reliable Algorithmic Recourse." *NeurIPS 2021*.
ROAR: min-max robust optimization for CEs under plausible model changes (bounded parameter perturbations). Inner maximization finds worst-case perturbation; outer minimization finds the best CE under it. Provides the adversarial training baseline for MC robustness. Our approach differs fundamentally: instead of hardening CEs against worst-case perturbations, we detect when drift has made existing CEs unreliable and trigger recomputation.

**Black, E., Wang, Z. & Fredrikson, M. (2022).** "Consistent Counterfactuals for Deep Models." *ICLR 2022*.
Shows that for non-linear models, high class scores alone are insufficient for robustness -- CEs must also be located in regions with low Lipschitz constants. Complementary to our drift-based approach: their method hardens individual CEs, while ours monitors the posterior globally.

**Hamman, F., Noorani, E., Mishra, S., Magazzeni, D. & Dutta, S. (2023).** "Robust Counterfactual Explanations for Neural Networks with Probabilistic Guarantees." *ICML 2023*.
Generates CEs robust to naturally-occurring model changes by requiring high class scores for both the CE and its neighbors. Provides probabilistic VaR guarantees. The probabilistic guarantee structure (via Chebyshev) is similar to Jamie's AVCG propositions, which we extend to the temporal setting.

**Jiang, J., Lan, J., Leofante, F., Rago, A. & Toni, F. (2023).** "Provably Robust and Plausible Counterfactual Explanations for Neural Networks via Robust Optimisation." *ACML 2023*.
MIP-based constrained optimization for provably robust CEs under bounded parameter changes. Provides deterministic (not probabilistic) guarantees but at higher computational cost. Our approach trades deterministic for probabilistic guarantees but gains the adaptive temporal dimension.

**Krishna, S., Ma, J. & Lakkaraju, H. (2023).** "Towards Bridging the Gaps between the Right to Explanation and the Right to Be Forgotten." *ICML 2023*.
Studies CE robustness under data deletion (right to be forgotten). Uses leave-k-out analysis to find CEs with higher class scores. Relevant because data deletion is a specific form of model drift that our framework subsumes.

### 1.4 Robustness against Model Multiplicity (MM)

**Breiman, L. (2001).** "Statistical Modeling: The Two Cultures." *Statistical Science*, 16(3), 199--231.
Coined the term "Rashomon set" -- the collection of models that perform equally well on a given task. The foundational observation that model choice is arbitrary when many equally-good models exist, with profound implications for explanation stability.

**Pawelczyk, M., Broelemann, K. & Kasneci, G. (2020).** "On Counterfactual Explanations under Predictive Multiplicity." *UAI 2020*.
First to demonstrate that CEs generated on one model have high probability of invalidation by other equally-performing models. Shows plausibility-cost tradeoff: more robust CEs require higher cost (further from decision boundary). Theoretical foundation for Jamie's AVCG work.

**Leofante, F., Botoeva, E. & Rajani, V. (2023).** "Counterfactual Explanations and Model Multiplicity: A Relational Verification View." *KR 2023*.
Uses product constructions from relational verification to generate CEs guaranteed valid across a set of ReLU neural networks. Proves that finding a CE valid across piece-wise linear models is NP-complete. Formal verification approach complementary to our statistical free-energy approach.

**Jiang, J., Leofante, F., Rago, A. & Toni, F. (2024).** "Recourse under Model Multiplicity via Argumentative Ensembling." *AAMAS 2024*.
Uses computational argumentation to find subsets of models and CEs satisfying non-emptiness, model agreement, counterfactual validity, and coherence. Nominated for Best Paper. Different solution strategy from AVCG: argumentative ensembling vs. variational amortization.

**Black, E., Raghavan, M. & Barocas, S. (2022).** "Model Multiplicity: Opportunities, Concerns, and Solutions." *FAccT 2022*.
Distinguishes procedural multiplicity (same accuracy, different parameters) from predictive multiplicity (different predictions for same inputs). Provides the theoretical framework for the Rashomon set that Jamie's AVCG operates over.

### 1.5 Robustness against Noisy Execution (NE) and Input Changes (IC)

**Dominguez-Olmedo, R., Karimi, A.-H. & Scholkopf, B. (2022).** "On the Adversarial Robustness of Causal Algorithmic Recourse." *ICML 2022*.
Robust optimization for CEs under execution noise, treating perturbations as adversarial attacks on the CE itself. For linear models, proves increasing class scores suffices.

**Pawelczyk, M., Datta, T., van den Heuvel, J., Kasneci, G. & Lakkaraju, H. (2023).** "Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse." *ICLR 2023*.
Introduces invalidation rate (IR) as a metric and allows users to specify acceptable IR levels, controlling the cost-robustness tradeoff. The IR concept is related to our Expected Counterfactual Invalidity (ECI) but ECI operates over temporal drift rather than execution noise.

**Slack, D., Hilgard, A., Lakkaraju, H. & Singh, S. (2021).** "Counterfactual Explanations Can Be Manipulated." *NeurIPS 2021*.
Shows CEs can be adversarially manipulated to produce unfair recourse costs across protected groups. Raises fairness concerns that connect to robustness -- our temporal monitoring could detect when CEs become discriminatory after model updates.

**Leofante, F. & Potyka, N. (2024).** "Promoting Counterfactual Robustness through Diversity." *AAAI 2024*.
Shows that single-instance CE robustness (||x'_1 - x'_2|| <= k||x_1 - x_2||) may be impossible for traditional methods. Proposes diverse CE sets as a relaxed robustness guarantee. Complementary to our approach: diversity addresses IC robustness, we address temporal MC/MM robustness.

---

## 2. Amortized Variational Inference

### 2.1 Foundations

**Kingma, D.P. & Welling, M. (2014).** "Auto-Encoding Variational Bayes." *ICLR 2014*. [arXiv:1312.6114]
Introduces the VAE: a recognition network (encoder) amortizes posterior inference by learning a parametric mapping from data to approximate posterior parameters, trained end-to-end via the reparameterization trick. The foundational framework for Jamie's AVCG -- the encoder q_psi and generator g_phi are exactly the VAE encoder-decoder, but the reconstruction objective is replaced by counterfactual validity across the Rashomon set.

### 2.2 The Amortization Gap

**Cremer, C., Li, X. & Duvenaud, D. (2018).** "Inference Suboptimality in Variational Autoencoders." *ICML 2018*. [arXiv:1801.03558]
Decomposes the VAE inference gap into the *approximation gap* (limited family of q) and the *amortization gap* (encoder can't optimize per-instance). Finds amortization is often the dominant failure mode. This is the theoretical foundation for why our epsilon-based switching is needed: the amortization gap is the specific failure mode that accumulates under model drift.

### 2.3 Closing the Gap

**Kim, Y., Wiseman, S., Miller, A.C., Sontag, D. & Rush, A.M. (2018).** "Semi-Amortized Variational Autoencoders." *ICML 2018*. [arXiv:1802.02550]
Hybrid approach: use amortized encoder to *initialize* variational parameters, then run SVI refinement steps. The refinement is differentiable, so training is end-to-end. Directly motivates a future extension of our work: use the free-energy signal not just for REUSE/RETRAIN but to gate semi-amortized refinement on specific instances.

**Marino, J., Yue, Y. & Mandt, S. (2018).** "Iterative Amortized Inference." *ICML 2018*. [arXiv:1807.09356]
Learns an inference network that iteratively refines posterior estimates by encoding gradient signals. Closes the amortization gap while retaining amortization's speed benefits. Could be combined with our drift detector: low drift -> single forward pass; medium drift -> iterative refinement; high drift -> full retraining.

### 2.4 Amortized Inference in Cognition

**Gershman, S.J. & Goodman, N.D. (2014).** "Amortized Inference in Probabilistic Reasoning." *CogSci 2014*.
Proposes that human cognition uses amortized inference -- learning to map observations directly to posterior beliefs without costly per-instance computation. The brain trades accuracy for speed, with the amortization gap as a cognitive bias. Our framework formalizes when this trade is acceptable (low drift) vs. when full inference is needed (high drift).

### 2.5 Surveys

**Ganguly, A., Jain, S. & Watchareeruetai, U. (2024).** "Amortized Variational Inference: A Systematic Review." *JAIR*, 78. [arXiv:2209.10888]
Comprehensive review of amortized VI: mathematical foundations, amortization gap, generalization, posterior collapse. Useful survey anchor for positioning our contribution.

---

## 3. Active Inference and the Free Energy Principle

### 3.1 Core Formalism

**Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P. & Pezzulo, G. (2017).** "Active Inference: A Process Theory." *Neural Computation*, 29(1), 1--49.
The comprehensive process-theory paper. Derives perception, action, and planning from variational free energy minimization under a POMDP generative model. Our paper shows Jamie's AVCG objective has identical structure to this variational free energy, making the AVCG a special case of active inference where the "action" is generating a counterfactual and the "perception" is inferring over the Rashomon set.

**Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V. & Friston, K. (2020).** "Active Inference on Discrete State-Spaces: A Synthesis." *Journal of Mathematical Psychology*, 99. [arXiv:2001.07203]
Complete mathematical synthesis of active inference on discrete state-spaces. Derives belief update, policy evaluation via expected free energy, and action selection from first principles. Technical reference for the discrete formulation underlying our adaptive decision rule.

**Parr, T. & Friston, K.J. (2019).** "Generalised Free Energy and Active Inference." *Biological Cybernetics*, 113(5--6), 495--513.
Shows that generalised free energy = VFE + EFE, unifying perception and planning. The EFE decomposes into epistemic (exploration) and pragmatic (exploitation) terms. Our adaptive rule can be understood through this lens: REUSE is pragmatic (exploit cached inference), RETRAIN is epistemic (explore new posterior).

### 3.2 VFE vs EFE Distinction

**Millidge, B., Tschantz, A. & Buckley, C.L. (2021).** "Whence the Expected Free Energy?" *Neural Computation*, 33(2), 447--482. [arXiv:2004.08128]
Critical clarification: EFE is NOT simply "VFE projected into the future" -- that would actually discourage exploration. EFE has a distinct derivation involving counterfactual outcomes under policies. **Important for our paper**: VFE is the right signal for detecting current inference failure (amortization staleness), while EFE governs the prospective decision of whether to retrain. We must be precise about which quantity we use where.

### 3.3 Emotion and Affect as Free Energy Dynamics

**Joffily, M. & Coricelli, G. (2013).** "Emotional Valence and the Free-Energy Principle." *PLoS Computational Biology*, 9(6), e1003094.
Defines emotional valence as the negative rate of change of free energy (dF/dt) and arousal as the absolute magnitude of this rate. The *dynamics* of free energy -- not just its magnitude -- carry functionally meaningful information. **This is the direct formal basis for our "anxiety" concept**: a rising free energy rate over the counterfactual generator indicates the amortized solution is degrading, triggering recomputation.

**Pattisapu, C., Verbelen, T., Pitliya, R.J., Kiefer, A.B. & Albarracin, M. (2024).** "Free Energy in a Circumplex Model of Emotion." *IWAI 2024*, CCIS vol. 2193, Springer. [arXiv:2407.02474]
Maps emotions into a two-dimensional circumplex (valence x arousal) derived from expected free energy: valence = utility gap, arousal = entropy of posterior beliefs. **Directly relevant**: provides the formal mapping from our drift measure D_t to an affective signal -- high D_t with uncertain validity maps to high arousal + negative valence (anxiety).

**Albarracin, M., Bouchard-Joly, G., Sheikhbahaee, Z., Miller, M., Pitliya, R.J. & Poirier, P. (2024).** "Feeling Our Place in the World: An Active Inference Account of Self-Esteem." *Neuroscience of Consciousness*, 2024(1), niae007.
Self-esteem as an inferential sociometer under active inference -- the capacity to interpret standing and modulate behavior through affective appraisal. Demonstrates that affective signals serve as meta-cognitive monitors of inference quality, directly supporting our use of "anxiety" as a computational signal for counterfactual staleness.

### 3.4 Distributional Robustness and Active Inference

**Tschantz, A. et al. (2025).** "Distributionally Robust Free Energy Principle for Decision-Making." *Nature Communications*, 2025. [arXiv:2503.13223]
DR-FREE: extends active inference with distributional robustness -- policies minimize the *maximum* free energy over an ambiguity set of environments. Parallel approach to ours: DR-FREE handles model ambiguity at the policy level, our work handles it at the inference level for counterfactuals. Could inform future extensions where the Rashomon set itself is uncertain.

---

## 4. Active Inference and Social Cognition (Albarracin et al.)

These papers establish the theoretical foundation for applying active inference to social and decision-theoretic contexts, which grounds the extension to counterfactual explainability.

### 4.1 Variational Approaches to Social Structures

**Albarracin, M., Constant, A., Friston, K.J. & Ramstead, M.J.D. (2021).** "A Variational Approach to Scripts." *Frontiers in Psychology*, 12, 585493.
Formalizes social scripts as sequences of perception-action loops under active inference -- social behavior as variational inference over cultural affordances. **Relevant because**: counterfactual explanations are themselves "scripts" for recourse -- sequences of changes an individual should make. The variational treatment of scripts maps onto our variational treatment of counterfactuals.

**Albarracin, M., Demekas, D., Ramstead, M.J.D. & Heins, C. (2022).** "Epistemic Communities under Active Inference." *Entropy*, 24(4), 476.
Computational model of confirmation bias and echo chambers: active inference agents preferentially sample information confirming existing beliefs. **Relevant because**: model multiplicity creates an "epistemic community" of models that agree on accuracy but disagree on explanations. The Rashomon set IS an epistemic community in this sense.

**Albarracin, M. & Pitliya, R.J. (2022).** "The Nature of Beliefs and Believing." *Frontiers in Psychology*, 13, 981925.
Reviews belief formation, updating, and social conformity effects under the free energy framework. Relevant to the question of how users update their beliefs about the reliability of counterfactual explanations -- trust in CEs is itself a belief-updating process.

**Hyland, D. & Albarracin, M. (2025).** "On the Variational Costs of Changing Our Minds." *IWAI 2025*. [arXiv:2509.17957]
Formalizes belief revision as a motivated variational decision where agents weigh belief utility against KL-divergence costs. Shows confirmation bias and polarization emerge as resource-rational strategies. **Directly relevant**: the decision to REUSE vs RETRAIN is a belief revision problem -- changing from "my cached CEs are valid" to "I need to recompute" has a variational cost (the retraining compute), and agents should only pay it when the evidence (drift D_t) is sufficient.

### 4.2 Multi-Agent Inference and Empathy

**Albarracin, M., Mikeda, A., Jimenez Rodriguez, A., Namjoshi, S., Sakthivadivel, D.A.R., Pae, H., Shah, H. & Wilson, P. (2026).** "Empathy Modeling in Active Inference Agents for Perspective-Taking and Alignment." [arXiv:2602.20936]
Self-other model transformation in active inference: reciprocal empathy induces cooperation without communication. **Relevant for future direction**: multi-agent counterfactuals where one agent's recourse affects another's outcomes. The empathy mechanism provides a framework for generating CEs that account for social externalities.

**Albarracin, M., Pitliya, R.J., Smithe, T.S.C., Friedman, D.A., Friston, K. & Ramstead, M.J.D. (2024).** "Shared Protentions in Multi-Agent Active Inference." *Entropy*, 26(4), 303.
Formalizes shared goal-directed behavior via shared protentions (anticipated future states) in multi-agent generative models using category theory. Relevant for extending the framework to settings where multiple stakeholders share expectations about what CEs should deliver.

### 4.3 Explainable AI under Active Inference

**Albarracin, M., Hipolito, I., Tremblay, S.E., Fox, J.G., Rene, G., Friston, K. & Ramstead, M.J.D. (2023).** "Designing Explainable Artificial Intelligence with Active Inference: A Framework for Transparent Introspection and Decision-Making." [arXiv:2306.04025]
Proposes an XAI architecture using active inference with hierarchical generative models, enabling AI systems to explain their own decisions through introspection. **This is the most directly relevant prior work from our group**: it establishes that active inference provides a natural framework for explainability. The current paper extends this by showing that *counterfactual* explanations specifically can be formalized as free energy minimization, and that the adaptive recomputation rule follows from the same framework.

### 4.4 Resilience and Adaptation

**Miller, M., Albarracin, M., Pitliya, R.J., Kiefer, A.B., Mago, J., Gorman, C., Friston, K. & Ramstead, M.J.D. (2022).** "Resilience and Active Inference." *Frontiers in Psychology*, 13, 1059117.
Threefold typology of resilience under active inference: inertia (resist change), elasticity (bounce back), plasticity (expand repertoire). **Relevant because**: our adaptive framework embodies all three -- REUSE is inertia (resist unnecessary recomputation), RETRAIN triggered by drift is elasticity (bounce back to valid CEs), and the decision rule itself is plastic (threshold adapts to drift statistics).

---

## 5. Amortized Inference for Causal and Counterfactual Reasoning

**Pawlowski, N., Castro, D.C. & Glocker, B. (2020).** "Deep Structural Causal Models for Tractable Counterfactual Inference." *NeurIPS 2020*. [arXiv:2006.06485]
Formulates deep SCMs using normalizing flows and variational inference for tractable abduction -- amortizing the inference step in causal counterfactual reasoning. Demonstrates that amortized inference can handle all three levels of Pearl's causal ladder. Key precedent for amortized counterfactual inference, though without the free-energy adaptation mechanism.

**Lorch, L., Sussex, S., Rothfuss, J., Krause, A. & Scholkopf, B. (2022).** "Amortized Inference for Causal Structure Learning." *NeurIPS 2022*. [arXiv:2205.12934]
AVICI: trains a variational model on simulated data to predict causal structure from observational data, bypassing combinatorial search. Demonstrates that amortization can acquire domain-specific inductive biases. Parallel to our amortization of counterfactual search: both replace exponential-time search with amortized forward passes.

**Paul, A., Isomura, T. & Razi, A. (2024).** "On Predictive Planning and Counterfactual Learning in Active Inference." *Entropy*, 26(6), 484. [arXiv:2403.12417]
Examines forward planning and backward counterfactual learning in active inference. Proposes a mixed model balancing both. **Most direct precedent** for connecting active inference with counterfactual reasoning -- though it addresses RL-style counterfactual policies rather than XAI counterfactual explanations.

---

## 6. Jamie Duell's Papers (Direct Predecessors)

**Duell, J. & Fan, X. (2026).** "Provably Robust Bayesian Counterfactual Explanations under Model Changes." [arXiv:2601.16659]
Introduces Probabilistically Safe CEs (PSCE): delta-safety (high predictive confidence) and epsilon-robustness (low predictive variance) guarantees under model retraining. Uses a (delta, epsilon)-set formulation. **Addresses MC robustness** with static Bayesian guarantees. Our paper extends the temporal dimension: PSCE guarantees hold at one time point; we provide guarantees that degrade gracefully with posterior drift and a decision rule for when they need refreshing.

**Duell, J. (2026).** "Bayesian Amortized Counterfactual Explanations under Model Multiplicity." *SIAM/ACM submission (under review)*.
The AVCG framework -- the direct predecessor of our work. Key contributions:
- Variational objective: G(x) = -R_epsilon(x') + D_KL(q_psi || p(z)) + lambda * E[d(g_phi(...), x)]
- Rashomon-restricted posterior P_R(theta) via MC dropout
- Propositions 3.3-3.4 (Markov/Chebyshev bounds) and Corollary 3.5 (validity transfer to new models)
- Evaluated on Adult Income, Credit, Spambase, PneumoniaMNIST
- **Addresses MM robustness** via amortized generation over the Rashomon set

**Gap between the two Duell papers**: PSCE handles model *changes* with static guarantees; AVCG handles model *multiplicity* with amortized generation. Neither addresses the temporal question of *when* to recompute. Our paper bridges this gap.

---

## 7. Adaptive Computation and Model Drift Detection

### 7.1 Adaptive Computation in Neural Networks

**Graves, A. (2016).** "Adaptive Computation Time for Recurrent Neural Networks." [arXiv:1603.08983]
ACT: RNNs learn how many computational steps per input via a differentiable halting mechanism. Conceptual ancestor of our approach -- ACT uses a learned halting probability, we use free energy as the halting signal for counterfactual recomputation.

**Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J. & Kaiser, L. (2019).** "Universal Transformers." *ICLR 2019*. [arXiv:1807.03819]
Combines Transformers with ACT for variable computation depth per token. Achieves Turing completeness. Demonstrates adaptive computation in modern architectures. Our work applies this principle to counterfactual inference with a principled variational objective rather than a learned halting unit.

**Laskaridis, S. et al. (2021).** "Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions." *EMDL 2021 (ACM Workshop)*.
Surveys early-exit networks with side classifiers at intermediate layers. Distinguishes static exit policies (entropy thresholds) from dynamic ones (learned controllers). Our free-energy threshold is analogous to an entropy-based exit policy but grounded in variational principles.

### 7.2 Concept Drift and Model Monitoring

**Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014).** "A Survey on Concept Drift Adaptation." *ACM Computing Surveys*, 46(4), Article 44.
Standard survey on concept drift: types (sudden, gradual, incremental, recurring), detection methods (CUSUM, DDM, ADWIN), adaptation strategies. Our drift measure D_t = D_KL(P_R^{t+1} || P_R^t) is a concept drift detector *specific to counterfactual validity* rather than general prediction accuracy.

**Rabanser, S., Gunnemann, S. & Lipton, Z. (2019).** "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift." *NeurIPS 2019*.
Empirical comparison of dataset shift detection methods. Finds that no single method dominates; the best depends on the type of shift. Relevant because our KL-based posterior drift measure could be compared against these general-purpose shift detectors.

**Regol, F., Schwinn, L., Sprague, K., Coates, M. & Markovich, T. (2025).** "When to Retrain a Machine Learning Model." [arXiv:2505.14903]
UPF: performance forecasting for retraining decisions with limited labeled data. **Most directly related to our adaptive rule** but from a different angle -- they ask "when to retrain the model?", we ask "when to retrain the counterfactual generator?". Both are resource-allocation decisions triggered by degradation signals.

### 7.3 Resource-Rational Computation

**Lieder, F. & Griffiths, T.L. (2020).** "Resource-Rational Analysis: Understanding Human Cognition as the Optimal Use of Limited Computational Resources." *Behavioral and Brain Sciences*, 43, e1.
Proposes that human cognition is bounded-optimal given finite computational resources. The five-step framework formalizes the tradeoff between inference quality and computational cost. **Our adaptive rule IS resource-rational**: invest retraining compute only when expected improvement in CE validity exceeds the cost. This connects our work to the broader cognitive science literature.

---

## 8. Energy-Based Approaches to Counterfactual Explanations

**Doncieux, S. et al. (2025).** "Exploring Energy Landscapes for Minimal Counterfactual Explanations: Applications in Cybersecurity and Beyond." [arXiv:2503.18185]
Reformulates counterfactual search as energy minimization using Taylor expansion + Boltzmann distribution + simulated annealing. **Closest existing work to using energy/free-energy for CEs**, but uses physical energy landscapes rather than variational free energy. Our work goes further by using the full variational apparatus (encoder, decoder, KL, Rashomon restriction) rather than post-hoc energy landscape exploration.

---

## 9. Summary: Positioning and Gaps

Our paper sits at the intersection of three literatures that have **not previously been connected**:

| Literature | What it provides | What it lacks |
|---|---|---|
| **Robust CEs** (Jiang et al. survey, Pawelczyk, Leofante, Hamman) | Problem formulation, robustness taxonomy, static guarantees | Temporal/adaptive dimension; no mechanism for detecting staleness |
| **Amortized VI** (Kingma & Welling, Cremer, Kim, Marino) | Inference engine, amortization gap analysis | No application to CEs; no drift-aware adaptation |
| **Active Inference / FEP** (Friston, Da Costa, Joffily & Coricelli, Albarracin) | Variational formalism, affective signals, resource-rational framing | Never applied to XAI counterfactual explanations |

**Our contribution unifies these** by showing:
1. AVCG *is* variational free energy minimization (connecting robust CEs to AIF)
2. The amortization gap under model drift can be detected via posterior KL divergence (connecting amortized VI to drift detection)
3. The recomputation decision is an active inference policy selection problem (connecting AIF to adaptive computation)

**No existing work**:
- Connects active inference to counterfactual explainability (Paul et al. 2024 is closest but addresses RL, not XAI)
- Provides adaptive recomputation rules for amortized CE generators
- Extends static Rashomon-validity guarantees to the temporal setting
- Uses affective inference (anxiety/valence) as a computational signal for explanation maintenance
