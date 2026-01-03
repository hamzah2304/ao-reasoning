# Experiment 2: Linear Probes on Reasoning Activations

**Goal:** Establish baseline - what information is linearly decodable from reasoning model activations?

---

## Overview

Before training activation oracles, we need to verify that the target information is actually encoded in activations. Linear probes (logistic regression classifiers) serve as a simple, interpretable baseline.

## Research Questions

1. **Can we predict function type from activations?** (8-way classification)
2. **Can we predict correctness from activations?** (binary: correct vs incorrect solution)
3. **Can we predict step importance from activations?** (regression + anchor detection)
4. **At which layers is this information encoded?** (layer analysis)

## Data

| Split | Steps | Problems |
|-------|-------|----------|
| Correct solutions | 4,003 | 20 |
| Incorrect solutions | 3,486 | 20 |
| **Total** | 7,489 | 20 unique |

### Function Tag Distribution

| Tag | Count | % |
|-----|-------|---|
| active_computation | 2,518 | 33.6% |
| fact_retrieval | 2,066 | 27.6% |
| uncertainty_management | 869 | 11.6% |
| result_consolidation | 788 | 10.5% |
| plan_generation | 743 | 9.9% |
| self_checking | 272 | 3.6% |
| problem_setup | 156 | 2.1% |
| final_answer_emission | 69 | 0.9% |

### Correctness Distribution

- Correct: 4,003 (53.5%)
- Incorrect: 3,486 (46.5%)

Nearly balanced - good for binary classification.

---

## Method

### Activation Extraction

- **Model:** DeepSeek-R1-Distill-Llama-8B
- **Primary Layer:** -8 (layer 24 of 32)
- **Additional Layers:** -4, -12, -16 for layer analysis
- **Position:** Last token of each reasoning step (cumulative trace)
- **Dimension:** 4096

### Probe Architecture

- **Classifier:** Logistic Regression (sklearn)
- **Regularization:** L2 with cross-validated C
- **Split:** 80/20 train/test, stratified by problem (no problem overlap)
- **Multi-label handling:** For function tags, use multi-label one-vs-rest

### Evaluation Metrics

- Accuracy (overall and per-class)
- Macro F1 (balances class imbalances)
- Confusion matrix
- Comparison to random baseline

---

## Experimental Design

### Probe 1: Function Tag Prediction (8-way)

**Task:** Given activation vector, predict the function tag(s) of that reasoning step.

**Challenges:**
- Multi-label (steps can have multiple tags) - use multi-label classification
- Imbalanced classes (0.9% to 33.6%) - use class weights
- High dimensionality (4096) vs samples (7489) - may need regularization

**Baselines:**
- Random: ~12.5% (1/8) for balanced, weighted random for actual distribution
- Majority class: 33.6% (always predict active_computation)

**Success criterion:** Significantly above majority class baseline.

### Probe 2: Correctness Prediction (binary)

**Task:** Given activation vector from a reasoning step, predict if this step is part of a correct or incorrect solution.

**Hypothesis:** Activations might encode "quality" signals even mid-trace. Steps from incorrect solutions might show different patterns (confusion, uncertainty).

**Baselines:**
- Random: 50%
- Majority class: 53.5%

**Success criterion:** Significantly above random (>55% with p<0.05).

### Probe 3: Importance Prediction

**Task:** Given activation vector, predict the counterfactual importance of that reasoning step.

**Approach:**
- **Regression:** Predict continuous importance score (-1 to 1)
- **Classification:** Predict top-10% "anchor" steps vs rest

**Why this is interesting:**
- Thought-anchors showed importance exists behaviorally
- If probes succeed → importance is encoded in activations (new finding)
- Could enable automatic anchor detection without expensive counterfactual testing

**Baselines:**
- Regression: R² = 0 (predict mean)
- Classification: Random = 50%, Majority = 90%

**Success criterion:** R² > 0 or F1 significantly above chance for anchor detection.

### Probe 4: Layer Analysis

**Task:** Run function tag probe at multiple layers to find where information is encoded.

**Layers:** -4, -8, -12, -16 (4, 8, 12, 16 from end)

**Hypothesis:** Semantic/abstract information (function tags) should be in later layers. Early layers may encode more syntactic features.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `1_extract_activations.py` | Extract activations for all 7489 steps at multiple layers |
| `2_train_probes.py` | Train logistic regression probes for all tasks |
| `3_layer_analysis.py` | Compare probe performance across layers |
| `4_visualize_results.py` | Generate figures and visualizations |

---

## Expected Outcomes

### Best Case
- Function tag probe achieves >50% accuracy (well above 33.6% baseline)
- Correctness probe achieves >60% accuracy
- Clear layer-dependent pattern emerges
- Strong evidence that activations encode reasoning properties

### Good Case
- Function tag probe achieves 40-50% accuracy
- Some tags easier than others (reveals what activations encode)
- Correctness probe marginal or fails
- Partial evidence for activation encoding

### Acceptable Case
- Both probes near baseline
- Clear negative result: reasoning process is not linearly decodable
- Still informative for the field

---

## Limitations

### Data Size Concerns (Critical)

| Metric | Value | Concern |
|--------|-------|---------|
| Unique problems | 20 | Very limited diversity |
| Total steps | 7,489 | Decent for probes, small for oracle |
| Test set (20%) | ~4 problems, ~1,500 steps | Small test set |
| Steps per problem | ~375 avg | High correlation within problem |

**Statistical implications:**
- Train/test split by problem means only 4 test problems - high variance in results
- Steps within a problem are highly correlated (same reasoning trace)
- 7,489 "samples" is misleading - effectively 20 independent datapoints at problem level
- Results should be considered **exploratory/pilot**, not definitive

**Mitigations:**
- Problem-stratified split avoids data leakage
- Cross-validation could help but still limited by 20 problems
- Report confidence intervals, not just point estimates
- Be conservative in claims

### Other Limitations

1. **Linear probes only** - Information might be encoded non-linearly
2. **Single token position** - Using last token only, information may be distributed
3. **Label quality** - Function tags from external labeling, may have noise
4. **Problem diversity** - Only 20 unique problems, limited generalization
5. **Model-specific** - Results may not transfer to other reasoning models

---

## References

- Thought Anchors paper (function tag source): arXiv:2506.19143
- Linear probing methodology: Belinkov (2022) "Probing Classifiers"
- math-rollouts dataset: huggingface.co/datasets/uzaymacar/math-rollouts
