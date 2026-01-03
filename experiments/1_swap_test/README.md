# Experiment 1: Activation Swap Test

**Goal:** Test whether pre-trained activation oracles transfer to different model architectures.

---

## Overview

This experiment tests whether activation oracles genuinely read information from neural activations or simply confabulate answers based on textual context. We deliberately mismatch activations and context to create a discriminative test.

## Research Question

**Do activation oracles extract information encoded in neural activations, or do they pattern-match on textual context and hallucinate plausible answers?**

## Background

Activation oracles (Karvonen et al., 2025) are LLMs trained to answer natural language questions about the internal states of other neural networks. The oracle receives:
1. Activation vectors from a target model
2. A textual prompt describing what to analyze
3. A question about the activations

The oracle then generates a natural language response. However, a critical concern is whether the oracle actually processes the activation information or merely uses contextual cues to generate plausible-sounding responses.

## Experimental Design

### Core Idea: Activation-Context Mismatch

We create scenarios where:
- **Activations** come from Problem A (which has answer X)
- **Context** describes Problem B (which has answer Y)
- **Question** asks: "What is the final answer?"

If the oracle:
- Answers **X** → Evidence it reads activations
- Answers **Y** → Evidence it confabulates from context
- Answers something else → Unclear (noise, confusion, or independent generation)

### Test Setup

| Component | Source |
|-----------|--------|
| Target Model | DeepSeek-R1-Distill-Llama-8B |
| Oracle Model | Llama-3.1-8B-Instruct (LoRA fine-tuned for activation interpretation) |
| Activation Layer | -8 (8th from end, 24th of 32 total) |
| Token Position | Last token of final reasoning step |
| Problems | 5 MATH Level 5 problems with unique numerical answers |

### Problem Set

Selected for unique answers to enable unambiguous discrimination:

| Problem ID | Answer | Math Type |
|------------|--------|-----------|
| problem_330 | 88572 | Number Theory |
| problem_2050 | 336 | Combinatorics |
| problem_3360 | 12.1 | Algebra |
| problem_4019 | 4 | Geometry |
| problem_6481 | 30.8 | Probability |

### Test Matrix

For 5 problems, we create a 5x5 matrix:
- **Diagonal (5 tests)**: Matched activations and context (baseline)
- **Off-diagonal (20 tests)**: Mismatched activations and context

Total: 25 test cases

## Hypotheses

### H0 (Null): Context-Based Confabulation
The oracle generates answers based primarily on textual context, ignoring or failing to extract meaningful information from activations.

**Prediction**: In mismatched cases, oracle answers will correlate with the **context problem's answer**, not the **activation source's answer**.

### H1 (Alternative): Activation Reading
The oracle extracts information genuinely encoded in the activation vectors.

**Prediction**: In mismatched cases, oracle answers will correlate with the **activation source's answer**, regardless of context.

### H2 (Hybrid): Partial Information Extraction
The oracle uses both activation information and context, with varying degrees of reliance.

**Prediction**: Mixed results, potentially influenced by answer salience, problem similarity, or other factors.

## Technical Decisions and Rationale

### 1. Oracle Model Choice: Llama-3.1-8B

**Decision**: Use `adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct`

**Rationale**:
- Same Llama architecture family as target model (DeepSeek-R1-Distill-Llama-8B is a Llama derivative)
- Pre-trained oracle available on HuggingFace with LoRA weights
- 8B parameters matches our GPU memory constraints

**Concern**: Architecture similarity might artificially inflate performance. A more rigorous test would use a different architecture family.

### 2. Layer Selection: -8 (24th of 32)

**Decision**: Extract from layer -8 (8 layers before output)

**Rationale**:
- Following activation oracles paper methodology
- Late layers encode more task-specific, abstract information
- Early layers contain more syntactic/surface features
- Layer -8 balances abstraction with sufficient processing

**Concern**: This is a somewhat arbitrary choice. Optimal layer may vary by task or model.

### 3. Token Position: Last Token

**Decision**: Use activation at the final token of the reasoning trace

**Rationale**:
- LLMs accumulate context through autoregressive processing
- Final token position should encode the "conclusion" of reasoning
- Standard practice in activation probing literature

**Concern**: Information may be distributed across multiple token positions. Single-token extraction loses spatial information.

### 4. Question Type: Final Answer Retrieval

**Decision**: Ask "What is the final numerical answer to this math problem?"

**Rationale**:
- Clear, unambiguous question with verifiable ground truth
- Final answer should be strongly encoded after complete reasoning
- Easy to evaluate (exact match or numerical comparison)

**Concern**: This tests one specific type of information. Oracle may encode other information (reasoning steps, confidence, error states) that we don't test.

### 5. Prompt Structure

**Decision**: Use minimal context - just problem statement without reasoning trace

**Rationale**:
- Maximizes discriminability between activation and context signals
- If reasoning trace included, oracle has too much information to confabulate from
- Tests whether activations alone carry answer information

**Concern**: This is an artificially adversarial setup. In realistic use, you'd have matching context.

## Evaluation Metrics

### Primary Metrics

1. **Activation Alignment Rate**: % of mismatched cases where answer matches activation source
2. **Context Alignment Rate**: % of mismatched cases where answer matches context
3. **Neither Rate**: % of cases matching neither (noise/confusion)
4. **Baseline Accuracy**: % correct on diagonal (matched) cases

### Statistical Analysis

- Chi-squared test for independence between answer source and activation/context
- Binomial test for activation alignment rate vs chance (20% for 5 problems)
- Confidence intervals for all rates

## Interpretation Guide

### Strong Evidence for Activation Reading
- Activation alignment >> context alignment (e.g., >70% vs <30%)
- Activation alignment significantly above chance (20%)
- Consistent across different problem pairs

### Strong Evidence for Confabulation
- Context alignment >> activation alignment
- Activation alignment near or below chance
- Consistent pattern across problems

### Ambiguous Results
- Similar activation and context alignment rates
- High "neither" rate
- Inconsistent patterns across problems

## Limitations (Critical Assessment)

### 1. Sample Size: Severely Underpowered
- **Only 5 problems** (25 total test cases, 20 mismatched)
- Cannot establish statistical significance for most effects
- No power analysis performed
- Results should be considered **exploratory pilot data**, not confirmatory evidence

### 2. Model Similarity: Confounded by Architecture
- Target (DeepSeek-R1-Distill-Llama-8B) and oracle base (Llama-3.1-8B) share Llama architecture
- Oracle may exploit architectural similarities rather than learning general activation interpretation
- Cannot distinguish "reading activations" from "recognizing familiar activation patterns"
- A proper test would use cross-architecture oracles

### 3. Training Data Contamination Risk
- Oracle trained on LatentQA dataset which may include similar math problems
- Oracle may have learned answer-activation associations during training
- Not testing true generalization

### 4. Single Layer, Single Token
- Only testing layer -8, last token position
- Information may be encoded differently at other layers/positions
- May miss distributed representations
- Oversimplifies the complexity of neural representations

### 5. One Question Type Only
- Only testing "final answer" retrieval
- Oracles might work for some queries but not others
- Doesn't test reasoning trace reconstruction, confidence estimation, or error detection

### 6. Artificial Adversarial Setup
- Deliberately mismatching activations and context is unrealistic
- Real use cases would have matching context
- May stress-test edge cases rather than typical performance

### 7. Answer Confounds
- Some answers more "obvious" than others (e.g., 4 is a common number)
- Oracle might have priors over typical math answers
- No control for answer frequency/salience

### 8. No Baseline Comparison
- Not comparing to random guessing, trivial baselines, or alternative methods
- Hard to contextualize absolute performance numbers

### 9. Reproducibility Concerns
- Single run without repetition
- No seed control documented
- Temperature/sampling parameters may affect results

### 10. Evaluation Brittleness
- Exact match evaluation may be too strict (e.g., "4" vs "4.0" vs "four")
- Fuzzy matching introduces subjectivity
- LLM-based evaluation adds another layer of uncertainty

## Expected Outcomes

### Optimistic Scenario
Oracle shows strong activation alignment (>60%), suggesting genuine activation reading capability that transfers to reasoning models. This would support the potential of activation oracles for interpretability research.

### Pessimistic Scenario
Oracle shows strong context alignment or random performance, suggesting the oracle confabulates based on context and doesn't meaningfully process activations. This would raise serious concerns about the activation oracles methodology.

### Realistic Expectation
Likely mixed results with moderate activation alignment (30-50%), suggesting partial activation reading with significant context influence. This would indicate cautious optimism with need for further investigation.

## Files

```
experiments/1_swap_test/
├── README.md                 # This file
├── scripts/
│   ├── 1_load_oracle.py     # Download and verify oracle model
│   ├── 2_run_swap_test.py   # Execute swap test experiment
│   └── 3_analyze_results.py # Analyze and visualize results
└── results/
    ├── oracle_info.json     # Oracle model configuration
    ├── swap_test_raw.json   # Raw experiment results
    ├── swap_test_analysis.json  # Computed metrics
    └── figures/             # Visualizations
```

## Running the Experiment

```bash
# Step 1: Download and verify oracle
python scripts/1_load_oracle.py

# Step 2: Run swap test
python scripts/2_run_swap_test.py

# Step 3: Analyze results
python scripts/3_analyze_results.py
```

## References

- Karvonen et al. (2025). "Activation Oracles." arXiv:2512.15674
- Original oracle checkpoints: https://huggingface.co/adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct
- Math rollouts dataset: https://huggingface.co/datasets/uzaymacar/math-rollouts
