# Experiment 3: Activation Oracle for Reasoning Steps

**Goal:** Train an activation oracle to predict reasoning step properties from activations

---

## Overview

This experiment trains an LLM to interpret its own activations during reasoning. The oracle predicts:
1. **Function tags** - What type of reasoning step (active_computation, fact_retrieval, etc.)
2. **Correctness** - Whether the solution is ultimately correct or incorrect
3. **Importance** - Counterfactual importance score (how much removing this step affects accuracy)

## Key Design Decisions

- **Same model as target**: DeepSeek-R1-Distill-Llama-8B interprets its own activations
- **Multi-task oracle**: Single model predicts all three properties in one response
- **Natural language output**: "This reasoning step performs active_computation. The solution is correct. Importance: moderate (0.3)."
- **LoRA fine-tuning**: Parameter-efficient adaptation

## Prerequisites

1. Activation extraction complete (from experiment 2)
2. Linear probe results showing positive signal (go/no-go criteria)

### Go/No-Go Criteria

Proceed if ANY of:
- Function tag F1 > random baseline + 0.10
- Correctness accuracy > 55%
- Importance R² > 0.05

## Usage

```bash
# 1. Prepare training data (requires activation cache)
python scripts/1_prepare_data.py

# 2. Train oracle (RTX 3090 24GB, ~2-3 hours)
python scripts/2_train_oracle.py

# 3. Evaluate (compare to linear probes)
python scripts/3_evaluate_oracle.py
```

## Configuration (RTX 3090 24GB)

```python
batch_size = 4
gradient_accumulation = 4  # Effective batch = 16
lora_r = 64
lora_alpha = 128
gradient_checkpointing = True
torch_dtype = "bfloat16"
```

## Directory Structure

```
3_oracle_training/
├── scripts/
│   ├── 1_prepare_data.py      # Convert activations to training format
│   ├── 2_train_oracle.py      # Train LoRA adapter
│   └── 3_evaluate_oracle.py   # Evaluate and compare to probes
├── results/
│   ├── oracle_train_data.pt   # Training data
│   ├── oracle_eval_data.pt    # Held-out eval data
│   ├── oracle_model/          # Trained LoRA weights
│   └── oracle_eval_results.json
└── README.md
```

## Expected Results

If linear probes show signal, oracle should:
- Match or exceed probe performance (captures non-linear patterns)
- Provide richer, interpretable outputs
- Enable qualitative analysis of reasoning steps

If oracle significantly outperforms probes → evidence that LLMs can usefully interpret their own activations during reasoning.

---

## Limitations

### Data Size Concerns (Critical)

| Metric | Value | Concern |
|--------|-------|---------|
| Training samples | ~6,000 steps | Small for LLM fine-tuning |
| Unique problems | 16 (train) | Very limited diversity |
| Test problems | 4 | High variance in evaluation |

**Key issues:**
- LLM fine-tuning typically uses 10K-100K+ examples; we have ~6K
- Only 20 unique problems total - oracle may memorize problem-specific patterns
- Steps within a problem are correlated - not truly independent samples
- Risk of overfitting to the small problem set

**Why we proceed anyway:**
- LoRA is parameter-efficient (fewer params to overfit)
- Single epoch training reduces memorization
- Comparison to probes uses same data, so relative performance is fair
- This is exploratory research, not production deployment

### Other Limitations

1. **Self-interpretation bias** - Model may have priors about its own behavior
2. **Natural language parsing** - Evaluation requires parsing free-form text
3. **Single layer** - Only uses activations from one layer
4. **Prompt sensitivity** - Results may depend on exact prompt wording
