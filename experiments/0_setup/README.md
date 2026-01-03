# Phase 0: Setup - Data Loading and Activation Extraction

**Goal:** Prepare data and extract neural activations from reasoning model.

---

## Goal

Prepare data and extract neural activations from DeepSeek-R1-Distill-Llama-8B for subsequent experiments.

## What We're Doing

1. **Load HuggingFace data** - 5 MATH problems with different answer types
2. **Extract activations** - Get hidden states at layer -8 for each reasoning step
3. **Cache to disk** - Save for reuse in Experiments 1-3

## Why Layer -8?

Following the activation oracles paper, we extract from middle-to-late layers where:
- Abstract reasoning is encoded
- Not too early (still syntactic)
- Not final layer (too task-specific)

## Data Structure

Each problem contains:
```
problem_XXX/
├── problem.json              # Problem text, ground truth answer, metadata
├── chunks_labeled.json       # Reasoning steps with pre-computed labels:
│                            #   - function_tags (8 categories)
│                            #   - counterfactual_importance_accuracy
│                            #   - resampling_importance_accuracy
│                            #   - depends_on, accuracy, etc.
└── chunk_N/solutions.json   # Rollouts (resampled continuations) from step N
```

## Problems for Swap Test

| Problem ID | Math Type | Answer | Chunks |
|------------|-----------|--------|--------|
| problem_330 | Algebra | 88572 | ~200 |
| problem_2050 | Counting & Probability | 336 | ~200 |
| problem_3360 | Geometry | 12.1 | ~200 |
| problem_4019 | Intermediate Algebra | 4 | ~200 |
| problem_6481 | Prealgebra | 30.8 | ~200 |

All have **different answers** → good for swap test pairs

## Scripts

Run in order:

1. **`1_verify_data.py`** - Check downloaded data structure and labels
2. **`2_load_model.py`** - Load target model and test inference
3. **`3_extract_activations.py`** - Extract and cache activations
4. **`4_verify_cache.py`** - Verify activation cache integrity

## Expected Outputs

```
results/
├── data_verification.json    # Data structure validation
├── model_info.json          # Model config and test inference
├── activation_cache.pt      # Cached activations (~2-3 GB)
└── cache_stats.json         # Cache validation stats
```

## Estimated Time

- Data verification: 1 minute
- Model loading: 2-5 minutes (first time downloads model)
- Activation extraction: 15-30 minutes (depends on GPU)
- Total: **~20-35 minutes**

## Next Steps

After Phase 0 completion → `experiments/1_swap_test/`
