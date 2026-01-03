#!/usr/bin/env python3
"""
Prepare oracle training data from activation cache.
Converts activations + labels to TrainingDataPoint format for oracle training.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, Mapping
from pydantic import BaseModel, ConfigDict, model_validator

# Paths
PROBE_RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results")
OUTPUT_DIR = Path("/workspace/activation-oracles-reasoning/experiments/3_oracle_training/results")
MODEL_NAME = "deepseek-ai/deepseek-r1-distill-llama-8b"

# Function tags for natural language
FUNCTION_TAG_DESCRIPTIONS = {
    "active_computation": "active computation (performing calculations)",
    "fact_retrieval": "fact retrieval (recalling known information)",
    "uncertainty_management": "uncertainty management (handling doubt or errors)",
    "result_consolidation": "result consolidation (combining intermediate results)",
    "plan_generation": "plan generation (creating a strategy)",
    "self_checking": "self-checking (verifying intermediate results)",
    "problem_setup": "problem setup (understanding the question)",
    "final_answer_emission": "final answer emission (producing the answer)",
}


class TrainingDataPoint(BaseModel):
    """Training data point compatible with activation_oracles."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None
    ds_label: str | None
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
        return values


def importance_to_text(importance: float | None) -> str:
    """Convert importance score to natural language."""
    if importance is None:
        importance = 0.0

    if importance < 0.1:
        level = "low"
    elif importance < 0.3:
        level = "moderate"
    else:
        level = "high"

    return f"Importance: {level} ({importance:.2f})"


def create_target_output(function_tags: list[str], is_correct: bool, importance: float | None) -> str:
    """Create natural language target output."""
    parts = []

    # Function tags
    if function_tags:
        tag_descriptions = [FUNCTION_TAG_DESCRIPTIONS.get(t, t) for t in function_tags]
        if len(tag_descriptions) == 1:
            parts.append(f"This reasoning step performs {tag_descriptions[0]}.")
        else:
            parts.append(f"This reasoning step performs {', '.join(tag_descriptions[:-1])} and {tag_descriptions[-1]}.")
    else:
        parts.append("This reasoning step has no identified function.")

    # Correctness
    correctness = "correct" if is_correct else "incorrect"
    parts.append(f"The solution is {correctness}.")

    # Importance
    parts.append(importance_to_text(importance))

    return " ".join(parts)


SPECIAL_TOKEN = "?"  # Placeholder for activation injection


def create_prompt(layer: int) -> str:
    """Create the prompt for oracle inference."""
    prefix = f"Layer: {layer}\n{SPECIAL_TOKEN} \n"  # Single position marker
    prompt = prefix + "Analyze this reasoning step from a language model. What function does it serve, is the overall solution correct, and how important is this step?"
    return prompt


def find_special_token_positions(input_ids: list[int], tokenizer) -> list[int]:
    """Find positions of the special '?' token in the tokenized sequence."""
    # Get the token ID for '?'
    special_token_ids = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)

    positions = []
    for i, token_id in enumerate(input_ids):
        if token_id in special_token_ids:
            positions.append(i)
            break  # We only have one position marker

    if not positions:
        # Fallback: use position after "Layer: X\n"
        positions = [0]

    return positions


def load_activation_cache(layer: int = -8):
    """Load activation cache from linear probe experiment."""
    cache_path = PROBE_RESULTS_DIR / "activation_cache_multilayer.pt"
    print(f"Loading cache from {cache_path}")
    cache = torch.load(cache_path, weights_only=False)

    activations = cache["activations"][layer]  # [N, D]
    labels = cache["labels"]

    print(f"Loaded {len(labels)} samples, activation shape: {activations.shape}")
    return activations, labels


def get_problem_stratified_split(labels, test_ratio=0.2, seed=42):
    """Split by problem to avoid data leakage (same as probes)."""
    np.random.seed(seed)
    problems = list(set(l["problem_id"] for l in labels))
    np.random.shuffle(problems)
    n_test = max(1, int(len(problems) * test_ratio))
    test_problems = set(problems[:n_test])
    train_idx = [i for i, l in enumerate(labels) if l["problem_id"] not in test_problems]
    test_idx = [i for i, l in enumerate(labels) if l["problem_id"] in test_problems]
    return train_idx, test_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPARING ORACLE TRAINING DATA")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load activations (use best layer from probes, default -8)
    print("\n2. Loading activations...")
    layer = -8  # TODO: Could read from probe results to get best layer
    activations, labels = load_activation_cache(layer=layer)

    # Create prompt template
    base_prompt = create_prompt(layer)

    # Tokenize prompt to get input_ids structure
    print("\n3. Creating training data points...")
    training_data = []

    for i, (act, label) in enumerate(zip(activations, labels)):
        # Create target output
        target = create_target_output(
            function_tags=label["function_tags"],
            is_correct=label["is_correct"],
            importance=label["counterfactual_importance"]
        )

        # Create full conversation
        messages = [
            {"role": "user", "content": base_prompt},
            {"role": "assistant", "content": target}
        ]

        # Tokenize with chat template
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        # Find where assistant response starts to create labels
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": base_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_only, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        # Labels: -100 for prompt, actual tokens for response
        labels_ids = [-100] * prompt_len + full_tokens[prompt_len:]

        # Ensure lengths match
        if len(labels_ids) != len(full_tokens):
            labels_ids = [-100] * len(full_tokens)  # Fallback
            labels_ids[prompt_len:] = full_tokens[prompt_len:]

        # Find position of special token for activation injection
        positions = find_special_token_positions(full_tokens, tokenizer)

        # Create training data point
        dp = TrainingDataPoint(
            datapoint_type="reasoning_step_analysis",
            input_ids=full_tokens,
            labels=labels_ids,
            layer=layer,
            steering_vectors=act.unsqueeze(0),  # [1, D]
            positions=positions,  # Position of '?' token
            feature_idx=i,
            target_output=target,
            context_input_ids=None,
            context_positions=None,
            ds_label=label["problem_id"],
            meta_info={
                "problem_id": label["problem_id"],
                "chunk_idx": label["chunk_idx"],
                "function_tags": label["function_tags"],
                "is_correct": label["is_correct"],
                "counterfactual_importance": label["counterfactual_importance"],
            }
        )
        training_data.append(dp)

        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(labels)} samples")

    print(f"   Created {len(training_data)} training data points")

    # Split into train/eval
    print("\n4. Splitting data...")
    train_idx, test_idx = get_problem_stratified_split(labels)
    train_data = [training_data[i] for i in train_idx]
    eval_data = [training_data[i] for i in test_idx]
    print(f"   Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Save
    print("\n5. Saving...")

    # Convert to serializable format
    def to_dict(dp: TrainingDataPoint) -> dict:
        return {
            "datapoint_type": dp.datapoint_type,
            "input_ids": dp.input_ids,
            "labels": dp.labels,
            "layer": dp.layer,
            "steering_vectors": dp.steering_vectors,  # Keep as tensor
            "positions": dp.positions,
            "feature_idx": dp.feature_idx,
            "target_output": dp.target_output,
            "context_input_ids": dp.context_input_ids,
            "context_positions": dp.context_positions,
            "ds_label": dp.ds_label,
            "meta_info": dict(dp.meta_info),
        }

    train_save = {
        "config": {
            "model": MODEL_NAME,
            "layer": layer,
            "n_samples": len(train_data),
        },
        "data": [to_dict(dp) for dp in train_data],
    }

    eval_save = {
        "config": {
            "model": MODEL_NAME,
            "layer": layer,
            "n_samples": len(eval_data),
        },
        "data": [to_dict(dp) for dp in eval_data],
    }

    train_path = OUTPUT_DIR / "oracle_train_data.pt"
    eval_path = OUTPUT_DIR / "oracle_eval_data.pt"

    torch.save(train_save, train_path)
    torch.save(eval_save, eval_path)

    print(f"   Saved train data: {train_path}")
    print(f"   Saved eval data: {eval_path}")

    # Verify positions are valid
    print("\n6. Verifying data...")
    n_valid_pos = sum(1 for dp in training_data if dp.positions[0] > 0)
    print(f"   Samples with valid '?' position: {n_valid_pos}/{len(training_data)}")
    if n_valid_pos < len(training_data) * 0.9:
        print("   WARNING: Many samples have fallback position=0")

    # Show example
    print("\n" + "=" * 60)
    print("EXAMPLE DATA POINT")
    print("=" * 60)
    example = training_data[0]
    print(f"Target output: {example.target_output}")
    print(f"Layer: {example.layer}")
    print(f"Activation shape: {example.steering_vectors.shape}")
    print(f"Positions: {example.positions}")
    print(f"Input tokens: {len(example.input_ids)}")
    print(f"Meta info: {dict(example.meta_info)}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
