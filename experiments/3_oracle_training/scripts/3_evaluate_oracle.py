#!/usr/bin/env python3
"""
Evaluate trained oracle on reasoning step analysis.
Compares oracle predictions to ground truth and linear probe baselines.
"""

import json
import re
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
DATA_DIR = Path("/workspace/activation-oracles-reasoning/experiments/3_oracle_training/results")
PROBE_RESULTS = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results/probe_results.json")
OUTPUT_DIR = DATA_DIR

# Function tags
FUNCTION_TAGS = [
    "active_computation", "fact_retrieval", "uncertainty_management",
    "result_consolidation", "plan_generation", "self_checking",
    "problem_setup", "final_answer_emission"
]


def get_hf_submodule(model, layer_idx: int):
    """Get the submodule to hook onto."""
    # Handle PEFT models by getting base model
    base = model.base_model.model if hasattr(model, "base_model") else model

    if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
        return base.model.embed_tokens
    elif hasattr(base, "transformer") and hasattr(base.transformer, "wte"):
        return base.transformer.wte
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


class SteeringHook:
    """Hook to inject activations."""

    def __init__(self, positions, steering_vectors, device, dtype):
        self.positions = positions
        self.steering_vectors = [sv.to(device=device, dtype=dtype) for sv in steering_vectors]
        self.handle = None

    def hook_fn(self, module, input, output):
        for batch_idx, (pos_list, sv) in enumerate(zip(self.positions, self.steering_vectors)):
            for pos_idx, pos in enumerate(pos_list):
                if pos < output.shape[1]:
                    output[batch_idx, pos] = sv[pos_idx]
        return output

    def register(self, submodule):
        self.handle = submodule.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


def parse_function_tags(text: str) -> list[str]:
    """Extract function tags from oracle output."""
    found_tags = []
    text_lower = text.lower()

    for tag in FUNCTION_TAGS:
        # Check for tag name or description
        if tag.replace("_", " ") in text_lower or tag in text_lower:
            found_tags.append(tag)

    return found_tags


def parse_correctness(text: str) -> bool | None:
    """Extract correctness from oracle output.

    Only matches definitive statements like:
    - "solution is correct/incorrect"
    - "the answer is correct/incorrect"
    - "this is correct/incorrect"
    - "step is correct/incorrect"

    Excludes meta-commentary like "whether the solution is correct".
    """
    text_lower = text.lower()

    # Patterns that indicate meta-commentary (NOT a prediction)
    meta_patterns = [
        "whether the solution is",
        "whether it is correct",
        "if the solution is",
        "determine what",
        "figure out",
        "asking me to",
    ]

    # If this looks like meta-commentary about the task, return None
    for pattern in meta_patterns:
        if pattern in text_lower:
            return None

    # Definitive negative patterns (check first - "incorrect" contains "correct")
    negative_patterns = [
        "solution is incorrect",
        "answer is incorrect",
        "this is incorrect",
        "step is incorrect",
        "is incorrect",
        "not correct",
    ]

    for pattern in negative_patterns:
        if pattern in text_lower:
            return False

    # Definitive positive patterns
    positive_patterns = [
        "solution is correct",
        "answer is correct",
        "this is correct",
        "step is correct",
    ]

    for pattern in positive_patterns:
        if pattern in text_lower:
            return True

    return None


def parse_importance(text: str) -> float | None:
    """Extract importance score from oracle output."""
    # Look for patterns like "Importance: high (0.45)" or "(0.45)"
    match = re.search(r'\(([0-9]+\.?[0-9]*)\)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Look for level words
    text_lower = text.lower()
    if "high" in text_lower:
        return 0.5
    elif "moderate" in text_lower:
        return 0.2
    elif "low" in text_lower:
        return 0.05

    return None


SPECIAL_TOKEN = "?"


def find_special_token_position(input_ids, tokenizer):
    """Find position of '?' token."""
    special_token_ids = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    input_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()

    for i, token_id in enumerate(input_list):
        if token_id in special_token_ids:
            return i
    return 0  # Fallback


def generate_oracle_response(model, tokenizer, submodule, data_point, device, dtype, max_new_tokens=100):
    """Generate oracle response for a single data point."""
    model.eval()

    # Create prompt (without response)
    layer = data_point["layer"]
    prefix = f"Layer: {layer}\n{SPECIAL_TOKEN} \n"
    prompt = prefix + "Analyze this reasoning step from a language model. What function does it serve, is the overall solution correct, and how important is this step?"

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    # Find position of special token
    special_pos = find_special_token_position(input_ids, tokenizer)

    # Set up steering hook
    hook = SteeringHook(
        positions=[[special_pos]],  # Position of '?' token
        steering_vectors=[data_point["steering_vectors"]],
        device=device,
        dtype=dtype
    )
    hook.register(submodule)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        hook.remove()

    return response


def evaluate_oracle(model, tokenizer, submodule, eval_data, device, dtype, max_samples=None):
    """Evaluate oracle on held-out data."""
    if max_samples:
        eval_data = eval_data[:max_samples]

    results = {
        "predictions": [],
        "function_tags": {"pred": [], "true": []},
        "correctness": {"pred": [], "true": []},
        "importance": {"pred": [], "true": []},
    }

    print(f"Evaluating on {len(eval_data)} samples...")

    for dp in tqdm(eval_data, desc="Generating"):
        response = generate_oracle_response(model, tokenizer, submodule, dp, device, dtype)
        meta = dp["meta_info"]

        # Parse predictions
        pred_tags = parse_function_tags(response)
        pred_correct = parse_correctness(response)
        pred_importance = parse_importance(response)

        # Store for metrics
        results["predictions"].append({
            "response": response,
            "target": dp["target_output"],
            "pred_tags": pred_tags,
            "true_tags": meta["function_tags"],
            "pred_correct": pred_correct,
            "true_correct": meta["is_correct"],
            "pred_importance": pred_importance,
            "true_importance": meta["counterfactual_importance"],
        })

        # For tag metrics (multi-label)
        for tag in FUNCTION_TAGS:
            results["function_tags"]["pred"].append(1 if tag in pred_tags else 0)
            results["function_tags"]["true"].append(1 if tag in meta["function_tags"] else 0)

        # For correctness metrics
        if pred_correct is not None:
            results["correctness"]["pred"].append(1 if pred_correct else 0)
            results["correctness"]["true"].append(1 if meta["is_correct"] else 0)

        # For importance metrics
        if pred_importance is not None and meta["counterfactual_importance"] is not None:
            results["importance"]["pred"].append(pred_importance)
            results["importance"]["true"].append(meta["counterfactual_importance"])

    return results


def compute_metrics(results):
    """Compute evaluation metrics."""
    metrics = {}

    # Function tag metrics
    if results["function_tags"]["pred"]:
        y_pred = np.array(results["function_tags"]["pred"])
        y_true = np.array(results["function_tags"]["true"])
        metrics["function_tags"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    # Correctness metrics
    if results["correctness"]["pred"]:
        y_pred = np.array(results["correctness"]["pred"])
        y_true = np.array(results["correctness"]["true"])
        metrics["correctness"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "n_samples": len(y_pred),
        }

    # Importance metrics
    if results["importance"]["pred"]:
        y_pred = np.array(results["importance"]["pred"])
        y_true = np.array(results["importance"]["true"])
        metrics["importance"] = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_pred) > 1 else 0.0,
            "n_samples": len(y_pred),
        }

    return metrics


def load_probe_baselines():
    """Load linear probe results for comparison."""
    if not PROBE_RESULTS.exists():
        return None

    with open(PROBE_RESULTS) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("ORACLE EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load model
    print("\n1. Loading model...")
    model_path = OUTPUT_DIR / "oracle_model" / "final"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 2_train_oracle.py first.")
        return

    # Load config
    config_path = OUTPUT_DIR / "oracle_model" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    submodule = get_hf_submodule(model, config.get("hook_onto_layer", 1))

    # Load eval data
    print("\n2. Loading eval data...")
    eval_path = DATA_DIR / "oracle_eval_data.pt"
    eval_save = torch.load(eval_path, weights_only=False)
    eval_data = eval_save["data"]
    print(f"   Loaded {len(eval_data)} eval samples")

    # Evaluate
    print("\n3. Evaluating oracle...")
    results = evaluate_oracle(model, tokenizer, submodule, eval_data, device, dtype, max_samples=200)

    # Compute metrics
    metrics = compute_metrics(results)

    # Load probe baselines
    print("\n4. Loading probe baselines...")
    probe_baselines = load_probe_baselines()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- Function Tags ---")
    if "function_tags" in metrics:
        m = metrics["function_tags"]
        print(f"Oracle F1 (macro): {m['f1_macro']:.3f}")
        print(f"Oracle F1 (micro): {m['f1_micro']:.3f}")
        if probe_baselines:
            probe_f1 = probe_baselines["function_tags"]["macro_f1"]
            print(f"Probe F1 (macro):  {probe_f1:.3f}")
            print(f"Improvement: {m['f1_macro'] - probe_f1:+.3f}")

    print("\n--- Correctness ---")
    if "correctness" in metrics:
        m = metrics["correctness"]
        print(f"Oracle accuracy: {m['accuracy']:.3f}")
        print(f"Oracle F1:       {m['f1']:.3f}")
        if probe_baselines:
            probe_acc = probe_baselines["correctness"]["accuracy"]
            print(f"Probe accuracy:  {probe_acc:.3f}")
            print(f"Improvement: {m['accuracy'] - probe_acc:+.3f}")

    print("\n--- Importance ---")
    if "importance" in metrics:
        m = metrics["importance"]
        print(f"Oracle R²:          {m['r2']:.3f}")
        print(f"Oracle correlation: {m['correlation']:.3f}")
        if probe_baselines:
            probe_r2 = probe_baselines["importance"]["regression"]["r2"]
            print(f"Probe R²:           {probe_r2:.3f}")
            print(f"Improvement: {m['r2'] - probe_r2:+.3f}")

    # Save results
    save_results = {
        "oracle_metrics": metrics,
        "probe_baselines": probe_baselines,
        "sample_predictions": results["predictions"][:10],  # First 10 examples
    }

    results_path = OUTPUT_DIR / "oracle_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Show example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    for i, pred in enumerate(results["predictions"][:3]):
        print(f"\n--- Example {i + 1} ---")
        print(f"Target:   {pred['target'][:200]}...")
        print(f"Response: {pred['response'][:200]}...")
        print(f"Tags: pred={pred['pred_tags']}, true={pred['true_tags']}")
        print(f"Correct: pred={pred['pred_correct']}, true={pred['true_correct']}")
        print(f"Importance: pred={pred['pred_importance']}, true={pred['true_importance']}")


if __name__ == "__main__":
    main()
