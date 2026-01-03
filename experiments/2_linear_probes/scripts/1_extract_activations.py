#!/usr/bin/env python3
"""
Extract activations from DeepSeek-R1-Distill-Llama-8B for all reasoning steps.

Extracts at multiple layers for subsequent layer analysis.
Saves activations with all associated labels.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc
import time

# Configuration
MODEL_NAME = "deepseek-ai/deepseek-r1-distill-llama-8b"
DATA_DIR = Path("/workspace/activation-oracles-reasoning/data/math_rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95")
OUTPUT_DIR = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results")

# Layers to extract (negative indices from end)
LAYERS_TO_EXTRACT = [-4, -8, -12, -16]  # 28, 24, 20, 16 of 32


def load_model():
    """Load model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers, {model.config.hidden_size} hidden dim")

    return model, tokenizer, num_layers


def get_activations(model, tokenizer, text, layers, num_layers):
    """Extract activations at specified layers for the last token."""
    activations = {}

    # Convert negative indices to positive
    layer_indices = [num_layers + l if l < 0 else l for l in layers]

    # Hook to capture activations
    captured = {}
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is a tuple, first element is hidden states
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu()
        return hook

    # Register hooks
    handles = []
    for layer_idx in layer_indices:
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    # Tokenize and run forward pass
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Map back to negative indices
    for neg_idx, pos_idx in zip(layers, layer_indices):
        activations[neg_idx] = captured[pos_idx]

    return activations


def load_all_data():
    """Load all problems with their chunks and labels."""
    all_data = []

    for solution_type in ["correct_base_solution", "incorrect_base_solution"]:
        is_correct = solution_type == "correct_base_solution"
        solution_dir = DATA_DIR / solution_type

        for problem_dir in sorted(solution_dir.iterdir()):
            problem_id = problem_dir.name

            # Load problem metadata
            problem_file = problem_dir / "problem.json"
            chunks_file = problem_dir / "chunks_labeled.json"

            if not chunks_file.exists():
                print(f"  Skipping {problem_id} ({solution_type}): no chunks_labeled.json")
                continue

            problem_data = json.load(open(problem_file))
            chunks = json.load(open(chunks_file))

            # Build cumulative trace for each chunk
            cumulative_text = ""
            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = chunk.get("chunk", "")
                cumulative_text += chunk_text

                all_data.append({
                    "problem_id": problem_id,
                    "is_correct": is_correct,
                    "chunk_idx": chunk_idx,
                    "cumulative_text": cumulative_text,
                    "function_tags": chunk.get("function_tags", []),
                    "counterfactual_importance": chunk.get("counterfactual_importance_accuracy"),
                    "resampling_importance": chunk.get("resampling_importance_accuracy"),
                    "ground_truth": problem_data.get("ground_truth"),
                })

    return all_data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACTIVATION EXTRACTION FOR LINEAR PROBES")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    all_data = load_all_data()
    print(f"   Total steps: {len(all_data)}")
    print(f"   Correct: {sum(1 for d in all_data if d['is_correct'])}")
    print(f"   Incorrect: {sum(1 for d in all_data if not d['is_correct'])}")

    # Load model
    print("\n2. Loading model...")
    model, tokenizer, num_layers = load_model()
    print(f"   Extracting from layers: {LAYERS_TO_EXTRACT}")
    print(f"   (Absolute indices: {[num_layers + l for l in LAYERS_TO_EXTRACT]})")

    # Extract activations
    print("\n3. Extracting activations...")
    start_time = time.time()

    # Store activations organized by layer
    activations_by_layer = {layer: [] for layer in LAYERS_TO_EXTRACT}
    labels = []

    for i, item in enumerate(tqdm(all_data, desc="Extracting")):
        # Get activations at all layers
        acts = get_activations(
            model, tokenizer,
            item["cumulative_text"],
            LAYERS_TO_EXTRACT,
            num_layers
        )

        # Store activations
        for layer in LAYERS_TO_EXTRACT:
            activations_by_layer[layer].append(acts[layer])

        # Store labels (same for all layers)
        labels.append({
            "problem_id": item["problem_id"],
            "is_correct": item["is_correct"],
            "chunk_idx": item["chunk_idx"],
            "function_tags": item["function_tags"],
            "counterfactual_importance": item["counterfactual_importance"],
            "resampling_importance": item["resampling_importance"],
        })

        # Periodic memory cleanup
        if (i + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    extraction_time = time.time() - start_time
    print(f"   Extraction complete in {extraction_time/60:.1f} minutes")

    # Stack activations into tensors
    print("\n4. Stacking and saving...")
    for layer in LAYERS_TO_EXTRACT:
        activations_by_layer[layer] = torch.cat(activations_by_layer[layer], dim=0)
        print(f"   Layer {layer}: {activations_by_layer[layer].shape}")

    # Save
    cache = {
        "activations": activations_by_layer,
        "labels": labels,
        "config": {
            "model": MODEL_NAME,
            "layers": LAYERS_TO_EXTRACT,
            "total_steps": len(all_data),
            "extraction_time_seconds": extraction_time,
        }
    }

    cache_path = OUTPUT_DIR / "activation_cache_multilayer.pt"
    torch.save(cache, cache_path)

    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"   Saved to: {cache_path}")
    print(f"   Cache size: {cache_size_mb:.1f} MB")

    # Verify
    print("\n5. Verification...")
    loaded = torch.load(cache_path, weights_only=False)
    for layer in LAYERS_TO_EXTRACT:
        acts = loaded["activations"][layer]
        print(f"   Layer {layer}: shape={acts.shape}, mean={acts.float().mean():.4f}, std={acts.float().std():.4f}")
        nan_count = torch.isnan(acts).sum().item()
        inf_count = torch.isinf(acts).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"   WARNING: {nan_count} NaN, {inf_count} Inf values!")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Total steps: {len(all_data)}")
    print(f"Layers: {LAYERS_TO_EXTRACT}")
    print(f"Time: {extraction_time/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
