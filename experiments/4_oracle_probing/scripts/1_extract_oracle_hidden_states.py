#!/usr/bin/env python3
"""
Extract hidden states from the paper's oracle when fed DeepSeek activations.

The idea (from Oscar): Even if the oracle can't verbalize correctly,
its internal hidden states might still learn useful representations.
We probe THOSE to see if the oracle learned to transform activations usefully.
"""

import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
ACTIVATION_CACHE = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results/activation_cache_multilayer.pt")
RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results")

# Oracle config (paper's pre-trained oracle)
ORACLE_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"


class HiddenStateExtractor:
    """Hook to extract hidden states from oracle layers."""

    def __init__(self, layers_to_extract):
        self.layers_to_extract = layers_to_extract
        self.hidden_states = {}
        self.handles = []

    def hook_fn(self, layer_idx):
        def hook(module, input, output):
            # output is tuple (hidden_states, ...) for transformer layers
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Take the last token's hidden state
            self.hidden_states[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook

    def register(self, model):
        """Register hooks on specified layers."""
        for layer_idx in self.layers_to_extract:
            # Get the actual layer (handling negative indices)
            if layer_idx < 0:
                actual_idx = len(model.model.layers) + layer_idx
            else:
                actual_idx = layer_idx

            layer = model.model.layers[actual_idx]
            handle = layer.register_forward_hook(self.hook_fn(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear(self):
        self.hidden_states = {}


class ActivationSteering:
    """Hook to inject DeepSeek activations into oracle."""

    def __init__(self, steering_vector, position, coefficient=1.0):
        self.steering_vector = steering_vector
        self.position = position
        self.coefficient = coefficient
        self.handle = None

    def hook_fn(self, module, input, output):
        """Inject activation at specified position."""
        # Additive steering with norm matching (like the paper)
        h = output[0] if isinstance(output, tuple) else output

        if self.position < h.shape[1]:
            original_norm = h[:, self.position, :].norm(dim=-1, keepdim=True)
            steering_norm = self.steering_vector.norm(dim=-1, keepdim=True)

            # Norm-matched addition
            if steering_norm > 0:
                normalized_steering = self.steering_vector * (original_norm / steering_norm)
                h[:, self.position, :] = h[:, self.position, :] + self.coefficient * normalized_steering

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


def load_oracle():
    """Load the paper's pre-trained oracle."""
    print("Loading oracle...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Load base model
    print(f"  Base model: {ORACLE_BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        ORACLE_BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Load LoRA adapter
    print(f"  LoRA adapter: {ORACLE_LORA_PATH}")
    model.load_adapter(
        ORACLE_LORA_PATH,
        adapter_name="oracle",
        is_trainable=False,
        low_cpu_mem_usage=True
    )
    model.set_adapter("oracle")
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ORACLE_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Layers: {len(model.model.layers)}")
    print(f"  Hidden size: {model.config.hidden_size}")

    return model, tokenizer, device, dtype


def load_activations():
    """Load DeepSeek activations from cache."""
    print(f"\nLoading activations from {ACTIVATION_CACHE}")
    cache = torch.load(ACTIVATION_CACHE, weights_only=False)

    # Get layer -8 activations (primary layer)
    activations = cache["activations"][-8]
    labels = cache["labels"]

    print(f"  Loaded {len(activations)} samples")
    print(f"  Activation shape: {activations.shape}")

    return activations, labels


def extract_oracle_hidden_states(model, tokenizer, activations, labels, device, dtype,
                                  layers_to_extract=[-4, -8, -12, -16],
                                  max_samples=None):
    """
    Feed each DeepSeek activation into oracle and extract hidden states.
    """
    if max_samples:
        activations = activations[:max_samples]
        labels = labels[:max_samples]

    n_samples = len(activations)
    hidden_dim = model.config.hidden_size
    n_layers = len(layers_to_extract)

    # Initialize storage
    oracle_hidden_states = {layer: torch.zeros(n_samples, hidden_dim) for layer in layers_to_extract}

    # Create a simple prompt - the oracle will process the steering vector
    prompt = "Analyze this activation and describe what the model is computing."
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Position to inject activation (early in sequence)
    inject_position = 5  # After a few tokens

    # Get embedding layer for steering
    embed_layer = model.model.embed_tokens

    # Extract hidden states for each sample
    print(f"\nExtracting oracle hidden states for {n_samples} samples...")

    extractor = HiddenStateExtractor(layers_to_extract)
    extractor.register(model)

    for i in tqdm(range(n_samples)):
        # Get this sample's activation
        activation = activations[i].to(device=device, dtype=dtype).unsqueeze(0)

        # Create steering hook
        steering = ActivationSteering(activation, inject_position, coefficient=1.0)
        steering.register(embed_layer)

        # Clear previous hidden states
        extractor.clear()

        # Forward pass (just process input, don't generate)
        with torch.no_grad():
            _ = model(input_ids)

        # Store hidden states
        for layer in layers_to_extract:
            oracle_hidden_states[layer][i] = extractor.hidden_states[layer].squeeze(0)

        # Remove steering hook
        steering.remove()

    extractor.remove()

    return oracle_hidden_states, labels


def main():
    print("=" * 60)
    print("ORACLE HIDDEN STATE EXTRACTION")
    print("=" * 60)
    print("\nIdea: Even if oracle can't verbalize correctly,")
    print("its internal hidden states might learn useful representations.")
    print("We'll probe those to see if they're more informative than")
    print("probing DeepSeek activations directly.")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load oracle
    model, tokenizer, device, dtype = load_oracle()

    # Load DeepSeek activations
    activations, labels = load_activations()

    # Extract oracle hidden states
    layers_to_extract = [-4, -8, -12, -16]
    oracle_hidden_states, labels = extract_oracle_hidden_states(
        model, tokenizer, activations, labels, device, dtype,
        layers_to_extract=layers_to_extract,
        max_samples=None  # Process all samples
    )

    # Save results
    save_data = {
        "oracle_hidden_states": oracle_hidden_states,
        "labels": labels,
        "config": {
            "oracle_base": ORACLE_BASE_MODEL,
            "oracle_lora": ORACLE_LORA_PATH,
            "layers_extracted": layers_to_extract,
            "n_samples": len(labels),
            "hidden_dim": model.config.hidden_size,
        }
    }

    save_path = RESULTS_DIR / "oracle_hidden_states.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved oracle hidden states to {save_path}")

    # Print stats
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Samples: {len(labels)}")
    print(f"Layers: {layers_to_extract}")
    for layer in layers_to_extract:
        h = oracle_hidden_states[layer]
        print(f"  Layer {layer}: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")

    print(f"\nNext: Run 2_train_probes_on_oracle.py to compare probe performance")


if __name__ == "__main__":
    main()
