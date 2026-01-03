#!/usr/bin/env python3
"""
Extract hidden states from OUR custom-trained oracle (experiment 3).

Comparison:
- Paper's oracle: Trained on 1M diverse interp tasks (Llama base)
- Our oracle: Trained on 5.5K reasoning labels (DeepSeek base)

Even if our oracle overfit and can't verbalize, its hidden states
might have learned useful task-specific transformations.
"""

import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
ACTIVATION_CACHE = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results/activation_cache_multilayer.pt")
CUSTOM_ORACLE_PATH = Path("/workspace/activation-oracles-reasoning/experiments/3_oracle_training/results/oracle_model/final")
RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results")

# Our oracle's base model
ORACLE_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def log(msg):
    print(msg, flush=True)


class HiddenStateExtractor:
    """Hook to extract hidden states from oracle layers."""

    def __init__(self, layers_to_extract):
        self.layers_to_extract = layers_to_extract
        self.hidden_states = {}
        self.handles = []

    def hook_fn(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.hidden_states[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook

    def register(self, model):
        # Handle PeftModel wrapping: PeftModel -> LoraModel -> LlamaForCausalLM -> LlamaModel
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            base = model.base_model.model.model  # PeftModel
        else:
            base = model.model  # Regular model
        for layer_idx in self.layers_to_extract:
            if layer_idx < 0:
                actual_idx = len(base.layers) + layer_idx
            else:
                actual_idx = layer_idx
            layer = base.layers[actual_idx]
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
        h = output[0] if isinstance(output, tuple) else output

        if self.position < h.shape[1]:
            original_norm = h[:, self.position, :].norm(dim=-1, keepdim=True)
            steering_norm = self.steering_vector.norm(dim=-1, keepdim=True)

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


def load_custom_oracle():
    """Load our custom-trained oracle from experiment 3."""
    log("Loading custom oracle...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Load base model (same as DeepSeek we extracted activations from)
    log(f"  Base model: {ORACLE_BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        ORACLE_BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Load our LoRA adapter
    log(f"  LoRA adapter: {CUSTOM_ORACLE_PATH}")
    model = PeftModel.from_pretrained(
        base_model,
        CUSTOM_ORACLE_PATH,
        is_trainable=False
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_ORACLE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PeftModel wraps: PeftModel -> LoraModel -> LlamaForCausalLM -> LlamaModel
    base = model.base_model.model.model  # -> LlamaModel with .layers
    log(f"  Layers: {len(base.layers)}")
    log(f"  Hidden size: {model.config.hidden_size}")

    return model, tokenizer, device, dtype


def load_activations():
    """Load DeepSeek activations from cache."""
    log(f"\nLoading activations from {ACTIVATION_CACHE}")
    cache = torch.load(ACTIVATION_CACHE, weights_only=False)

    activations = cache["activations"][-8]  # Primary layer
    labels = cache["labels"]

    log(f"  Loaded {len(activations)} samples")
    log(f"  Activation shape: {activations.shape}")

    return activations, labels


def extract_oracle_hidden_states(model, tokenizer, activations, labels, device, dtype,
                                  layers_to_extract=[-4, -8, -12, -16],
                                  max_samples=None):
    """Feed each DeepSeek activation into our oracle and extract hidden states."""
    if max_samples:
        activations = activations[:max_samples]
        labels = labels[:max_samples]

    n_samples = len(activations)
    hidden_dim = model.config.hidden_size

    oracle_hidden_states = {layer: torch.zeros(n_samples, hidden_dim) for layer in layers_to_extract}

    # Simple prompt
    prompt = "Analyze this activation and describe what the model is computing."
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    inject_position = 5
    # Handle PeftModel wrapping: PeftModel -> LoraModel -> LlamaForCausalLM -> LlamaModel
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model.model  # PeftModel
    else:
        base = model.model  # Regular model
    embed_layer = base.embed_tokens

    log(f"\nExtracting custom oracle hidden states for {n_samples} samples...")

    extractor = HiddenStateExtractor(layers_to_extract)
    extractor.register(model)

    for i in tqdm(range(n_samples)):
        activation = activations[i].to(device=device, dtype=dtype).unsqueeze(0)

        steering = ActivationSteering(activation, inject_position, coefficient=1.0)
        steering.register(embed_layer)

        extractor.clear()

        with torch.no_grad():
            _ = model(input_ids)

        for layer in layers_to_extract:
            oracle_hidden_states[layer][i] = extractor.hidden_states[layer].squeeze(0)

        steering.remove()

    extractor.remove()

    return oracle_hidden_states, labels


def main():
    log("=" * 60)
    log("CUSTOM ORACLE HIDDEN STATE EXTRACTION")
    log("=" * 60)
    log("\nThis extracts hidden states from OUR oracle (experiment 3),")
    log("not the paper's pre-trained oracle.")
    log("\nEven if our oracle overfit, its hidden states might have")
    log("learned useful task-specific representations.")
    log("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load our custom oracle
    model, tokenizer, device, dtype = load_custom_oracle()

    # Load DeepSeek activations
    activations, labels = load_activations()

    # Extract hidden states
    layers_to_extract = [-4, -8, -12, -16]
    oracle_hidden_states, labels = extract_oracle_hidden_states(
        model, tokenizer, activations, labels, device, dtype,
        layers_to_extract=layers_to_extract,
        max_samples=None
    )

    # Save results
    save_data = {
        "oracle_hidden_states": oracle_hidden_states,
        "labels": labels,
        "config": {
            "oracle_base": ORACLE_BASE_MODEL,
            "oracle_lora": str(CUSTOM_ORACLE_PATH),
            "layers_extracted": layers_to_extract,
            "n_samples": len(labels),
            "hidden_dim": model.config.hidden_size,
            "oracle_type": "custom_trained"
        }
    }

    save_path = RESULTS_DIR / "custom_oracle_hidden_states.pt"
    torch.save(save_data, save_path)
    log(f"\nSaved custom oracle hidden states to {save_path}")

    # Print stats
    log("\n" + "=" * 60)
    log("EXTRACTION COMPLETE")
    log("=" * 60)
    log(f"Samples: {len(labels)}")
    log(f"Layers: {layers_to_extract}")
    for layer in layers_to_extract:
        h = oracle_hidden_states[layer]
        log(f"  Layer {layer}: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")


if __name__ == "__main__":
    main()
