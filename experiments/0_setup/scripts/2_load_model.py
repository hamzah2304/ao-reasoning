"""
Step 2: Load target model and test inference

Checks:
- Model downloads and loads successfully
- Can run inference
- Can extract hidden states
- Model config is as expected

IMPORTANT: Sets HF_HOME to /workspace/.cache/huggingface for persistence
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# IMPORTANT: Set cache to persistent location
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

# Configuration
TARGET_MODEL = "deepseek-ai/deepseek-r1-distill-llama-8b"
RESULTS_DIR = Path("../results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test prompt
TEST_PROMPT = "What is 2 + 2?"


def main():
    print("=" * 60)
    print("Step 2: Loading Target Model")
    print("=" * 60)
    print()

    print(f"HuggingFace cache: {os.environ['HF_HOME']}")
    print()

    results = {
        "model_name": TARGET_MODEL,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hf_cache": os.environ['HF_HOME']
    }

    if torch.cuda.is_available():
        results["cuda_device"] = torch.cuda.get_device_name(0)
        results["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"Model: {TARGET_MODEL}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {results['cuda_device']}")
        print(f"VRAM: {results['cuda_memory_gb']:.1f} GB")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    elapsed = time.time() - start
    results["tokenizer_load_time"] = elapsed
    print(f"  ✓ Loaded in {elapsed:.2f}s")
    print()

    # Load model
    print("Loading model...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    model.eval()
    elapsed = time.time() - start
    results["model_load_time"] = elapsed
    print(f"  ✓ Loaded in {elapsed:.1f}s")
    print()

    # Model info
    print("Model Configuration:")
    config = model.config
    results["config"] = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size
    }
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    print()

    # Test inference
    print("Testing inference...")
    print(f"  Prompt: '{TEST_PROMPT}'")
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results["test_inference"] = {
        "prompt": TEST_PROMPT,
        "response": response,
        "time": elapsed
    }
    print(f"  Response: '{response}'")
    print(f"  Time: {elapsed:.2f}s")
    print()

    # Test hidden state extraction
    print("Testing hidden state extraction...")
    layer = -8
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer]
    activation_shape = hidden_states.shape
    results["activation_extraction"] = {
        "layer": layer,
        "shape": list(activation_shape),
        "dtype": str(hidden_states.dtype)
    }
    print(f"  Layer {layer} shape: {activation_shape}")
    print(f"  dtype: {hidden_states.dtype}")
    print(f"  ✓ Successfully extracted hidden states")
    print()

    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        results["memory_usage_gb"] = {
            "allocated": allocated,
            "reserved": reserved
        }
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print()

    # Save results
    output_file = RESULTS_DIR / "model_info.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Model loaded successfully")
    print(f"✓ Inference working")
    print(f"✓ Hidden state extraction working")
    print()
    print(f"Results saved to {output_file}")
    print("✓ Ready for Step 3: Activation Extraction")
    print("=" * 60)


if __name__ == "__main__":
    main()
