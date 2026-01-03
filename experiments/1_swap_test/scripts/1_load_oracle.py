"""
Step 1: Download and verify oracle model

Downloads the Llama-3.1-8B oracle checkpoint from HuggingFace
and verifies it loads correctly.
"""

import json
import os
import sys
from pathlib import Path
import time

# IMPORTANT: Set cache to persistent location BEFORE importing transformers
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
ORACLE_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"
RESULTS_DIR = Path("../results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("Step 1: Loading Oracle Model")
    print("=" * 60)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "oracle_base_model": ORACLE_BASE_MODEL,
        "oracle_lora_path": ORACLE_LORA_PATH,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hf_cache": os.environ['HF_HOME']
    }

    if torch.cuda.is_available():
        results["cuda_device"] = torch.cuda.get_device_name(0)
        results["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {results['cuda_device']}")
        print(f"VRAM: {results['cuda_memory_gb']:.1f} GB")
        print()

    print(f"Base model: {ORACLE_BASE_MODEL}")
    print(f"LoRA adapter: {ORACLE_LORA_PATH}")
    print(f"Device: {DEVICE}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ORACLE_BASE_MODEL)
    tokenizer.padding_side = "left"  # Required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elapsed = time.time() - start
    results["tokenizer_load_time"] = elapsed
    print(f"  Loaded in {elapsed:.2f}s")
    print()

    # Load base model
    print("Loading base model...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        ORACLE_BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    elapsed = time.time() - start
    results["base_model_load_time"] = elapsed
    print(f"  Loaded in {elapsed:.1f}s")
    print()

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    start = time.time()
    model.load_adapter(
        ORACLE_LORA_PATH,
        adapter_name="oracle",
        is_trainable=False,
        low_cpu_mem_usage=True
    )
    model.set_adapter("oracle")
    model.eval()
    elapsed = time.time() - start
    results["lora_load_time"] = elapsed
    print(f"  Loaded in {elapsed:.1f}s")
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
    print()

    # Test basic inference (without activation steering)
    print("Testing basic inference...")
    test_prompt = "What is 2 + 2? Answer with just the number."
    messages = [{"role": "user", "content": test_prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results["test_inference"] = {
        "prompt": test_prompt,
        "response": response.strip(),
        "time": elapsed
    }
    print(f"  Prompt: '{test_prompt}'")
    print(f"  Response: '{response.strip()}'")
    print(f"  Time: {elapsed:.2f}s")
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
    output_file = RESULTS_DIR / "oracle_info.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Oracle model loaded successfully")
    print(f"Base: {ORACLE_BASE_MODEL}")
    print(f"LoRA: {ORACLE_LORA_PATH}")
    print()
    print(f"Results saved to {output_file}")
    print("Ready for Step 2: Swap Test Execution")
    print("=" * 60)


if __name__ == "__main__":
    main()
