"""
Step 3: Extract activations from target model

For each reasoning step in each problem:
1. Build cumulative text (reasoning trace so far)
2. Run through model
3. Extract hidden states at layer -8, last token
4. Cache to disk

Usage:
    python 3_extract_activations.py           # All 5 problems
    python 3_extract_activations.py --test    # 1 problem only (for timing)
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
import sys
import os

# IMPORTANT: Set cache to persistent location
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

# Configuration
TARGET_MODEL = "deepseek-ai/deepseek-r1-distill-llama-8b"
ACTIVATION_LAYER = -8
DATA_DIR = Path("../../../data/math_rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95")
RESULTS_DIR = Path("../results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SWAP_TEST_PROBLEMS = [
    "problem_330",
    "problem_2050",
    "problem_3360",
    "problem_4019",
    "problem_6481",
]


def load_problem_data(problem_id, solution_type="correct_base_solution"):
    """Load problem and chunks."""
    problem_dir = DATA_DIR / solution_type / problem_id

    with open(problem_dir / "problem.json") as f:
        problem = json.load(f)

    with open(problem_dir / "chunks_labeled.json") as f:
        chunks = json.load(f)

    return {
        "problem_id": problem_id,
        "problem": problem,
        "chunks": chunks
    }


def extract_activations(model, tokenizer, text, layer=-8):
    """Extract activations at specified layer for given text."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Last token at specified layer
    hidden_states = outputs.hidden_states[layer]
    last_token_activations = hidden_states[0, -1, :]

    return last_token_activations.cpu()


def main():
    # Check for test mode
    test_mode = "--test" in sys.argv

    print("=" * 60)
    if test_mode:
        print("Step 3: Extracting Activations (TEST MODE - 1 problem)")
    else:
        print("Step 3: Extracting Activations (ALL 5 problems)")
    print("=" * 60)
    print()

    # Load problems
    print("Loading problems...")
    if test_mode:
        problems_to_process = [SWAP_TEST_PROBLEMS[0]]  # Just first problem
        cache_filename = "activation_cache_test.pt"
    else:
        problems_to_process = SWAP_TEST_PROBLEMS
        cache_filename = "activation_cache.pt"

    problems = []
    for prob_id in problems_to_process:
        data = load_problem_data(prob_id)
        problems.append(data)
        print(f"  ✓ {prob_id}: {len(data['chunks'])} chunks")

    total_chunks = sum(len(p['chunks']) for p in problems)
    print(f"\nTotal steps to process: {total_chunks}")
    print()

    # Load model
    print("Loading model...")
    print(f"  Model: {TARGET_MODEL}")
    print(f"  Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"  ✓ Model loaded")
    print()

    # Extract activations
    print("Extracting activations...")
    print(f"  Layer: {ACTIVATION_LAYER}")
    print()

    activation_cache = {}
    total_extracted = 0
    start_time = time.time()

    for problem in tqdm(problems, desc="Problems"):
        problem_id = problem["problem_id"]
        activation_cache[problem_id] = {}

        # Build cumulative text
        text_so_far = ""

        for chunk in tqdm(problem["chunks"], desc=f"  {problem_id}", leave=False):
            chunk_idx = chunk["chunk_idx"]
            text_so_far += chunk["chunk"]

            # Extract
            activations = extract_activations(model, tokenizer, text_so_far, layer=ACTIVATION_LAYER)
            activation_cache[problem_id][chunk_idx] = activations
            total_extracted += 1

    elapsed = time.time() - start_time

    print()
    print(f"✓ Extracted {total_extracted} activation vectors")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Average: {elapsed/total_extracted:.2f}s per step")

    if test_mode:
        # Extrapolate to all 5 problems
        print()
        print(f"Estimated time for all 5 problems:")
        print(f"  ~{(elapsed/total_chunks) * 1245:.0f}s ({(elapsed/total_chunks) * 1245 / 60:.1f} min)")

    print()

    # Save cache
    print("Saving activation cache...")
    cache_file = RESULTS_DIR / cache_filename

    torch.save({
        "activation_cache": activation_cache,
        "problems": problems,
        "config": {
            "target_model": TARGET_MODEL,
            "activation_layer": ACTIVATION_LAYER,
            "device": DEVICE,
            "num_problems": len(problems),
            "total_chunks": total_extracted,
            "extraction_time": elapsed,
            "test_mode": test_mode
        }
    }, cache_file)

    cache_size = cache_file.stat().st_size / (1024 ** 2)
    print(f"  ✓ Saved to {cache_file}")
    print(f"  Size: {cache_size:.1f} MB")

    if test_mode:
        print(f"  Estimated size for all 5 problems: {cache_size * (1245/total_chunks):.1f} MB")

    print()

    # Memory cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Problems processed: {len(problems)}")
    print(f"Activation vectors: {total_extracted}")
    print(f"Cache size: {cache_size:.1f} MB")
    print(f"Extraction time: {elapsed/60:.1f} minutes")

    if test_mode:
        print()
        print("Test mode complete. To process all problems, run:")
        print("  python 3_extract_activations.py")
    else:
        print()
        print("✓ Ready for Step 4: Cache Verification")

    print("=" * 60)


if __name__ == "__main__":
    main()
