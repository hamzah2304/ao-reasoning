"""
Step 4: Verify activation cache integrity

Checks:
- Cache file loads successfully
- All problems present
- All chunks have activations
- Activation shapes are correct
- No NaN or Inf values
"""

import json
from pathlib import Path
import torch
import numpy as np

# Configuration
RESULTS_DIR = Path("../results")
CACHE_FILE = RESULTS_DIR / "activation_cache.pt"

EXPECTED_PROBLEMS = [
    "problem_330",
    "problem_2050",
    "problem_3360",
    "problem_4019",
    "problem_6481",
]


def main():
    print("=" * 60)
    print("Step 4: Verifying Activation Cache")
    print("=" * 60)
    print()

    if not CACHE_FILE.exists():
        print(f"✗ Cache file not found: {CACHE_FILE}")
        print("  Run step 3_extract_activations.py first")
        return

    # Load cache
    print(f"Loading cache from {CACHE_FILE}...")
    cache_data = torch.load(CACHE_FILE)
    print(f"  ✓ Cache loaded")
    print()

    # Extract components
    activation_cache = cache_data["activation_cache"]
    problems = cache_data["problems"]
    config = cache_data["config"]

    # Display config
    print("Configuration:")
    print(f"  Model: {config['target_model']}")
    print(f"  Layer: {config['activation_layer']}")
    print(f"  Problems: {config['num_problems']}")
    print(f"  Total chunks: {config['total_chunks']}")
    print(f"  Extraction time: {config['extraction_time']:.1f}s")
    print()

    # Verify problems
    print("Verifying problems...")
    cached_problems = set(activation_cache.keys())
    expected_problems = set(EXPECTED_PROBLEMS)

    missing = expected_problems - cached_problems
    extra = cached_problems - expected_problems

    if not missing and not extra:
        print(f"  ✓ All {len(EXPECTED_PROBLEMS)} problems present")
    else:
        if missing:
            print(f"  ✗ Missing problems: {missing}")
        if extra:
            print(f"  ! Extra problems: {extra}")

    print()

    # Verify each problem
    print("Verifying activations per problem...")
    stats = []

    for problem in problems:
        problem_id = problem["problem_id"]
        expected_chunks = len(problem["chunks"])
        cached_chunks = len(activation_cache[problem_id])

        # Get activation shapes and check for issues
        chunk_ids = list(activation_cache[problem_id].keys())
        sample_activation = activation_cache[problem_id][chunk_ids[0]]
        activation_shape = sample_activation.shape
        activation_dtype = sample_activation.dtype

        # Check for NaN/Inf
        all_activations = [activation_cache[problem_id][i] for i in chunk_ids]
        has_nan = any(torch.isnan(act).any().item() for act in all_activations)
        has_inf = any(torch.isinf(act).any().item() for act in all_activations)

        # Stats
        all_vals = torch.cat([act.flatten() for act in all_activations])
        mean_val = all_vals.mean().item()
        std_val = all_vals.std().item()
        min_val = all_vals.min().item()
        max_val = all_vals.max().item()

        problem_stats = {
            "problem_id": problem_id,
            "expected_chunks": expected_chunks,
            "cached_chunks": cached_chunks,
            "activation_shape": list(activation_shape),
            "dtype": str(activation_dtype),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "stats": {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val
            }
        }
        stats.append(problem_stats)

        # Print
        status = "✓" if cached_chunks == expected_chunks and not has_nan and not has_inf else "✗"
        print(f"  {status} {problem_id}:")
        print(f"    Chunks: {cached_chunks}/{expected_chunks}")
        print(f"    Shape: {activation_shape}")
        print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")

        if has_nan:
            print(f"    ✗ Contains NaN values!")
        if has_inf:
            print(f"    ✗ Contains Inf values!")

    print()

    # Overall summary
    total_cached = sum(s["cached_chunks"] for s in stats)
    total_expected = sum(s["expected_chunks"] for s in stats)
    any_nan = any(s["has_nan"] for s in stats)
    any_inf = any(s["has_inf"] for s in stats)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total activations: {total_cached}/{total_expected}")

    if total_cached == total_expected:
        print("✓ All chunks have activations")
    else:
        print(f"✗ Missing {total_expected - total_cached} activations")

    if not any_nan:
        print("✓ No NaN values")
    else:
        print("✗ Some activations contain NaN")

    if not any_inf:
        print("✓ No Inf values")
    else:
        print("✗ Some activations contain Inf")

    print()

    # Save stats
    output_file = RESULTS_DIR / "cache_stats.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": config,
            "problem_stats": stats,
            "summary": {
                "total_problems": len(stats),
                "total_cached": total_cached,
                "total_expected": total_expected,
                "any_nan": any_nan,
                "any_inf": any_inf
            }
        }, f, indent=2)

    print(f"Stats saved to {output_file}")

    if total_cached == total_expected and not any_nan and not any_inf:
        print()
        print("✓ All verification checks passed!")
        print("✓ Ready to proceed to Experiment 1: Swap Test")
    else:
        print()
        print("✗ Some checks failed. Review errors above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
