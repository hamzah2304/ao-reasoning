"""
Step 1: Verify downloaded data structure and labels

Checks:
- All 5 problems downloaded
- Required files exist (problem.json, chunks_labeled.json)
- Labels are present (function_tags, importance scores)
- Answers are unique (for swap test)
"""

import json
from pathlib import Path
from collections import Counter

# Configuration
DATA_DIR = Path("../../../data/math_rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

SWAP_TEST_PROBLEMS = [
    "problem_330",   # Algebra
    "problem_2050",  # Counting & Probability
    "problem_3360",  # Geometry
    "problem_4019",  # Intermediate Algebra
    "problem_6481",  # Prealgebra
]

EXPECTED_LABELS = [
    "chunk",
    "chunk_idx",
    "function_tags",
    "counterfactual_importance_accuracy",
    "resampling_importance_accuracy",
    "accuracy"
]


def verify_problem(problem_id, solution_type="correct_base_solution"):
    """Verify a single problem's data."""
    problem_dir = DATA_DIR / solution_type / problem_id

    result = {
        "problem_id": problem_id,
        "exists": problem_dir.exists(),
        "files": {},
        "labels": {},
        "errors": []
    }

    if not problem_dir.exists():
        result["errors"].append(f"Directory not found: {problem_dir}")
        return result

    # Check problem.json
    problem_file = problem_dir / "problem.json"
    if problem_file.exists():
        with open(problem_file) as f:
            problem_data = json.load(f)
        result["files"]["problem.json"] = True
        result["problem_type"] = problem_data.get("type", "unknown")
        result["gt_answer"] = problem_data.get("gt_answer", "unknown")
        result["level"] = problem_data.get("level", "unknown")
    else:
        result["files"]["problem.json"] = False
        result["errors"].append("Missing problem.json")

    # Check chunks_labeled.json
    chunks_file = problem_dir / "chunks_labeled.json"
    if chunks_file.exists():
        with open(chunks_file) as f:
            chunks = json.load(f)
        result["files"]["chunks_labeled.json"] = True
        result["num_chunks"] = len(chunks)

        # Check labels in first chunk
        if chunks:
            first_chunk = chunks[0]
            chunk_keys = set(first_chunk.keys())
            result["labels"] = {label: label in chunk_keys for label in EXPECTED_LABELS}

            # Check if any labels missing
            missing = [label for label in EXPECTED_LABELS if label not in chunk_keys]
            if missing:
                result["errors"].append(f"Missing labels: {missing}")

            # Sample function tags
            function_tags = [c.get("function_tags", []) for c in chunks[:10]]
            result["sample_function_tags"] = function_tags

    else:
        result["files"]["chunks_labeled.json"] = False
        result["errors"].append("Missing chunks_labeled.json")

    return result


def main():
    print("=" * 60)
    print("Step 1: Verifying Downloaded Data")
    print("=" * 60)
    print()

    results = []
    all_answers = []

    for prob_id in SWAP_TEST_PROBLEMS:
        print(f"Checking {prob_id}...")
        result = verify_problem(prob_id)
        results.append(result)

        if result["exists"]:
            print(f"  ✓ Directory exists")
            print(f"    Type: {result.get('problem_type', 'N/A')}")
            print(f"    Answer: {result.get('gt_answer', 'N/A')}")
            print(f"    Chunks: {result.get('num_chunks', 0)}")
            all_answers.append(result.get('gt_answer'))

            # Check for errors
            if result["errors"]:
                for error in result["errors"]:
                    print(f"    ✗ {error}")
        else:
            print(f"  ✗ Directory not found")

        print()

    # Check answer uniqueness
    print("Answer Uniqueness Check:")
    answer_counts = Counter(all_answers)
    unique_answers = len(set(all_answers))
    print(f"  Total problems: {len(all_answers)}")
    print(f"  Unique answers: {unique_answers}")

    if unique_answers == len(all_answers):
        print("  ✓ All answers are unique (good for swap test)")
    else:
        print("  ✗ Some answers are duplicated:")
        for ans, count in answer_counts.items():
            if count > 1:
                print(f"    {ans}: {count} problems")

    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    successful = sum(1 for r in results if r["exists"] and not r["errors"])
    total_chunks = sum(r.get("num_chunks", 0) for r in results)

    print(f"Problems verified: {successful}/{len(SWAP_TEST_PROBLEMS)}")
    print(f"Total reasoning steps: {total_chunks}")
    print()

    # Save results
    output_file = RESULTS_DIR / "data_verification.json"
    with open(output_file, "w") as f:
        json.dump({
            "problems": results,
            "summary": {
                "total_problems": len(SWAP_TEST_PROBLEMS),
                "successful": successful,
                "total_chunks": total_chunks,
                "unique_answers": unique_answers
            }
        }, f, indent=2)

    print(f"Results saved to {output_file}")

    if successful == len(SWAP_TEST_PROBLEMS):
        print("✓ All checks passed! Ready for Step 2.")
    else:
        print("✗ Some checks failed. Fix issues before proceeding.")

    print("=" * 60)


if __name__ == "__main__":
    main()
