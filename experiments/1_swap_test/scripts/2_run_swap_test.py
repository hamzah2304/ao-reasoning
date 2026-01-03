"""
Step 2: Run Swap Test Experiment

Creates a 5x5 matrix of tests where:
- Diagonal: Matched activations and context (baseline)
- Off-diagonal: Mismatched activations and context

For each test:
1. Load activation from problem A (final reasoning step)
2. Create prompt with context from problem B
3. Ask oracle "What is the final numerical answer?"
4. Record response

Usage:
    python 2_run_swap_test.py           # Full 5x5 matrix (25 tests)
    python 2_run_swap_test.py --quick   # Quick test (4 cases only)
"""

import json
import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from typing import Optional

# IMPORTANT: Set cache to persistent location BEFORE importing transformers
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add activation_oracles to path for utilities
sys.path.insert(0, '/workspace/activation_oracles')
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

# Configuration
ORACLE_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"
ACTIVATION_CACHE = Path("../../0_setup/results/activation_cache.pt")
RESULTS_DIR = Path("../results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Oracle configuration
INJECTION_LAYER = 1  # Layer to inject activations into oracle
STEERING_COEFFICIENT = 1.0

# Problem answers (ground truth)
PROBLEM_ANSWERS = {
    "problem_330": "88572",
    "problem_2050": "336",
    "problem_3360": "12.1",
    "problem_4019": "4",
    "problem_6481": "30.8",
}


@dataclass
class SwapTestResult:
    """Result of a single swap test."""
    activation_source: str  # Problem ID for activation
    context_source: str     # Problem ID for context
    activation_answer: str  # Ground truth for activation source
    context_answer: str     # Ground truth for context source
    is_matched: bool        # True if activation_source == context_source
    oracle_response: str    # Raw oracle output
    parsed_answer: str      # Extracted numerical answer
    matches_activation: bool  # Response matches activation source's answer
    matches_context: bool     # Response matches context source's answer
    matches_neither: bool     # Matches neither
    inference_time: float


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Generate the prefix that tells the oracle about the activation source.

    This prefix format is used by the activation oracles training.
    """
    return f"[Activations from layer {layer}, {num_positions} token position(s)]\n\n"


def create_oracle_prompt(problem_text: str, layer: int, num_positions: int) -> str:
    """Create the full prompt for the oracle.

    Args:
        problem_text: The math problem text to use as context
        layer: The layer the activations came from (for prefix)
        num_positions: Number of activation positions

    Returns:
        Full prompt string
    """
    prefix = get_introspection_prefix(layer, num_positions)

    question = """Based on the activations provided, what is the final numerical answer to the math problem being solved?

Context (problem being reasoned about):
{problem}

Answer with ONLY the numerical answer, nothing else.""".format(problem=problem_text)

    return prefix + question


def parse_answer(response: str) -> str:
    """Extract numerical answer from oracle response."""
    # Clean up the response
    response = response.strip()

    # Try to extract first number-like string
    import re

    # Match integers, decimals, fractions
    patterns = [
        r'^[\d,]+\.?\d*',  # Numbers at start
        r'[\d,]+\.?\d*',   # Any number
    ]

    for pattern in patterns:
        match = re.search(pattern, response.replace(',', ''))
        if match:
            return match.group(0)

    return response[:20]  # Fallback: first 20 chars


def answers_match(parsed: str, ground_truth: str) -> bool:
    """Check if parsed answer matches ground truth (with tolerance for floats)."""
    try:
        parsed_float = float(parsed.replace(',', ''))
        truth_float = float(ground_truth.replace(',', ''))
        return abs(parsed_float - truth_float) < 0.1
    except ValueError:
        return parsed.strip() == ground_truth.strip()


def run_oracle_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    submodule: torch.nn.Module,
    prompt: str,
    activation: torch.Tensor,
    position: int = 1,  # Inject at this position in the prompt
) -> tuple[str, float]:
    """Run oracle inference with activation steering.

    Args:
        model: The oracle model
        tokenizer: Tokenizer
        submodule: The layer to inject into
        prompt: The text prompt
        activation: Activation vector to inject [d_model]
        position: Token position to inject at

    Returns:
        (response_text, inference_time)
    """
    # Tokenize prompt
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    # Prepare steering vectors
    # activation shape: [d_model], need [1, d_model] for single position
    steering_vector = activation.unsqueeze(0).to(DEVICE)  # [1, d_model]

    # Create hook
    vectors = [steering_vector]  # Batch of 1
    positions = [[position]]     # Inject at position 1

    hook_fn = get_hf_activation_steering_hook(
        vectors=vectors,
        positions=positions,
        steering_coefficient=STEERING_COEFFICIENT,
        device=DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )

    # Generate with steering
    start = time.time()
    with torch.no_grad():
        with add_hook(submodule, hook_fn):
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.time() - start

    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip(), elapsed


def main():
    quick_mode = "--quick" in sys.argv

    print("=" * 60)
    if quick_mode:
        print("Step 2: Running Swap Test (QUICK MODE - 4 tests)")
    else:
        print("Step 2: Running Swap Test (FULL - 25 tests)")
    print("=" * 60)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load activation cache
    print("Loading activation cache...")
    cache_data = torch.load(ACTIVATION_CACHE, weights_only=False)
    activation_cache = cache_data["activation_cache"]
    problems = cache_data["problems"]
    config = cache_data["config"]

    print(f"  Loaded {len(activation_cache)} problems")
    print(f"  Source layer: {config['activation_layer']}")
    print()

    # Load oracle model
    print("Loading oracle model...")
    tokenizer = AutoTokenizer.from_pretrained(ORACLE_BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ORACLE_BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )

    # Load oracle LoRA
    model.load_adapter(
        ORACLE_LORA_PATH,
        adapter_name="oracle",
        is_trainable=False,
        low_cpu_mem_usage=True
    )
    model.set_adapter("oracle")
    model.eval()
    print("  Oracle loaded")
    print()

    # Get injection submodule (layer 1)
    submodule = get_hf_submodule(model, INJECTION_LAYER)
    print(f"  Injection layer: {INJECTION_LAYER}")
    print()

    # Get problem list
    problem_ids = list(activation_cache.keys())
    print(f"Problems: {problem_ids}")
    print()

    # Build test matrix
    if quick_mode:
        # Quick mode: 2 matched + 2 mismatched
        test_pairs = [
            (problem_ids[0], problem_ids[0]),  # Matched
            (problem_ids[1], problem_ids[1]),  # Matched
            (problem_ids[0], problem_ids[1]),  # Mismatched
            (problem_ids[1], problem_ids[0]),  # Mismatched
        ]
    else:
        # Full 5x5 matrix
        test_pairs = []
        for act_prob in problem_ids:
            for ctx_prob in problem_ids:
                test_pairs.append((act_prob, ctx_prob))

    print(f"Running {len(test_pairs)} tests...")
    print()

    # Get problem texts for context
    problem_texts = {}
    for prob_data in problems:
        prob_id = prob_data["problem_id"]
        problem_texts[prob_id] = prob_data["problem"]["problem"]

    # Run tests
    results = []
    start_time = time.time()

    for act_prob, ctx_prob in tqdm(test_pairs, desc="Swap tests"):
        # Get final reasoning step activation
        chunk_ids = sorted(activation_cache[act_prob].keys())
        final_chunk_idx = chunk_ids[-1]
        activation = activation_cache[act_prob][final_chunk_idx]  # [4096]

        # Create prompt with context from ctx_prob
        source_layer = abs(config["activation_layer"])  # -8 -> 8
        prompt = create_oracle_prompt(
            problem_text=problem_texts[ctx_prob],
            layer=source_layer,
            num_positions=1
        )

        # Run oracle
        response, inf_time = run_oracle_inference(
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            prompt=prompt,
            activation=activation,
        )

        # Parse and evaluate
        parsed = parse_answer(response)
        act_answer = PROBLEM_ANSWERS[act_prob]
        ctx_answer = PROBLEM_ANSWERS[ctx_prob]

        matches_act = answers_match(parsed, act_answer)
        matches_ctx = answers_match(parsed, ctx_answer)

        result = SwapTestResult(
            activation_source=act_prob,
            context_source=ctx_prob,
            activation_answer=act_answer,
            context_answer=ctx_answer,
            is_matched=(act_prob == ctx_prob),
            oracle_response=response,
            parsed_answer=parsed,
            matches_activation=matches_act,
            matches_context=matches_ctx,
            matches_neither=(not matches_act and not matches_ctx),
            inference_time=inf_time,
        )
        results.append(result)

        # Print progress
        match_str = "ACT" if matches_act else ("CTX" if matches_ctx else "???")
        tqdm.write(f"  [{act_prob[-3:]}→{ctx_prob[-3:]}] Response: {parsed:>10} | Match: {match_str}")

    total_time = time.time() - start_time
    print()
    print(f"Total time: {total_time:.1f}s")
    print()

    # Save raw results
    output_file = RESULTS_DIR / "swap_test_raw.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "oracle_base": ORACLE_BASE_MODEL,
                "oracle_lora": ORACLE_LORA_PATH,
                "injection_layer": INJECTION_LAYER,
                "steering_coefficient": STEERING_COEFFICIENT,
                "source_layer": config["activation_layer"],
                "num_tests": len(results),
                "quick_mode": quick_mode,
            },
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    print(f"Raw results saved to {output_file}")
    print()

    # Quick summary
    matched = [r for r in results if r.is_matched]
    mismatched = [r for r in results if not r.is_matched]

    print("=" * 60)
    print("Quick Summary")
    print("=" * 60)

    if matched:
        matched_correct = sum(1 for r in matched if r.matches_activation)
        print(f"Matched (baseline): {matched_correct}/{len(matched)} correct")

    if mismatched:
        act_matches = sum(1 for r in mismatched if r.matches_activation)
        ctx_matches = sum(1 for r in mismatched if r.matches_context)
        neither = sum(1 for r in mismatched if r.matches_neither)
        print(f"Mismatched tests: {len(mismatched)}")
        print(f"  Matches activation: {act_matches} ({100*act_matches/len(mismatched):.1f}%)")
        print(f"  Matches context: {ctx_matches} ({100*ctx_matches/len(mismatched):.1f}%)")
        print(f"  Matches neither: {neither} ({100*neither/len(mismatched):.1f}%)")

    print()
    print("Ready for Step 3: Analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
