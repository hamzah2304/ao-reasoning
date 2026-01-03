#!/usr/bin/env python3
"""
Analyze attention patterns by function tag.

Question: Do certain attention heads preferentially attend to/from
certain types of reasoning steps (function tags)?
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add thought-anchors to path
sys.path.append('/workspace/thought-anchors/whitebox-analyses')
sys.path.append('/workspace/thought-anchors/whitebox-analyses/scripts')

from attention_analysis.receiver_head_funcs import (
    get_problem_text_sentences,
    get_all_heads_vert_scores,
)
from attention_analysis.attn_funcs import get_avg_attention_matrix


def log(msg):
    print(msg, flush=True)


# Paths
PROBLEM_DIR = Path("/workspace/thought-anchors/math-rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution/problem_1591")
RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results")

FUNCTION_TAGS = [
    "active_computation", "fact_retrieval", "uncertainty_management",
    "result_consolidation", "plan_generation", "self_checking",
    "problem_setup", "final_answer_emission"
]


def load_chunks_with_tags():
    """Load chunks and their function tags."""
    with open(PROBLEM_DIR / "chunks_labeled.json") as f:
        chunks = json.load(f)
    return chunks


def get_attention_matrix(layer, head, text, sentences, model_name="llama-8b"):
    """Get the sentence-averaged attention matrix for a layer/head."""
    return get_avg_attention_matrix(
        text,
        model_name=model_name,
        layer=layer,
        head=head,
        sentences=sentences,
    )


def analyze_attention_by_tag(chunks, text, sentences, n_layers=32, n_heads=32, sample_layers=None):
    """
    Analyze which heads attend strongly to which function tag types.

    For each head, compute:
    - Average attention received by sentences of each function tag
    - Average attention given by sentences of each function tag
    """
    if sample_layers is None:
        sample_layers = [0, 8, 16, 24, 31]  # Sample layers to speed up

    n_sentences = len(sentences)

    # Build sentence -> tag mapping (a sentence can have multiple tags)
    sentence_tags = []
    for chunk in chunks:
        tags = chunk.get("function_tags", [])
        sentence_tags.append(tags if tags else ["unknown"])

    # Ensure we have the right number
    if len(sentence_tags) != n_sentences:
        log(f"Warning: {len(sentence_tags)} chunks but {n_sentences} sentences")
        # Pad or truncate
        while len(sentence_tags) < n_sentences:
            sentence_tags.append(["unknown"])
        sentence_tags = sentence_tags[:n_sentences]

    # Results storage
    # attention_to_tag[layer][head][tag] = list of attention values received
    # attention_from_tag[layer][head][tag] = list of attention values given
    attention_to_tag = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    attention_from_tag = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    log(f"\nAnalyzing attention patterns for {len(sample_layers)} layers...")

    for layer in sample_layers:
        log(f"  Layer {layer}...")
        for head in range(n_heads):
            # Get attention matrix (n_sentences x n_sentences)
            # attn[i, j] = attention from sentence i to sentence j
            try:
                attn = get_attention_matrix(layer, head, text, sentences)
            except Exception as e:
                log(f"    Error getting attention for L{layer}H{head}: {e}")
                continue

            # For each target sentence, how much attention does it receive?
            for j in range(n_sentences):
                attn_received = attn[:, j].mean()  # Average attention from all sentences
                for tag in sentence_tags[j]:
                    attention_to_tag[layer][head][tag].append(attn_received)

            # For each source sentence, how much attention does it give to earlier sentences?
            for i in range(n_sentences):
                if i > 0:
                    attn_given = attn[i, :i].mean()  # Average attention to earlier sentences
                    for tag in sentence_tags[i]:
                        attention_from_tag[layer][head][tag].append(attn_given)

    return attention_to_tag, attention_from_tag, sentence_tags


def summarize_results(attention_to_tag, attention_from_tag):
    """Summarize attention patterns by tag."""
    log("\n" + "=" * 70)
    log("ATTENTION PATTERNS BY FUNCTION TAG")
    log("=" * 70)

    # Aggregate across all layers/heads
    tag_received = defaultdict(list)
    tag_given = defaultdict(list)

    for layer in attention_to_tag:
        for head in attention_to_tag[layer]:
            for tag in attention_to_tag[layer][head]:
                tag_received[tag].extend(attention_to_tag[layer][head][tag])
            for tag in attention_from_tag[layer][head]:
                tag_given[tag].extend(attention_from_tag[layer][head][tag])

    log("\n[Attention RECEIVED by function tag]")
    log("(Higher = sentences of this type are attended to more)")
    log("-" * 50)
    results_received = []
    for tag in FUNCTION_TAGS:
        if tag in tag_received and tag_received[tag]:
            mean_attn = np.mean(tag_received[tag])
            std_attn = np.std(tag_received[tag])
            results_received.append((tag, mean_attn, std_attn, len(tag_received[tag])))

    results_received.sort(key=lambda x: -x[1])
    for tag, mean, std, n in results_received:
        log(f"  {tag:<25} mean={mean:.4f} (std={std:.4f}, n={n})")

    log("\n[Attention GIVEN by function tag]")
    log("(Higher = sentences of this type attend more to earlier context)")
    log("-" * 50)
    results_given = []
    for tag in FUNCTION_TAGS:
        if tag in tag_given and tag_given[tag]:
            mean_attn = np.mean(tag_given[tag])
            std_attn = np.std(tag_given[tag])
            results_given.append((tag, mean_attn, std_attn, len(tag_given[tag])))

    results_given.sort(key=lambda x: -x[1])
    for tag, mean, std, n in results_given:
        log(f"  {tag:<25} mean={mean:.4f} (std={std:.4f}, n={n})")

    return results_received, results_given


def find_specialized_heads(attention_to_tag, attention_from_tag, top_k=5):
    """Find heads that are most specialized for certain function tags."""
    log("\n" + "=" * 70)
    log("SPECIALIZED ATTENTION HEADS")
    log("=" * 70)

    # For each head, compute which tag it attends to most (relative to others)
    head_specialization = []

    for layer in attention_to_tag:
        for head in attention_to_tag[layer]:
            tag_means = {}
            for tag in FUNCTION_TAGS:
                if tag in attention_to_tag[layer][head]:
                    tag_means[tag] = np.mean(attention_to_tag[layer][head][tag])

            if len(tag_means) >= 2:
                # Find the tag with highest attention and compute relative strength
                best_tag = max(tag_means, key=tag_means.get)
                best_val = tag_means[best_tag]
                other_vals = [v for t, v in tag_means.items() if t != best_tag]
                if other_vals:
                    avg_other = np.mean(other_vals)
                    if avg_other > 0:
                        specialization = best_val / avg_other
                        head_specialization.append((layer, head, best_tag, specialization, best_val))

    # Sort by specialization
    head_specialization.sort(key=lambda x: -x[3])

    log(f"\n[Top {top_k} specialized heads for receiving attention]")
    log("-" * 60)
    for layer, head, tag, spec, val in head_specialization[:top_k]:
        log(f"  L{layer:02d}H{head:02d}: {tag:<25} (spec={spec:.2f}x, val={val:.4f})")

    return head_specialization


def main():
    log("=" * 70)
    log("ATTENTION ANALYSIS BY FUNCTION TAG")
    log("=" * 70)
    log("\nAnalyzing how attention patterns differ by reasoning function type.")
    log("Using problem 1591 (101 sentences with function tags).")
    log("=" * 70)

    os.chdir('/workspace/thought-anchors/whitebox-analyses/scripts')

    # Load data
    chunks = load_chunks_with_tags()
    log(f"\nLoaded {len(chunks)} chunks with function tags")

    # Get text and sentences
    text, sentences = get_problem_text_sentences(1591, True, "llama-8b")
    log(f"Text has {len(sentences)} sentences")

    # Analyze attention patterns (sample 5 layers to speed up)
    sample_layers = [0, 8, 16, 24, 31]
    attention_to_tag, attention_from_tag, sentence_tags = analyze_attention_by_tag(
        chunks, text, sentences,
        n_layers=32, n_heads=32,
        sample_layers=sample_layers
    )

    # Summarize results
    results_received, results_given = summarize_results(attention_to_tag, attention_from_tag)

    # Find specialized heads
    head_specialization = find_specialized_heads(attention_to_tag, attention_from_tag)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {
        "attention_received_by_tag": {tag: {"mean": float(m), "std": float(s), "n": int(n)}
                                       for tag, m, s, n in results_received},
        "attention_given_by_tag": {tag: {"mean": float(m), "std": float(s), "n": int(n)}
                                    for tag, m, s, n in results_given},
        "top_specialized_heads": [
            {"layer": int(l), "head": int(h), "tag": t, "specialization": float(sp), "value": float(v)}
            for l, h, t, sp, v in head_specialization[:20]
        ],
        "sample_layers": sample_layers,
        "n_sentences": len(sentences),
        "n_chunks": len(chunks),
    }

    save_path = RESULTS_DIR / "attention_by_function_tag.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    log(f"\nResults saved to {save_path}")

    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
