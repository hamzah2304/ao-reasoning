#!/usr/bin/env python3
"""
Visualize probe results: per-tag F1, confusion matrices, layer analysis.
Run after 2_train_probes.py and 3_layer_analysis.py complete.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results")

# Color scheme
COLORS = {
    "probe": "#2ecc71",      # green
    "baseline": "#e74c3c",   # red
    "accent": "#3498db",     # blue
    "neutral": "#95a5a6",    # gray
}


def plot_function_tag_f1(results):
    """Bar chart of per-tag F1 scores vs baselines."""
    fig, ax = plt.subplots(figsize=(12, 6))

    tags = list(results["per_tag_metrics"].keys())
    f1_scores = [results["per_tag_metrics"][t]["f1"] for t in tags]

    # Sort by F1 score
    sorted_idx = np.argsort(f1_scores)[::-1]
    tags = [tags[i] for i in sorted_idx]
    f1_scores = [f1_scores[i] for i in sorted_idx]

    x = np.arange(len(tags))
    bars = ax.bar(x, f1_scores, color=COLORS["probe"], alpha=0.8, label="Probe F1")

    # Add baseline lines
    ax.axhline(y=results["random_baseline_f1"], color=COLORS["baseline"],
               linestyle="--", linewidth=2, label=f"Random baseline ({results['random_baseline_f1']:.3f})")
    ax.axhline(y=results["macro_f1"], color=COLORS["accent"],
               linestyle="-", linewidth=2, label=f"Macro F1 ({results['macro_f1']:.3f})")

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Function Tag Probe: Per-Tag F1 Scores")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels, title):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_layer_analysis(layer_results):
    """Line plots showing probe performance across layers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    layers = layer_results["layers"]

    # Function tag F1
    ax = axes[0, 0]
    ax.plot(layers, layer_results["function_tag_macro_f1"], "o-",
            color=COLORS["probe"], linewidth=2, markersize=8, label="Macro F1")
    ax.plot(layers, layer_results["function_tag_micro_f1"], "s--",
            color=COLORS["accent"], linewidth=2, markersize=8, label="Micro F1")
    ax.set_xlabel("Layer (negative index)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Function Tag Probe by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correctness accuracy
    ax = axes[0, 1]
    ax.plot(layers, layer_results["correctness_accuracy"], "o-",
            color=COLORS["probe"], linewidth=2, markersize=8, label="Accuracy")
    ax.plot(layers, layer_results["correctness_f1"], "s--",
            color=COLORS["accent"], linewidth=2, markersize=8, label="F1")
    ax.axhline(y=0.5, color=COLORS["baseline"], linestyle="--", label="Random (0.5)")
    ax.set_xlabel("Layer (negative index)")
    ax.set_ylabel("Score")
    ax.set_title("Correctness Probe by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Importance R²
    ax = axes[1, 0]
    ax.plot(layers, layer_results["importance_r2"], "o-",
            color=COLORS["probe"], linewidth=2, markersize=8)
    ax.axhline(y=0, color=COLORS["baseline"], linestyle="--", label="Baseline (R²=0)")
    ax.set_xlabel("Layer (negative index)")
    ax.set_ylabel("R² Score")
    ax.set_title("Importance Regression by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Importance anchor F1
    ax = axes[1, 1]
    ax.plot(layers, layer_results["importance_anchor_f1"], "o-",
            color=COLORS["probe"], linewidth=2, markersize=8)
    ax.set_xlabel("Layer (negative index)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Anchor Detection (Top-10%) by Layer")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_summary_comparison(probe_results):
    """Summary bar chart comparing all probes to baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Function Tags\n(Macro F1)", "Correctness\n(Accuracy)",
               "Importance\n(R²)", "Anchor Detection\n(F1)"]

    probe_scores = [
        probe_results["function_tags"]["macro_f1"],
        probe_results["correctness"]["accuracy"],
        max(0, probe_results["importance"]["regression"]["r2"]),  # Clamp negative R²
        probe_results["importance"]["classification"]["f1"],
    ]

    baseline_scores = [
        probe_results["function_tags"]["random_baseline_f1"],
        probe_results["correctness"]["random_baseline"],
        0,  # R² baseline
        0.5,  # Random F1 for balanced
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, probe_scores, width, label="Probe", color=COLORS["probe"], alpha=0.8)
    bars2 = ax.bar(x + width/2, baseline_scores, width, label="Baseline", color=COLORS["baseline"], alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Linear Probe Results vs Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("VISUALIZING PROBE RESULTS")
    print("=" * 60)

    # Create figures directory
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load results
    probe_path = RESULTS_DIR / "probe_results.json"
    layer_path = RESULTS_DIR / "layer_analysis_results.json"

    if not probe_path.exists():
        print(f"ERROR: {probe_path} not found. Run 2_train_probes.py first.")
        return

    with open(probe_path) as f:
        probe_results = json.load(f)
    print(f"Loaded probe results from {probe_path}")

    # 1. Function tag F1 bar chart
    print("\n1. Plotting function tag F1 scores...")
    fig = plot_function_tag_f1(probe_results["function_tags"])
    fig.savefig(figures_dir / "function_tag_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {figures_dir / 'function_tag_f1.png'}")

    # 2. Correctness confusion matrix
    print("\n2. Plotting correctness confusion matrix...")
    cm = np.array(probe_results["correctness"]["confusion_matrix"])
    fig = plot_confusion_matrix(cm, ["Incorrect", "Correct"], "Correctness Probe Confusion Matrix")
    fig.savefig(figures_dir / "correctness_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {figures_dir / 'correctness_confusion.png'}")

    # 3. Importance confusion matrix
    print("\n3. Plotting importance (anchor) confusion matrix...")
    cm = np.array(probe_results["importance"]["classification"]["confusion_matrix"])
    fig = plot_confusion_matrix(cm, ["Not Anchor", "Anchor"], "Anchor Detection Confusion Matrix")
    fig.savefig(figures_dir / "importance_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {figures_dir / 'importance_confusion.png'}")

    # 4. Summary comparison
    print("\n4. Plotting summary comparison...")
    fig = plot_summary_comparison(probe_results)
    fig.savefig(figures_dir / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {figures_dir / 'summary_comparison.png'}")

    # 5. Layer analysis (if available)
    if layer_path.exists():
        print("\n5. Plotting layer analysis...")
        with open(layer_path) as f:
            layer_results = json.load(f)
        fig = plot_layer_analysis(layer_results)
        fig.savefig(figures_dir / "layer_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {figures_dir / 'layer_analysis.png'}")
    else:
        print(f"\n5. Skipping layer analysis (run 3_layer_analysis.py first)")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    func = probe_results["function_tags"]
    corr = probe_results["correctness"]
    imp = probe_results["importance"]

    print(f"\nFunction Tag Probe:")
    print(f"  Macro F1: {func['macro_f1']:.3f} (baseline: {func['random_baseline_f1']:.3f})")
    print(f"  Improvement: +{func['macro_f1'] - func['random_baseline_f1']:.3f}")

    print(f"\nCorrectness Probe:")
    print(f"  Accuracy: {corr['accuracy']:.3f} (baseline: {corr['random_baseline']:.3f})")
    print(f"  Improvement: +{corr['accuracy'] - corr['random_baseline']:.3f}")

    print(f"\nImportance Probes:")
    print(f"  Regression R²: {imp['regression']['r2']:.3f}")
    print(f"  Anchor F1: {imp['classification']['f1']:.3f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if func['macro_f1'] > func['random_baseline_f1'] + 0.1:
        print("✓ Function tags ARE linearly decodable from activations")
    elif func['macro_f1'] > func['random_baseline_f1'] + 0.05:
        print("~ Function tags show WEAK linear decodability")
    else:
        print("✗ Function tags NOT linearly decodable")

    if corr['accuracy'] > 0.55:
        print("✓ Correctness IS encoded in activations")
    elif corr['accuracy'] > 0.52:
        print("~ Correctness shows WEAK encoding")
    else:
        print("✗ Correctness NOT encoded (or not linearly)")

    if imp['regression']['r2'] > 0.1:
        print("✓ Importance IS predictable from activations")
    elif imp['regression']['r2'] > 0:
        print("~ Importance shows WEAK predictability")
    else:
        print("✗ Importance NOT predictable from activations")

    print(f"\nFigures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
