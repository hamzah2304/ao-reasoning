"""
Step 3: Analyze Swap Test Results

Produces:
1. Confusion matrix heatmap (activation source vs context source)
2. Statistical analysis (chi-squared test, binomial test)
3. Summary metrics and interpretation
4. Visualizations for the report

Usage:
    python 3_analyze_results.py
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RESULTS_DIR = Path("../results")
RAW_RESULTS_FILE = RESULTS_DIR / "swap_test_raw.json"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_results():
    """Load raw results from JSON."""
    with open(RAW_RESULTS_FILE) as f:
        data = json.load(f)
    return data["config"], data["results"]


def compute_metrics(results: list) -> dict:
    """Compute key metrics from results."""
    matched = [r for r in results if r["is_matched"]]
    mismatched = [r for r in results if not r["is_matched"]]

    metrics = {
        "total_tests": len(results),
        "matched_tests": len(matched),
        "mismatched_tests": len(mismatched),
    }

    # Matched (baseline) accuracy
    if matched:
        matched_correct = sum(1 for r in matched if r["matches_activation"])
        metrics["baseline_accuracy"] = matched_correct / len(matched)
        metrics["baseline_correct"] = matched_correct

    # Mismatched analysis
    if mismatched:
        act_matches = sum(1 for r in mismatched if r["matches_activation"])
        ctx_matches = sum(1 for r in mismatched if r["matches_context"])
        neither = sum(1 for r in mismatched if r["matches_neither"])

        metrics["activation_alignment_rate"] = act_matches / len(mismatched)
        metrics["context_alignment_rate"] = ctx_matches / len(mismatched)
        metrics["neither_rate"] = neither / len(mismatched)
        metrics["activation_matches"] = act_matches
        metrics["context_matches"] = ctx_matches
        metrics["neither_matches"] = neither

    return metrics


def compute_statistics(metrics: dict, num_problems: int = 5) -> dict:
    """Compute statistical tests."""
    stats_results = {}

    n_mismatched = metrics.get("mismatched_tests", 0)
    if n_mismatched == 0:
        return stats_results

    # Binomial test: Is activation alignment significantly above chance (1/5 = 20%)?
    act_matches = metrics.get("activation_matches", 0)
    chance_prob = 1 / num_problems

    # One-sided binomial test
    binom_result = stats.binomtest(
        act_matches,
        n_mismatched,
        chance_prob,
        alternative='greater'
    )
    stats_results["binomial_test"] = {
        "successes": act_matches,
        "trials": n_mismatched,
        "null_probability": chance_prob,
        "p_value": binom_result.pvalue,
        "significant_at_05": binom_result.pvalue < 0.05,
        "ci_lower": binom_result.proportion_ci(confidence_level=0.95).low,
        "ci_upper": binom_result.proportion_ci(confidence_level=0.95).high,
    }

    # Chi-squared test: Is there association between match type and answer source?
    # Contingency table: [matches_activation, matches_context, neither]
    observed = np.array([
        metrics.get("activation_matches", 0),
        metrics.get("context_matches", 0),
        metrics.get("neither_matches", 0)
    ])

    # Expected under null (equal probability)
    expected = np.array([n_mismatched / 3] * 3)

    if all(expected > 0):
        chi2, p_value = stats.chisquare(observed, expected)
        stats_results["chi_squared_test"] = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "significant_at_05": p_value < 0.05,
        }

    return stats_results


def create_confusion_matrix(results: list, problem_ids: list) -> np.ndarray:
    """Create a confusion matrix showing oracle answers.

    Rows: Activation source
    Cols: Context source
    Values: What the oracle answered (encoded as: 0=neither, 1=activation, 2=context)
    """
    n = len(problem_ids)
    matrix = np.zeros((n, n), dtype=int)

    prob_to_idx = {p: i for i, p in enumerate(problem_ids)}

    for r in results:
        i = prob_to_idx[r["activation_source"]]
        j = prob_to_idx[r["context_source"]]

        if r["matches_activation"]:
            matrix[i, j] = 1  # Matches activation
        elif r["matches_context"]:
            matrix[i, j] = 2  # Matches context
        else:
            matrix[i, j] = 0  # Neither

    return matrix


def create_response_matrix(results: list, problem_ids: list) -> dict:
    """Create matrices of oracle responses and correct answers."""
    n = len(problem_ids)
    responses = [[None for _ in range(n)] for _ in range(n)]
    correct = [[None for _ in range(n)] for _ in range(n)]

    prob_to_idx = {p: i for i, p in enumerate(problem_ids)}

    for r in results:
        i = prob_to_idx[r["activation_source"]]
        j = prob_to_idx[r["context_source"]]
        responses[i][j] = r["parsed_answer"]
        correct[i][j] = (r["matches_activation"], r["matches_context"])

    return {"responses": responses, "correct": correct}


def plot_confusion_heatmap(results: list, problem_ids: list, output_path: Path):
    """Create and save confusion matrix heatmap."""
    n = len(problem_ids)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get response data
    response_data = create_response_matrix(results, problem_ids)
    responses = response_data["responses"]
    correct = response_data["correct"]

    # Subplot 1: What the oracle answered
    ax1 = axes[0]
    response_display = [[responses[i][j] if responses[i][j] else "" for j in range(n)] for i in range(n)]

    # Color based on match type
    colors = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if correct[i][j]:
                matches_act, matches_ctx = correct[i][j]
                if matches_act and matches_ctx:  # Diagonal
                    colors[i, j] = 3
                elif matches_act:
                    colors[i, j] = 1
                elif matches_ctx:
                    colors[i, j] = 2
                else:
                    colors[i, j] = 0

    cmap = plt.cm.colors.ListedColormap(['lightgray', 'lightgreen', 'lightcoral', 'lightblue'])
    sns.heatmap(colors, ax=ax1, cmap=cmap, annot=response_display, fmt='',
                xticklabels=[p[-3:] for p in problem_ids],
                yticklabels=[p[-3:] for p in problem_ids],
                cbar=False, linewidths=0.5, linecolor='gray')
    ax1.set_xlabel('Context Source (Problem ID)')
    ax1.set_ylabel('Activation Source (Problem ID)')
    ax1.set_title('Oracle Responses\n(Green=Activation, Red=Context, Blue=Both, Gray=Neither)')

    # Subplot 2: Summary bar chart
    ax2 = axes[1]
    metrics = compute_metrics(results)

    if metrics.get("mismatched_tests", 0) > 0:
        categories = ['Matches\nActivation', 'Matches\nContext', 'Matches\nNeither']
        values = [
            metrics.get("activation_alignment_rate", 0) * 100,
            metrics.get("context_alignment_rate", 0) * 100,
            metrics.get("neither_rate", 0) * 100,
        ]
        colors = ['lightgreen', 'lightcoral', 'lightgray']

        bars = ax2.bar(categories, values, color=colors, edgecolor='black')
        ax2.axhline(y=20, color='red', linestyle='--', label='Chance (20%)')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Mismatched Test Results\n(Off-diagonal cases)')
        ax2.set_ylim(0, 100)
        ax2.legend()

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_detailed_results(results: list, output_path: Path):
    """Create detailed results visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort results by test case
    results_sorted = sorted(results, key=lambda r: (r["activation_source"], r["context_source"]))

    x = range(len(results_sorted))
    colors = []
    labels = []

    for r in results_sorted:
        if r["is_matched"]:
            if r["matches_activation"]:
                colors.append('blue')
            else:
                colors.append('orange')
        else:
            if r["matches_activation"]:
                colors.append('green')
            elif r["matches_context"]:
                colors.append('red')
            else:
                colors.append('gray')

        act_short = r["activation_source"][-3:]
        ctx_short = r["context_source"][-3:]
        labels.append(f"{act_short}→{ctx_short}")

    ax.scatter(x, [1]*len(results_sorted), c=colors, s=200, marker='s')

    # Add labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Matched (correct)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Matched (wrong)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Mismatched: matches ACT'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Mismatched: matches CTX'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Mismatched: neither'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

    ax.set_title('All Test Cases (Activation→Context)')
    ax.set_xlabel('Test Case')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def interpret_results(metrics: dict, statistics: dict) -> str:
    """Generate interpretation text."""
    lines = []

    # Baseline
    if "baseline_accuracy" in metrics:
        acc = metrics["baseline_accuracy"] * 100
        lines.append(f"BASELINE (Matched): {acc:.1f}% accuracy ({metrics['baseline_correct']}/{metrics['matched_tests']})")
        if acc < 50:
            lines.append("  Warning: Low baseline accuracy suggests oracle may not reliably extract answers even with matched conditions.")

    # Main question: Does oracle read activations?
    if "activation_alignment_rate" in metrics:
        act_rate = metrics["activation_alignment_rate"] * 100
        ctx_rate = metrics["context_alignment_rate"] * 100
        neither_rate = metrics["neither_rate"] * 100

        lines.append("")
        lines.append(f"MISMATCHED TEST RESULTS:")
        lines.append(f"  Activation alignment: {act_rate:.1f}%")
        lines.append(f"  Context alignment: {ctx_rate:.1f}%")
        lines.append(f"  Neither: {neither_rate:.1f}%")

        # Interpret
        lines.append("")
        lines.append("INTERPRETATION:")

        if act_rate > ctx_rate + 20:
            lines.append("  Strong evidence for ACTIVATION READING")
            lines.append("  The oracle appears to extract information from activations rather than confabulating from context.")
        elif ctx_rate > act_rate + 20:
            lines.append("  Strong evidence for CONTEXT CONFABULATION")
            lines.append("  The oracle appears to ignore activations and generate answers based on context alone.")
        elif neither_rate > 50:
            lines.append("  HIGH NOISE/CONFUSION")
            lines.append("  The oracle is not reliably extracting information from either source.")
        else:
            lines.append("  AMBIGUOUS RESULTS")
            lines.append("  No clear evidence for either hypothesis. Oracle may use both sources or neither reliably.")

        # Statistical significance
        if "binomial_test" in statistics:
            binom = statistics["binomial_test"]
            lines.append("")
            lines.append(f"STATISTICAL ANALYSIS:")
            lines.append(f"  Binomial test (activation > chance):")
            lines.append(f"    p-value: {binom['p_value']:.4f}")
            lines.append(f"    95% CI: [{binom['ci_lower']:.2f}, {binom['ci_upper']:.2f}]")
            if binom["significant_at_05"]:
                lines.append(f"    SIGNIFICANT at p<0.05")
            else:
                lines.append(f"    NOT significant at p<0.05")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Step 3: Analyzing Swap Test Results")
    print("=" * 60)
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    config, results = load_results()
    print(f"  Loaded {len(results)} test results")
    print(f"  Quick mode: {config.get('quick_mode', False)}")
    print()

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)
    print(f"  Total tests: {metrics['total_tests']}")
    print(f"  Matched (diagonal): {metrics['matched_tests']}")
    print(f"  Mismatched (off-diagonal): {metrics['mismatched_tests']}")
    print()

    # Get problem IDs
    problem_ids = sorted(set(r["activation_source"] for r in results))

    # Compute statistics
    print("Running statistical tests...")
    statistics = compute_statistics(metrics, len(problem_ids))
    print()

    # Generate plots
    print("Generating visualizations...")
    plot_confusion_heatmap(results, problem_ids, FIGURES_DIR / "confusion_heatmap.png")
    plot_detailed_results(results, FIGURES_DIR / "detailed_results.png")
    print()

    # Interpret results
    interpretation = interpret_results(metrics, statistics)
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(interpretation)
    print()

    # Save analysis
    analysis_output = {
        "config": config,
        "metrics": metrics,
        "statistics": statistics,
        "interpretation": interpretation,
    }

    output_file = RESULTS_DIR / "swap_test_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)

    print(f"Analysis saved to {output_file}")
    print(f"Figures saved to {FIGURES_DIR}/")
    print()
    print("=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
