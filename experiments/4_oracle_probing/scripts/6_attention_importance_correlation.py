#!/usr/bin/env python3
"""
Correlate receiver head scores (attention received) with counterfactual importance.

Novel analysis: Does attention flow predict causal importance?

Hypothesis: Sentences that receive more attention from later sentences
should have higher counterfactual importance (removing them hurts more).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


def log(msg):
    print(msg, flush=True)


# Paths
IMPORTANCE_PATH = Path("/workspace/thought-anchors/analysis/problem_1591/rollouts_analysis/correct_base_solution/analysis_results.json")
RECEIVER_CSV = Path("/workspace/thought-anchors/csvs/receiver_head_scores_all_llama-8b_k16_pi4.csv")
RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results")


def load_importance_data():
    """Load counterfactual importance scores."""
    with open(IMPORTANCE_PATH) as f:
        data = json.load(f)

    # Should be a list with one problem
    problem = data[0] if isinstance(data, list) else data

    rows = []
    for chunk in problem["labeled_chunks"]:
        rows.append({
            "sentence_idx": chunk["chunk_idx"],
            "text": chunk["chunk"][:50],  # First 50 chars
            "function_tag": chunk["function_tags"][0] if chunk["function_tags"] else "unknown",
            "counterfactual_importance_accuracy": chunk["counterfactual_importance_accuracy"],
            "counterfactual_importance_kl": chunk["counterfactual_importance_kl"],
            "forced_importance_accuracy": chunk["forced_importance_accuracy"],
            "forced_importance_kl": chunk["forced_importance_kl"],
            "resampling_importance_accuracy": chunk["resampling_importance_accuracy"],
        })

    return pd.DataFrame(rows)


def load_receiver_scores():
    """Load receiver head scores."""
    df = pd.read_csv(RECEIVER_CSV)
    # Filter to problem 1591, correct solution
    df = df[(df["problem_number"] == 1591) & (df["is_correct"] == True)]
    return df[["sentence_idx", "receiver_head_score", "accuracy"]]


def correlate_and_plot(df, importance_col, save_name):
    """Compute correlation and create scatter plot."""
    # Remove NaN values
    mask = ~(df["receiver_head_score"].isna() | df[importance_col].isna())
    x = df.loc[mask, "receiver_head_score"].values
    y = np.abs(df.loc[mask, importance_col].values)  # Use absolute importance

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    log(f"\n  {importance_col}:")
    log(f"    Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})")
    log(f"    Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.4f})")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by function tag
    tags = df.loc[mask, "function_tag"].values
    unique_tags = list(set(tags))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tags)))
    tag_to_color = dict(zip(unique_tags, colors))

    for tag in unique_tags:
        tag_mask = tags == tag
        ax.scatter(x[tag_mask], y[tag_mask],
                   c=[tag_to_color[tag]], label=tag, alpha=0.6, s=50)

    # Add regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5,
            label=f'r={pearson_r:.3f}')

    ax.set_xlabel("Receiver Head Score (attention received)")
    ax.set_ylabel(f"|{importance_col}|")
    ax.set_title(f"Attention Received vs Counterfactual Importance\n(Pearson r={pearson_r:.3f}, p={pearson_p:.3f})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{save_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "metric": importance_col,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": int(len(x))
    }


def analyze_by_position(df):
    """Check if the correlation holds after controlling for position."""
    log("\n[Position Analysis]")
    log("-" * 50)

    # Add normalized position
    n = len(df)
    df = df.copy()
    df["position"] = df["sentence_idx"] / n

    # Correlation between position and receiver score
    r_pos_recv, p_pos_recv = stats.pearsonr(df["position"], df["receiver_head_score"])
    log(f"  Position ↔ Receiver score: r={r_pos_recv:.4f} (p={p_pos_recv:.4f})")

    # Correlation between position and importance
    r_pos_imp, p_pos_imp = stats.pearsonr(df["position"], np.abs(df["counterfactual_importance_accuracy"]))
    log(f"  Position ↔ |Importance|: r={r_pos_imp:.4f} (p={p_pos_imp:.4f})")

    # Partial correlation: receiver score ↔ importance, controlling for position
    from scipy.stats import pearsonr

    # Residualize both variables on position
    def residualize(y, x):
        slope, intercept = np.polyfit(x, y, 1)
        return y - (slope * x + intercept)

    recv_resid = residualize(df["receiver_head_score"].values, df["position"].values)
    imp_resid = residualize(np.abs(df["counterfactual_importance_accuracy"].values), df["position"].values)

    r_partial, p_partial = pearsonr(recv_resid, imp_resid)
    log(f"\n  Partial correlation (controlling for position):")
    log(f"    Receiver ↔ |Importance|: r={r_partial:.4f} (p={p_partial:.4f})")

    return {
        "position_receiver_r": float(r_pos_recv),
        "position_importance_r": float(r_pos_imp),
        "partial_r": float(r_partial),
        "partial_p": float(p_partial)
    }


def main():
    log("=" * 70)
    log("ATTENTION RECEIVED vs COUNTERFACTUAL IMPORTANCE")
    log("=" * 70)
    log("\nHypothesis: Sentences that receive more attention from later")
    log("sentences should have higher counterfactual importance.")
    log("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    log("\n[Loading Data]")
    importance_df = load_importance_data()
    log(f"  Importance data: {len(importance_df)} sentences")

    receiver_df = load_receiver_scores()
    log(f"  Receiver scores: {len(receiver_df)} sentences")

    # Merge
    df = importance_df.merge(receiver_df, on="sentence_idx", how="inner")
    log(f"  Merged: {len(df)} sentences")

    # Correlations
    log("\n[Correlations]")
    log("-" * 50)

    results = []

    # Main correlation: receiver score vs counterfactual importance
    results.append(correlate_and_plot(df, "counterfactual_importance_accuracy", "attn_vs_importance_accuracy"))
    results.append(correlate_and_plot(df, "counterfactual_importance_kl", "attn_vs_importance_kl"))
    results.append(correlate_and_plot(df, "forced_importance_accuracy", "attn_vs_forced_accuracy"))

    # Position analysis
    position_results = analyze_by_position(df)

    # By function tag
    log("\n[Correlation by Function Tag]")
    log("-" * 50)
    tag_correlations = {}
    for tag in df["function_tag"].unique():
        tag_df = df[df["function_tag"] == tag]
        if len(tag_df) >= 5:  # Need enough points
            x = tag_df["receiver_head_score"].values
            y = np.abs(tag_df["counterfactual_importance_accuracy"].values)
            if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
                r, p = stats.pearsonr(x, y)
                log(f"  {tag}: r={r:.3f} (p={p:.3f}, n={len(tag_df)})")
                tag_correlations[tag] = {"r": float(r), "p": float(p), "n": len(tag_df)}

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    main_r = results[0]["pearson_r"]
    main_p = results[0]["pearson_p"]

    if main_p < 0.05:
        if main_r > 0:
            log(f"\n✓ POSITIVE correlation found: r={main_r:.3f} (p={main_p:.4f})")
            log("  Sentences receiving more attention DO have higher importance!")
            log("  This supports the hypothesis that attention flow encodes causal structure.")
        else:
            log(f"\n✗ NEGATIVE correlation found: r={main_r:.3f} (p={main_p:.4f})")
            log("  Sentences receiving more attention have LOWER importance!")
    else:
        log(f"\n○ NO significant correlation: r={main_r:.3f} (p={main_p:.4f})")
        log("  Attention received does not predict counterfactual importance.")
        log("  This suggests importance is truly causal, not encoded in attention patterns.")

    # Save results
    save_data = {
        "correlations": results,
        "position_analysis": position_results,
        "by_function_tag": tag_correlations,
        "n_sentences": len(df),
        "problem": 1591
    }

    save_path = RESULTS_DIR / "attention_importance_correlation.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    log(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
