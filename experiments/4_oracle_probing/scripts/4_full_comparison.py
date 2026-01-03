#!/usr/bin/env python3
"""
Full 3-way comparison: DeepSeek vs Paper's Oracle vs Our Custom Oracle.

Also adds importance prediction (even though we expect it to fail).
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')


def log(msg):
    print(msg, flush=True)


# Paths
PAPER_ORACLE_PATH = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results/oracle_hidden_states.pt")
CUSTOM_ORACLE_PATH = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results/custom_oracle_hidden_states.pt")
DEEPSEEK_CACHE_PATH = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results/activation_cache_multilayer.pt")
RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results")

FUNCTION_TAGS = [
    "active_computation", "fact_retrieval", "uncertainty_management",
    "result_consolidation", "plan_generation", "self_checking",
    "problem_setup", "final_answer_emission"
]


def load_all_data():
    """Load all three sources of hidden states."""
    log("Loading all data sources...")

    # DeepSeek raw activations
    deepseek_data = torch.load(DEEPSEEK_CACHE_PATH, weights_only=False)
    deepseek_acts = deepseek_data["activations"]
    labels = deepseek_data["labels"]
    log(f"  DeepSeek: {len(labels)} samples")

    # Paper's oracle hidden states
    paper_data = torch.load(PAPER_ORACLE_PATH, weights_only=False)
    paper_states = paper_data["oracle_hidden_states"]
    log(f"  Paper oracle: {len(paper_data['labels'])} samples")

    # Our custom oracle hidden states
    if CUSTOM_ORACLE_PATH.exists():
        custom_data = torch.load(CUSTOM_ORACLE_PATH, weights_only=False)
        custom_states = custom_data["oracle_hidden_states"]
        log(f"  Custom oracle: {len(custom_data['labels'])} samples")
    else:
        custom_states = None
        log(f"  Custom oracle: NOT YET EXTRACTED")

    return deepseek_acts, paper_states, custom_states, labels


def prepare_labels(labels, label_type="function_tags"):
    """Prepare labels for probing."""
    if label_type == "function_tags":
        all_tags = [l["function_tags"] for l in labels]
        mlb = MultiLabelBinarizer(classes=FUNCTION_TAGS)
        y = mlb.fit_transform(all_tags)
        return y, mlb
    elif label_type == "correctness":
        y = np.array([1 if l["is_correct"] else 0 for l in labels])
        return y, None
    elif label_type == "importance":
        # Get counterfactual importance scores
        y = np.array([l.get("counterfactual_importance", 0.0) for l in labels])
        return y, None


def get_problem_split(labels, test_ratio=0.2, seed=42):
    """Split by problem to avoid leakage."""
    np.random.seed(seed)
    problems = list(set(l["problem_id"] for l in labels))
    np.random.shuffle(problems)
    n_test = max(1, int(len(problems) * test_ratio))
    test_problems = set(problems[:n_test])

    train_idx = [i for i, l in enumerate(labels) if l["problem_id"] not in test_problems]
    test_idx = [i for i, l in enumerate(labels) if l["problem_id"] in test_problems]

    return train_idx, test_idx


def train_probe(X, y, train_idx, test_idx, task_type="classification"):
    """Train a probe and return metrics."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if task_type == "multilabel":
        clf = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        return {"macro_f1": macro_f1}
    elif task_type == "binary":
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc}
    elif task_type == "regression":
        reg = Ridge(alpha=1.0)
        reg.fit(X_train_scaled, y_train)
        y_pred = reg.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        return {"r2": r2}


def run_comparison(deepseek_acts, paper_states, custom_states, labels, layers):
    """Run full 3-way comparison across all tasks."""
    results = {
        "function_tags": {"deepseek": {}, "paper_oracle": {}, "custom_oracle": {}},
        "correctness": {"deepseek": {}, "paper_oracle": {}, "custom_oracle": {}},
        "importance": {"deepseek": {}, "paper_oracle": {}, "custom_oracle": {}},
    }

    train_idx, test_idx = get_problem_split(labels)
    log(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")

    # Prepare all labels
    y_func, _ = prepare_labels(labels, "function_tags")
    y_corr, _ = prepare_labels(labels, "correctness")
    y_imp, _ = prepare_labels(labels, "importance")

    has_custom = custom_states is not None

    for layer in layers:
        log(f"\n{'='*60}")
        log(f"LAYER {layer}")
        log(f"{'='*60}")

        # Get activations for this layer
        X_deepseek = deepseek_acts[layer].float().numpy()
        X_paper = paper_states[layer].float().numpy()
        X_custom = custom_states[layer].float().numpy() if has_custom else None

        # ---- Function Tags ----
        log("\n[Function Tags - Macro F1]")

        ds_func = train_probe(X_deepseek, y_func, train_idx, test_idx, "multilabel")
        results["function_tags"]["deepseek"][layer] = ds_func
        log(f"  DeepSeek:      {ds_func['macro_f1']:.3f}")

        paper_func = train_probe(X_paper, y_func, train_idx, test_idx, "multilabel")
        results["function_tags"]["paper_oracle"][layer] = paper_func
        diff = paper_func['macro_f1'] - ds_func['macro_f1']
        log(f"  Paper Oracle:  {paper_func['macro_f1']:.3f} ({diff:+.3f})")

        if has_custom:
            custom_func = train_probe(X_custom, y_func, train_idx, test_idx, "multilabel")
            results["function_tags"]["custom_oracle"][layer] = custom_func
            diff = custom_func['macro_f1'] - ds_func['macro_f1']
            log(f"  Custom Oracle: {custom_func['macro_f1']:.3f} ({diff:+.3f})")

        # ---- Correctness ----
        log("\n[Correctness - Accuracy]")

        ds_corr = train_probe(X_deepseek, y_corr, train_idx, test_idx, "binary")
        results["correctness"]["deepseek"][layer] = ds_corr
        log(f"  DeepSeek:      {ds_corr['accuracy']:.3f}")

        paper_corr = train_probe(X_paper, y_corr, train_idx, test_idx, "binary")
        results["correctness"]["paper_oracle"][layer] = paper_corr
        diff = paper_corr['accuracy'] - ds_corr['accuracy']
        log(f"  Paper Oracle:  {paper_corr['accuracy']:.3f} ({diff:+.3f})")

        if has_custom:
            custom_corr = train_probe(X_custom, y_corr, train_idx, test_idx, "binary")
            results["correctness"]["custom_oracle"][layer] = custom_corr
            diff = custom_corr['accuracy'] - ds_corr['accuracy']
            log(f"  Custom Oracle: {custom_corr['accuracy']:.3f} ({diff:+.3f})")

        # ---- Importance ----
        log("\n[Importance - R² (regression)]")

        ds_imp = train_probe(X_deepseek, y_imp, train_idx, test_idx, "regression")
        results["importance"]["deepseek"][layer] = ds_imp
        log(f"  DeepSeek:      {ds_imp['r2']:.3f}")

        paper_imp = train_probe(X_paper, y_imp, train_idx, test_idx, "regression")
        results["importance"]["paper_oracle"][layer] = paper_imp
        diff = paper_imp['r2'] - ds_imp['r2']
        log(f"  Paper Oracle:  {paper_imp['r2']:.3f} ({diff:+.3f})")

        if has_custom:
            custom_imp = train_probe(X_custom, y_imp, train_idx, test_idx, "regression")
            results["importance"]["custom_oracle"][layer] = custom_imp
            diff = custom_imp['r2'] - ds_imp['r2']
            log(f"  Custom Oracle: {custom_imp['r2']:.3f} ({diff:+.3f})")

    return results


def print_summary(results, layers, has_custom):
    """Print summary tables."""
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    # Function Tags
    log("\n[Function Tags - Macro F1]")
    header = f"{'Layer':<8} {'DeepSeek':<12} {'Paper Oracle':<14} {'Diff':<8}"
    if has_custom:
        header += f" {'Custom Oracle':<14} {'Diff':<8}"
    log(header)
    log("-" * len(header))

    for layer in layers:
        ds = results["function_tags"]["deepseek"][layer]["macro_f1"]
        paper = results["function_tags"]["paper_oracle"][layer]["macro_f1"]
        row = f"{layer:<8} {ds:<12.3f} {paper:<14.3f} {paper-ds:+.3f}"
        if has_custom and layer in results["function_tags"]["custom_oracle"]:
            custom = results["function_tags"]["custom_oracle"][layer]["macro_f1"]
            row += f"    {custom:<14.3f} {custom-ds:+.3f}"
        log(row)

    # Correctness
    log("\n[Correctness - Accuracy]")
    header = f"{'Layer':<8} {'DeepSeek':<12} {'Paper Oracle':<14} {'Diff':<8}"
    if has_custom:
        header += f" {'Custom Oracle':<14} {'Diff':<8}"
    log(header)
    log("-" * len(header))

    for layer in layers:
        ds = results["correctness"]["deepseek"][layer]["accuracy"]
        paper = results["correctness"]["paper_oracle"][layer]["accuracy"]
        row = f"{layer:<8} {ds:<12.3f} {paper:<14.3f} {paper-ds:+.3f}"
        if has_custom and layer in results["correctness"]["custom_oracle"]:
            custom = results["correctness"]["custom_oracle"][layer]["accuracy"]
            row += f"    {custom:<14.3f} {custom-ds:+.3f}"
        log(row)

    # Importance
    log("\n[Importance - R²]")
    header = f"{'Layer':<8} {'DeepSeek':<12} {'Paper Oracle':<14} {'Diff':<8}"
    if has_custom:
        header += f" {'Custom Oracle':<14} {'Diff':<8}"
    log(header)
    log("-" * len(header))

    for layer in layers:
        ds = results["importance"]["deepseek"][layer]["r2"]
        paper = results["importance"]["paper_oracle"][layer]["r2"]
        row = f"{layer:<8} {ds:<12.3f} {paper:<14.3f} {paper-ds:+.3f}"
        if has_custom and layer in results["importance"]["custom_oracle"]:
            custom = results["importance"]["custom_oracle"][layer]["r2"]
            row += f"    {custom:<14.3f} {custom-ds:+.3f}"
        log(row)


def main():
    log("=" * 70)
    log("FULL 3-WAY COMPARISON: DeepSeek vs Paper Oracle vs Custom Oracle")
    log("=" * 70)

    # Load all data
    deepseek_acts, paper_states, custom_states, labels = load_all_data()

    layers = [-4, -8, -12, -16]
    has_custom = custom_states is not None

    # Run comparison
    results = run_comparison(deepseek_acts, paper_states, custom_states, labels, layers)

    # Print summary
    print_summary(results, layers, has_custom)

    # Save results
    save_path = RESULTS_DIR / "full_comparison.json"

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(save_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    log(f"\nResults saved to {save_path}")

    # Verdict
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)

    # Check if oracle hidden states help
    best_ds_corr = max(results["correctness"]["deepseek"][l]["accuracy"] for l in layers)
    best_paper_corr = max(results["correctness"]["paper_oracle"][l]["accuracy"] for l in layers)

    if best_paper_corr > best_ds_corr:
        log(f"\nPaper Oracle hidden states ARE more informative for correctness!")
        log(f"  Best DeepSeek: {best_ds_corr:.3f}")
        log(f"  Best Paper Oracle: {best_paper_corr:.3f}")
        log(f"  Improvement: {best_paper_corr - best_ds_corr:+.3f}")
    else:
        log(f"\nPaper Oracle hidden states are NOT more informative.")

    if has_custom:
        best_custom_corr = max(results["correctness"]["custom_oracle"][l]["accuracy"] for l in layers)
        if best_custom_corr > best_ds_corr:
            log(f"\nCustom Oracle hidden states ARE more informative for correctness!")
            log(f"  Best DeepSeek: {best_ds_corr:.3f}")
            log(f"  Best Custom Oracle: {best_custom_corr:.3f}")
            log(f"  Improvement: {best_custom_corr - best_ds_corr:+.3f}")

    # Check importance (expect failure)
    best_ds_imp = max(results["importance"]["deepseek"][l]["r2"] for l in layers)
    log(f"\nImportance prediction (expected to fail):")
    log(f"  Best DeepSeek R²: {best_ds_imp:.3f}")
    if best_ds_imp < 0:
        log("  Confirmed: Importance is NOT predictable from activations (causal property)")


if __name__ == "__main__":
    main()
