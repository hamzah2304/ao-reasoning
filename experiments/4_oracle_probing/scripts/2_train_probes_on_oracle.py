#!/usr/bin/env python3
"""
Train probes on oracle hidden states and compare to direct DeepSeek probes.

This tests Oscar's hypothesis: even if the oracle can't verbalize correctly,
its internal hidden states might be more informative for downstream probing.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# Paths
ORACLE_STATES_PATH = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results/oracle_hidden_states.pt")
DEEPSEEK_CACHE_PATH = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results/activation_cache_multilayer.pt")
DEEPSEEK_RESULTS_PATH = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results/probe_results.json")
RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/4_oracle_probing/results")

FUNCTION_TAGS = [
    "active_computation", "fact_retrieval", "uncertainty_management",
    "result_consolidation", "plan_generation", "self_checking",
    "problem_setup", "final_answer_emission"
]


def load_data():
    """Load oracle hidden states and original DeepSeek activations."""
    log("Loading data...")

    # Load oracle hidden states
    oracle_data = torch.load(ORACLE_STATES_PATH, weights_only=False)
    oracle_states = oracle_data["oracle_hidden_states"]
    labels = oracle_data["labels"]
    config = oracle_data["config"]

    log(f"  Oracle hidden states: {len(labels)} samples")

    # Load original DeepSeek activations for comparison
    deepseek_data = torch.load(DEEPSEEK_CACHE_PATH, weights_only=False)
    deepseek_acts = deepseek_data["activations"]

    print(f"  DeepSeek activations: {len(deepseek_data['labels'])} samples")

    # Load previous probe results for comparison
    with open(DEEPSEEK_RESULTS_PATH) as f:
        deepseek_results = json.load(f)

    return oracle_states, deepseek_acts, labels, config, deepseek_results


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


def train_probe(X, y, train_idx, test_idx, multilabel=False):
    """Train a probe and return metrics."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if multilabel:
        clf = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        return {"macro_f1": macro_f1}
    else:
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc}


def compare_probes(oracle_states, deepseek_acts, labels, layers):
    """Compare probe performance on oracle hidden states vs DeepSeek activations."""
    results = {
        "function_tags": {"oracle": {}, "deepseek": {}},
        "correctness": {"oracle": {}, "deepseek": {}},
    }

    train_idx, test_idx = get_problem_split(labels)
    print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")

    # Prepare labels
    y_func, _ = prepare_labels(labels, "function_tags")
    y_corr, _ = prepare_labels(labels, "correctness")

    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        # Get activations
        X_oracle = oracle_states[layer].float().numpy()
        X_deepseek = deepseek_acts[layer].float().numpy()

        # Function tags
        print("  Function tags:")
        oracle_func = train_probe(X_oracle, y_func, train_idx, test_idx, multilabel=True)
        deepseek_func = train_probe(X_deepseek, y_func, train_idx, test_idx, multilabel=True)

        results["function_tags"]["oracle"][layer] = oracle_func
        results["function_tags"]["deepseek"][layer] = deepseek_func

        diff = oracle_func["macro_f1"] - deepseek_func["macro_f1"]
        print(f"    Oracle F1:   {oracle_func['macro_f1']:.3f}")
        print(f"    DeepSeek F1: {deepseek_func['macro_f1']:.3f}")
        print(f"    Difference:  {diff:+.3f} {'(Oracle better)' if diff > 0 else '(DeepSeek better)'}")

        # Correctness
        print("  Correctness:")
        oracle_corr = train_probe(X_oracle, y_corr, train_idx, test_idx, multilabel=False)
        deepseek_corr = train_probe(X_deepseek, y_corr, train_idx, test_idx, multilabel=False)

        results["correctness"]["oracle"][layer] = oracle_corr
        results["correctness"]["deepseek"][layer] = deepseek_corr

        diff = oracle_corr["accuracy"] - deepseek_corr["accuracy"]
        print(f"    Oracle Acc:   {oracle_corr['accuracy']:.3f}")
        print(f"    DeepSeek Acc: {deepseek_corr['accuracy']:.3f}")
        print(f"    Difference:   {diff:+.3f} {'(Oracle better)' if diff > 0 else '(DeepSeek better)'}")

    return results


def main():
    print("=" * 60)
    print("PROBE COMPARISON: Oracle Hidden States vs DeepSeek Activations")
    print("=" * 60)
    print("\nHypothesis: Oracle's internal hidden states might be more")
    print("informative than DeepSeek's raw activations, even if the")
    print("oracle can't generate correct text output.")
    print("=" * 60)

    # Load data
    oracle_states, deepseek_acts, labels, config, deepseek_baseline = load_data()

    layers = config["layers_extracted"]

    # Compare probes
    results = compare_probes(oracle_states, deepseek_acts, labels, layers)

    # Save results
    save_data = {
        "comparison": results,
        "config": config,
        "deepseek_baseline": deepseek_baseline,
    }

    save_path = RESULTS_DIR / "probe_comparison.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nFunction Tags (Macro F1):")
    print(f"{'Layer':<10} {'Oracle':<10} {'DeepSeek':<10} {'Diff':<10}")
    print("-" * 40)
    for layer in layers:
        o = results["function_tags"]["oracle"][layer]["macro_f1"]
        d = results["function_tags"]["deepseek"][layer]["macro_f1"]
        print(f"{layer:<10} {o:<10.3f} {d:<10.3f} {o-d:+.3f}")

    print("\nCorrectness (Accuracy):")
    print(f"{'Layer':<10} {'Oracle':<10} {'DeepSeek':<10} {'Diff':<10}")
    print("-" * 40)
    for layer in layers:
        o = results["correctness"]["oracle"][layer]["accuracy"]
        d = results["correctness"]["deepseek"][layer]["accuracy"]
        print(f"{layer:<10} {o:<10.3f} {d:<10.3f} {o-d:+.3f}")

    # Overall verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Check best layer for each
    best_oracle_func = max(layers, key=lambda l: results["function_tags"]["oracle"][l]["macro_f1"])
    best_deepseek_func = max(layers, key=lambda l: results["function_tags"]["deepseek"][l]["macro_f1"])

    oracle_best_f1 = results["function_tags"]["oracle"][best_oracle_func]["macro_f1"]
    deepseek_best_f1 = results["function_tags"]["deepseek"][best_deepseek_func]["macro_f1"]

    if oracle_best_f1 > deepseek_best_f1:
        print(f"\nOracle hidden states ARE more informative!")
        print(f"Best Oracle F1: {oracle_best_f1:.3f} (layer {best_oracle_func})")
        print(f"Best DeepSeek F1: {deepseek_best_f1:.3f} (layer {best_deepseek_func})")
        print(f"Improvement: {oracle_best_f1 - deepseek_best_f1:+.3f}")
    else:
        print(f"\nOracle hidden states are NOT more informative.")
        print(f"Best Oracle F1: {oracle_best_f1:.3f} (layer {best_oracle_func})")
        print(f"Best DeepSeek F1: {deepseek_best_f1:.3f} (layer {best_deepseek_func})")
        print(f"Difference: {oracle_best_f1 - deepseek_best_f1:+.3f}")
        print("\nThe oracle didn't learn useful cross-model representations.")


if __name__ == "__main__":
    main()
