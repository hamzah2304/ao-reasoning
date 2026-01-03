#!/usr/bin/env python3
"""
Analyze how probe performance varies across layers.
Runs function tag and correctness probes at multiple layers.
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

RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results")
CACHE_PATH = RESULTS_DIR / "activation_cache_multilayer.pt"

FUNCTION_TAGS = [
    "active_computation", "fact_retrieval", "uncertainty_management",
    "result_consolidation", "plan_generation", "self_checking",
    "problem_setup", "final_answer_emission"
]


def load_cache():
    """Load full cache with all layers."""
    print(f"Loading cache from {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, weights_only=False)
    layers = list(cache["activations"].keys())
    print(f"Available layers: {layers}")
    return cache


def get_problem_stratified_split(labels, test_ratio=0.2, seed=42):
    """Split by problem to avoid data leakage."""
    np.random.seed(seed)
    problems = list(set(l["problem_id"] for l in labels))
    np.random.shuffle(problems)
    n_test = max(1, int(len(problems) * test_ratio))
    test_problems = set(problems[:n_test])
    train_idx = [i for i, l in enumerate(labels) if l["problem_id"] not in test_problems]
    test_idx = [i for i, l in enumerate(labels) if l["problem_id"] in test_problems]
    return train_idx, test_idx


def run_function_tag_probe(X_train, X_test, y_train, y_test):
    """Train and evaluate function tag probe."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    return macro_f1, micro_f1


def run_correctness_probe(X_train, X_test, y_train, y_test):
    """Train and evaluate correctness probe."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, f1


def run_importance_probe(X_train, X_test, y_cont_train, y_cont_test, y_bin_train, y_bin_test):
    """Train and evaluate importance probes (regression + classification)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Regression
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_s, y_cont_train)
    y_pred = reg.predict(X_test_s)
    r2 = r2_score(y_cont_test, y_pred)

    # Classification
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced', n_jobs=-1)
    clf.fit(X_train_s, y_bin_train)
    y_pred_bin = clf.predict(X_test_s)
    f1 = f1_score(y_bin_test, y_pred_bin)

    return r2, f1


def main():
    print("=" * 60)
    print("LAYER ANALYSIS")
    print("=" * 60)

    # Load cache
    cache = load_cache()
    layers = sorted(cache["activations"].keys())
    labels = cache["labels"]

    # Prepare labels
    mlb = MultiLabelBinarizer(classes=FUNCTION_TAGS)
    y_func = mlb.fit_transform([l["function_tags"] for l in labels])
    y_corr = np.array([1 if l["is_correct"] else 0 for l in labels])

    # Importance labels
    y_imp_cont = np.array([l["counterfactual_importance"] if l["counterfactual_importance"] is not None else 0.0
                           for l in labels])
    threshold_10pct = np.percentile(y_imp_cont, 90)
    y_imp_bin = (y_imp_cont >= threshold_10pct).astype(int)

    # Get split
    train_idx, test_idx = get_problem_stratified_split(labels)
    y_func_train, y_func_test = y_func[train_idx], y_func[test_idx]
    y_corr_train, y_corr_test = y_corr[train_idx], y_corr[test_idx]
    y_imp_cont_train, y_imp_cont_test = y_imp_cont[train_idx], y_imp_cont[test_idx]
    y_imp_bin_train, y_imp_bin_test = y_imp_bin[train_idx], y_imp_bin[test_idx]

    print(f"\nTrain: {len(train_idx)} samples")
    print(f"Test:  {len(test_idx)} samples")

    # Run probes at each layer
    results = {"layers": [], "function_tag_macro_f1": [], "function_tag_micro_f1": [],
               "correctness_accuracy": [], "correctness_f1": [],
               "importance_r2": [], "importance_anchor_f1": []}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        X = cache["activations"][layer].float().numpy()
        X_train, X_test = X[train_idx], X[test_idx]

        # Function tags
        func_macro, func_micro = run_function_tag_probe(X_train, X_test, y_func_train, y_func_test)
        print(f"Function tags: macro_f1={func_macro:.3f}, micro_f1={func_micro:.3f}")

        # Correctness
        corr_acc, corr_f1 = run_correctness_probe(X_train, X_test, y_corr_train, y_corr_test)
        print(f"Correctness:   acc={corr_acc:.3f}, f1={corr_f1:.3f}")

        # Importance
        imp_r2, imp_f1 = run_importance_probe(X_train, X_test,
                                               y_imp_cont_train, y_imp_cont_test,
                                               y_imp_bin_train, y_imp_bin_test)
        print(f"Importance:    r2={imp_r2:.3f}, anchor_f1={imp_f1:.3f}")

        results["layers"].append(layer)
        results["function_tag_macro_f1"].append(float(func_macro))
        results["function_tag_micro_f1"].append(float(func_micro))
        results["correctness_accuracy"].append(float(corr_acc))
        results["correctness_f1"].append(float(corr_f1))
        results["importance_r2"].append(float(imp_r2))
        results["importance_anchor_f1"].append(float(imp_f1))

    # Save results
    results_path = RESULTS_DIR / "layer_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("LAYER ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n{'Layer':>8} | {'Func F1':>8} | {'Corr Acc':>9} | {'Imp R²':>8} | {'Anchor F1':>10}")
    print("-" * 55)
    for i, layer in enumerate(results["layers"]):
        print(f"{layer:>8} | {results['function_tag_macro_f1'][i]:>8.3f} | {results['correctness_accuracy'][i]:>9.3f} | {results['importance_r2'][i]:>8.3f} | {results['importance_anchor_f1'][i]:>10.3f}")

    # Find best layers
    best_func_idx = np.argmax(results["function_tag_macro_f1"])
    best_corr_idx = np.argmax(results["correctness_accuracy"])
    best_imp_idx = np.argmax(results["importance_r2"])
    print(f"\nBest layer for function tags: {results['layers'][best_func_idx]} (F1={results['function_tag_macro_f1'][best_func_idx]:.3f})")
    print(f"Best layer for correctness: {results['layers'][best_corr_idx]} (Acc={results['correctness_accuracy'][best_corr_idx]:.3f})")
    print(f"Best layer for importance: {results['layers'][best_imp_idx]} (R²={results['importance_r2'][best_imp_idx]:.3f})")


if __name__ == "__main__":
    main()
