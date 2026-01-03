#!/usr/bin/env python3
"""
Train linear probes on activations to predict:
1. Function tags (8-way multi-label classification)
2. Correctness (binary: correct vs incorrect solution)
3. Importance (regression + top-10% classification)

Uses stratified splitting by problem to avoid data leakage.
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, mean_squared_error, r2_score,
                             mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("/workspace/activation-oracles-reasoning/experiments/2_linear_probes/results")
CACHE_PATH = RESULTS_DIR / "activation_cache_multilayer.pt"

# Function tags to predict
FUNCTION_TAGS = [
    "active_computation",
    "fact_retrieval",
    "uncertainty_management",
    "result_consolidation",
    "plan_generation",
    "self_checking",
    "problem_setup",
    "final_answer_emission"
]


def load_data(layer=-8):
    """Load activations and labels for specified layer."""
    print(f"Loading data from {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, weights_only=False)

    X = cache["activations"][layer].float().numpy()
    labels = cache["labels"]

    print(f"Loaded {len(X)} samples, {X.shape[1]} features")
    return X, labels


def prepare_function_tag_labels(labels):
    """Convert function tags to multi-label binary format."""
    all_tags = [l["function_tags"] for l in labels]

    # Use only known tags
    mlb = MultiLabelBinarizer(classes=FUNCTION_TAGS)
    y = mlb.fit_transform(all_tags)

    return y, mlb


def prepare_correctness_labels(labels):
    """Convert correctness to binary format."""
    y = np.array([1 if l["is_correct"] else 0 for l in labels])
    return y


def get_problem_stratified_split(labels, test_ratio=0.2, seed=42):
    """Split by problem to avoid data leakage."""
    np.random.seed(seed)

    # Group by problem
    problems = list(set(l["problem_id"] for l in labels))
    np.random.shuffle(problems)

    # Split problems
    n_test = max(1, int(len(problems) * test_ratio))
    test_problems = set(problems[:n_test])
    train_problems = set(problems[n_test:])

    train_idx = [i for i, l in enumerate(labels) if l["problem_id"] in train_problems]
    test_idx = [i for i, l in enumerate(labels) if l["problem_id"] in test_problems]

    print(f"Split: {len(train_problems)} train problems, {len(test_problems)} test problems")
    print(f"       {len(train_idx)} train samples, {len(test_idx)} test samples")

    return train_idx, test_idx


def train_function_tag_probe(X, labels, layer):
    """Train multi-label classifier for function tags."""
    print("\n" + "=" * 50)
    print("FUNCTION TAG PROBE")
    print("=" * 50)

    # Prepare labels
    y, mlb = prepare_function_tag_labels(labels)
    print(f"Label shape: {y.shape}")

    # Split by problem
    train_idx, test_idx = get_problem_stratified_split(labels)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multi-label classifier
    print("\nTraining OneVsRest Logistic Regression...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1
        )
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)

    # Metrics
    results = {
        "layer": layer,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "per_tag_metrics": {},
    }

    print("\n--- Per-Tag Results ---")
    for i, tag in enumerate(FUNCTION_TAGS):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        support = y_test[:, i].sum()

        results["per_tag_metrics"][tag] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "support": int(support),
        }
        print(f"{tag:30s}: acc={acc:.3f}, f1={f1:.3f}, support={support}")

    # Overall metrics
    overall_acc = accuracy_score(y_test.flatten(), y_pred.flatten())
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

    results["overall_accuracy"] = float(overall_acc)
    results["macro_f1"] = float(macro_f1)
    results["micro_f1"] = float(micro_f1)

    print(f"\n--- Overall ---")
    print(f"Accuracy: {overall_acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Micro F1: {micro_f1:.3f}")

    # Baseline comparisons
    # Random baseline: predict based on class frequency
    class_freqs = y_train.mean(axis=0)
    y_random = (np.random.rand(len(y_test), len(FUNCTION_TAGS)) < class_freqs).astype(int)
    random_macro_f1 = f1_score(y_test, y_random, average='macro', zero_division=0)

    # Majority baseline: always predict most frequent class
    y_majority = np.zeros_like(y_test)
    y_majority[:, np.argmax(class_freqs)] = 1
    majority_macro_f1 = f1_score(y_test, y_majority, average='macro', zero_division=0)

    results["random_baseline_f1"] = float(random_macro_f1)
    results["majority_baseline_f1"] = float(majority_macro_f1)

    print(f"\n--- Baselines ---")
    print(f"Random baseline F1: {random_macro_f1:.3f}")
    print(f"Majority baseline F1: {majority_macro_f1:.3f}")
    print(f"Improvement over random: {macro_f1 - random_macro_f1:.3f}")

    return results, clf, scaler


def train_correctness_probe(X, labels, layer):
    """Train binary classifier for correctness."""
    print("\n" + "=" * 50)
    print("CORRECTNESS PROBE")
    print("=" * 50)

    # Prepare labels
    y = prepare_correctness_labels(labels)
    print(f"Correct: {y.sum()}, Incorrect: {len(y) - y.sum()}")

    # Split by problem
    train_idx, test_idx = get_problem_stratified_split(labels)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {
        "layer": layer,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": float(acc),
        "f1": float(f1),
    }

    print(f"\n--- Results ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Incorr  Corr")
    print(f"Actual Incorr   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Corr     {cm[1,0]:4d}  {cm[1,1]:4d}")

    results["confusion_matrix"] = cm.tolist()

    # Baselines
    random_acc = 0.5
    majority_acc = max(y_train.mean(), 1 - y_train.mean())

    results["random_baseline"] = float(random_acc)
    results["majority_baseline"] = float(majority_acc)

    print(f"\n--- Baselines ---")
    print(f"Random: {random_acc:.3f}")
    print(f"Majority: {majority_acc:.3f}")
    print(f"Improvement over random: {acc - random_acc:.3f}")

    return results, clf, scaler


def train_importance_probes(X, labels, layer):
    """Train importance probes: regression + top-10% classification."""
    print("\n" + "=" * 50)
    print("IMPORTANCE PROBES")
    print("=" * 50)

    # Extract importance values
    y_cont = np.array([l["counterfactual_importance"] if l["counterfactual_importance"] is not None else 0.0
                       for l in labels])

    # Create top-10% binary labels
    threshold_10pct = np.percentile(y_cont, 90)
    y_binary = (y_cont >= threshold_10pct).astype(int)

    print(f"Continuous importance: mean={y_cont.mean():.4f}, std={y_cont.std():.4f}")
    print(f"Top-10% threshold: {threshold_10pct:.4f}")
    print(f"Binary labels: {y_binary.sum()} positive ({100*y_binary.mean():.1f}%)")

    # Split by problem
    train_idx, test_idx = get_problem_stratified_split(labels)
    X_train, X_test = X[train_idx], X[test_idx]
    y_cont_train, y_cont_test = y_cont[train_idx], y_cont[test_idx]
    y_binary_train, y_binary_test = y_binary[train_idx], y_binary[test_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {
        "layer": layer,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "threshold_10pct": float(threshold_10pct),
    }

    # --- Regression ---
    print("\n--- Regression (predicting continuous importance) ---")
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_scaled, y_cont_train)
    y_pred_cont = reg.predict(X_test_scaled)

    mse = mean_squared_error(y_cont_test, y_pred_cont)
    mae = mean_absolute_error(y_cont_test, y_pred_cont)
    r2 = r2_score(y_cont_test, y_pred_cont)

    # Baseline: predict mean
    baseline_pred = np.full_like(y_cont_test, y_cont_train.mean())
    baseline_mse = mean_squared_error(y_cont_test, baseline_pred)
    baseline_r2 = r2_score(y_cont_test, baseline_pred)

    results["regression"] = {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "baseline_mse": float(baseline_mse),
        "baseline_r2": float(baseline_r2),
    }

    print(f"MSE: {mse:.6f} (baseline: {baseline_mse:.6f})")
    print(f"MAE: {mae:.6f}")
    print(f"R²:  {r2:.4f} (baseline: {baseline_r2:.4f})")

    # Check correlation between predicted and actual
    correlation = np.corrcoef(y_cont_test, y_pred_cont)[0, 1]
    results["regression"]["correlation"] = float(correlation)
    print(f"Correlation: {correlation:.4f}")

    # --- Top-10% Classification ---
    print("\n--- Top-10% Classification (anchor step detection) ---")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',  # Handle imbalance
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_binary_train)
    y_pred_binary = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_binary_test, y_pred_binary)
    f1 = f1_score(y_binary_test, y_pred_binary)

    # Confusion matrix
    cm = confusion_matrix(y_binary_test, y_pred_binary)

    # Baselines
    random_acc = 0.5
    majority_acc = max(y_binary_train.mean(), 1 - y_binary_train.mean())

    results["classification"] = {
        "accuracy": float(acc),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "random_baseline": float(random_acc),
        "majority_baseline": float(majority_acc),
        "n_positive_test": int(y_binary_test.sum()),
        "n_negative_test": int(len(y_binary_test) - y_binary_test.sum()),
    }

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              NotAnch  Anchor")
    print(f"Actual NotAnch  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Anchor   {cm[1,0]:4d}    {cm[1,1]:4d}")
    print(f"\nBaselines: random={random_acc:.3f}, majority={majority_acc:.3f}")

    return results, {"regression": reg, "classification": clf}, scaler


def main():
    print("=" * 60)
    print("LINEAR PROBE TRAINING")
    print("=" * 60)

    # Use primary layer
    primary_layer = -8

    # Load data
    X, labels = load_data(layer=primary_layer)

    # Train function tag probe
    func_results, func_clf, func_scaler = train_function_tag_probe(X, labels, primary_layer)

    # Train correctness probe
    corr_results, corr_clf, corr_scaler = train_correctness_probe(X, labels, primary_layer)

    # Train importance probes
    imp_results, imp_models, imp_scaler = train_importance_probes(X, labels, primary_layer)

    # Save results
    all_results = {
        "function_tags": func_results,
        "correctness": corr_results,
        "importance": imp_results,
    }

    results_path = RESULTS_DIR / "probe_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save models
    import pickle
    models_path = RESULTS_DIR / "probe_models.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump({
            "function_tags": {"clf": func_clf, "scaler": func_scaler},
            "correctness": {"clf": corr_clf, "scaler": corr_scaler},
            "importance": {"models": imp_models, "scaler": imp_scaler},
        }, f)
    print(f"Models saved to: {models_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nFunction Tag Probe (layer {primary_layer}):")
    print(f"  Macro F1: {func_results['macro_f1']:.3f} (random: {func_results['random_baseline_f1']:.3f})")

    print(f"\nCorrectness Probe (layer {primary_layer}):")
    print(f"  Accuracy: {corr_results['accuracy']:.3f} (random: {corr_results['random_baseline']:.3f})")

    print(f"\nImportance Probes (layer {primary_layer}):")
    print(f"  Regression R²: {imp_results['regression']['r2']:.3f} (baseline: {imp_results['regression']['baseline_r2']:.3f})")
    print(f"  Anchor Classification F1: {imp_results['classification']['f1']:.3f}")


if __name__ == "__main__":
    main()
