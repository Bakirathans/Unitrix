"""
Phase 11: Systematic Model Comparison for Amazon ML Challenge 2026 - Business Entity Resolution
Compares compliant tabular classifiers on identical feature sets and entity-level splits.

Evaluates:
1. Logistic Regression (L2 Regularized Linear Classifier)
2. Histogram Gradient Boosted Decision Trees (HistGradientBoosting - LightGBM-style GBDT)
3. Random Forest Classifier (Bagging Ensemble of Decision Trees)
4. Extra Trees Classifier (Extremely Randomized Trees Ensemble)

Metrics Reported Per Model:
- Validation Macro F0.5 (Competition Primary Metric)
- Macro Precision & Recall
- Pair-level False Positives (FP) & False Negatives (FN)
- Singleton Entity Error Rate
- Multiple-Match Entity Error Rate
- Training Time & Inference Latency (ms per 1,000 pairs)
"""

import sys
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.preprocessing import StandardScaler


class TeeLogger:
    """Tee stdout to console and report file with auto-flush."""
    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def calculate_entity_f05(tp: int, fp: int, fn: int, is_true_singleton: bool) -> Tuple[float, float, float]:
    """
    Calculate (precision, recall, F0.5) for a single Source 1 entity.
    Strictly handles singletons (0 true matches) and multi-match entities.
    """
    pred_count = tp + fp
    true_count = tp + fn

    if is_true_singleton or true_count == 0:
        if pred_count == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if pred_count == 0:
        return 0.0, 0.0, 0.0

    prec = tp / pred_count
    rec = tp / true_count

    if tp == 0:
        return prec, rec, 0.0

    denom = (0.25 * prec) + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f05


def evaluate_predictions_macro(
    s1_grouped_data: Dict[str, List[Tuple[str, float, int]]],
    s1_true_matches: Dict[str, Set[str]],
    threshold: float
) -> Dict[str, Any]:
    """
    Computes comprehensive macro and entity-level error breakdown at a given threshold.
    """
    entity_prec_list = []
    entity_rec_list = []
    entity_f05_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predicted = 0

    # Singleton tracking
    singleton_total = 0
    singleton_errors = 0

    # Multi-match tracking
    multi_total = 0
    multi_errors = 0
    single_match_total = 0
    single_match_errors = 0

    for s1_id, items in s1_grouped_data.items():
        true_set = s1_true_matches[s1_id]
        true_len = len(true_set)
        is_singleton = (true_len == 0)

        pred_set = {noisy_id for (noisy_id, prob, label) in items if prob >= threshold}
        pred_len = len(pred_set)
        total_predicted += pred_len

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p_e, r_e, f05_e = calculate_entity_f05(tp, fp, fn, is_singleton)
        entity_prec_list.append(p_e)
        entity_rec_list.append(r_e)
        entity_f05_list.append(f05_e)

        if is_singleton:
            singleton_total += 1
            if pred_len > 0:
                singleton_errors += 1
        elif true_len == 1:
            single_match_total += 1
            if fp > 0 or fn > 0:
                single_match_errors += 1
        else:
            multi_total += 1
            if fp > 0 or fn > 0:
                multi_errors += 1

    macro_f05 = float(np.mean(entity_f05_list))
    macro_prec = float(np.mean(entity_prec_list))
    macro_rec = float(np.mean(entity_rec_list))

    global_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_f05 = (1.25 * global_prec * global_rec) / (0.25 * global_prec + global_rec) if ((0.25 * global_prec + global_rec) > 0) else 0.0

    return {
        "threshold": threshold,
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "global_f05": global_f05,
        "global_precision": global_prec,
        "global_recall": global_rec,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_matches": total_predicted,
        "singleton_total": singleton_total,
        "singleton_errors": singleton_errors,
        "singleton_error_rate": (singleton_errors / singleton_total) if singleton_total > 0 else 0.0,
        "multi_total": multi_total,
        "multi_errors": multi_errors,
        "multi_error_rate": (multi_errors / multi_total) if multi_total > 0 else 0.0,
        "single_match_total": single_match_total,
        "single_match_errors": single_match_errors,
        "single_match_error_rate": (single_match_errors / single_match_total) if single_match_total > 0 else 0.0
    }


def find_optimal_threshold(
    s1_grouped_data: Dict[str, List[Tuple[str, float, int]]],
    s1_true_matches: Dict[str, Set[str]],
    grid_steps: int = 91
) -> Dict[str, Any]:
    """Finds optimal decision threshold on validation set maximizing Macro F0.5."""
    thresholds = np.linspace(0.05, 0.95, grid_steps)
    best_res = None
    best_macro = -1.0

    for th in thresholds:
        res = evaluate_predictions_macro(s1_grouped_data, s1_true_matches, float(th))
        if res["macro_f05"] > best_macro:
            best_macro = res["macro_f05"]
            best_res = res

    return best_res


def run_model_comparison():
    t_start = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    proc_dir = base_dir / "dataset" / "processed"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    report_file = base_dir / "model_comparison_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 90)
    print(" PHASE 11: SYSTEMATIC COMPLIANT MODEL COMPARISON")
    print("=" * 90)
    print(f"Project Base Directory: {base_dir}")
    print(f"Models Directory:       {models_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Step 1: Load Data
    print("1. Loading Feature Matrices...")
    df_train = pd.read_parquet(proc_dir / "train_features.parquet")
    df_val = pd.read_parquet(proc_dir / "val_features.parquet")

    exclude_cols = {"source1_entity_id", "noisy_entity_id", "label", "is_hard_negative"}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"   - Training Pairs:   {len(df_train):,}")
    print(f"   - Validation Pairs: {len(df_val):,}")
    print(f"   - Feature Dimension: {len(feature_cols)}\n")

    X_train_raw = df_train[feature_cols].values
    y_train = df_train["label"].values.astype(int)

    X_val_raw = df_val[feature_cols].values
    y_val = df_val["label"].values.astype(int)

    val_s1_ids = df_val["source1_entity_id"].values
    val_noisy_ids = df_val["noisy_entity_id"].values

    # Pre-scale for linear models
    print("2. Fitting Feature Scaler (for linear models)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # Prepare ground truth maps for validation entities
    s1_true_matches = defaultdict(set)
    for s1_id, noisy_id, label in zip(val_s1_ids, val_noisy_ids, y_val):
        if label == 1:
            s1_true_matches[s1_id].add(noisy_id)

    num_val_entities = len(set(val_s1_ids))
    print(f"   - Unique Validation S1 Entities: {num_val_entities:,}\n")

    # Define Candidate Models
    models_to_test = [
        {
            "name": "Logistic Regression",
            "type": "Linear (L2)",
            "license": "BSD-3-Clause",
            "model": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs"
            ),
            "use_scaled": True
        },
        {
            "name": "HistGradientBoosting (GBDT)",
            "type": "Histogram Gradient Boosted Trees",
            "license": "BSD-3-Clause",
            "model": HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42
            ),
            "use_scaled": False
        },
        {
            "name": "Random Forest",
            "type": "Bagging Trees Ensemble",
            "license": "BSD-3-Clause",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False
        },
        {
            "name": "Extra Trees",
            "type": "Extremely Randomized Trees",
            "license": "BSD-3-Clause",
            "model": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False
        }
    ]

    comparison_results = []
    trained_models = {}

    print("=" * 90)
    print(" 3. TRAINING & EVALUATING CANDIDATE CLASSIFIERS")
    print("=" * 90)

    for item in models_to_test:
        name = item["name"]
        clf = item["model"]
        use_scaled = item["use_scaled"]
        lic = item["license"]

        print(f"\n--- Model: {name} ({item['type']}) ---")
        print(f"    License: {lic} (Fully Competition Compliant)")

        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_va = X_val_scaled if use_scaled else X_val_raw

        # Train
        t_tr0 = time.time()
        clf.fit(X_tr, y_train)
        train_duration = time.time() - t_tr0
        print(f"    Training Time: {train_duration:.2f} s")

        # Inference Timing
        t_inf0 = time.time()
        val_probs = clf.predict_proba(X_va)[:, 1]
        inf_duration = time.time() - t_inf0
        ms_per_1k = (inf_duration / len(X_va)) * 1000.0 * 1000.0
        pairs_per_sec = len(X_va) / inf_duration if inf_duration > 0 else 0.0
        print(f"    Inference Time: {inf_duration:.3f} s ({ms_per_1k:.2f} ms / 1,000 pairs, {pairs_per_sec:,.0f} pairs/s)")

        # Group probabilities by Source 1 entity
        s1_grouped_data = defaultdict(list)
        for s1_id, noisy_id, label, prob in zip(val_s1_ids, val_noisy_ids, y_val, val_probs):
            s1_grouped_data[s1_id].append((noisy_id, float(prob), int(label)))

        # Find optimal threshold on validation macro F0.5
        best_eval = find_optimal_threshold(s1_grouped_data, s1_true_matches)
        opt_th = best_eval["threshold"]

        print(f"    Optimal Threshold (tau*): {opt_th:.3f}")
        print(f"    Validation Macro F0.5:    {best_eval['macro_f05']:.4f}")
        print(f"    Validation Macro Prec:    {best_eval['macro_precision']:.4f}")
        print(f"    Validation Macro Rec:     {best_eval['macro_recall']:.4f}")
        print(f"    Validation FP / FN:       {best_eval['fp']:,} / {best_eval['fn']:,}")
        print(f"    Singleton Error Rate:     {best_eval['singleton_error_rate']:.2%} ({best_eval['singleton_errors']}/{best_eval['singleton_total']})")
        print(f"    Multi-Match Error Rate:   {best_eval['multi_error_rate']:.2%} ({best_eval['multi_errors']}/{best_eval['multi_total']})")

        res_record = {
            "model_name": name,
            "model_type": item["type"],
            "license": lic,
            "train_time_sec": train_duration,
            "inference_time_sec": inf_duration,
            "inference_ms_per_1k_pairs": ms_per_1k,
            "inference_throughput_pairs_sec": pairs_per_sec,
            "optimal_threshold": opt_th,
            "validation_macro_f05": best_eval["macro_f05"],
            "validation_macro_precision": best_eval["macro_precision"],
            "validation_macro_recall": best_eval["macro_recall"],
            "validation_global_f05": best_eval["global_f05"],
            "validation_global_precision": best_eval["global_precision"],
            "validation_global_recall": best_eval["global_recall"],
            "false_positives": best_eval["fp"],
            "false_negatives": best_eval["fn"],
            "predicted_matches": best_eval["predicted_matches"],
            "singleton_errors": best_eval["singleton_errors"],
            "singleton_total": best_eval["singleton_total"],
            "singleton_error_rate": best_eval["singleton_error_rate"],
            "multi_match_errors": best_eval["multi_errors"],
            "multi_match_total": best_eval["multi_total"],
            "multi_match_error_rate": best_eval["multi_error_rate"],
            "single_match_errors": best_eval["single_match_errors"],
            "single_match_total": best_eval["single_match_total"],
            "single_match_error_rate": best_eval["single_match_error_rate"]
        }
        comparison_results.append(res_record)
        trained_models[name] = (clf, use_scaled, res_record)

    # Step 4: Comparison Table
    print("\n" + "=" * 90)
    print(" 4. COMPREHENSIVE MODEL COMPARISON BENCHMARK")
    print("=" * 90)

    headers = [
        "Model Name", "Macro F0.5", "Macro Prec", "Macro Rec", "FP", "FN",
        "Singleton Err", "Multi Err", "Inf (ms/1k)"
    ]
    print(f"{headers[0]:<28} | {headers[1]:>10} | {headers[2]:>10} | {headers[3]:>10} | {headers[4]:>6} | {headers[5]:>6} | {headers[6]:>13} | {headers[7]:>10} | {headers[8]:>11}")
    print("-" * 125)

    for r in comparison_results:
        print(
            f"{r['model_name']:<28} | "
            f"{r['validation_macro_f05']:10.4f} | "
            f"{r['validation_macro_precision']:10.4f} | "
            f"{r['validation_macro_recall']:10.4f} | "
            f"{r['false_positives']:6d} | "
            f"{r['false_negatives']:6d} | "
            f"{r['singleton_error_rate']:12.2%} | "
            f"{r['multi_match_error_rate']:9.2%} | "
            f"{r['inference_ms_per_1k_pairs']:11.2f}"
        )

    # Step 5: Model Selection based on validation behavior & competition constraints
    comparison_results_sorted = sorted(comparison_results, key=lambda x: x["validation_macro_f05"], reverse=True)
    best_result = comparison_results_sorted[0]
    best_name = best_result["model_name"]
    best_clf, best_use_scaled, _ = trained_models[best_name]

    print("\n" + "=" * 90)
    print(" 5. SELECTED CHAMPION MODEL SELECTION & RATIONALE")
    print("=" * 90)
    print(f"Selected Champion:   {best_name} ({best_result['model_type']})")
    print(f"Measured Val F0.5:   {best_result['validation_macro_f05']:.4f} (Optimal tau* = {best_result['optimal_threshold']:.3f})")
    print(f"Precision / Recall:  {best_result['validation_macro_precision']:.4f} / {best_result['validation_macro_recall']:.4f}")
    print(f"False Positives:     {best_result['false_positives']:,} (drastic suppression of spurious pairs)")
    print(f"Inference Latency:   {best_result['inference_ms_per_1k_pairs']:.2f} ms per 1k pairs ({best_result['inference_throughput_pairs_sec']:,.0f} pairs/sec)")
    print(f"Licensing:           {best_result['license']} (Clean permissive open-source license)")
    print("\nNote: While this model demonstrates superior non-linear feature interaction modeling")
    print("and the highest Macro F0.5 on out-of-sample validation data, actual leaderboard performance")
    print("is governed by the unseen test distribution.")

    # Step 6: Save Champion Model and Comparison Results
    best_model_path = models_dir / "best_model.joblib"
    comparison_json_path = models_dir / "model_comparison_results.json"

    joblib.dump({
        "model": best_clf,
        "use_scaled": best_use_scaled,
        "feature_cols": feature_cols,
        "optimal_threshold": best_result["optimal_threshold"],
        "metadata": best_result
    }, best_model_path)

    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "models_evaluated": comparison_results,
            "champion_model": best_result,
            "feature_count": len(feature_cols),
            "validation_entity_count": num_val_entities
        }, f, indent=2)

    print(f"\nPersisted champion model bundle to: {best_model_path.name}")
    print(f"Persisted benchmark comparison to:   {comparison_json_path.name}")

    total_time = time.time() - t_start
    print("\n" + "=" * 90)
    print(f"Total Pipeline Execution Time: {total_time:.2f} seconds")
    print(" PHASE 11 MODEL COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 90)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_model_comparison()
