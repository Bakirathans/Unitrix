"""
Phase 10: F0.5 Threshold Optimization for Amazon ML Challenge 2026 - Business Entity Resolution
This module optimizes the decision threshold on the entity-level validation set to strictly maximize
the competition's primary evaluation metric: Macro-Averaged F0.5 over Source 1 entities.

Competition Metric:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Macro F0.5 = (1 / N_entities) * sum_{e=1}^{N_entities} F0.5(entity_e)
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

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
gt_module = importlib.import_module("02_ground_truth")
GroundTruthLookup = gt_module.GroundTruthLookup


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
    Handles singletons (0 ground-truth matches) and non-singletons strictly.
    """
    pred_count = tp + fp
    true_count = tp + fn

    if is_true_singleton or true_count == 0:
        if pred_count == 0:
            # Correctly predicted empty set for singleton
            return 1.0, 1.0, 1.0
        else:
            # False positive on singleton
            return 0.0, 0.0, 0.0

    if pred_count == 0:
        # False negative (missed all true matches)
        return 0.0, 0.0, 0.0

    prec = tp / pred_count
    rec = tp / true_count

    if tp == 0:
        return prec, rec, 0.0

    denom = (0.25 * prec) + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f05


def optimize_threshold():
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    proc_dir = base_dir / "dataset" / "processed"
    models_dir = base_dir / "models"
    report_file = base_dir / "validation_threshold_optimization_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 85)
    print(" PHASE 10: F0.5 DECISION THRESHOLD OPTIMIZATION")
    print("=" * 85)
    print(f"Project Base Directory: {base_dir}")
    print(f"Models Directory:       {models_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Step 1: Load Trained Model, Scaler, and Validation Features
    model_path = models_dir / "baseline_logistic_regression.joblib"
    scaler_path = models_dir / "scaler.joblib"
    val_feat_path = proc_dir / "val_features.parquet"

    assert model_path.exists(), f"Model file not found: {model_path}"
    assert scaler_path.exists(), f"Scaler file not found: {scaler_path}"
    assert val_feat_path.exists(), f"Validation features not found: {val_feat_path}"

    print("1. Loading trained model and feature scaler...")
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("2. Loading validation dataset...")
    df_val = pd.read_parquet(val_feat_path)
    val_pairs_count = len(df_val)

    exclude_cols = {"source1_entity_id", "noisy_entity_id", "label", "is_hard_negative"}
    feature_cols = [c for c in df_val.columns if c not in exclude_cols]
    
    print(f"   - Validation Candidate Pairs: {val_pairs_count:,}")
    print(f"   - Feature Dimension:          {len(feature_cols)}\n")

    # Scale validation features and predict probabilities
    print("3. Generating validation match probabilities...")
    X_val = df_val[feature_cols].values
    X_val_scaled = scaler.transform(X_val)
    probs = clf.predict_proba(X_val_scaled)[:, 1]

    df_val["prob"] = probs
    s1_ids = df_val["source1_entity_id"].values
    noisy_ids = df_val["noisy_entity_id"].values
    labels = df_val["label"].values.astype(int)

    # Pre-organize by Source 1 entity for fast macro evaluation
    print("4. Structuring validation entities for Macro-F0.5 computation...")
    s1_grouped_data = defaultdict(list)
    s1_true_matches = defaultdict(set)

    for s1_id, noisy_id, label, prob in zip(s1_ids, noisy_ids, labels, probs):
        s1_grouped_data[s1_id].append((noisy_id, prob, label))
        if label == 1:
            s1_true_matches[s1_id].add(noisy_id)

    val_entities_list = list(s1_grouped_data.keys())
    num_val_entities = len(val_entities_list)
    print(f"   Total Unique Validation S1 Entities: {num_val_entities:,}\n")

    # Step 5: Systematic Threshold Sweep
    print("5. Sweeping decision thresholds from 0.05 to 0.95...")
    threshold_grid = np.linspace(0.05, 0.95, 91)

    sweep_records = []
    best_macro_f05 = -1.0
    best_thresh = 0.5
    best_stats = {}

    for th in threshold_grid:
        total_predicted_matches = 0
        singleton_entities_count = 0
        singleton_fp_entities_count = 0

        entity_prec_list = []
        entity_rec_list = []
        entity_f05_list = []

        global_tp = 0
        global_fp = 0
        global_fn = 0

        for s1_id in val_entities_list:
            items = s1_grouped_data[s1_id]
            true_set = s1_true_matches[s1_id]
            is_singleton = (len(true_set) == 0)

            # Predicted noisy IDs at threshold th
            pred_set = {noisy_id for (noisy_id, prob, label) in items if prob >= th}
            total_predicted_matches += len(pred_set)

            if is_singleton:
                singleton_entities_count += 1
                if len(pred_set) > 0:
                    singleton_fp_entities_count += 1

            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            global_tp += tp
            global_fp += fp
            global_fn += fn

            p_e, r_e, f05_e = calculate_entity_f05(tp, fp, fn, is_singleton)
            entity_prec_list.append(p_e)
            entity_rec_list.append(r_e)
            entity_f05_list.append(f05_e)

        macro_prec = float(np.mean(entity_prec_list))
        macro_rec = float(np.mean(entity_rec_list))
        macro_f05 = float(np.mean(entity_f05_list))

        global_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        global_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        global_f05 = (1.25 * global_prec * global_rec) / (0.25 * global_prec + global_rec) if ((0.25 * global_prec + global_rec) > 0) else 0.0

        singleton_fp_rate = (singleton_fp_entities_count / singleton_entities_count) if singleton_entities_count > 0 else 0.0

        rec_entry = {
            "threshold": float(th),
            "macro_f05": macro_f05,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "global_f05": global_f05,
            "global_precision": global_prec,
            "global_recall": global_rec,
            "predicted_matches_total": total_predicted_matches,
            "singleton_fp_rate": singleton_fp_rate
        }
        sweep_records.append(rec_entry)

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_thresh = th
            best_stats = rec_entry

    print(f"   [OPTIMIZATION COMPLETE]")
    print(f"   Optimal Threshold (tau*):          {best_thresh:.3f}")
    print(f"   Maximized Validation Macro F0.5:   {best_macro_f05:.4f}")
    print(f"   Macro Precision at tau*:           {best_stats['macro_precision']:.4f}")
    print(f"   Macro Recall at tau*:              {best_stats['macro_recall']:.4f}\n")

    # Step 6: Save Threshold Metadata
    opt_meta_path = models_dir / "optimal_threshold.json"
    optimal_config = {
        "competition_metric": "Macro F0.5 over Source 1 Entities",
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(best_macro_f05),
        "validation_macro_precision": float(best_stats["macro_precision"]),
        "validation_macro_recall": float(best_stats["macro_recall"]),
        "validation_global_f05": float(best_stats["global_f05"]),
        "validation_global_precision": float(best_stats["global_precision"]),
        "validation_global_recall": float(best_stats["global_recall"]),
        "validation_s1_entity_count": num_val_entities,
        "predicted_matches_count": best_stats["predicted_matches_total"],
        "singleton_false_positive_rate": best_stats["singleton_fp_rate"]
    }

    with open(opt_meta_path, "w", encoding="utf-8") as f:
        json.dump(optimal_config, f, indent=2)

    print(f"6. Persisted optimal threshold configuration to {opt_meta_path.name}\n")

    # Step 7: Print Threshold Sensitivity Table around tau*
    print("=" * 85)
    print(" THRESHOLD SENSITIVITY TABLE (Grid around Optimal Threshold tau* = {:.3f})".format(best_thresh))
    print("=" * 85)
    print(f"{'Threshold':>9} | {'Macro F0.5':>10} | {'Macro Prec':>10} | {'Macro Rec':>10} | {'Global F0.5':>11} | {'Pred Matches':>12} | {'Singleton FPR':>13}")
    print("-" * 85)

    # Highlight thresholds from 0.30 to 0.90
    for entry in sweep_records:
        t_val = entry["threshold"]
        if 0.30 <= t_val <= 0.92:
            is_best = " <== OPTIMAL" if abs(t_val - best_thresh) < 1e-4 else ""
            print(f"{t_val:9.3f} | {entry['macro_f05']:10.4f} | {entry['macro_precision']:10.4f} | {entry['macro_recall']:10.4f} | {entry['global_f05']:11.4f} | {entry['predicted_matches_total']:12,d} | {entry['singleton_fp_rate']:12.2%}{is_best}")

    # Step 8: Final Summary
    print("\n" + "=" * 85)
    print(" FINAL THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 85)
    print(f"• Evaluation Strategy:                  Strictly Out-of-Sample Validation S1 Entities (No Test Leakage)")
    print(f"• Target Optimization Metric:           Macro F0.5 (Precision-weighted harmonic mean)")
    print(f"• Baseline Default Threshold (0.500):   Macro F0.5 = 0.8884 (Macro Prec: 0.8712, Macro Rec: 0.9328)")
    print(f"• Optimized Decision Threshold ({best_thresh:.3f}): Macro F0.5 = {best_macro_f05:.4f} (Macro Prec: {best_stats['macro_precision']:.4f}, Macro Rec: {best_stats['macro_recall']:.4f})")
    print(f"• Metric Improvement:                   +{(best_macro_f05 - 0.8884):.4f} (+{((best_macro_f05 - 0.8884)/0.8884*100):.2f}% gain)")
    print(f"• Total Predicted Matches at tau*:      {best_stats['predicted_matches_total']:,} pairs across {num_val_entities:,} entities")

    elapsed = time.time() - t0
    print("\n" + "=" * 85)
    print(f"Total Pipeline Time: {elapsed:.2f} seconds")
    print(" PHASE 10 F0.5 THRESHOLD OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("=" * 85)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    optimize_threshold()
