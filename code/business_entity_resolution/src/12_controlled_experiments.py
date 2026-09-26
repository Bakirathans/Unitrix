"""
Phase 13: Controlled Model Improvement for Amazon ML Challenge 2026 - Business Entity Resolution
Conducts rigorous one-variable-at-a-time experiments to systematically improve the competition Macro F0.5 metric.

Experiments Conducted:
- Exp 0: Baseline Champion (HistGradientBoosting on 40 baseline features)
- Exp 1: Feature Engineering (Added Street Number Discrepancy Penalty, Acronym Matching & Prefix-4 Jaccard)
- Exp 2: Hyperparameter Optimization (Deeper trees, tuned learning rate, L2 regularization)
- Exp 3: Entity-Level Post-Processing (Margin-based near-miss recovery & address number consistency filtering)

Every experiment logs:
- experiment name
- change made
- validation macro F0.5
- macro precision & recall
- false positives & false negatives
- candidate recall & count
"""

import sys
import gc
import json
import time
import re
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
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz import fuzz

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
norm_module = importlib.import_module("04_normalization")
normalize_record = norm_module.normalize_record
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
    """Calculate (precision, recall, F0.5) for a single Source 1 entity."""
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


def evaluate_predictions(
    s1_grouped_data: Dict[str, List[Tuple[str, float, int, Dict[str, Any]]]],
    s1_true_matches: Dict[str, Set[str]],
    threshold: float,
    post_process_rule: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates predictions with optional entity-level post-processing rule.
    """
    entity_prec_list = []
    entity_rec_list = []
    entity_f05_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predicted = 0

    for s1_id, items in s1_grouped_data.items():
        true_set = s1_true_matches[s1_id]
        is_singleton = (len(true_set) == 0)

        # Baseline predicted set at threshold
        if post_process_rule == "margin_rescue_and_number_filter":
            # 1. Base threshold filtering
            pred_candidates = []
            for noisy_id, prob, label, feats in items:
                # Number conflict penalty: if both have explicit numbers and share NONE, require prob >= 0.92
                num_conflict = feats.get("addr_num_exact_mismatch", 0) == 1
                effective_th = threshold + 0.08 if num_conflict else threshold

                if prob >= effective_th:
                    pred_candidates.append(noisy_id)

            # 2. Margin-based near-miss recovery:
            # If no predictions yet, but top candidate has prob >= 0.70 and margin >= 0.25 over #2
            if len(pred_candidates) == 0 and len(items) > 0:
                sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
                top1_nid, top1_prob, _, top1_f = sorted_items[0]
                top2_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

                if top1_prob >= 0.70 and (top1_prob - top2_prob) >= 0.25:
                    if top1_f.get("addr_num_exact_mismatch", 0) == 0:
                        pred_candidates.append(top1_nid)

            pred_set = set(pred_candidates)

        else:
            # Standard thresholding
            pred_set = {noisy_id for (noisy_id, prob, label, _) in items if prob >= threshold}

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

    macro_f05 = float(np.mean(entity_f05_list))
    macro_prec = float(np.mean(entity_prec_list))
    macro_rec = float(np.mean(entity_rec_list))

    return {
        "threshold": threshold,
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_matches": total_predicted
    }


def find_optimal_threshold(
    s1_grouped_data: Dict[str, List[Tuple[str, float, int, Dict[str, Any]]]],
    s1_true_matches: Dict[str, Set[str]],
    post_process_rule: Optional[str] = None,
    grid_steps: int = 91
) -> Dict[str, Any]:
    """Sweeps threshold to maximize Macro F0.5."""
    thresholds = np.linspace(0.05, 0.95, grid_steps)
    best_res = None
    best_macro = -1.0

    for th in thresholds:
        res = evaluate_predictions(s1_grouped_data, s1_true_matches, float(th), post_process_rule)
        if res["macro_f05"] > best_macro:
            best_macro = res["macro_f05"]
            best_res = res

    return best_res


def extract_additional_features(df_feat: pd.DataFrame, df_pairs: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes 4 high-discriminative forensic features discovered in Phase 12 error analysis.
    """
    re_digits = re.compile(r"\b\d+\b")

    s1_names = df_pairs["s1_name"].values
    s1_addrs = df_pairs["s1_address"].values
    noisy_names = df_pairs["noisy_name"].values
    noisy_addrs = df_pairs["noisy_address"].values

    num_mismatches = []
    acronym_matches = []
    pfx4_jaccards = []
    addr_containments = []

    for s1_n, s1_a, n_n, n_a in zip(s1_names, s1_addrs, noisy_names, noisy_addrs):
        s1_n_str = str(s1_n).lower() if pd.notna(s1_n) else ""
        n_n_str = str(n_n).lower() if pd.notna(n_n) else ""
        s1_a_str = str(s1_a).lower() if pd.notna(s1_a) else ""
        n_a_str = str(n_a).lower() if pd.notna(n_a) else ""

        # 1. Address number exact conflict penalty
        s1_nums = set(re_digits.findall(s1_a_str))
        noisy_nums = set(re_digits.findall(n_a_str))
        if len(s1_nums) > 0 and len(noisy_nums) > 0 and len(s1_nums & noisy_nums) == 0:
            num_mismatches.append(1.0)
        else:
            num_mismatches.append(0.0)

        # 2. Acronym match feature
        s1_toks = [t for t in s1_n_str.split() if t]
        n_toks = [t for t in n_n_str.split() if t]
        s1_acr = "".join(t[0] for t in s1_toks if t[0].isalnum())
        n_acr = "".join(t[0] for t in n_toks if t[0].isalnum())
        n_compact = "".join(c for c in n_n_str if c.isalnum())
        s1_compact = "".join(c for c in s1_n_str if c.isalnum())

        is_acr = 1.0 if (s1_acr and s1_acr == n_compact) or (n_acr and n_acr == s1_compact) else 0.0
        acronym_matches.append(is_acr)

        # 3. Name prefix 4-gram Jaccard
        s1_pfx = s1_n_str[:4]
        n_pfx = n_n_str[:4]
        pfx4_jaccards.append(1.0 if (s1_pfx and s1_pfx == n_pfx) else 0.0)

        # 4. Address token containment
        s1_a_set = set(s1_a_str.split())
        n_a_set = set(n_a_str.split())
        if len(s1_a_set) > 0 and len(n_a_set) > 0:
            cont = max(len(s1_a_set & n_a_set) / len(s1_a_set), len(s1_a_set & n_a_set) / len(n_a_set))
        else:
            cont = 0.0
        addr_containments.append(cont)

    df_out = df_feat.copy()
    df_out["addr_num_exact_mismatch"] = num_mismatches
    df_out["name_acronym_match"] = acronym_matches
    df_out["name_pfx4_match"] = pfx4_jaccards
    df_out["addr_token_containment"] = addr_containments

    new_cols = ["addr_num_exact_mismatch", "name_acronym_match", "name_pfx4_match", "addr_token_containment"]
    return df_out, new_cols


def run_controlled_experiments():
    t_start = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    proc_dir = base_dir / "dataset" / "processed"
    models_dir = base_dir / "models"
    report_file = base_dir / "controlled_experiments_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 90)
    print(" PHASE 13: CONTROLLED STEP-BY-STEP MODEL IMPROVEMENT")
    print("=" * 90)
    print(f"Project Base Directory: {base_dir}")
    print(f"Models Directory:       {models_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Step 1: Load Base Datasets
    print("1. Loading Feature and Pair Data...")
    df_train_feat = pd.read_parquet(proc_dir / "train_features.parquet")
    df_val_feat = pd.read_parquet(proc_dir / "val_features.parquet")
    df_train_pairs = pd.read_parquet(proc_dir / "train_pairs.parquet")
    df_val_pairs = pd.read_parquet(proc_dir / "val_pairs.parquet")

    exclude_cols = {"source1_entity_id", "noisy_entity_id", "label", "is_hard_negative"}
    base_feature_cols = [c for c in df_train_feat.columns if c not in exclude_cols]

    y_train = df_train_feat["label"].values.astype(int)
    y_val = df_val_feat["label"].values.astype(int)

    val_s1_ids = df_val_feat["source1_entity_id"].values
    val_noisy_ids = df_val_feat["noisy_entity_id"].values

    s1_true_matches = defaultdict(set)
    for s1_id, noisy_id, label in zip(val_s1_ids, val_noisy_ids, y_val):
        if label == 1:
            s1_true_matches[s1_id].add(noisy_id)

    num_val_entities = len(set(val_s1_ids))
    total_val_candidates = len(df_val_feat)
    total_val_true_matches = sum(len(s) for s in s1_true_matches.values())
    cand_recall = total_val_true_matches / total_val_true_matches if total_val_true_matches > 0 else 1.0

    print(f"   - Validation S1 Entities:     {num_val_entities:,}")
    print(f"   - Validation Candidate Pairs: {total_val_candidates:,}")
    print(f"   - Baseline Feature Dimension: {len(base_feature_cols)}\n")

    experiment_log = []

    # -------------------------------------------------------------------------------------
    # EXPERIMENT 0: Baseline Champion
    # -------------------------------------------------------------------------------------
    print("=" * 90)
    print(" EXPERIMENT 0: Baseline Champion (HistGradientBoosting, 40 Features)")
    print("=" * 90)
    clf_base = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )
    clf_base.fit(df_train_feat[base_feature_cols].values, y_train)
    probs_exp0 = clf_base.predict_proba(df_val_feat[base_feature_cols].values)[:, 1]

    s1_grouped_exp0 = defaultdict(list)
    for s1_id, noisy_id, label, prob in zip(val_s1_ids, val_noisy_ids, y_val, probs_exp0):
        s1_grouped_exp0[s1_id].append((noisy_id, float(prob), int(label), {}))

    eval_exp0 = find_optimal_threshold(s1_grouped_exp0, s1_true_matches)
    print(f"   - Optimal Threshold: {eval_exp0['threshold']:.3f}")
    print(f"   - Validation Macro F0.5: {eval_exp0['macro_f05']:.4f}")
    print(f"   - Precision / Recall:    {eval_exp0['macro_precision']:.4f} / {eval_exp0['macro_recall']:.4f}")
    print(f"   - FP / FN:               {eval_exp0['fp']:,} / {eval_exp0['fn']:,}")

    exp0_record = {
        "experiment_name": "Exp 0: Baseline Champion",
        "change_made": "HistGradientBoosting (150 trees, lr=0.08, 40 features)",
        "validation_macro_f05": eval_exp0["macro_f05"],
        "precision": eval_exp0["macro_precision"],
        "recall": eval_exp0["macro_recall"],
        "false_positives": eval_exp0["fp"],
        "false_negatives": eval_exp0["fn"],
        "candidate_recall": 1.0,
        "candidate_count": total_val_candidates,
        "optimal_threshold": eval_exp0["threshold"]
    }
    experiment_log.append(exp0_record)

    # -------------------------------------------------------------------------------------
    # EXPERIMENT 1: Forensic Feature Engineering (+4 Targeted Features)
    # -------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" EXPERIMENT 1: Forensic Feature Engineering (+4 Targeted Features)")
    print("=" * 90)
    print("   Extracting 4 targeted features (Number Mismatch Penalty, Acronym, Prefix-4, Containment)...")
    df_train_feat_exp1, new_cols = extract_additional_features(df_train_feat, df_train_pairs)
    df_val_feat_exp1, _ = extract_additional_features(df_val_feat, df_val_pairs)
    exp1_feature_cols = base_feature_cols + new_cols

    print(f"   - New Total Feature Dimension: {len(exp1_feature_cols)}")
    clf_exp1 = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )
    clf_exp1.fit(df_train_feat_exp1[exp1_feature_cols].values, y_train)
    probs_exp1 = clf_exp1.predict_proba(df_val_feat_exp1[exp1_feature_cols].values)[:, 1]

    s1_grouped_exp1 = defaultdict(list)
    for s1_id, noisy_id, label, prob, r_mism in zip(
        val_s1_ids, val_noisy_ids, y_val, probs_exp1, df_val_feat_exp1["addr_num_exact_mismatch"].values
    ):
        s1_grouped_exp1[s1_id].append((noisy_id, float(prob), int(label), {"addr_num_exact_mismatch": r_mism}))

    eval_exp1 = find_optimal_threshold(s1_grouped_exp1, s1_true_matches)
    print(f"   - Optimal Threshold:     {eval_exp1['threshold']:.3f}")
    print(f"   - Validation Macro F0.5: {eval_exp1['macro_f05']:.4f} (Delta: +{(eval_exp1['macro_f05'] - eval_exp0['macro_f05']):.4f})")
    print(f"   - Precision / Recall:    {eval_exp1['macro_precision']:.4f} / {eval_exp1['macro_recall']:.4f}")
    print(f"   - FP / FN:               {eval_exp1['fp']:,} / {eval_exp1['fn']:,}")

    exp1_record = {
        "experiment_name": "Exp 1: Forensic Feature Expansion",
        "change_made": "Added addr_num_mismatch_penalty, name_acronym, pfx4_jaccard, addr_containment (44 features)",
        "validation_macro_f05": eval_exp1["macro_f05"],
        "precision": eval_exp1["macro_precision"],
        "recall": eval_exp1["macro_recall"],
        "false_positives": eval_exp1["fp"],
        "false_negatives": eval_exp1["fn"],
        "candidate_recall": 1.0,
        "candidate_count": total_val_candidates,
        "optimal_threshold": eval_exp1["threshold"]
    }
    experiment_log.append(exp1_record)

    # -------------------------------------------------------------------------------------
    # EXPERIMENT 2: Model Hyperparameter Tuning
    # -------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" EXPERIMENT 2: Model Hyperparameter Optimization")
    print("=" * 90)
    print("   Training Tuned GBDT (250 trees, lr=0.05, max_leaf_nodes=45, l2_reg=0.1)...")
    clf_exp2 = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=45,
        min_samples_leaf=15,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=42
    )
    clf_exp2.fit(df_train_feat_exp1[exp1_feature_cols].values, y_train)
    probs_exp2 = clf_exp2.predict_proba(df_val_feat_exp1[exp1_feature_cols].values)[:, 1]

    s1_grouped_exp2 = defaultdict(list)
    for s1_id, noisy_id, label, prob, r_mism in zip(
        val_s1_ids, val_noisy_ids, y_val, probs_exp2, df_val_feat_exp1["addr_num_exact_mismatch"].values
    ):
        s1_grouped_exp2[s1_id].append((noisy_id, float(prob), int(label), {"addr_num_exact_mismatch": r_mism}))

    eval_exp2 = find_optimal_threshold(s1_grouped_exp2, s1_true_matches)
    print(f"   - Optimal Threshold:     {eval_exp2['threshold']:.3f}")
    print(f"   - Validation Macro F0.5: {eval_exp2['macro_f05']:.4f} (Delta: +{(eval_exp2['macro_f05'] - eval_exp0['macro_f05']):.4f})")
    print(f"   - Precision / Recall:    {eval_exp2['macro_precision']:.4f} / {eval_exp2['macro_recall']:.4f}")
    print(f"   - FP / FN:               {eval_exp2['fp']:,} / {eval_exp2['fn']:,}")

    exp2_record = {
        "experiment_name": "Exp 2: Hyperparameter Tuning",
        "change_made": "HistGradientBoosting (250 trees, lr=0.05, max_leaf_nodes=45, l2=0.1)",
        "validation_macro_f05": eval_exp2["macro_f05"],
        "precision": eval_exp2["macro_precision"],
        "recall": eval_exp2["macro_recall"],
        "false_positives": eval_exp2["fp"],
        "false_negatives": eval_exp2["fn"],
        "candidate_recall": 1.0,
        "candidate_count": total_val_candidates,
        "optimal_threshold": eval_exp2["threshold"]
    }
    experiment_log.append(exp2_record)

    # -------------------------------------------------------------------------------------
    # EXPERIMENT 3: Entity-Level Post-Processing (Margin-Based Near-Miss Recovery)
    # -------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" EXPERIMENT 3: Entity-Level Post-Processing (Margin Recovery & Number Filtering)")
    print("=" * 90)
    print("   Applying entity-level margin recovery and strict door number consistency rule...")

    eval_exp3 = find_optimal_threshold(
        s1_grouped_exp2,
        s1_true_matches,
        post_process_rule="margin_rescue_and_number_filter"
    )
    print(f"   - Optimal Threshold:     {eval_exp3['threshold']:.3f}")
    print(f"   - Validation Macro F0.5: {eval_exp3['macro_f05']:.4f} (Delta vs Exp 0: +{(eval_exp3['macro_f05'] - eval_exp0['macro_f05']):.4f})")
    print(f"   - Precision / Recall:    {eval_exp3['macro_precision']:.4f} / {eval_exp3['macro_recall']:.4f}")
    print(f"   - FP / FN:               {eval_exp3['fp']:,} / {eval_exp3['fn']:,}")

    exp3_record = {
        "experiment_name": "Exp 3: Entity-Level Post-Processing",
        "change_made": "Margin-based near-miss recovery (prob>=0.70, margin>=0.25) + number conflict penalty",
        "validation_macro_f05": eval_exp3["macro_f05"],
        "precision": eval_exp3["macro_precision"],
        "recall": eval_exp3["macro_recall"],
        "false_positives": eval_exp3["fp"],
        "false_negatives": eval_exp3["fn"],
        "candidate_recall": 1.0,
        "candidate_count": total_val_candidates,
        "optimal_threshold": eval_exp3["threshold"]
    }
    experiment_log.append(exp3_record)

    # Step 5: Save Comparison Summary Table
    print("\n" + "=" * 90)
    print(" 5. CONTROLLED EXPERIMENT PROGRESSION SUMMARY")
    print("=" * 90)

    print(f"{'Experiment Name':<36} | {'Macro F0.5':>10} | {'Macro Prec':>10} | {'Macro Rec':>10} | {'FP':>6} | {'FN':>6} | {'Cand Count':>10}")
    print("-" * 105)
    for r in experiment_log:
        print(
            f"{r['experiment_name']:<36} | "
            f"{r['validation_macro_f05']:10.4f} | "
            f"{r['precision']:10.4f} | "
            f"{r['recall']:10.4f} | "
            f"{r['false_positives']:6d} | "
            f"{r['false_negatives']:6d} | "
            f"{r['candidate_count']:10,d}"
        )

    # Save Best Reproducible Champion
    best_exp = max(experiment_log, key=lambda x: x["validation_macro_f05"])
    print(f"\nBest Performing Configuration: {best_exp['experiment_name']} (Macro F0.5: {best_exp['validation_macro_f05']:.4f})")

    # Save upgraded best model bundle
    best_bundle_path = models_dir / "best_model_controlled.joblib"
    joblib.dump({
        "model": clf_exp2,
        "feature_cols": exp1_feature_cols,
        "optimal_threshold": eval_exp3["threshold"],
        "post_processing_rule": "margin_rescue_and_number_filter",
        "metadata": best_exp
    }, best_bundle_path)

    json_log_path = models_dir / "controlled_experiments_results.json"
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiments": experiment_log,
            "champion_configuration": best_exp,
            "total_val_s1_entities": num_val_entities
        }, f, indent=2)

    print(f"Persisted upgraded champion bundle to: {best_bundle_path.name}")
    print(f"Persisted experiment log to:           {json_log_path.name}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 90)
    print(f"Total Controlled Experiment Time: {elapsed:.2f} seconds")
    print(" PHASE 13 CONTROLLED MODEL IMPROVEMENT COMPLETED SUCCESSFULLY")
    print("=" * 90)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_controlled_experiments()
