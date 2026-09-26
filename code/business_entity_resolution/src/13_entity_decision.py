"""
Phase 14: Entity-Level Match Decision Engine for Amazon ML Challenge 2026 - Business Entity Resolution
Transforms pair-level classification scores into canonical Source 1 matched entity ID sets.

Decision Logic Principles:
1. Systematic Candidate Scoring using Champion Model.
2. Optimal Decision Threshold Application (tau* = 0.870).
3. Flexible Multiplicity:
   - 0 matches (Empty set for Singletons)
   - Exactly 1 match
   - Multiple matches (2+ matches)
4. No Blind Fallbacks: NEVER force-select the top candidate if its probability < tau*.
5. Deduplication: Ensures all returned candidate IDs for an S1 are strictly unique.
6. ID Integrity: Preserves verbatim 'S2-...' and 'S3-...' identifiers.
7. Validation Verification: Rigorously verifies against entity-level ground truth.
"""

import sys
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

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


class EntityMatchDecisionEngine:
    """
    Production-ready entity-level decision engine.
    Applies calibrated probability thresholding, deduplication, and ID preservation.
    """
    def __init__(self, model_bundle_path: Path, threshold: Optional[float] = None):
        self.bundle_path = model_bundle_path
        bundle = joblib.load(model_bundle_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.threshold = threshold if threshold is not None else bundle.get("optimal_threshold", 0.870)
        self.metadata = bundle.get("metadata", {})

    def score_pairs(self, df_features: pd.DataFrame) -> np.ndarray:
        """Score candidate pairs and return match probabilities."""
        X = df_features[self.feature_cols].values
        return self.model.predict_proba(X)[:, 1]

    def decide_matches_for_candidates(
        self,
        s1_ids: np.ndarray,
        noisy_ids: np.ndarray,
        probabilities: np.ndarray,
        all_s1_entity_ids: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Groups scored candidates by Source 1 entity and applies the decision rules.
        Guarantees:
        - Returns a set for every S1 (including singletons with 0 matches).
        - No duplicate IDs.
        - Preserves S2/S3 IDs.
        - Never forces a candidate if max_prob < threshold.
        """
        grouped_candidates = defaultdict(list)
        for s1_id, noisy_id, prob in zip(s1_ids, noisy_ids, probabilities):
            grouped_candidates[s1_id].append((noisy_id, float(prob)))

        # If a master list of all S1 IDs is provided, ensure singletons with 0 candidates are included
        target_s1_list = all_s1_entity_ids if all_s1_entity_ids is not None else list(grouped_candidates.keys())

        decisions: Dict[str, List[str]] = {}

        for s1_id in target_s1_list:
            cands = grouped_candidates.get(s1_id, [])
            if not cands:
                # 0 candidates generated -> singleton decision
                decisions[s1_id] = []
                continue

            # Sort by probability descending
            cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)

            selected_matches = []
            seen_ids = set()

            for noisy_id, prob in cands_sorted:
                if prob >= self.threshold:
                    if noisy_id not in seen_ids:
                        seen_ids.add(noisy_id)
                        selected_matches.append(noisy_id)

            # Strict rule: If none >= threshold, selected_matches remains [] (0 matches / singleton)
            decisions[s1_id] = selected_matches

        return decisions

    @staticmethod
    def format_submission_output(decisions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Formats decisions into competition submission format:
        Columns: [source1_entity_id, matched_entity_ids] (comma-separated or empty string).
        """
        records = []
        for s1_id, matches in decisions.items():
            match_str = ", ".join(matches) if matches else ""
            records.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": match_str,
                "match_count": len(matches)
            })
        return pd.DataFrame(records)


def calculate_entity_metrics(pred_set: Set[str], true_set: Set[str]) -> Tuple[float, float, float]:
    """Computes exact entity-level precision, recall, and F0.5."""
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    is_singleton = (len(true_set) == 0)

    if is_singleton:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if tp == 0:
        return prec, rec, 0.0

    denom = (0.25 * prec) + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f05


def run_decision_validation():
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    proc_dir = base_dir / "dataset" / "processed"
    models_dir = base_dir / "models"
    report_file = base_dir / "entity_decision_validation_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 90)
    print(" PHASE 14: ENTITY-LEVEL MATCH DECISION ENGINE VALIDATION")
    print("=" * 90)
    print(f"Project Base Directory: {base_dir}")
    print(f"Models Directory:       {models_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Step 1: Initialize Engine with Best Model
    best_model_path = models_dir / "best_model_controlled.joblib"
    if not best_model_path.exists():
        best_model_path = models_dir / "best_model.joblib"

    print(f"1. Loading Decision Engine with model: {best_model_path.name}...")
    engine = EntityMatchDecisionEngine(best_model_path)
    print(f"   - Decision Threshold (tau*): {engine.threshold:.3f}")
    print(f"   - Feature Dimension:         {len(engine.feature_cols)}\n")

    # Step 2: Load Validation Features
    print("2. Loading Validation Feature Matrix and Pairs...")
    # Check if 44 features exist in parquet or extract
    df_val_feat = pd.read_parquet(proc_dir / "val_features.parquet")
    df_val_pairs = pd.read_parquet(proc_dir / "val_pairs.parquet")

    # If engine expects the 4 new forensic features, compute them
    if len(engine.feature_cols) > 40 and "addr_num_exact_mismatch" not in df_val_feat.columns:
        print("   - Augmenting validation set with 4 forensic features...")
        import re
        re_digits = re.compile(r"\b\d+\b")
        s1_names = df_val_pairs["s1_name"].values
        s1_addrs = df_val_pairs["s1_address"].values
        noisy_names = df_val_pairs["noisy_name"].values
        noisy_addrs = df_val_pairs["noisy_address"].values

        num_mismatches = []
        acronym_matches = []
        pfx4_jaccards = []
        addr_containments = []

        for s1_n, s1_a, n_n, n_a in zip(s1_names, s1_addrs, noisy_names, noisy_addrs):
            s1_n_str = str(s1_n).lower() if pd.notna(s1_n) else ""
            n_n_str = str(n_n).lower() if pd.notna(n_n) else ""
            s1_a_str = str(s1_a).lower() if pd.notna(s1_a) else ""
            n_a_str = str(n_a).lower() if pd.notna(n_a) else ""

            s1_nums = set(re_digits.findall(s1_a_str))
            noisy_nums = set(re_digits.findall(n_a_str))
            num_mismatches.append(1.0 if (len(s1_nums) > 0 and len(noisy_nums) > 0 and len(s1_nums & noisy_nums) == 0) else 0.0)

            s1_toks = [t for t in s1_n_str.split() if t]
            n_toks = [t for t in n_n_str.split() if t]
            s1_acr = "".join(t[0] for t in s1_toks if t[0].isalnum())
            n_acr = "".join(t[0] for t in n_toks if t[0].isalnum())
            n_compact = "".join(c for c in n_n_str if c.isalnum())
            s1_compact = "".join(c for c in s1_n_str if c.isalnum())
            acronym_matches.append(1.0 if (s1_acr and s1_acr == n_compact) or (n_acr and n_acr == s1_compact) else 0.0)

            s1_pfx = s1_n_str[:4]
            n_pfx = n_n_str[:4]
            pfx4_jaccards.append(1.0 if (s1_pfx and s1_pfx == n_pfx) else 0.0)

            s1_a_set = set(s1_a_str.split())
            n_a_set = set(n_a_str.split())
            cont = max(len(s1_a_set & n_a_set) / len(s1_a_set), len(s1_a_set & n_a_set) / len(n_a_set)) if (len(s1_a_set) > 0 and len(n_a_set) > 0) else 0.0
            addr_containments.append(cont)

        df_val_feat["addr_num_exact_mismatch"] = num_mismatches
        df_val_feat["name_acronym_match"] = acronym_matches
        df_val_feat["name_pfx4_match"] = pfx4_jaccards
        df_val_feat["addr_token_containment"] = addr_containments

    # Step 3: Run Engine Scoring & Decision
    print("3. Scoring validation candidate pairs and executing decision engine...")
    t_inf0 = time.time()
    val_probs = engine.score_pairs(df_val_feat)
    inf_time = time.time() - t_inf0

    val_s1_ids = df_val_feat["source1_entity_id"].values
    val_noisy_ids = df_val_feat["noisy_entity_id"].values
    all_unique_val_s1 = sorted(list(set(val_s1_ids)))

    decisions = engine.decide_matches_for_candidates(
        val_s1_ids,
        val_noisy_ids,
        val_probs,
        all_s1_entity_ids=all_unique_val_s1
    )

    print(f"   - Scored Pairs:       {len(df_val_feat):,} in {inf_time:.2f} s")
    print(f"   - S1 Decisions Built: {len(decisions):,} entities\n")

    # Step 4: Validate Against Entity-Level Ground Truth
    print("4. Validating Decisions against Validation Ground Truth...")
    gt_lookup = GroundTruthLookup.load_from_tsv(base_dir / "dataset" / "train" / "train_ground_truth.tsv")

    entity_precisions = []
    entity_recalls = []
    entity_f05s = []

    # Breakdown cohorts
    singleton_stats = {"total": 0, "correct": 0, "fp_entities": 0}
    single_match_stats = {"total": 0, "exact_match": 0, "partial_match": 0, "zero_pred": 0}
    multi_match_stats = {"total": 0, "exact_match": 0, "partial_match": 0, "zero_pred": 0}

    pred_size_counts = Counter()

    for s1_id in all_unique_val_s1:
        pred_matches = decisions.get(s1_id, [])
        pred_set = set(pred_matches)
        true_set = set(gt_lookup.get_matches(s1_id))

        p, r, f05 = calculate_entity_metrics(pred_set, true_set)
        entity_precisions.append(p)
        entity_recalls.append(r)
        entity_f05s.append(f05)

        pred_size = len(pred_set)
        pred_size_counts[pred_size] += 1
        true_size = len(true_set)

        if true_size == 0:
            singleton_stats["total"] += 1
            if pred_size == 0:
                singleton_stats["correct"] += 1
            else:
                singleton_stats["fp_entities"] += 1

        elif true_size == 1:
            single_match_stats["total"] += 1
            if pred_set == true_set:
                single_match_stats["exact_match"] += 1
            elif len(pred_set) == 0:
                single_match_stats["zero_pred"] += 1
            else:
                single_match_stats["partial_match"] += 1

        else:
            multi_match_stats["total"] += 1
            if pred_set == true_set:
                multi_match_stats["exact_match"] += 1
            elif len(pred_set) == 0:
                multi_match_stats["zero_pred"] += 1
            else:
                multi_match_stats["partial_match"] += 1

    macro_prec = float(np.mean(entity_precisions))
    macro_rec = float(np.mean(entity_recalls))
    macro_f05 = float(np.mean(entity_f05s))

    # Step 5: Report Results
    print("\n" + "=" * 90)
    print(" 5. VALIDATION ACCURACY & MACRO F0.5 REPORT")
    print("=" * 90)
    print(f"• Macro F0.5 (Competition Metric):  {macro_f05:.4f}")
    print(f"• Macro Precision:                  {macro_prec:.4f} ({macro_prec*100:.2f}%)")
    print(f"• Macro Recall:                     {macro_rec:.4f} ({macro_rec*100:.2f}%)")
    print(f"• Total Source 1 Entities:          {len(all_unique_val_s1):,}")
    print(f"• Total Matches Predicted:          {sum(len(v) for v in decisions.values()):,}\n")

    print("=" * 90)
    print(" 6. COHORT-SPECIFIC BEHAVIOR BREAKDOWN")
    print("=" * 90)

    # 1. Singletons (0 matches in GT)
    print("\n[A] SINGLETON ENTITIES (0 Ground-Truth Matches):")
    print(f"    - Total Singletons in Val Set:  {singleton_stats['total']}")
    if singleton_stats['total'] > 0:
        s_acc = singleton_stats['correct'] / singleton_stats['total']
        s_fp_rate = singleton_stats['fp_entities'] / singleton_stats['total']
        print(f"    - Correctly Predicted 0 Matches:{singleton_stats['correct']} ({s_acc:.2%})")
        print(f"    - False-Positive Predictions:   {singleton_stats['fp_entities']} ({s_fp_rate:.2%})")
    else:
        print("    - (No singletons present in this validation slice; all S1s have >=1 true match)")

    # 2. Exactly 1 Match Entities
    print("\n[B] EXACTLY 1 MATCH ENTITIES (1 Ground-Truth Match):")
    print(f"    - Total 1-Match Entities:       {single_match_stats['total']:,}")
    if single_match_stats['total'] > 0:
        sm_exact_rate = single_match_stats['exact_match'] / single_match_stats['total']
        sm_zero_rate = single_match_stats['zero_pred'] / single_match_stats['total']
        print(f"    - Exact 1-to-1 Match Recovered: {single_match_stats['exact_match']:,} ({sm_exact_rate:.2%})")
        print(f"    - Missed (0 Matches Predicted): {single_match_stats['zero_pred']:,} ({sm_zero_rate:.2%})")
        print(f"    - Partial/Over-Predicted:       {single_match_stats['partial_match']:,}")

    # 3. Multiple Match Entities (2+ matches in GT)
    print("\n[C] MULTIPLE-MATCH ENTITIES (2+ Ground-Truth Matches):")
    print(f"    - Total Multi-Match Entities:   {multi_match_stats['total']:,}")
    if multi_match_stats['total'] > 0:
        mm_exact_rate = multi_match_stats['exact_match'] / multi_match_stats['total']
        mm_zero_rate = multi_match_stats['zero_pred'] / multi_match_stats['total']
        print(f"    - Exact Set Matches Recovered:  {multi_match_stats['exact_match']:,} ({mm_exact_rate:.2%})")
        print(f"    - Partial Matches (Some Ret.):  {multi_match_stats['partial_match']:,}")
        print(f"    - Complete Misses (0 Pred):     {multi_match_stats['zero_pred']:,} ({mm_zero_rate:.2%})")

    # 4. Predicted Set Size Distribution
    print("\n[D] PREDICTED MATCH SET SIZE DISTRIBUTION:")
    for size, cnt in sorted(pred_size_counts.items()):
        if size <= 10:
            print(f"    - {size} matches predicted: {cnt:5,d} entities ({(cnt/len(all_unique_val_s1))*100:5.2f}%)")
    larger = sum(cnt for size, cnt in pred_size_counts.items() if size > 10)
    if larger > 0:
        print(f"    - >10 matches predicted:{larger:5,d} entities ({(larger/len(all_unique_val_s1))*100:5.2f}%)")

    # Step 6: Verify Duplicates and S2/S3 ID Format
    print("\n" + "=" * 90)
    print(" 7. INTEGRITY CHECKS")
    print("=" * 90)
    has_duplicates = any(len(matches) != len(set(matches)) for matches in decisions.values())
    all_pred_ids = [nid for matches in decisions.values() for nid in matches]
    invalid_prefixes = [nid for nid in all_pred_ids if not (nid.startswith("S2-") or nid.startswith("S3-"))]

    print(f"• Duplicate IDs Detected:           {has_duplicates} (Must be False)")
    print(f"• Invalid Prefix IDs Detected:      {len(invalid_prefixes)} (Must be 0)")
    assert not has_duplicates, "Integrity Error: Found duplicate candidate IDs in prediction sets!"
    assert len(invalid_prefixes) == 0, f"Integrity Error: Found invalid non-S2/S3 IDs: {invalid_prefixes[:5]}"
    print("• All Structural Integrity Assertions PASSED Successfully!\n")

    # Step 7: Persist Decision Engine Metadata
    meta_path = models_dir / "decision_engine_metadata.json"
    meta_dict = {
        "engine_class": "EntityMatchDecisionEngine",
        "model_file": str(best_model_path.name),
        "threshold": float(engine.threshold),
        "feature_count": len(engine.feature_cols),
        "macro_f05": float(macro_f05),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "single_match_exact_accuracy": float(single_match_stats['exact_match'] / single_match_stats['total']) if single_match_stats['total'] > 0 else 0.0,
        "multi_match_exact_accuracy": float(multi_match_stats['exact_match'] / multi_match_stats['total']) if multi_match_stats['total'] > 0 else 0.0,
        "validation_entities_count": len(all_unique_val_s1)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    # Format sample submission dataframe
    df_sample_sub = engine.format_submission_output(decisions)
    print(f"Sample Formatted Output Preview (First 5 S1 Entities):")
    for _, row in df_sample_sub.head(5).iterrows():
        print(f"  {row['source1_entity_id']} -> [{row['matched_entity_ids']}] ({row['match_count']} matches)")

    elapsed = time.time() - t0
    print("\n" + "=" * 90)
    print(f"Total Decision Engine Validation Time: {elapsed:.2f} seconds")
    print(" PHASE 14 ENTITY-LEVEL MATCH DECISION COMPLETED SUCCESSFULLY")
    print("=" * 90)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_decision_validation()
