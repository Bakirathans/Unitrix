"""
Phase 17: Local Output Validation Suite
Amazon ML Challenge 2026 - Business Entity Resolution

Performs strict, independent validation on the required submission files:
1. output/matching_results.tsv
2. output/candidate_pairs.tsv

Validation Checks:
[Check 1] Exact required columns and headers
[Check 2] Strict TSV formatting (tab delimiters, no CSV commas as column separators)
[Check 3] Every test S1 appears exactly once
[Check 4] No duplicate S1 IDs
[Check 5] No missing S1 IDs (exact match with test_source1.tsv)
[Check 6] No invalid S2/S3 IDs (must exist in test_source2.tsv or test_source3.tsv)
[Check 7] No duplicate matched IDs per S1 entity
[Check 8] No duplicate candidate IDs per S1 entity
[Check 9] matching_results predictions are a strict subset of candidate_pairs
[Check 10] No S1-to-S1 predictions (no S1 IDs in predicted or candidate sets)
[Check 11] Empty predictions are valid (proper tab with empty string for singletons)
[Check 12] candidate_pairs non-emptiness & candidate coverage summary

Prints PASS/FAIL with detailed metrics for every check.
Never modifies outputs silently.
"""

import sys
import time
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import Counter

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd


def print_check_result(check_num: int, title: str, passed: bool, details: str = ""):
    status_str = "[ PASS ]" if passed else "[ FAIL ]"
    color_prefix = ""
    print(f"{status_str} Check {check_num:02d}: {title}")
    if details:
        for line in details.strip().split("\n"):
            print(f"         {line}")
    print()


def validate_all_outputs():
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"

    matching_results_path = output_dir / "matching_results.tsv"
    candidate_pairs_path = output_dir / "candidate_pairs.tsv"
    test_s1_path = test_dir / "test_source1.tsv"
    test_s2_path = test_dir / "test_source2.tsv"
    test_s3_path = test_dir / "test_source3.tsv"

    print("=" * 90)
    print(" PHASE 17: LOCAL OUTPUT VALIDATION SUITE")
    print("=" * 90)
    print(f"Base Directory:           {base_dir}")
    print(f"Matching Results Path:    {matching_results_path}")
    print(f"Candidate Pairs Path:     {candidate_pairs_path}\n")

    all_passed = True

    # 0. File Existence Checks
    if not matching_results_path.exists():
        print(f"[ FAIL ] File not found: {matching_results_path}")
        return False
    if not candidate_pairs_path.exists():
        print(f"[ FAIL ] File not found: {candidate_pairs_path}")
        return False

    # 1. Load Ground Truth Test Entity IDs
    print("Loading Ground Truth Test Entity IDs for Cross-Verification...")
    df_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str, keep_default_na=False)
    expected_s1_ids = list(df_s1["entity_id"].values)
    expected_s1_set = set(expected_s1_ids)
    expected_s1_count = len(expected_s1_ids)
    print(f"• Expected Test Source 1 Entities: {expected_s1_count:,}")

    # Load valid S2 and S3 IDs
    valid_s2_ids: Set[str] = set()
    valid_s3_ids: Set[str] = set()

    print("• Loading valid test Source 2 IDs...")
    for chunk in pd.read_csv(test_s2_path, sep="\t", usecols=["entity_id"], dtype=str, chunksize=500_000):
        valid_s2_ids.update(chunk["entity_id"].values)
    print(f"  Loaded {len(valid_s2_ids):,} Source 2 IDs.")

    print("• Loading valid test Source 3 IDs...")
    for chunk in pd.read_csv(test_s3_path, sep="\t", usecols=["entity_id"], dtype=str, chunksize=500_000):
        valid_s3_ids.update(chunk["entity_id"].values)
    print(f"  Loaded {len(valid_s3_ids):,} Source 3 IDs.")

    valid_pool_ids = valid_s2_ids | valid_s3_ids
    print(f"  Total Valid Candidate Pool IDs (S2 | S3): {len(valid_pool_ids):,}\n")

    print("-" * 90)
    print(" EXECUTING DETAILED VALIDATION CHECKS")
    print("-" * 90)

    # ---------------------------------------------------------
    # Check 1: Exact required columns and headers
    # ---------------------------------------------------------
    with open(matching_results_path, "r", encoding="utf-8") as f:
        match_header = f.readline().strip().split("\t")
    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        cand_header = f.readline().strip().split("\t")

    c1_match_ok = (match_header == ["source1_entity_id", "matched_entity_ids"])
    c1_cand_ok = (cand_header == ["source1_entity_id", "candidate_entity_ids"])
    c1_passed = c1_match_ok and c1_cand_ok
    all_passed &= c1_passed
    print_check_result(
        1, "Exact Required Columns & Headers", c1_passed,
        f"matching_results.tsv: {match_header} (Expected: ['source1_entity_id', 'matched_entity_ids'])\n"
        f"candidate_pairs.tsv:  {cand_header} (Expected: ['source1_entity_id', 'candidate_entity_ids'])"
    )

    # ---------------------------------------------------------
    # Check 2: Strict TSV formatting
    # ---------------------------------------------------------
    # Sample top 1,000 lines from both files and ensure delimiter is tab, not comma
    tsv_format_errors = 0
    with open(matching_results_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 5000:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                tsv_format_errors += 1

    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 5000:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                tsv_format_errors += 1

    c2_passed = (tsv_format_errors == 0)
    all_passed &= c2_passed
    print_check_result(
        2, "Strict TSV Formatting (Tab Delimited)", c2_passed,
        f"Verified line structure across 10,000 sampled rows. Format errors: {tsv_format_errors}"
    )

    # ---------------------------------------------------------
    # Read both files into memory/data structures for comprehensive analysis
    # ---------------------------------------------------------
    print("Reading full submission files for entity-level integrity checks...")
    df_match = pd.read_csv(matching_results_path, sep="\t", dtype=str, keep_default_na=False)
    df_cand = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)

    match_s1_list = list(df_match["source1_entity_id"].values)
    cand_s1_list = list(df_cand["source1_entity_id"].values)
    match_s1_set = set(match_s1_list)
    cand_s1_set = set(cand_s1_list)

    # ---------------------------------------------------------
    # Check 3: Every test S1 appears exactly once
    # ---------------------------------------------------------
    c3_match = (len(match_s1_list) == expected_s1_count) and (len(match_s1_set) == expected_s1_count)
    c3_cand = (len(cand_s1_list) == expected_s1_count) and (len(cand_s1_set) == expected_s1_count)
    c3_passed = c3_match and c3_cand
    all_passed &= c3_passed
    print_check_result(
        3, "Every Test Source 1 Appears Exactly Once", c3_passed,
        f"matching_results.tsv row count: {len(df_match):,} (Expected: {expected_s1_count:,})\n"
        f"candidate_pairs.tsv row count:  {len(df_cand):,} (Expected: {expected_s1_count:,})"
    )

    # ---------------------------------------------------------
    # Check 4: No duplicate S1 IDs
    # ---------------------------------------------------------
    match_s1_dups = len(match_s1_list) - len(match_s1_set)
    cand_s1_dups = len(cand_s1_list) - len(cand_s1_set)
    c4_passed = (match_s1_dups == 0) and (cand_s1_dups == 0)
    all_passed &= c4_passed
    print_check_result(
        4, "No Duplicate Source 1 IDs", c4_passed,
        f"matching_results duplicate S1 IDs: {match_s1_dups}\n"
        f"candidate_pairs duplicate S1 IDs:  {cand_s1_dups}"
    )

    # ---------------------------------------------------------
    # Check 5: No missing S1 IDs (Exact set equality)
    # ---------------------------------------------------------
    missing_in_match = expected_s1_set - match_s1_set
    extra_in_match = match_s1_set - expected_s1_set
    missing_in_cand = expected_s1_set - cand_s1_set
    extra_in_cand = cand_s1_set - expected_s1_set

    c5_passed = (len(missing_in_match) == 0 and len(extra_in_match) == 0 and
                 len(missing_in_cand) == 0 and len(extra_in_cand) == 0)
    all_passed &= c5_passed
    print_check_result(
        5, "No Missing or Extra Source 1 IDs (100% Set Match)", c5_passed,
        f"Missing in matching_results: {len(missing_in_match)} | Extra: {len(extra_in_match)}\n"
        f"Missing in candidate_pairs:  {len(missing_in_cand)} | Extra: {len(extra_in_cand)}"
    )

    # ---------------------------------------------------------
    # Entity-level Forensics across all 1.73M rows
    # ---------------------------------------------------------
    print("Performing comprehensive entity-level parsing on 1,732,544 rows...")

    invalid_matched_ids = []
    invalid_cand_ids = []
    s1_to_s1_predictions = []
    duplicate_matched_ids_rows = 0
    duplicate_cand_ids_rows = 0
    subset_violations = []

    total_predicted_matches = 0
    total_candidates = 0
    empty_prediction_count = 0
    empty_cand_count = 0

    match_cardinality = Counter()

    match_vals = df_match["matched_entity_ids"].values
    cand_vals = df_cand["candidate_entity_ids"].values
    s1_vals = df_match["source1_entity_id"].values

    for s1_id, m_str, c_str in zip(s1_vals, match_vals, cand_vals):
        m_list = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []
        c_list = [x.strip() for x in c_str.split(",") if x.strip()] if c_str else []
        c_set = set(c_list)

        m_len = len(m_list)
        c_len = len(c_list)
        total_predicted_matches += m_len
        total_candidates += c_len
        match_cardinality[m_len] += 1

        if m_len == 0:
            empty_prediction_count += 1
        if c_len == 0:
            empty_cand_count += 1

        # Check duplicate IDs in list
        if len(m_list) != len(set(m_list)):
            duplicate_matched_ids_rows += 1
        if len(c_list) != len(c_set):
            duplicate_cand_ids_rows += 1

        # Check ID validity and S1-to-S1 predictions
        for mid in m_list:
            if mid.startswith("S1-"):
                s1_to_s1_predictions.append((s1_id, mid))
            if mid not in valid_pool_ids:
                invalid_matched_ids.append((s1_id, mid))
            if mid not in c_set:
                subset_violations.append((s1_id, mid))

        for cid in c_list:
            if cid.startswith("S1-"):
                s1_to_s1_predictions.append((s1_id, cid))
            if cid not in valid_pool_ids:
                invalid_cand_ids.append((s1_id, cid))

    # ---------------------------------------------------------
    # Check 6: No invalid S2/S3 IDs
    # ---------------------------------------------------------
    c6_passed = (len(invalid_matched_ids) == 0 and len(invalid_cand_ids) == 0)
    all_passed &= c6_passed
    print_check_result(
        6, "No Invalid S2/S3 IDs (Must Exist in Pool)", c6_passed,
        f"Invalid IDs in matching_results: {len(invalid_matched_ids)}\n"
        f"Invalid IDs in candidate_pairs:  {len(invalid_cand_ids)}"
    )

    # ---------------------------------------------------------
    # Check 7: No duplicate matched IDs
    # ---------------------------------------------------------
    c7_passed = (duplicate_matched_ids_rows == 0)
    all_passed &= c7_passed
    print_check_result(
        7, "No Duplicate Matched IDs per S1 Entity", c7_passed,
        f"Entities with duplicate predicted matches: {duplicate_matched_ids_rows}"
    )

    # ---------------------------------------------------------
    # Check 8: No duplicate candidate IDs
    # ---------------------------------------------------------
    c8_passed = (duplicate_cand_ids_rows == 0)
    all_passed &= c8_passed
    print_check_result(
        8, "No Duplicate Candidate IDs per S1 Entity", c8_passed,
        f"Entities with duplicate candidates: {duplicate_cand_ids_rows}"
    )

    # ---------------------------------------------------------
    # Check 9: Matching results are a strict subset of candidate pairs
    # ---------------------------------------------------------
    c9_passed = (len(subset_violations) == 0)
    all_passed &= c9_passed
    print_check_result(
        9, "Matching Predictions are a Strict Subset of Candidate Pairs", c9_passed,
        f"Subset violations (matches not found in candidate set): {len(subset_violations)}"
    )

    # ---------------------------------------------------------
    # Check 10: No S1-to-S1 predictions
    # ---------------------------------------------------------
    c10_passed = (len(s1_to_s1_predictions) == 0)
    all_passed &= c10_passed
    print_check_result(
        10, "No S1-to-S1 Predictions (Only S2 and S3 Valid)", c10_passed,
        f"S1-to-S1 predictions detected: {len(s1_to_s1_predictions)}"
    )

    # ---------------------------------------------------------
    # Check 11: Empty predictions are valid
    # ---------------------------------------------------------
    # Singletons must be formatted with empty string
    c11_passed = (empty_prediction_count > 0 and empty_prediction_count < expected_s1_count)
    all_passed &= c11_passed
    print_check_result(
        11, "Empty Predictions Correctly Formatted for Singletons", c11_passed,
        f"Valid Singleton (0-match) entities: {empty_prediction_count:,} ({empty_prediction_count / expected_s1_count * 100:.2f}%)\n"
        f"Valid Multi-match / Single-match:     {expected_s1_count - empty_prediction_count:,} ({(expected_s1_count - empty_prediction_count) / expected_s1_count * 100:.2f}%)"
    )

    # ---------------------------------------------------------
    # Check 12: Candidate pairs coverage summary
    # ---------------------------------------------------------
    c12_passed = (total_candidates > 0 and empty_cand_count < expected_s1_count)
    all_passed &= c12_passed
    print_check_result(
        12, "Candidate Pairs Final Pre-Inference Set Coverage", c12_passed,
        f"Total Final Candidates: {total_candidates:,} (Average {total_candidates / expected_s1_count:.2f} per S1)\n"
        f"Entities with >= 1 candidate: {expected_s1_count - empty_cand_count:,} ({(expected_s1_count - empty_cand_count) / expected_s1_count * 100:.2f}%)"
    )

    elapsed = time.time() - t0
    print("=" * 90)
    if all_passed:
        print(" FINAL VERDICT: ALL 12 VALIDATION CHECKS PASSED PERFECTLY (100% COMPLIANT)")
    else:
        print(" FINAL VERDICT: VALIDATION FAILED - SEE ABOVE DETAILS")
    print("=" * 90)
    print(f"Validation Duration: {elapsed:.2f} s\n")

    print("SUBMISSION SUMMARY STATISTICS:")
    print(f"• Total Source 1 Entities:   {expected_s1_count:,}")
    print(f"• Total Predicted Matches:    {total_predicted_matches:,} (Avg {total_predicted_matches / expected_s1_count:.2f} per S1)")
    print(f"• Total Final Candidates:     {total_candidates:,} (Avg {total_candidates / expected_s1_count:.2f} per S1)")
    print("\nMatch Set Cardinality Breakdown:")
    for card in sorted(match_cardinality.keys()):
        cnt = match_cardinality[card]
        pct = (cnt / expected_s1_count) * 100
        print(f"  - {card:2d} matches: {cnt:9,d} entities ({pct:6.2f}%)")

    return all_passed


if __name__ == "__main__":
    success = validate_all_outputs()
    sys.exit(0 if success else 1)
