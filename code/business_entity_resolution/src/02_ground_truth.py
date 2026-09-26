"""
Phase 2: Ground Truth Analysis for Amazon ML Challenge 2026 - Business Entity Resolution
This script performs in-depth analysis of the ground-truth mappings, parses matched_entity_ids safely,
analyzes match cardinality, prefix distribution, coverage across sources, anomaly detection,
and provides reusable lookup structures for downstream blocking and training phases.
"""

import sys
import gc
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter, defaultdict

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np


def parse_matched_entity_ids(val: Any) -> List[str]:
    """
    Safely parse matched_entity_ids column value into a list of strings.
    Empty values (NaN, None, empty string, whitespace) become an empty list [].
    """
    if val is None or pd.isna(val):
        return []
    
    val_str = str(val).strip()
    if not val_str:
        return []
    
    # Split by comma, strip whitespace, and discard empty strings
    tokens = [token.strip() for token in val_str.split(",") if token.strip()]
    return tokens


class GroundTruthLookup:
    """
    Reusable lookup structure for Ground Truth mappings.
    Provides O(1) queries for S1->Matches, S1->S2, S1->S3, S2->S1, S3->S1, and pairwise checks.
    """
    def __init__(self, s1_to_matches: Dict[str, List[str]]):
        self.s1_to_matches = s1_to_matches
        self.s1_to_s2: Dict[str, List[str]] = {}
        self.s1_to_s3: Dict[str, List[str]] = {}
        self.s2_to_s1: Dict[str, Set[str]] = defaultdict(set)
        self.s3_to_s1: Dict[str, Set[str]] = defaultdict(set)
        
        self._build_indices()

    def _build_indices(self):
        for s1_id, matches in self.s1_to_matches.items():
            s2_list = []
            s3_list = []
            for m in matches:
                if m.startswith("S2-"):
                    s2_list.append(m)
                    self.s2_to_s1[m].add(s1_id)
                elif m.startswith("S3-"):
                    s3_list.append(m)
                    self.s3_to_s1[m].add(s1_id)
            self.s1_to_s2[s1_id] = s2_list
            self.s1_to_s3[s1_id] = s3_list

    def get_matches(self, s1_id: str) -> List[str]:
        """Return all matched entity IDs for a given S1 entity ID."""
        return self.s1_to_matches.get(s1_id, [])

    def get_s2_matches(self, s1_id: str) -> List[str]:
        """Return S2 matched entity IDs for a given S1 entity ID."""
        return self.s1_to_s2.get(s1_id, [])

    def get_s3_matches(self, s1_id: str) -> List[str]:
        """Return S3 matched entity IDs for a given S1 entity ID."""
        return self.s1_to_s3.get(s1_id, [])

    def is_match(self, s1_id: str, candidate_id: str) -> bool:
        """Check if candidate_id is a ground-truth match for s1_id."""
        return candidate_id in self.s1_to_matches.get(s1_id, [])

    def get_s1_for_noisy(self, noisy_id: str) -> Set[str]:
        """Return S1 entity IDs linked to a given noisy entity ID (S2 or S3)."""
        if noisy_id.startswith("S2-"):
            return self.s2_to_s1.get(noisy_id, set())
        elif noisy_id.startswith("S3-"):
            return self.s3_to_s1.get(noisy_id, set())
        return set()

    @classmethod
    def load_from_tsv(cls, filepath: Path | str) -> "GroundTruthLookup":
        """Load ground truth from TSV file into GroundTruthLookup object."""
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        s1_to_matches: Dict[str, List[str]] = {}
        for s1_id, matches_raw in zip(df["source1_entity_id"], df["matched_entity_ids"]):
            s1_to_matches[str(s1_id).strip()] = parse_matched_entity_ids(matches_raw)
        return cls(s1_to_matches)


class TeeLogger:
    """Tee stdout to both console and a report file with autoflush."""
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


def run_ground_truth_analysis():
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "dataset" / "train"
    report_file = base_dir / "ground_truth_analysis_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 85)
    print(" PHASE 2: GROUND TRUTH ANALYSIS & VALIDATION")
    print("=" * 85)
    print(f"Project Base Directory: {base_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Paths
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print(f"1. Loading ground truth from {gt_path.name}...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    # Parse matched_entity_ids safely
    print("   Parsing matched_entity_ids into structured lists...")
    raw_s1_ids = gt_df["source1_entity_id"].astype(str).tolist()
    raw_matched = gt_df["matched_entity_ids"].tolist()
    
    parsed_matches: List[List[str]] = [parse_matched_entity_ids(v) for v in raw_matched]
    match_counts = np.array([len(m) for m in parsed_matches], dtype=np.int32)
    gt_df["match_count"] = match_counts
    
    total_gt_s1 = len(gt_df)
    unique_gt_s1 = len(set(raw_s1_ids))
    print(f"\n[1] Total Source 1 Entities in Ground Truth: {total_gt_s1:,} rows (Unique IDs: {unique_gt_s1:,})")

    # 2. Singleton entities with zero matches
    zero_matches_mask = (match_counts == 0)
    num_singletons = int(zero_matches_mask.sum())
    pct_singletons = (num_singletons / total_gt_s1) * 100
    print(f"\n[2] Singleton Entities (Zero Matches): {num_singletons:,} ({pct_singletons:.2f}%)")

    # 3. Entities with exactly one match
    one_match_mask = (match_counts == 1)
    num_one_match = int(one_match_mask.sum())
    pct_one_match = (num_one_match / total_gt_s1) * 100
    print(f"\n[3] Entities with Exactly One Match: {num_one_match:,} ({pct_one_match:.2f}%)")

    # 4. Entities with multiple matches
    multi_match_mask = (match_counts > 1)
    num_multi_match = int(multi_match_mask.sum())
    pct_multi_match = (num_multi_match / total_gt_s1) * 100
    print(f"\n[4] Entities with Multiple Matches (> 1): {num_multi_match:,} ({pct_multi_match:.2f}%)")

    # 5. Distribution of number of matches
    print("\n[5] Distribution of Number of Matches per S1 Entity:")
    print(f"    - Min Matches:    {match_counts.min()}")
    print(f"    - Max Matches:    {match_counts.max()}")
    print(f"    - Mean Matches:   {match_counts.mean():.4f}")
    print(f"    - Median Matches: {np.median(match_counts):.0f}")
    print(f"    - Std Dev:        {match_counts.std():.4f}")
    
    percentiles = [25, 50, 75, 90, 95, 99, 99.9]
    print("    - Percentiles:")
    for p in percentiles:
        val = np.percentile(match_counts, p)
        print(f"      * {p:5.1f}th percentile: {val:.1f}")

    print("\n    - Match Count Breakdown (Histogram / Frequency):")
    freq_counter = Counter(match_counts)
    for count_val in sorted(freq_counter.keys()):
        freq = freq_counter[count_val]
        pct = (freq / total_gt_s1) * 100
        bar = "#" * int(pct / 2)
        print(f"      Count = {count_val:2d}: {freq:8,d} ({pct:6.2f}%)  {bar}")

    # Build match breakdown (S2 vs S3 vs others)
    print("\nAnalyzing S2/S3 match counts, duplicates, and mappings...")
    total_s2_matches = 0
    total_s3_matches = 0
    s1_with_s2_count = 0
    s1_with_s3_count = 0
    s1_with_both_count = 0

    all_s2_ids: List[str] = []
    all_s3_ids: List[str] = []
    invalid_ids: List[str] = []
    intra_record_duplicates = 0

    s2_to_s1_map = defaultdict(list)
    s3_to_s1_map = defaultdict(list)

    for s1_id, matches in zip(raw_s1_ids, parsed_matches):
        if len(matches) != len(set(matches)):
            intra_record_duplicates += 1

        has_s2 = False
        has_s3 = False

        for m in matches:
            if m.startswith("S2-"):
                total_s2_matches += 1
                has_s2 = True
                all_s2_ids.append(m)
                s2_to_s1_map[m].append(s1_id)
            elif m.startswith("S3-"):
                total_s3_matches += 1
                has_s3 = True
                all_s3_ids.append(m)
                s3_to_s1_map[m].append(s1_id)
            else:
                invalid_ids.append(m)

        if has_s2:
            s1_with_s2_count += 1
        if has_s3:
            s1_with_s3_count += 1
        if has_s2 and has_s3:
            s1_with_both_count += 1

    total_matches_all = total_s2_matches + total_s3_matches + len(invalid_ids)

    # 6. Total S2 matches
    unique_s2_matched_set = set(all_s2_ids)
    unique_s2_matched = len(unique_s2_matched_set)
    print(f"\n[6] Total S2 Matches: {total_s2_matches:,} ({total_s2_matches / total_matches_all * 100:.2f}% of all match links)")
    print(f"    Unique S2 Entities Matched: {unique_s2_matched:,}")

    # 7. Total S3 matches
    unique_s3_matched_set = set(all_s3_ids)
    unique_s3_matched = len(unique_s3_matched_set)
    print(f"\n[7] Total S3 Matches: {total_s3_matches:,} ({total_s3_matches / total_matches_all * 100:.2f}% of all match links)")
    print(f"    Unique S3 Entities Matched: {unique_s3_matched:,}")

    # 8. S1 entities having S2 matches
    pct_s1_s2 = (s1_with_s2_count / total_gt_s1) * 100
    print(f"\n[8] S1 Entities having S2 Matches: {s1_with_s2_count:,} ({pct_s1_s2:.2f}%)")

    # 9. S1 entities having S3 matches
    pct_s1_s3 = (s1_with_s3_count / total_gt_s1) * 100
    print(f"\n[9] S1 Entities having S3 Matches: {s1_with_s3_count:,} ({pct_s1_s3:.2f}%)")
    print(f"    S1 Entities having BOTH S2 & S3 Matches: {s1_with_both_count:,} ({s1_with_both_count / total_gt_s1 * 100:.2f}%)")

    # 10. Invalid matched IDs check
    print("\n[10] Invalid Matched IDs Check:")
    print(f"     - Syntax / prefix anomalies (not starting with S2- or S3-): {len(invalid_ids)}")
    if invalid_ids:
        print(f"       Examples of invalid IDs: {invalid_ids[:10]}")

    # Load Source 1, 2, 3 ID sets efficiently (usecols=['entity_id'])
    print("\nLoading source datasets for integrity verification and coverage...")
    s1_all_ids = set(pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], dtype=str)["entity_id"].dropna().unique())
    s2_all_ids = set(pd.read_csv(s2_path, sep="\t", usecols=["entity_id"], dtype=str)["entity_id"].dropna().unique())
    s3_all_ids = set(pd.read_csv(s3_path, sep="\t", usecols=["entity_id"], dtype=str)["entity_id"].dropna().unique())

    # Check unresolvable / dangling IDs in ground truth
    missing_s2_in_data = unique_s2_matched_set - s2_all_ids
    missing_s3_in_data = unique_s3_matched_set - s3_all_ids

    print(f"     - S2 IDs in Ground Truth missing from train_source2.tsv: {len(missing_s2_in_data)}")
    print(f"     - S3 IDs in Ground Truth missing from train_source3.tsv: {len(missing_s3_in_data)}")

    # 11. Duplicate matched IDs check
    print("\n[11] Duplicate Matched IDs Check:")
    print(f"     - Intra-record duplicate occurrences (same ID repeated in single S1 row): {intra_record_duplicates}")
    
    # Cross-record duplicates (1 S2/S3 entity linked to multiple S1 entities)
    s2_multi_parent = {k: v for k, v in s2_to_s1_map.items() if len(v) > 1}
    s3_multi_parent = {k: v for k, v in s3_to_s1_map.items() if len(v) > 1}
    print(f"     - S2 entities linked to MULTIPLE S1 entities: {len(s2_multi_parent):,}")
    if s2_multi_parent:
        sample_k = next(iter(s2_multi_parent))
        print(f"       Example: {sample_k} -> S1 parents: {s2_multi_parent[sample_k]}")
    print(f"     - S3 entities linked to MULTIPLE S1 entities: {len(s3_multi_parent):,}")
    if s3_multi_parent:
        sample_k = next(iter(s3_multi_parent))
        print(f"       Example: {sample_k} -> S1 parents: {s3_multi_parent[sample_k]}")

    # 12. Ground-truth coverage
    print("\n[12] Ground-Truth Coverage Analysis:")
    gt_s1_set = set(raw_s1_ids)
    s1_cov = (len(gt_s1_set) / len(s1_all_ids)) * 100 if s1_all_ids else 0
    s2_cov = (unique_s2_matched / len(s2_all_ids)) * 100 if s2_all_ids else 0
    s3_cov = (unique_s3_matched / len(s3_all_ids)) * 100 if s3_all_ids else 0

    print(f"     - Source 1 Coverage in Ground Truth: {len(gt_s1_set):,} / {len(s1_all_ids):,} ({s1_cov:.2f}%)")
    print(f"     - Source 2 Coverage in Ground Truth: {unique_s2_matched:,} / {len(s2_all_ids):,} ({s2_cov:.2f}%)")
    print(f"     - Source 3 Coverage in Ground Truth: {unique_s3_matched:,} / {len(s3_all_ids):,} ({s3_cov:.2f}%)")
    print(f"     - Unmatched Source 2 Entities: {len(s2_all_ids) - unique_s2_matched:,} ({(100 - s2_cov):.2f}%)")
    print(f"     - Unmatched Source 3 Entities: {len(s3_all_ids) - unique_s3_matched:,} ({(100 - s3_cov):.2f}%)")

    # Pick samples for detailed inspection
    singleton_indices = np.where(zero_matches_mask)[0][:3].tolist()
    multi_indices = np.where(multi_match_mask)[0][:3].tolist()
    both_indices = [i for i, m in enumerate(parsed_matches) if any(x.startswith("S2-") for x in m) and any(x.startswith("S3-") for x in m)][:3]

    needed_s1_ids = set()
    needed_s2_ids = set()
    needed_s3_ids = set()

    all_sample_idx = set(singleton_indices + multi_indices + both_indices)
    for idx in all_sample_idx:
        needed_s1_ids.add(raw_s1_ids[idx])
        for m in parsed_matches[idx]:
            if m.startswith("S2-"):
                needed_s2_ids.add(m)
            elif m.startswith("S3-"):
                needed_s3_ids.add(m)

    # Load only necessary sample rows from files for fast lookup
    print("\nFetching record attributes for targeted sample visualization...")
    s1_samples_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    s1_sample_dict = s1_samples_df[s1_samples_df["entity_id"].isin(needed_s1_ids)].set_index("entity_id").to_dict(orient="index")
    del s1_samples_df
    gc.collect()

    s2_samples_df = pd.read_csv(s2_path, sep="\t", dtype=str)
    s2_sample_dict = s2_samples_df[s2_samples_df["entity_id"].isin(needed_s2_ids)].set_index("entity_id").to_dict(orient="index")
    del s2_samples_df
    gc.collect()

    s3_samples_df = pd.read_csv(s3_path, sep="\t", dtype=str)
    s3_sample_dict = s3_samples_df[s3_samples_df["entity_id"].isin(needed_s3_ids)].set_index("entity_id").to_dict(orient="index")
    del s3_samples_df
    gc.collect()

    # 13. Examples of singleton entities
    print("\n[13] Examples of Singleton Entities (0 matches in noisy sources):")
    for i, idx in enumerate(singleton_indices, 1):
        s1_id = raw_s1_ids[idx]
        info = s1_sample_dict.get(s1_id, {})
        print(f"     --- Singleton Example {i} ---")
        print(f"       Entity ID: {s1_id}")
        print(f"       Name:      {info.get('business_name', 'N/A')}")
        print(f"       Address:   {info.get('business_address', 'N/A')}")
        print(f"       Country:   {info.get('country', 'N/A')}")

    # 14. Examples of multi-match entities
    print("\n[14] Examples of Multi-Match Entities (>1 matches):")
    for i, idx in enumerate(multi_indices, 1):
        s1_id = raw_s1_ids[idx]
        matches = parsed_matches[idx]
        info = s1_sample_dict.get(s1_id, {})
        s2_sub = [m for m in matches if m.startswith("S2-")]
        s3_sub = [m for m in matches if m.startswith("S3-")]
        print(f"     --- Multi-Match Example {i} ---")
        print(f"       S1 ID:         {s1_id}")
        print(f"       Name:          {info.get('business_name', 'N/A')}")
        print(f"       Address:       {info.get('business_address', 'N/A')}")
        print(f"       Country:       {info.get('country', 'N/A')}")
        print(f"       Total Matches: {len(matches)} (S2: {len(s2_sub)}, S3: {len(s3_sub)})")
        print(f"       Matched IDs:   {matches}")

    # 15. Examples of S1 records and their true S2/S3 records side-by-side
    print("\n[15] Detailed Side-by-Side Comparison of S1 vs True S2/S3 Matches:")
    for i, idx in enumerate(both_indices, 1):
        s1_id = raw_s1_ids[idx]
        matches = parsed_matches[idx]
        s1_info = s1_sample_dict.get(s1_id, {})
        
        print(f"\n     ==========================================================================")
        print(f"     Comparison Case {i}: S1 Entity [{s1_id}]")
        print(f"     --------------------------------------------------------------------------")
        print(f"     [SOURCE 1]  Name:    {s1_info.get('business_name', 'N/A')}")
        print(f"                 Address: {s1_info.get('business_address', 'N/A')}")
        print(f"                 Country: {s1_info.get('country', 'N/A')}")
        
        for m_id in matches:
            if m_id.startswith("S2-"):
                m_info = s2_sample_dict.get(m_id, {})
                print(f"     [MATCH S2]  ID:      {m_id}")
                print(f"                 Name:    {m_info.get('business_name', 'N/A')}")
                print(f"                 Address: {m_info.get('business_address', 'N/A')}")
                print(f"                 Country: {m_info.get('country', 'N/A')}")
            elif m_id.startswith("S3-"):
                m_info = s3_sample_dict.get(m_id, {})
                print(f"     [MATCH S3]  ID:      {m_id}")
                print(f"                 Name:    {m_info.get('business_name', 'N/A')}")
                print(f"                 Address: {m_info.get('business_address', 'N/A')}")
                print(f"                 Country: {m_info.get('country', 'N/A')}")

    print("\n" + "=" * 85)
    print(" REUSABLE LOOKUP STRUCTURE VALIDATION")
    print("=" * 85)
    # Instantiate GroundTruthLookup class from the parsed ground truth
    lookup = GroundTruthLookup(dict(zip(raw_s1_ids, parsed_matches)))
    sample_test_id = raw_s1_ids[0]
    sample_res = lookup.get_matches(sample_test_id)
    print(f"Lookup structure instantiated successfully.")
    print(f"Sample test lookup for '{sample_test_id}': {len(sample_res)} matches found.")
    print(f"  - S2 matches: {lookup.get_s2_matches(sample_test_id)}")
    print(f"  - S3 matches: {lookup.get_s3_matches(sample_test_id)}")

    elapsed = time.time() - t0
    print(f"\nTotal Elapsed Time: {elapsed:.2f} seconds")
    print("=" * 85)
    print(" PHASE 2 GROUND TRUTH ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 85)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_ground_truth_analysis()
