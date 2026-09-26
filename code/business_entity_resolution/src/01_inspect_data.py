"""
Phase 1: Dataset Inspection for Amazon ML Challenge 2026 - Business Entity Resolution
This script inspects train and test datasets, computes comprehensive profiling metrics,
checks data consistency and integrity, and reports anomalies.
"""

import os
import sys
import gc
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np


def inspect_dataframe(df: pd.DataFrame, name: str):
    print("=" * 80)
    print(f" DATASET PROFILE: {name}")
    print("=" * 80)

    # 1. Shape
    num_rows, num_cols = df.shape
    print(f"1. Shape: {num_rows:,} rows x {num_cols} columns")

    # 2. Column names & 3. Data types
    print("\n2 & 3. Column Names, Data Types & Missingness:")
    stats_list = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = (n_missing / num_rows) * 100 if num_rows > 0 else 0
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)
        stats_list.append({
            "Column": col,
            "Dtype": dtype,
            "Missing": f"{n_missing:,}",
            "Missing %": f"{pct_missing:.3f}%",
            "Unique Count": f"{n_unique:,}"
        })
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))

    # 7. Duplicate rows
    # Check duplicate rows across all columns
    print("\n7. Duplicate Rows Check:")
    num_dup_rows = df.duplicated().sum()
    print(f"   Total exact duplicate rows: {num_dup_rows:,} ({(num_dup_rows / num_rows * 100):.4f}%)")

    # 8. Duplicate entity IDs & 9. Entity ID prefix distribution
    if "entity_id" in df.columns:
        print("\n8. Duplicate Entity IDs:")
        dup_ids = df["entity_id"].duplicated().sum()
        print(f"   Duplicate entity_id count: {dup_ids:,}")

        print("\n9. Entity ID Prefix Distribution:")
        prefixes = df["entity_id"].dropna().astype(str).str.split("-").str[0].value_counts()
        for prefix, count in prefixes.items():
            print(f"   Prefix '{prefix}': {count:,} ({count / len(df) * 100:.2f}%)")
    elif "source1_entity_id" in df.columns:
        print("\n8. Duplicate Source1 Entity IDs:")
        dup_ids = df["source1_entity_id"].duplicated().sum()
        print(f"   Duplicate source1_entity_id count: {dup_ids:,}")

        print("\n9. Source1 Entity ID Prefix Distribution:")
        prefixes = df["source1_entity_id"].dropna().astype(str).str.split("-").str[0].value_counts()
        for prefix, count in prefixes.items():
            print(f"   Prefix '{prefix}': {count:,} ({count / len(df) * 100:.2f}%)")

    # 10. Country distribution
    if "country" in df.columns:
        print("\n10. Country Distribution:")
        country_counts = df["country"].value_counts(dropna=False)
        for country, count in country_counts.items():
            c_name = "<MISSING>" if pd.isna(country) else str(country)
            print(f"   {c_name}: {count:,} ({count / len(df) * 100:.2f}%)")

    # 11. Sample records
    print("\n11. Sample Records (First 3):")
    for idx, row in df.head(3).iterrows():
        print(f"   --- Record {idx} ---")
        for col in df.columns:
            print(f"     {col}: {repr(row[col])}")

    # 12 & 13. Business name and Address examples
    if "business_name" in df.columns:
        print("\n12. Sample Business Names:")
        sample_names = df["business_name"].dropna().sample(min(5, len(df)), random_state=42).tolist()
        for name_val in sample_names:
            print(f"     * {name_val}")

    if "business_address" in df.columns:
        print("\n13. Sample Addresses:")
        sample_addrs = df["business_address"].dropna().sample(min(5, len(df)), random_state=42).tolist()
        for addr_val in sample_addrs:
            print(f"     * {addr_val}")

    # 14. Text length statistics
    print("\n14. Text Length Statistics (Character Count):")
    for text_col in ["business_name", "business_address"]:
        if text_col in df.columns:
            lengths = df[text_col].dropna().astype(str).str.len()
            print(f"   [{text_col}]")
            print(f"     Min: {lengths.min()} | Max: {lengths.max()} | Mean: {lengths.mean():.2f} | Median: {lengths.median():.0f} | Std: {lengths.std():.2f}")


def inspect_ground_truth(gt_df: pd.DataFrame, s1_df: pd.DataFrame):
    print("=" * 80)
    print(" GROUND TRUTH & SOURCE 1 COVERAGE ANALYSIS")
    print("=" * 80)

    # 15. Basic consistency checks
    s1_ids = set(s1_df["entity_id"].dropna().unique())
    gt_s1_ids = set(gt_df["source1_entity_id"].dropna().unique())

    print("\n16. Source 1 vs Ground Truth ID Coverage:")
    print(f"   Unique entity_ids in train_source1: {len(s1_ids):,}")
    print(f"   Unique source1_entity_ids in train_ground_truth: {len(gt_s1_ids):,}")

    missing_in_gt = s1_ids - gt_s1_ids
    extra_in_gt = gt_s1_ids - s1_ids
    print(f"   Source 1 IDs missing in Ground Truth: {len(missing_in_gt):,}")
    print(f"   Ground Truth IDs not in Source 1: {len(extra_in_gt):,}")

    # Ground truth match parsing
    print("\n Ground Truth Match Cardinality Distribution:")
    gt_df["matched_entity_ids_clean"] = gt_df["matched_entity_ids"].fillna("").astype(str).str.strip()
    
    # Check empty vs non-empty
    is_empty = (gt_df["matched_entity_ids_clean"] == "")
    zero_matches_count = is_empty.sum()
    
    # Count matches per S1 entity
    match_counts = gt_df["matched_entity_ids_clean"].apply(
        lambda x: 0 if len(x) == 0 else len(x.split(","))
    )
    
    val_counts = match_counts.value_counts().sort_index()
    for num_m, count in val_counts.items():
        print(f"   Matches = {num_m:2d}: {count:8,d} ({count / len(gt_df) * 100:6.2f}%)")

    total_matches = match_counts.sum()
    print(f"\n   Total Match Pairs in Ground Truth: {total_matches:,}")
    print(f"   Average Matches per S1 Entity: {match_counts.mean():.3f}")

    # Breakdown of matched sources (S2 vs S3 vs others)
    print("\n Matched Entity ID Source Breakdown (S2 vs S3):")
    all_matched_tokens = []
    # Sample or iterate efficiently
    all_matched_list = gt_df[~is_empty]["matched_entity_ids_clean"].str.cat(sep=",").split(",")
    matched_prefixes = pd.Series([x.split("-")[0] for x in all_matched_list if "-" in x]).value_counts()
    for pfx, count in matched_prefixes.items():
        print(f"   Source Prefix '{pfx}': {count:,} ({count / len(all_matched_list) * 100:.2f}%)")
    
    unique_matched_ids = set(all_matched_list)
    print(f"   Unique noisy entity IDs matched: {len(unique_matched_ids):,}")


class TeeLogger:
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    base_dir = Path(__file__).resolve().parents[3]
    dataset_dir = base_dir / "dataset"
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    report_file = base_dir / "dataset_inspection_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print(f"Project Base Directory: {base_dir}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Report Output File: {report_file}\n")

    files_to_inspect = [
        ("Train Source 1", train_dir / "train_source1.tsv"),
        ("Train Source 2", train_dir / "train_source2.tsv"),
        ("Train Source 3", train_dir / "train_source3.tsv"),
        ("Train Ground Truth", train_dir / "train_ground_truth.tsv"),
        ("Test Source 1", test_dir / "test_source1.tsv"),
        ("Test Source 2", test_dir / "test_source2.tsv"),
        ("Test Source 3", test_dir / "test_source3.tsv"),
    ]

    s1_train_df = None
    gt_train_df = None

    for label, filepath in files_to_inspect:
        if not filepath.exists():
            print(f"WARNING: File not found: {filepath}")
            continue
        
        print(f"\nLoading {label} from {filepath.name} ({filepath.stat().st_size / (1024*1024):.1f} MB)...")
        # Read TSV using sep='\t'
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        
        inspect_dataframe(df, label)
        
        if label == "Train Source 1":
            s1_train_df = df
        elif label == "Train Ground Truth":
            gt_train_df = df
        else:
            del df
            gc.collect()

    if s1_train_df is not None and gt_train_df is not None:
        inspect_ground_truth(gt_train_df, s1_train_df)
        del s1_train_df, gt_train_df
        gc.collect()

    print("\n" + "=" * 80)
    print(" PHASE 1 INSPECTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
