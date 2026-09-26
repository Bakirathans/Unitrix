"""
Phase 7: Training Pair Generation & Hard-Negative Mining for Amazon ML Challenge 2026 - Business Entity Resolution
This module constructs clean, leak-free pair-level training and validation datasets.

Key Design Rules:
1. Strict Entity-Level Splitting: Split at the Source 1 entity level (never randomly at pair level).
   Zero S1 entity leakage between Train and Validation.
2. Ground-Truth Positives: Include all valid positive pairs (Label = 1).
3. Hard-Negative Mining: Mine candidates from blocking that have high name/address similarity
   or share distinct brand tokens but are NOT true matches (Label = 0).
4. Stratified Negative Sampling: Avoid overwhelming trivial negatives by sampling balanced
   hard and medium negatives (e.g., 3-5 negatives per positive).
5. Output Persistence: Save train_pairs and val_pairs to parquet/tsv for Phase 8 feature engineering.
"""

import sys
import gc
import re
import time
import random
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
from rapidfuzz import fuzz

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
norm_module = importlib.import_module("04_normalization")
gt_module = importlib.import_module("02_ground_truth")
block_module = importlib.import_module("05_blocking")

clean_str_input = norm_module.clean_str_input
normalize_record = norm_module.normalize_record
GroundTruthLookup = gt_module.GroundTruthLookup
CandidateBlocker = block_module.CandidateBlocker
fast_extract_keys_from_raw = block_module.fast_extract_keys_from_raw


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


def generate_labeled_pairs(
    sample_s1_size: int = 40000,
    neg_to_pos_ratio: int = 4,
    val_split_ratio: float = 0.20,
    random_seed: int = 42
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    t0 = time.time()

    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "dataset" / "train"
    out_dir = base_dir / "dataset" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = base_dir / "pair_generation_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 85)
    print(" PHASE 7: TRAINING PAIR GENERATION & HARD-NEGATIVE MINING")
    print("=" * 85)
    print(f"Project Base Directory: {base_dir}")
    print(f"Output Directory:       {out_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Step 1: Load Ground Truth
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print("1. Loading ground truth lookups...")
    gt_lookup = GroundTruthLookup.load_from_tsv(gt_path)

    # Step 2: Load S1 Entities and Split at Entity Level
    print("2. Loading Source 1 records and performing Entity-Level Train/Validation Split...")
    df_s1_all = pd.read_csv(s1_path, sep="\t", dtype=str)
    
    # Filter S1 entities that have at least one match for high-signal training pairs
    df_s1_with_matches = df_s1_all[df_s1_all["entity_id"].apply(lambda eid: len(gt_lookup.get_matches(eid)) > 0)]
    print(f"   Total S1 entities with ground-truth matches: {len(df_s1_with_matches):,}")

    if sample_s1_size and sample_s1_size < len(df_s1_with_matches):
        df_s1_sample = df_s1_with_matches.sample(n=sample_s1_size, random_state=random_seed)
    else:
        df_s1_sample = df_s1_with_matches

    all_sampled_s1_ids = df_s1_sample["entity_id"].tolist()
    random.shuffle(all_sampled_s1_ids)

    # Strict Entity-Level Split
    num_val_s1 = int(len(all_sampled_s1_ids) * val_split_ratio)
    val_s1_ids = set(all_sampled_s1_ids[:num_val_s1])
    train_s1_ids = set(all_sampled_s1_ids[num_val_s1:])

    print(f"   Split {len(all_sampled_s1_ids):,} S1 entities into:")
    print(f"     - Train S1 Entities:      {len(train_s1_ids):,} ({(1 - val_split_ratio)*100:.1f}%)")
    print(f"     - Validation S1 Entities: {len(val_s1_ids):,} ({val_split_ratio*100:.1f}%)")
    assert len(train_s1_ids & val_s1_ids) == 0, "DATA LEAKAGE ERROR: Overlapping S1 entities between Train and Validation!"

    # Index S1 records
    s1_records_dict = df_s1_sample.set_index("entity_id").to_dict(orient="index")
    del df_s1_all, df_s1_with_matches, df_s1_sample
    gc.collect()

    # Step 3: Stream and Index Noisy Sources
    print("\n3. Indexing Noisy Sources (S2 & S3) for Candidate Generation...")
    blocker = CandidateBlocker(max_bucket_size=500)
    t_idx = time.time()
    blocker.stream_index_tsv(s2_path)
    blocker.stream_index_tsv(s3_path)
    blocker.finalize_index()
    print(f"   Inverted Index built in {time.time() - t_idx:.2f}s.\n")

    # Step 4: Block S1 Entities to Get Candidate Sets
    print("4. Generating candidate pools for all sampled S1 entities...")
    s1_norm_objs = [
        normalize_record(eid, s1_records_dict[eid]["business_name"], s1_records_dict[eid]["business_address"], s1_records_dict[eid]["country"])
        for eid in all_sampled_s1_ids
    ]
    t_blk = time.time()
    s1_candidate_pools = blocker.block_s1_batch(s1_norm_objs, max_candidates_per_s1=150)
    print(f"   Blocking completed in {time.time() - t_blk:.2f}s.\n")

    # Step 5: Collect Noisy Attributes for Similarity Calculation
    print("5. Loading noisy attributes for candidate pair feature scoring...")
    needed_noisy_ids = set()
    for eid in all_sampled_s1_ids:
        # All true matches
        for m in gt_lookup.get_matches(eid):
            needed_noisy_ids.add(m)
        # All candidate matches
        for c in s1_candidate_pools.get(eid, set()):
            needed_noisy_ids.add(c)

    needed_s2 = {nid for nid in needed_noisy_ids if nid.startswith("S2-")}
    needed_s3 = {nid for nid in needed_noisy_ids if nid.startswith("S3-")}

    print(f"   Total unique noisy entities needed: {len(needed_noisy_ids):,} (S2: {len(needed_s2):,}, S3: {len(needed_s3):,})")

    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    s2_lookup = df_s2[df_s2["entity_id"].isin(needed_s2)].set_index("entity_id").to_dict(orient="index")
    del df_s2
    gc.collect()

    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
    s3_lookup = df_s3[df_s3["entity_id"].isin(needed_s3)].set_index("entity_id").to_dict(orient="index")
    del df_s3
    gc.collect()
    print("   Attribute retrieval complete.\n")

    # Step 6: Construct Positive, Negative, and Hard-Negative Pairs
    print("6. Constructing labeled training and validation pairs with hard-negative mining...")
    
    train_pairs: List[Dict[str, Any]] = []
    val_pairs: List[Dict[str, Any]] = []

    # Counters
    stats = {
        "train_pos": 0,
        "train_neg": 0,
        "train_hard_neg": 0,
        "val_pos": 0,
        "val_neg": 0,
        "val_hard_neg": 0,
    }

    for s1_rec in s1_norm_objs:
        s1_id = s1_rec.entity_id
        is_val = s1_id in val_s1_ids

        s1_n = s1_rec.original_business_name
        s1_a = s1_rec.original_business_address
        s1_c = s1_rec.country

        true_matches = set(gt_lookup.get_matches(s1_id))
        cand_set = s1_candidate_pools.get(s1_id, set())

        # 1. Positives: All true matches
        for m_id in true_matches:
            m_row = s2_lookup.get(m_id) if m_id.startswith("S2-") else s3_lookup.get(m_id)
            if not m_row:
                continue
            pair_obj = {
                "source1_entity_id": s1_id,
                "noisy_entity_id": m_id,
                "label": 1,
                "is_hard_negative": 0,
                "s1_name": s1_n,
                "s1_address": s1_a,
                "s1_country": s1_c,
                "noisy_name": clean_str_input(m_row.get("business_name")),
                "noisy_address": clean_str_input(m_row.get("business_address")),
                "noisy_country": clean_str_input(m_row.get("country"))
            }
            if is_val:
                val_pairs.append(pair_obj)
                stats["val_pos"] += 1
            else:
                train_pairs.append(pair_obj)
                stats["train_pos"] += 1

        # 2. Non-Matches from Candidate Pool (Mining Hard Negatives)
        non_matches = list(cand_set - true_matches)
        if not non_matches:
            continue

        scored_negatives = []
        for cand_id in non_matches:
            c_row = s2_lookup.get(cand_id) if cand_id.startswith("S2-") else s3_lookup.get(cand_id)
            if not c_row:
                continue
            c_name = clean_str_input(c_row.get("business_name"))
            c_addr = clean_str_input(c_row.get("business_address"))
            c_country = clean_str_input(c_row.get("country"))

            # Calculate fast similarity for difficulty categorization
            n_sim = fuzz.ratio(s1_n.lower(), c_name.lower()) if (s1_n and c_name) else 0
            a_sim = fuzz.ratio(s1_a.lower(), c_addr.lower()) if (s1_a and c_addr) else 0
            combined_score = max(n_sim, a_sim) * 0.7 + (n_sim + a_sim) * 0.15

            # Hard negative criteria: High similarity (n_sim >= 60 or a_sim >= 65 or combined >= 55)
            is_hard = 1 if (n_sim >= 60 or a_sim >= 65 or combined_score >= 55) else 0

            scored_negatives.append((
                combined_score,
                is_hard,
                cand_id,
                c_name,
                c_addr,
                c_country
            ))

        # Sort negatives by difficulty (highest similarity score first = hardest negatives)
        scored_negatives.sort(key=lambda x: x[0], reverse=True)

        # Target number of negatives for this S1 entity
        target_negs = max(2, len(true_matches) * neg_to_pos_ratio)
        selected_negs = scored_negatives[:target_negs]

        for score, is_hard, cand_id, c_name, c_addr, c_country in selected_negs:
            pair_obj = {
                "source1_entity_id": s1_id,
                "noisy_entity_id": cand_id,
                "label": 0,
                "is_hard_negative": is_hard,
                "s1_name": s1_n,
                "s1_address": s1_a,
                "s1_country": s1_c,
                "noisy_name": c_name,
                "noisy_address": c_addr,
                "noisy_country": c_country
            }
            if is_val:
                val_pairs.append(pair_obj)
                stats["val_neg"] += 1
                if is_hard: stats["val_hard_neg"] += 1
            else:
                train_pairs.append(pair_obj)
                stats["train_neg"] += 1
                if is_hard: stats["train_hard_neg"] += 1

    # Convert to DataFrames
    df_train_pairs = pd.DataFrame(train_pairs)
    df_val_pairs = pd.DataFrame(val_pairs)

    # Save to disk
    train_parquet_path = out_dir / "train_pairs.parquet"
    val_parquet_path = out_dir / "val_pairs.parquet"
    train_tsv_path = out_dir / "train_pairs_sample.tsv"
    val_tsv_path = out_dir / "val_pairs_sample.tsv"

    print("7. Persisting pair datasets to disk...")
    df_train_pairs.to_parquet(train_parquet_path, index=False)
    df_val_pairs.to_parquet(val_parquet_path, index=False)

    # Also save head sample TSVs for inspection
    df_train_pairs.head(100).to_csv(train_tsv_path, sep="\t", index=False)
    df_val_pairs.head(100).to_csv(val_tsv_path, sep="\t", index=False)

    print(f"   Saved {len(df_train_pairs):,} train pairs to: {train_parquet_path.name}")
    print(f"   Saved {len(df_val_pairs):,} val pairs to:   {val_parquet_path.name}\n")

    # Step 7: Print Report
    total_train = len(df_train_pairs)
    total_val = len(df_val_pairs)
    total_all = total_train + total_val

    print("=" * 85)
    print(" TRAINING & VALIDATION PAIR GENERATION SUMMARY")
    print("=" * 85)
    print(f"1. Entity-Level Split Breakdown:")
    print(f"   - Total S1 Entities Sampled:               {len(all_sampled_s1_ids):10,d}")
    print(f"   - Train S1 Entities:                       {len(train_s1_ids):10,d} ({(len(train_s1_ids)/len(all_sampled_s1_ids))*100:.1f}%)")
    print(f"   - Validation S1 Entities:                  {len(val_s1_ids):10,d} ({(len(val_s1_ids)/len(all_sampled_s1_ids))*100:.1f}%)")
    print("-------------------------------------------------------------------------------------")
    print(f"2. Training Pair Statistics (N = {total_train:,}):")
    print(f"   - Positive Pairs (Label = 1):              {stats['train_pos']:10,d} ({(stats['train_pos']/total_train*100):5.2f}%)")
    print(f"   - Negative Pairs (Label = 0):              {stats['train_neg']:10,d} ({(stats['train_neg']/total_train*100):5.2f}%)")
    print(f"   - Hard-Negative Count:                     {stats['train_hard_neg']:10,d} ({(stats['train_hard_neg']/stats['train_neg']*100):5.2f}% of negatives)")
    print(f"   - Class Balance (Pos : Neg):               1 : {(stats['train_neg']/stats['train_pos']):.2f}")
    print("-------------------------------------------------------------------------------------")
    print(f"3. Validation Pair Statistics (N = {total_val:,}):")
    print(f"   - Positive Pairs (Label = 1):              {stats['val_pos']:10,d} ({(stats['val_pos']/total_val*100):5.2f}%)")
    print(f"   - Negative Pairs (Label = 0):              {stats['val_neg']:10,d} ({(stats['val_neg']/total_val*100):5.2f}%)")
    print(f"   - Hard-Negative Count:                     {stats['val_hard_neg']:10,d} ({(stats['val_hard_neg']/stats['val_neg']*100):5.2f}% of negatives)")
    print(f"   - Class Balance (Pos : Neg):               1 : {(stats['val_neg']/stats['val_pos']):.2f}")
    print("=====================================================================================\n")

    # Step 8: Show Examples of Hard Negatives vs True Positives
    print("Hard-Negative vs True-Positive Comparison Examples:")
    sample_hard_negs = df_train_pairs[df_train_pairs["is_hard_negative"] == 1].head(3)
    for idx, (_, r) in enumerate(sample_hard_negs.iterrows(), 1):
        print(f"\n  [Hard Negative Example {idx}]")
        print(f"    S1 Entity [{r['source1_entity_id']}]:   '{r['s1_name']}' | Addr: '{r['s1_address']}' ({r['s1_country']})")
        print(f"    Noisy Match [{r['noisy_entity_id']}]: '{r['noisy_name']}' | Addr: '{r['noisy_address']}' ({r['noisy_country']})")
        print(f"    Label: {r['label']} (HARD NEGATIVE - High surface similarity but distinct real-world entity)")

    elapsed = time.time() - t0
    print("\n" + "=" * 85)
    print(f"Total Elapsed Time: {elapsed:.2f} seconds")
    print(" PHASE 7 TRAINING PAIR GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 85)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    generate_labeled_pairs(sample_s1_size=40000, neg_to_pos_ratio=4, val_split_ratio=0.20)
