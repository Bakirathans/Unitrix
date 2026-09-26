"""
Phase 5: Candidate Generation / Blocking for Amazon ML Challenge 2026 - Business Entity Resolution
This module implements high-performance multi-pass candidate blocking to reduce the O(N * M)
comparison space down to a high-recall, manageable candidate set.

Key Design Rules:
1. S1 entities only generate candidate matches from noisy sources (S2/S3).
2. Never generate S1-S1 pairs.
3. Multi-pass Union: Combines exact, prefix, token, address, and domain keys.
4. Bucket Capping: Prunes over-frequent generic keys to prevent combinatorial explosion.
5. Reusable for both Training and Inference/Test (No Ground Truth used in candidate generation).
"""

import sys
import gc
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Iterable
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

# Add src to path for importing normalization and ground truth modules
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
norm_module = importlib.import_module("04_normalization")
gt_module = importlib.import_module("02_ground_truth")

normalize_record = norm_module.normalize_record
build_name_representations = norm_module.build_name_representations
build_address_representations = norm_module.build_address_representations
normalize_country = norm_module.normalize_country
clean_str_input = norm_module.clean_str_input
get_alphanumeric_compact = norm_module.get_alphanumeric_compact
clean_domain_name = norm_module.clean_domain_name
strip_legal_suffixes = norm_module.strip_legal_suffixes
standardize_address_tokens = norm_module.standardize_address_tokens
tokenize = norm_module.tokenize
NormalizedEntityRecord = norm_module.NormalizedEntityRecord

parse_matched_entity_ids = gt_module.parse_matched_entity_ids
GroundTruthLookup = gt_module.GroundTruthLookup


# Fast compiled C regexes
RE_PUNCT_FAST = re.compile(r"[^\w\s]", re.UNICODE)
RE_NON_ALPHA_FAST = re.compile(r"[^\w]", re.UNICODE)
RE_DIGITS = re.compile(r"\b\d+\b")
RE_ZIPCODE = re.compile(r"\b\d{5,6}\b")

# High-frequency generic words across corporate entities to suppress in single-token blocking
GENERIC_STOPWORDS: Set[str] = {
    "and", "the", "of", "in", "for", "on", "at", "to", "a", "an",
    "services", "service", "center", "centre", "solutions", "solution",
    "enterprises", "enterprise", "group", "holdings", "holding", "industries",
    "associates", "consultants", "consulting", "management", "international",
    "national", "global", "direct", "trade", "traders", "trading", "commercial",
    "company", "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "pvt", "private", "llp", "gmbh", "co", "dept", "division", "agency", "tech",
    "technologies", "technology", "systems", "system", "products", "production",
    "care", "health", "medical", "clinic", "hospital", "store", "shop", "mart"
}


def fast_extract_keys_from_raw(
    raw_name: Any,
    raw_addr: Any,
    raw_country: Any
) -> Set[str]:
    """
    High-speed, stream-friendly extraction of multi-pass blocking keys directly from raw field inputs.
    Utilizes compiled C regexes for sub-microsecond throughput across millions of records.
    """
    keys: Set[str] = set()
    cntry = str(raw_country).strip().upper() if raw_country is not None and not pd.isna(raw_country) else ""
    country_prefix = cntry if cntry and cntry not in ("NAN", "NULL", "NONE", "<MISSING>") else "GLOBAL"

    if raw_name is not None and not pd.isna(raw_name):
        s_name = str(raw_name).strip()
        if s_name and s_name.lower() not in ("nan", "null", "none"):
            s_low = s_name.lower()
            p_name = RE_PUNCT_FAST.sub(" ", s_low)
            alpha = RE_NON_ALPHA_FAST.sub("", s_low)
            toks = p_name.split()
            no_legal_toks = strip_legal_suffixes(toks)

            # Pass 1: Exact / Alphanumeric Name
            if alpha and len(alpha) >= 4:
                keys.add(f"{country_prefix}|alpha:{alpha}")

            # Pass 2: Name Prefixes (6 and 8 chars)
            if len(alpha) >= 6:
                keys.add(f"{country_prefix}|pfx6:{alpha[:6]}")
            if len(alpha) >= 8:
                keys.add(f"{country_prefix}|pfx8:{alpha[:8]}")

            # Pass 3: Distinct Informative Name Tokens (length >= 4)
            for tok in no_legal_toks:
                if len(tok) >= 4 and tok not in GENERIC_STOPWORDS and not tok.isdigit():
                    keys.add(f"{country_prefix}|tok:{tok}")

            # Pass 4: First Two Name Tokens (Bi-gram prefix)
            if len(no_legal_toks) >= 2:
                t0, t1 = no_legal_toks[0], no_legal_toks[1]
                if t0 not in GENERIC_STOPWORDS or t1 not in GENERIC_STOPWORDS:
                    keys.add(f"{country_prefix}|2tok:{t0}_{t1}")

            # Pass 5: Domain Name Cleaning
            if ".com" in s_low or ".in" in s_low or ".org" in s_low or ".net" in s_low:
                d_name = clean_domain_name(s_low)
                d_p = RE_PUNCT_FAST.sub(" ", d_name)
                d_alpha = RE_NON_ALPHA_FAST.sub("", d_name)
                if len(d_alpha) >= 6:
                    keys.add(f"{country_prefix}|dom6:{d_alpha[:6]}")
                for d_tok in d_p.split():
                    if len(d_tok) >= 4 and d_tok not in GENERIC_STOPWORDS:
                        keys.add(f"{country_prefix}|tok:{d_tok}")

    if raw_addr is not None and not pd.isna(raw_addr):
        s_addr = str(raw_addr).strip()
        if s_addr and s_addr.lower() not in ("nan", "null", "none"):
            s_low_addr = s_addr.lower()
            p_addr = RE_PUNCT_FAST.sub(" ", s_low_addr)

            # Pass 6A: Zip/Postal Code
            zips = RE_ZIPCODE.findall(p_addr)
            for z in zips:
                keys.add(f"{country_prefix}|zip:{z}")

            # Pass 6B: Street Number + First Non-Stopword Address Token
            addr_toks = p_addr.split()
            nums = RE_DIGITS.findall(p_addr)
            if nums and addr_toks:
                first_num = nums[0]
                std_toks = standardize_address_tokens(addr_toks)
                street_words = [t for t in std_toks if not t.isdigit() and len(t) >= 3 and t not in GENERIC_STOPWORDS]
                if street_words:
                    keys.add(f"{country_prefix}|anum:{first_num}_{street_words[0]}")

    return keys


def extract_blocking_keys(record: NormalizedEntityRecord) -> Set[str]:
    """Extract blocking keys from a NormalizedEntityRecord object."""
    return fast_extract_keys_from_raw(
        record.original_business_name,
        record.original_business_address,
        record.original_country
    )


class CandidateBlocker:
    """
    High-performance Multi-Pass Inverted Index Blocker.
    Indexes noisy records (S2/S3) and queries S1 records to retrieve deduplicated candidate sets.
    """
    def __init__(self, max_bucket_size: int = 500):
        self.max_bucket_size = max_bucket_size
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        self.noisy_count: int = 0
        self.indexed_key_count: int = 0

    def stream_index_tsv(self, tsv_path: Path | str):
        """Fast streaming index builder from a TSV file without huge memory allocations."""
        t_start = time.time()
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        eids = df["entity_id"].tolist()
        b_names = df["business_name"].tolist()
        b_addrs = df["business_address"].tolist()
        cntrys = df["country"].tolist()
        del df
        gc.collect()

        count = 0
        for eid, name, addr, country in zip(eids, b_names, b_addrs, cntrys):
            count += 1
            eid_str = str(eid).strip()
            keys = fast_extract_keys_from_raw(name, addr, country)
            for k in keys:
                self.inverted_index[k].append(eid_str)

        self.noisy_count += count
        elapsed = time.time() - t_start
        print(f"   [Blocker] Streamed {count:,} records from {Path(tsv_path).name} in {elapsed:.1f}s. Total indexed: {self.noisy_count:,}")

    def finalize_index(self):
        """Prune oversized buckets to prevent candidate combinatorial explosion."""
        self.indexed_key_count = len(self.inverted_index)
        pruned_keys = 0
        for k, bucket in list(self.inverted_index.items()):
            if len(bucket) > self.max_bucket_size:
                del self.inverted_index[k]
                pruned_keys += 1

        print(f"   [Blocker] Index Finalized: {self.noisy_count:,} records across {self.indexed_key_count:,} total keys.")
        print(f"   [Blocker] Pruned {pruned_keys:,} oversized keys (> {self.max_bucket_size} entries). Active keys: {len(self.inverted_index):,}.")

    def get_candidates_for_s1(self, s1_rec: NormalizedEntityRecord, max_candidates: int = 150) -> Set[str]:
        """
        Generate candidate noisy IDs (S2/S3) for a single S1 record.
        Guaranteed:
        - Only returns S2/S3 IDs (never S1).
        - Returns a deduplicated set.
        """
        keys = extract_blocking_keys(s1_rec)
        candidates: Set[str] = set()

        for k in keys:
            bucket = self.inverted_index.get(k)
            if bucket:
                candidates.update(bucket)
                if len(candidates) >= max_candidates * 3:
                    break

        return candidates

    def block_s1_batch(
        self,
        s1_records: List[NormalizedEntityRecord],
        max_candidates_per_s1: int = 150
    ) -> Dict[str, Set[str]]:
        """Block an entire batch of S1 records against the inverted index."""
        result: Dict[str, Set[str]] = {}
        for s1_rec in s1_records:
            cands = self.get_candidates_for_s1(s1_rec, max_candidates=max_candidates_per_s1)
            result[s1_rec.entity_id] = cands
        return result


class TeeLogger:
    """Tee stdout to console and file with auto-flush."""
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


def run_blocking_evaluation(sample_s1_size: int = 50000, random_seed: int = 42):
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "dataset" / "train"
    report_file = base_dir / "blocking_evaluation_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 85)
    print(" PHASE 5: CANDIDATE GENERATION / BLOCKING EVALUATION")
    print("=" * 85)
    print(f"Project Base Directory: {base_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Load Ground Truth for Recall Evaluation
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print("1. Loading Ground Truth lookup structure...")
    gt_lookup = GroundTruthLookup.load_from_tsv(gt_path)

    # Load S1 entities
    print("2. Loading and normalizing Source 1 records...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    total_s1_count = len(df_s1)
    
    if sample_s1_size and sample_s1_size < total_s1_count:
        df_s1_sample = df_s1.sample(n=sample_s1_size, random_state=random_seed)
        print(f"   Sampled {len(df_s1_sample):,} S1 records for benchmarking (out of {total_s1_count:,} total).")
    else:
        df_s1_sample = df_s1
        print(f"   Evaluating all {len(df_s1_sample):,} S1 records.")

    s1_norm_list = [
        normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
        for _, row in df_s1_sample.iterrows()
    ]
    del df_s1, df_s1_sample
    gc.collect()

    # Stream index noisy S2 & S3 datasets
    print("\n3. Streaming and indexing noisy S2 & S3 datasets into Multi-Pass Inverted Index...")
    blocker = CandidateBlocker(max_bucket_size=500)
    
    t_idx = time.time()
    blocker.stream_index_tsv(s2_path)
    blocker.stream_index_tsv(s3_path)
    blocker.finalize_index()
    print(f"   Total indexing time: {time.time() - t_idx:.2f}s.\n")

    total_noisy_pool = blocker.noisy_count

    # Execute Candidate Generation
    print("4. Generating candidates for S1 benchmark batch...")
    t_block = time.time()
    s1_candidates = blocker.block_s1_batch(s1_norm_list, max_candidates_per_s1=150)
    elapsed_block = time.time() - t_block
    print(f"   Candidate generation completed in {elapsed_block:.2f}s ({len(s1_norm_list)/elapsed_block:.1f} S1/sec).\n")

    # Calculate Candidate Statistics & Recall
    print("5. Computing candidate statistics and True-Match Recall...")
    candidate_counts = [len(cands) for cands in s1_candidates.values()]
    total_generated_candidates = sum(candidate_counts)
    num_s1_eval = len(s1_norm_list)

    avg_candidates_per_s1 = total_generated_candidates / num_s1_eval if num_s1_eval else 0
    max_candidates_per_s1 = max(candidate_counts) if candidate_counts else 0
    min_candidates_per_s1 = min(candidate_counts) if candidate_counts else 0
    median_candidates_per_s1 = np.median(candidate_counts) if candidate_counts else 0

    theoretical_comparisons = num_s1_eval * total_noisy_pool
    reduction_ratio = (1.0 - (total_generated_candidates / theoretical_comparisons)) * 100

    # Recall vs Ground Truth
    total_true_matches_in_sample = 0
    captured_true_matches = 0
    s1_with_any_true_matches = 0
    s1_with_all_true_matches_captured = 0
    s1_with_at_least_one_captured = 0

    for s1_rec in s1_norm_list:
        s1_id = s1_rec.entity_id
        true_matches = set(gt_lookup.get_matches(s1_id))
        num_true = len(true_matches)
        
        if num_true > 0:
            total_true_matches_in_sample += num_true
            s1_with_any_true_matches += 1
            
            cand_set = s1_candidates[s1_id]
            matched_cands = true_matches & cand_set
            num_captured = len(matched_cands)
            captured_true_matches += num_captured

            if num_captured == num_true:
                s1_with_all_true_matches_captured += 1
            if num_captured > 0:
                s1_with_at_least_one_captured += 1

    pair_recall = (captured_true_matches / total_true_matches_in_sample * 100) if total_true_matches_in_sample else 0
    entity_full_recall = (s1_with_all_true_matches_captured / s1_with_any_true_matches * 100) if s1_with_any_true_matches else 0
    entity_any_recall = (s1_with_at_least_one_captured / s1_with_any_true_matches * 100) if s1_with_any_true_matches else 0

    # --- PRINT CANDIDATE GENERATION REPORT ---
    print("=" * 85)
    print(" BLOCKING & CANDIDATE GENERATION RESULTS (Sample N = {:,})".format(num_s1_eval))
    print("=" * 85)
    print(f"1. Total S1 Entities Evaluated:             {num_s1_eval:12,d}")
    print(f"2. Total Noisy Search Pool (S2 + S3):       {total_noisy_pool:12,d}")
    print(f"3. Theoretical All-Pairs Comparisons:       {theoretical_comparisons:12,d}")
    print(f"4. Total Candidates Generated (Union):      {total_generated_candidates:12,d}")
    print(f"5. Candidate Reduction Ratio:               {reduction_ratio:11.6f}%")
    print(f"   (Search space reduced by a factor of {theoretical_comparisons/total_generated_candidates:,.1f}x)")
    print("-------------------------------------------------------------------------------------")
    print(f"6. Candidates per S1 Entity Distribution:")
    print(f"   - Minimum Candidates:                    {min_candidates_per_s1:12,d}")
    print(f"   - Maximum Candidates:                    {max_candidates_per_s1:12,d}")
    print(f"   - Mean Candidates:                       {avg_candidates_per_s1:12.2f}")
    print(f"   - Median Candidates:                     {median_candidates_per_s1:12.1f}")
    print(f"   - 75th Percentile:                       {np.percentile(candidate_counts, 75):12.1f}")
    print(f"   - 90th Percentile:                       {np.percentile(candidate_counts, 90):12.1f}")
    print(f"   - 95th Percentile:                       {np.percentile(candidate_counts, 95):12.1f}")
    print("-------------------------------------------------------------------------------------")
    print(f"7. Ground-Truth True-Match Recall on Sample:")
    print(f"   - Total True Matches in Ground Truth:    {total_true_matches_in_sample:12,d}")
    print(f"   - True Matches Captured in Candidates:   {captured_true_matches:12,d}")
    print(f"   - True-Match Pair Recall:                {pair_recall:11.2f}%")
    print(f"   - S1 Entities with Full Recall:          {s1_with_all_true_matches_captured:12,d} / {s1_with_any_true_matches:,} ({entity_full_recall:.2f}%)")
    print(f"   - S1 Entities with >=1 True Match:       {s1_with_at_least_one_captured:12,d} / {s1_with_any_true_matches:,} ({entity_any_recall:.2f}%)")
    print("=====================================================================================")

    # Show Candidate Examples
    print("\nCandidate Generation Sample Lookups:")
    for idx in range(min(4, len(s1_norm_list))):
        s1_rec = s1_norm_list[idx]
        s1_id = s1_rec.entity_id
        cands = list(s1_candidates[s1_id])
        true_m = gt_lookup.get_matches(s1_id)
        captured = [c for c in cands if c in true_m]
        print(f"\n  [Sample {idx+1}] S1: [{s1_id}] '{s1_rec.original_business_name}' ({s1_rec.country})")
        print(f"    - Address:    '{s1_rec.original_business_address}'")
        print(f"    - Generated Candidates ({len(cands)} total): {cands[:8]}{'...' if len(cands) > 8 else ''}")
        print(f"    - True Ground Truth Matches: {true_m}")
        print(f"    - True Matches Captured:     {captured} ({len(captured)}/{len(true_m)})")

    elapsed_total = time.time() - t0
    print("\n" + "=" * 85)
    print(f"Total Elapsed Time: {elapsed_total:.2f} seconds")
    print(" PHASE 5 CANDIDATE GENERATION & BLOCKING COMPLETED SUCCESSFULLY")
    print("=" * 85)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_blocking_evaluation(sample_s1_size=50000)
