"""
Phase 6: V3 Candidate Ranking Experiment (High-Speed Engine)
Amazon ML Challenge 2026 - Business Entity Resolution

This optimized module executes the candidate ranking stage across all 2,206,821 Source 1
entities, evaluates K in [25, 50, 75, 100] against V2 baseline, computes true-match rank distributions,
and generates reports/v3_ranking_audit.txt and reports/v3_lost_match_ranks.tsv.
"""

import sys
import gc
import re
import time
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
norm_module = importlib.import_module("04_normalization")
clean_str_input = norm_module.clean_str_input
strip_legal_suffixes = norm_module.strip_legal_suffixes
standardize_address_tokens = norm_module.standardize_address_tokens
clean_domain_name = norm_module.clean_domain_name

# Fast regexes
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_NON_ALPHA = re.compile(r"[^\w]", re.UNICODE)
RE_DIGITS = re.compile(r"\b\d+\b")

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

KEY_TYPE_WEIGHTS: Dict[str, float] = {
    "alpha": 1.25,   # Compact alphanumeric exact name match (highest specificity)
    "dom6": 1.15,    # Domain stem match
    "2tok": 1.05,    # First two name tokens bigram
    "pfx8": 0.95,    # 8-character name prefix
    "pfx6": 0.80,    # 6-character name prefix
    "anum": 0.75,    # Physical street number + street token
    "tok": 0.50      # Single distinct name token
}


def fast_extract_weighted_keys(name_val: str, addr_val: str, country_val: str) -> List[Tuple[str, float]]:
    """Extracts high-precision multi-pass blocking keys with pre-computed specificity weights."""
    keys: List[Tuple[str, float]] = []
    cntry = country_val.strip().upper() if country_val else "GLOBAL"
    if not cntry or cntry in ("NAN", "NULL", "NONE", "<MISSING>"):
        cntry = "GLOBAL"

    if name_val:
        s_low = name_val.lower()
        p_name = RE_PUNCT.sub(" ", s_low)
        alpha = RE_NON_ALPHA.sub("", s_low)
        toks = p_name.split()
        no_legal_toks = strip_legal_suffixes(toks)

        # Pass 1: Compact alphanumeric (>= 4)
        if len(alpha) >= 4:
            keys.append((f"{cntry}|alpha:{alpha}", KEY_TYPE_WEIGHTS["alpha"]))

        # Pass 2: Name prefixes (6 & 8 chars)
        if len(alpha) >= 6:
            keys.append((f"{cntry}|pfx6:{alpha[:6]}", KEY_TYPE_WEIGHTS["pfx6"]))
        if len(alpha) >= 8:
            keys.append((f"{cntry}|pfx8:{alpha[:8]}", KEY_TYPE_WEIGHTS["pfx8"]))

        # Pass 3: Distinct informative name tokens
        for tok in no_legal_toks:
            if len(tok) >= 4 and tok not in GENERIC_STOPWORDS and not tok.isdigit():
                keys.append((f"{cntry}|tok:{tok}", KEY_TYPE_WEIGHTS["tok"]))

        # Pass 4: First two tokens bigram
        if len(no_legal_toks) >= 2:
            t0, t1 = no_legal_toks[0], no_legal_toks[1]
            if t0 not in GENERIC_STOPWORDS or t1 not in GENERIC_STOPWORDS:
                keys.append((f"{cntry}|2tok:{t0}_{t1}", KEY_TYPE_WEIGHTS["2tok"]))

        # Pass 5: Domain name
        if ".com" in s_low or ".in" in s_low or ".org" in s_low or ".net" in s_low:
            d_name = clean_domain_name(s_low)
            d_alpha = RE_NON_ALPHA.sub("", d_name)
            if len(d_alpha) >= 6:
                keys.append((f"{cntry}|dom6:{d_alpha[:6]}", KEY_TYPE_WEIGHTS["dom6"]))

    if addr_val:
        s_low_a = addr_val.lower()
        p_addr = RE_PUNCT.sub(" ", s_low_a)

        # Pass 6: Street Number + First Non-Stopword Address Token
        addr_toks = p_addr.split()
        nums = RE_DIGITS.findall(p_addr)
        if nums and addr_toks:
            first_num = nums[0]
            std_toks = standardize_address_tokens(addr_toks)
            street_words = [t for t in std_toks if not t.isdigit() and len(t) >= 3 and t not in GENERIC_STOPWORDS]
            if street_words:
                keys.append((f"{cntry}|anum:{first_num}_{street_words[0]}", KEY_TYPE_WEIGHTS["anum"]))

    return keys


def main():
    t_global_start = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "dataset" / "train"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    v3_audit_txt_path = reports_dir / "v3_ranking_audit.txt"
    v3_lost_ranks_tsv_path = reports_dir / "v3_lost_match_ranks.tsv"

    print("=" * 90)
    print(" PHASE 6: V3 CANDIDATE RANKING EXPERIMENT (K = 25, 50, 75, 100)")
    print("=" * 90)
    print(f"Base Directory:           {base_dir}")
    print(f"Train Directory:          {train_dir}")
    print(f"Audit V3 Output Target:   {v3_audit_txt_path}")
    print(f"Lost Match Ranks Target:  {v3_lost_ranks_tsv_path}\n")

    # =========================================================================
    # PART 1: LOAD GROUND TRUTH
    # =========================================================================
    print("1. Loading ground truth positive match relations...")
    t0 = time.time()
    gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    s1_to_true_matches: Dict[str, Set[str]] = {}
    s1_to_true_s2: Dict[str, Set[str]] = {}
    s1_to_true_s3: Dict[str, Set[str]] = {}
    
    total_true_matches = 0
    total_true_s2 = 0
    total_true_s3 = 0
    s1_with_matches_count = 0
    s1_singleton_count = 0

    for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        s1_id = s1_id.strip()
        m_str = m_str.strip()
        if m_str:
            tokens = [t.strip() for t in m_str.split(",") if t.strip()]
            t_set = set(tokens)
            s1_to_true_matches[s1_id] = t_set
            
            s2_set = {t for t in tokens if t.startswith("S2-")}
            s3_set = {t for t in tokens if t.startswith("S3-")}
            s1_to_true_s2[s1_id] = s2_set
            s1_to_true_s3[s1_id] = s3_set
            
            total_true_matches += len(tokens)
            total_true_s2 += len(s2_set)
            total_true_s3 += len(s3_set)
            s1_with_matches_count += 1
        else:
            s1_to_true_matches[s1_id] = set()
            s1_to_true_s2[s1_id] = set()
            s1_to_true_s3[s1_id] = set()
            s1_singleton_count += 1

    del gt_df
    gc.collect()

    print(f"   - Loaded ground truth in {time.time() - t0:.2f}s")
    print(f"   - Total S1 Entities:         {len(s1_to_true_matches):,}")
    print(f"   - Total True Matches:        {total_true_matches:,} (S2: {total_true_s2:,}, S3: {total_true_s3:,})\n")

    # =========================================================================
    # PART 2: BUILD INVERTED INDEX OVER TRAIN SOURCE 2 & SOURCE 3
    # =========================================================================
    print("2. Building Multi-Pass Inverted Index over Train Source 2 & Source 3...")
    t0_idx = time.time()
    
    noisy_entity_ids: List[str] = []
    inverted_index: Dict[str, List[int]] = defaultdict(list)
    MAX_BUCKET_SIZE = 300

    noisy_record_idx = 0
    s2_count = 0
    s3_count = 0

    for s_idx, fname in [(2, "train_source2.tsv"), (3, "train_source3.tsv")]:
        path = train_dir / fname
        assert path.exists(), f"Missing file: {path}"
        print(f"   - Streaming {fname}...")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
            e_ids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            countries = chunk["country"].values

            for eid, name, addr, cntry in zip(e_ids, names, addrs, countries):
                noisy_entity_ids.append(eid)
                if s_idx == 2: s2_count += 1
                else: s3_count += 1

                keys = fast_extract_weighted_keys(name, addr, cntry)
                for k, _ in keys:
                    b = inverted_index[k]
                    if len(b) < MAX_BUCKET_SIZE:
                        b.append(noisy_record_idx)

                noisy_record_idx += 1

    # Prune oversized keys
    pruned_keys = 0
    for k in list(inverted_index.keys()):
        if len(inverted_index[k]) >= MAX_BUCKET_SIZE:
            del inverted_index[k]
            pruned_keys += 1

    idx_time = time.time() - t0_idx
    total_noisy = noisy_record_idx
    print(f"   - Indexed {total_noisy:,} noisy records in {idx_time:.2f}s")
    print(f"   - Active Inverted Keys: {len(inverted_index):,} (Pruned {pruned_keys:,})\n")

    # =========================================================================
    # PART 3: HIGH-SPEED VECTORIZED CANDIDATE RANKING & MULTI-K EVALUATION
    # =========================================================================
    print("3. Executing high-speed candidate ranking across all K in [25, 50, 75, 100] and V2...")
    t0_eval = time.time()

    K_VALUES = [25, 50, 75, 100]

    # Metrics collectors
    k_stats = {k: {
        "cand_counts": [],
        "total_cands": 0,
        "zero_cands": 0,
        "captured": 0,
        "captured_s2": 0,
        "captured_s3": 0,
        "captured_us": 0,
        "captured_india": 0,
        "full_retention": 0,
        "partial_retention": 0,
        "complete_loss": 0,
    } for k in K_VALUES}

    v2_stats = {
        "cand_counts": [],
        "total_cands": 0,
        "zero_cands": 0,
        "captured": 0,
        "captured_s2": 0,
        "captured_s3": 0,
        "captured_us": 0,
        "captured_india": 0,
        "full_retention": 0,
        "partial_retention": 0,
        "complete_loss": 0,
    }

    rank_bins = {
        "Rank 1-10": 0,
        "Rank 11-25": 0,
        "Rank 26-50": 0,
        "Rank 51-75": 0,
        "Rank 76-100": 0,
        "Rank >100": 0,
        "Not in V2": 0
    }

    true_match_ranks: List[int] = []
    diagnostic_lost_rank_samples = []
    MAX_DIAGNOSTIC_SAMPLES = 2000

    s1_train_path = train_dir / "train_source1.tsv"
    total_s1_processed = 0
    BATCH_SIZE = 100_000

    us_true_total = 4_578_522
    in_true_total = 3_059_843

    for chunk in pd.read_csv(s1_train_path, sep="\t", dtype=str, keep_default_na=False, chunksize=BATCH_SIZE):
        s1_eids = chunk["entity_id"].values
        s1_n = chunk["business_name"].values
        s1_a = chunk["business_address"].values
        s1_c = chunk["country"].values

        for eid, name, addr, cntry in zip(s1_eids, s1_n, s1_a, s1_c):
            true_matches = s1_to_true_matches.get(eid, set())
            true_s2 = s1_to_true_s2.get(eid, set())
            true_s3 = s1_to_true_s3.get(eid, set())
            num_true = len(true_matches)
            cntry_up = cntry.strip().upper() if cntry else "GLOBAL"
            is_us = (cntry_up == "US")
            is_india = (cntry_up == "INDIA")

            keys_with_weights = fast_extract_weighted_keys(name, addr, cntry)

            # Accumulate candidate multi-evidence scores
            cand_score_map: Dict[int, float] = defaultdict(float)
            for k, base_w in keys_with_weights:
                b = inverted_index.get(k)
                if b:
                    # Specificity bonus for compact/rare buckets
                    b_len = len(b)
                    w = base_w * (1.0 / math.log2(2.0 + b_len))
                    for idx_item in b:
                        cand_score_map[idx_item] += w

            v2_cand_count = len(cand_score_map)
            v2_stats["cand_counts"].append(v2_cand_count)
            v2_stats["total_cands"] += v2_cand_count
            if v2_cand_count == 0: v2_stats["zero_cands"] += 1

            # Sort candidate pool descending by multi-evidence ranking score
            ranked_items = sorted(cand_score_map.items(), key=lambda x: x[1], reverse=True)

            # Map noisy entity ID to rank (1-indexed)
            cand_id_to_rank: Dict[str, int] = {}
            ranked_eids: List[str] = []
            for r_idx, (idx_item, _) in enumerate(ranked_items, 1):
                c_eid = noisy_entity_ids[idx_item]
                cand_id_to_rank[c_eid] = r_idx
                ranked_eids.append(c_eid)

            # 1. Evaluate Unconstrained V2 Baseline
            v2_cand_eids_set = set(cand_id_to_rank.keys())
            if num_true > 0:
                v2_cap = true_matches & v2_cand_eids_set
                num_v2_cap = len(v2_cap)
                v2_stats["captured"] += num_v2_cap
                v2_stats["captured_s2"] += len(true_s2 & v2_cand_eids_set)
                v2_stats["captured_s3"] += len(true_s3 & v2_cand_eids_set)
                if is_us: v2_stats["captured_us"] += num_v2_cap
                if is_india: v2_stats["captured_india"] += num_v2_cap

                if num_v2_cap == num_true: v2_stats["full_retention"] += 1
                elif num_v2_cap > 0: v2_stats["partial_retention"] += 1
                else: v2_stats["complete_loss"] += 1

                # Track rank distribution for each ground-truth true match
                for tm_id in true_matches:
                    if tm_id in cand_id_to_rank:
                        tm_rank = cand_id_to_rank[tm_id]
                        true_match_ranks.append(tm_rank)
                        if tm_rank <= 10: rank_bins["Rank 1-10"] += 1
                        elif tm_rank <= 25: rank_bins["Rank 11-25"] += 1
                        elif tm_rank <= 50: rank_bins["Rank 26-50"] += 1
                        elif tm_rank <= 75: rank_bins["Rank 51-75"] += 1
                        elif tm_rank <= 100: rank_bins["Rank 76-100"] += 1
                        else: rank_bins["Rank >100"] += 1

                        if tm_rank > 100 and len(diagnostic_lost_rank_samples) < MAX_DIAGNOSTIC_SAMPLES:
                            diagnostic_lost_rank_samples.append({
                                "source1_entity_id": eid,
                                "s1_name": name,
                                "s1_address": addr,
                                "country": cntry,
                                "true_noisy_id": tm_id,
                                "true_match_rank": tm_rank,
                                "total_v2_candidates": v2_cand_count,
                                "reason": f"Ranked at position {tm_rank} in pool of {v2_cand_count} candidates"
                            })
                    else:
                        rank_bins["Not in V2"] += 1

            # 2. Evaluate K in [25, 50, 75, 100]
            for k_val in K_VALUES:
                k_selected_eids = ranked_eids[:k_val]
                k_count = len(k_selected_eids)
                ks = k_stats[k_val]
                ks["cand_counts"].append(k_count)
                ks["total_cands"] += k_count
                if k_count == 0: ks["zero_cands"] += 1

                if num_true > 0:
                    k_eids_set = set(k_selected_eids)
                    k_cap = true_matches & k_eids_set
                    num_k_cap = len(k_cap)
                    ks["captured"] += num_k_cap
                    ks["captured_s2"] += len(true_s2 & k_eids_set)
                    ks["captured_s3"] += len(true_s3 & k_eids_set)
                    if is_us: ks["captured_us"] += num_k_cap
                    if is_india: ks["captured_india"] += num_k_cap

                    if num_k_cap == num_true: ks["full_retention"] += 1
                    elif num_k_cap > 0: ks["partial_retention"] += 1
                    else: ks["complete_loss"] += 1

        total_s1_processed += len(s1_eids)
        if total_s1_processed % 500_000 == 0:
            print(f"   - Processed {total_s1_processed:,} / {len(s1_to_true_matches):,} S1 records in {time.time() - t0_eval:.1f}s ({total_s1_processed/(time.time()-t0_eval):,.0f} S1/s)...")

    eval_time = time.time() - t0_eval
    print(f"   - Candidate Ranking and Evaluation completed in {eval_time:.2f}s ({total_s1_processed/eval_time:,.0f} S1/s)\n")

    # =========================================================================
    # PART 4: COMPUTE AUDIT METRICS & WRITE REPORTS
    # =========================================================================
    print("4. Saving V3 ranking diagnostic reports...")
    
    # Save v3_lost_match_ranks.tsv
    df_lost_ranks = pd.DataFrame(diagnostic_lost_rank_samples)
    df_lost_ranks.to_csv(v3_lost_ranks_tsv_path, sep="\t", index=False)
    print(f"   - Saved {len(df_lost_ranks):,} rank diagnostic records to: {v3_lost_ranks_tsv_path.name}")

    tot_gt = total_true_matches
    N1 = len(s1_to_true_matches)

    # Rank metrics for captured true matches
    tm_ranks_arr = np.array(true_match_ranks) if true_match_ranks else np.array([1])
    med_rank = float(np.median(tm_ranks_arr))
    p75_rank = float(np.percentile(tm_ranks_arr, 75))
    p90_rank = float(np.percentile(tm_ranks_arr, 90))
    p95_rank = float(np.percentile(tm_ranks_arr, 95))
    p99_rank = float(np.percentile(tm_ranks_arr, 99))
    max_rank = int(np.max(tm_ranks_arr))

    # Top-K Rates (% of total ground truth true matches)
    rate_top10 = (rank_bins["Rank 1-10"] / tot_gt) * 100
    rate_top25 = ((rank_bins["Rank 1-10"] + rank_bins["Rank 11-25"]) / tot_gt) * 100
    rate_top50 = (k_stats[50]["captured"] / tot_gt) * 100
    rate_top75 = (k_stats[75]["captured"] / tot_gt) * 100
    rate_top100 = (k_stats[100]["captured"] / tot_gt) * 100

    v2_mean = float(np.mean(v2_stats["cand_counts"]))
    v2_p95 = float(np.percentile(v2_stats["cand_counts"], 95))
    v2_rec = (v2_stats["captured"] / tot_gt) * 100
    v2_full = (v2_stats["full_retention"] / s1_with_matches_count) * 100
    v2_s2 = (v2_stats["captured_s2"] / total_true_s2) * 100
    v2_s3 = (v2_stats["captured_s3"] / total_true_s3) * 100
    v2_us = (v2_stats["captured_us"] / us_true_total) * 100
    v2_in = (v2_stats["captured_india"] / in_true_total) * 100

    rec_k = {k: (k_stats[k]["captured"] / tot_gt * 100) for k in K_VALUES}
    full_k = {k: (k_stats[k]["full_retention"] / s1_with_matches_count * 100) for k in K_VALUES}
    s2_k = {k: (k_stats[k]["captured_s2"] / total_true_s2 * 100) for k in K_VALUES}
    s3_k = {k: (k_stats[k]["captured_s3"] / total_true_s3 * 100) for k in K_VALUES}
    us_k = {k: (k_stats[k]["captured_us"] / us_true_total * 100) for k in K_VALUES}
    in_k = {k: (k_stats[k]["captured_india"] / in_true_total * 100) for k in K_VALUES}

    with open(v3_audit_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write(" PHASE 6: V3 CANDIDATE RANKING EXPERIMENT AUDIT REPORT\n")
        f.write("=" * 95 + "\n\n")

        f.write("1. MULTI-K VS V2 BASELINE COMPARISON TABLE\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Metric':<25} | {'V2 (Unconstrained)':<18} | {'K = 25':<12} | {'K = 50':<12} | {'K = 75':<12} | {'K = 100':<12}\n")
        f.write("-" * 95 + "\n")

        f.write(f"{'Candidate pairs':<25} | {v2_stats['total_cands']:>18,d} | {k_stats[25]['total_cands']:>12,d} | {k_stats[50]['total_cands']:>12,d} | {k_stats[75]['total_cands']:>12,d} | {k_stats[100]['total_cands']:>12,d}\n")
        f.write(f"{'Mean candidates/S1':<25} | {v2_mean:>18.1f} | {np.mean(k_stats[25]['cand_counts']):>12.1f} | {np.mean(k_stats[50]['cand_counts']):>12.1f} | {np.mean(k_stats[75]['cand_counts']):>12.1f} | {np.mean(k_stats[100]['cand_counts']):>12.1f}\n")
        f.write(f"{'P95 candidates/S1':<25} | {v2_p95:>18.1f} | {np.percentile(k_stats[25]['cand_counts'], 95):>12.1f} | {np.percentile(k_stats[50]['cand_counts'], 95):>12.1f} | {np.percentile(k_stats[75]['cand_counts'], 95):>12.1f} | {np.percentile(k_stats[100]['cand_counts'], 95):>12.1f}\n")
        f.write(f"{'Pair recall':<25} | {v2_rec:>17.2f}% | {rec_k[25]:>11.2f}% | {rec_k[50]:>11.2f}% | {rec_k[75]:>11.2f}% | {rec_k[100]:>11.2f}%\n")
        f.write(f"{'Full entity retention':<25} | {v2_full:>17.2f}% | {full_k[25]:>11.2f}% | {full_k[50]:>11.2f}% | {full_k[75]:>11.2f}% | {full_k[100]:>11.2f}%\n")
        f.write(f"{'S1 -> S2 recall':<25} | {v2_s2:>17.2f}% | {s2_k[25]:>11.2f}% | {s2_k[50]:>11.2f}% | {s2_k[75]:>11.2f}% | {s2_k[100]:>11.2f}%\n")
        f.write(f"{'S1 -> S3 recall':<25} | {v2_s3:>17.2f}% | {s3_k[25]:>11.2f}% | {s3_k[50]:>11.2f}% | {s3_k[75]:>11.2f}% | {s3_k[100]:>11.2f}%\n")
        f.write(f"{'US recall':<25} | {v2_us:>17.2f}% | {us_k[25]:>11.2f}% | {us_k[50]:>11.2f}% | {us_k[75]:>11.2f}% | {us_k[100]:>11.2f}%\n")
        f.write(f"{'India recall':<25} | {v2_in:>17.2f}% | {in_k[25]:>11.2f}% | {in_k[50]:>11.2f}% | {in_k[75]:>11.2f}% | {in_k[100]:>11.2f}%\n")
        f.write("-" * 95 + "\n\n")

        f.write("2. CANDIDATE SET SUMMARY PER K\n")
        f.write("-" * 60 + "\n")
        for k_val in K_VALUES:
            ks = k_stats[k_val]
            arr = np.array(ks["cand_counts"])
            f.write(f"• K = {k_val:<3}: Total Pairs = {ks['total_cands']:11,d} | Mean = {np.mean(arr):5.1f} | Med = {np.median(arr):4.1f} | P90 = {np.percentile(arr, 90):4.1f} | P95 = {np.percentile(arr, 95):4.1f} | Max = {np.max(arr):3d}\n")
        f.write("\n")

        f.write("3. GROUND-TRUTH MATCH RANK DISTRIBUTION IN V2 POOL\n")
        f.write("-" * 60 + "\n")
        for r_bin, cnt in rank_bins.items():
            pct = (cnt / tot_gt) * 100
            f.write(f"• {r_bin:<15}: {cnt:10,d} true matches ({pct:5.2f}% of all GT matches)\n")
        f.write("\n")

        f.write("4. RANKING QUALITY DIAGNOSTIC METRICS (FOR CAPTURED MATCHES)\n")
        f.write("-" * 60 + "\n")
        f.write(f"• Median True-Match Rank:     {med_rank:6.1f}\n")
        f.write(f"• 75th Percentile Rank:        {p75_rank:6.1f}\n")
        f.write(f"• 90th Percentile Rank:        {p90_rank:6.1f}\n")
        f.write(f"• 95th Percentile Rank:        {p95_rank:6.1f}\n")
        f.write(f"• 99th Percentile Rank:        {p99_rank:6.1f}\n")
        f.write(f"• Maximum True-Match Rank:     {max_rank:6,d}\n\n")

        f.write(f"• True Match Top-10 Rate:      {rate_top10:6.2f}%\n")
        f.write(f"• True Match Top-25 Rate:      {rate_top25:6.2f}%\n")
        f.write(f"• True Match Top-50 Rate:      {rate_top50:6.2f}%\n")
        f.write(f"• True Match Top-75 Rate:      {rate_top75:6.2f}%\n")
        f.write(f"• True Match Top-100 Rate:     {rate_top100:6.2f}%\n\n")

        f.write("5. EXECUTIVE SUMMARY & CONCLUSION\n")
        f.write("-" * 60 + "\n")
        f.write(f"V2 BASELINE RECALL = {v2_rec:.2f}%\n\n")
        f.write(f"K=25 RECALL = {rec_k[25]:.2f}%\n")
        f.write(f"K=50 RECALL = {rec_k[50]:.2f}%\n")
        f.write(f"K=75 RECALL = {rec_k[75]:.2f}%\n")
        f.write(f"K=100 RECALL = {rec_k[100]:.2f}%\n\n")
        f.write(f"TRUE MATCH TOP-25 RATE = {rate_top25:.2f}%\n")
        f.write(f"TRUE MATCH TOP-50 RATE = {rate_top50:.2f}%\n")
        f.write(f"TRUE MATCH TOP-75 RATE = {rate_top75:.2f}%\n")
        f.write(f"TRUE MATCH TOP-100 RATE = {rate_top100:.2f}%\n")

    print(f"   - Successfully saved V3 ranking audit report to: {v3_audit_txt_path.name}\n")

    elapsed_all = time.time() - t_global_start
    print("=" * 90)
    print(f"Phase 6 V3 Ranking Experiment Completed in {elapsed_all:.2f}s ({elapsed_all/60:.2f} minutes)")
    print("=" * 90)


if __name__ == "__main__":
    main()
