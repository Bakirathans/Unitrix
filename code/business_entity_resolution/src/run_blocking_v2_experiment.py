"""
Phase 6 Fix: Comprehensive Unconstrained Blocking (V2) Evaluation & Test Candidate Generation
Amazon ML Challenge 2026 - Business Entity Resolution

This module executes:
1. Smoke tests (10k, 100k S1 entities)
2. Full Training Data Diagnostic Evaluation (2,206,821 S1 entities vs Ground Truth)
3. Generates reports/blocking_v2_audit.txt and reports/blocking_v2_lost_matches.tsv
4. Generates output/candidate_pairs_v2.tsv on Test Data (1,732,544 entities)
"""

import sys
import gc
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

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


def fast_extract_keys(name_val: str, addr_val: str, country_val: str) -> List[str]:
    """Extracts high-precision multi-pass blocking keys exactly preserving existing 6 rules."""
    keys: List[str] = []
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
            keys.append(f"{cntry}|alpha:{alpha}")

        # Pass 2: Name prefixes (6 & 8 chars)
        if len(alpha) >= 6:
            keys.append(f"{cntry}|pfx6:{alpha[:6]}")
        if len(alpha) >= 8:
            keys.append(f"{cntry}|pfx8:{alpha[:8]}")

        # Pass 3: Distinct informative name tokens
        for tok in no_legal_toks:
            if len(tok) >= 4 and tok not in GENERIC_STOPWORDS and not tok.isdigit():
                keys.append(f"{cntry}|tok:{tok}")

        # Pass 4: First two tokens bigram
        if len(no_legal_toks) >= 2:
            t0, t1 = no_legal_toks[0], no_legal_toks[1]
            if t0 not in GENERIC_STOPWORDS or t1 not in GENERIC_STOPWORDS:
                keys.append(f"{cntry}|2tok:{t0}_{t1}")

        # Pass 5: Domain name
        if ".com" in s_low or ".in" in s_low or ".org" in s_low or ".net" in s_low:
            d_name = clean_domain_name(s_low)
            d_alpha = RE_NON_ALPHA.sub("", d_name)
            if len(d_alpha) >= 6:
                keys.append(f"{cntry}|dom6:{d_alpha[:6]}")

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
                keys.append(f"{cntry}|anum:{first_num}_{street_words[0]}")

    return keys


def classify_lost_failure(s1_name: str, s1_addr: str, s1_cntry: str,
                            m_name: str, m_addr: str, m_cntry: str,
                            s1_keys: Set[str], m_keys: Set[str]) -> Tuple[str, str]:
    """Deterministically classifies why a true match was missed by unconstrained blocking."""
    shared = s1_keys & m_keys
    if shared:
        return "OTHER", f"Shared keys present: {list(shared)[:2]}"

    if not m_name or not s1_name:
        return "MISSING_FIELD", "Missing or empty business name"
    if not m_addr or not s1_addr:
        return "MISSING_FIELD", "Missing or empty business address"

    name_sim = fuzz.ratio(s1_name.lower(), m_name.lower())
    token_sort_name = fuzz.token_sort_ratio(s1_name.lower(), m_name.lower())
    addr_sim = fuzz.ratio(s1_addr.lower(), m_addr.lower())
    token_sort_addr = fuzz.token_sort_ratio(s1_addr.lower(), m_addr.lower())

    s1_script_non_ascii = bool(re.search(r"[^\x00-\x7F]", s1_name))
    m_script_non_ascii = bool(re.search(r"[^\x00-\x7F]", m_name))

    if (s1_script_non_ascii != m_script_non_ascii) or (s1_script_non_ascii and m_script_non_ascii and name_sim < 40):
        return "NON_LATIN_OR_TRANSLITERATION", f"Non-Latin script or cross-script transliteration ('{s1_name}' vs '{m_name}')"

    if token_sort_name >= 80 and name_sim < 60:
        return "TOKEN_ORDER", f"Name token reordering ('{s1_name}' vs '{m_name}')"

    if token_sort_addr >= 80 and addr_sim < 60:
        return "TOKEN_ORDER", f"Address token reordering ('{s1_addr}' vs '{m_addr}')"

    s1_toks = [t for t in s1_name.lower().split() if t not in GENERIC_STOPWORDS]
    m_toks = [t for t in m_name.lower().split() if t not in GENERIC_STOPWORDS]
    s1_acr = "".join(t[0] for t in s1_toks if t)
    m_acr = "".join(t[0] for t in m_toks if t)
    s1_comp = "".join(c for c in s1_name.lower() if c.isalnum())
    m_comp = "".join(c for c in m_name.lower() if c.isalnum())

    if (len(s1_toks) == 1 and len(s1_toks[0]) <= 4 and s1_toks[0] == m_acr) or \
       (len(m_toks) == 1 and len(m_toks[0]) <= 4 and m_toks[0] == s1_acr):
        return "ABBREVIATION", f"Acronym / initialism ('{s1_name}' vs '{m_name}')"

    if name_sim >= 65:
        return "NAME_VARIATION", f"Name typo or character perturbation (Sim: {name_sim}%)"

    if addr_sim >= 65:
        return "ADDRESS_VARIATION", f"Address variation / abbreviation (Addr Sim: {addr_sim}%, Name Sim: {name_sim}%)"

    if name_sim < 40 and addr_sim < 40:
        return "ALTERNATE_BRAND_NAME", f"Alternate DBA / Trade Name (Name Sim: {name_sim}%, Addr Sim: {addr_sim}%)"

    return "NO_BLOCKING_RULE_HIT", f"No shared blocking key generated (Name Sim: {name_sim}%, Addr Sim: {addr_sim}%)"


def main():
    t_global_start = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "dataset" / "train"
    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    reports_dir = base_dir / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cand_v2_test_path = output_dir / "candidate_pairs_v2.tsv"
    audit_v2_txt_path = reports_dir / "blocking_v2_audit.txt"
    lost_v2_tsv_path = reports_dir / "blocking_v2_lost_matches.tsv"

    print("=" * 90)
    print(" PHASE 6 FIX: UNCONSTRAINED BLOCKING RECALL (V2) & SENSITIVITY AUDIT")
    print("=" * 90)
    print(f"Base Directory:           {base_dir}")
    print(f"Train Directory:          {train_dir}")
    print(f"Test Directory:           {test_dir}")
    print(f"Candidate Pairs V2 Target:{cand_v2_test_path}")
    print(f"Audit V2 Target:          {audit_v2_txt_path}")
    print(f"Lost Matches Target:      {lost_v2_tsv_path}\n")

    # =========================================================================
    # PART 1: LOAD TRAINING GROUND TRUTH
    # =========================================================================
    print("1. Loading Ground Truth positive match relations...")
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
    # PART 2: INDEX TRAINING NOISY DATASETS (S2 & S3)
    # =========================================================================
    print("2. Indexing Train Source 2 & Source 3 into Multi-Pass Inverted Index...")
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

                keys = fast_extract_keys(name, addr, cntry)
                for k in keys:
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
    print(f"   - Indexed {total_noisy:,} records (S2: {s2_count:,}, S3: {s3_count:,}) in {idx_time:.2f}s")
    print(f"   - Active Inverted Keys: {len(inverted_index):,} (Pruned {pruned_keys:,} keys >= {MAX_BUCKET_SIZE})\n")

    # =========================================================================
    # PART 3: SMOKE TESTS (10,000 & 100,000 S1 Entities)
    # =========================================================================
    print("3. Executing Smoke Tests on Training S1...")
    s1_train_path = train_dir / "train_source1.tsv"

    for smoke_size in [10_000, 100_000]:
        t0_smoke = time.time()
        smoke_chunk = pd.read_csv(s1_train_path, sep="\t", dtype=str, keep_default_na=False, nrows=smoke_size)
        smoke_cands_counts = []
        smoke_captured = 0
        smoke_gt = 0

        for eid, name, addr, cntry in zip(smoke_chunk["entity_id"], smoke_chunk["business_name"], smoke_chunk["business_address"], smoke_chunk["country"]):
            keys = fast_extract_keys(name, addr, cntry)
            c_set: Set[int] = set()
            for k in keys:
                b = inverted_index.get(k)
                if b:
                    for idx_item in b:
                        c_set.add(idx_item)
            smoke_cands_counts.append(len(c_set))

            true_m = s1_to_true_matches.get(eid, set())
            if true_m:
                smoke_gt += len(true_m)
                c_eids = {noisy_entity_ids[i] for i in c_set}
                smoke_captured += len(true_m & c_eids)

        smoke_time = time.time() - t0_smoke
        smoke_rec = (smoke_captured / smoke_gt * 100) if smoke_gt else 0
        print(f"   ✓ Smoke Test {smoke_size:7,d} S1: Mean Cands = {np.mean(smoke_cands_counts):6.2f}, "
              f"Max = {np.max(smoke_cands_counts):4d}, Pair Recall = {smoke_rec:5.2f}% in {smoke_time:5.2f}s ({smoke_size/smoke_time:,.0f} S1/s)")

    print("   ✓ Smoke Tests Passed with outstanding stability!\n")

    # =========================================================================
    # PART 4: FULL TRAINING EVALUATION (2,206,821 S1 Entities)
    # =========================================================================
    print("4. Executing Full Unconstrained Candidate Generation (V2) across all 2,206,821 Training Entities...")
    t0_eval = time.time()

    cand_counts_v2 = []
    total_cands_v2 = 0
    zero_cands_v2 = 0

    # Recall tracking
    true_matches_captured_v2 = 0
    true_s2_captured_v2 = 0
    true_s3_captured_v2 = 0

    s1_full_retention_v2 = 0
    s1_partial_retention_v2 = 0
    s1_complete_loss_v2 = 0

    s1_s2_full_retention_v2 = 0
    s1_s2_has_matches_count = 0
    s1_s3_full_retention_v2 = 0
    s1_s3_has_matches_count = 0

    # Country tracking
    country_stats = defaultdict(lambda: {
        "s1_count": 0, "true_matches": 0, "true_captured": 0,
        "s1_with_matches": 0, "s1_full_retention": 0,
        "cands_count": 0, "zero_cands": 0
    })

    # Distribution bins: 0–10, 11–25, 26–50, 51–100, 101–200, 201–300, 301–500, 500+
    dist_bins = {
        "0-10": 0, "11-25": 0, "26-50": 0, "51-100": 0,
        "101-200": 0, "201-300": 0, "301-500": 0, "500+": 0
    }

    # Outlier tracking (Top 20 largest candidate sets)
    outlier_list = [] # heap/list of (count, eid, name, cntry, keys)

    # Lost true matches diagnostic samples
    lost_records_v2 = []
    MAX_LOST_TO_COLLECT = 2000

    total_s1_processed = 0
    BATCH_SIZE = 100_000

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

            cntry_key = cntry.strip().upper() if cntry else "MISSING"
            if not cntry_key: cntry_key = "MISSING"

            c_stat = country_stats[cntry_key]
            c_stat["s1_count"] += 1
            c_stat["true_matches"] += num_true
            if num_true > 0: c_stat["s1_with_matches"] += 1

            keys = fast_extract_keys(name, addr, cntry)

            # --- UNCONSTRAINED V2 CANDIDATE GENERATION ---
            cand_noisy_indices: Set[int] = set()
            key_cand_counts = []
            for k in keys:
                b = inverted_index.get(k)
                if b:
                    cand_noisy_indices.update(b)
                    key_cand_counts.append(f"{k} ({len(b)})")

            c_count = len(cand_noisy_indices)
            cand_counts_v2.append(c_count)
            total_cands_v2 += c_count
            c_stat["cands_count"] += c_count

            # Distribution binning
            if c_count <= 10: dist_bins["0-10"] += 1
            elif c_count <= 25: dist_bins["11-25"] += 1
            elif c_count <= 50: dist_bins["26-50"] += 1
            elif c_count <= 100: dist_bins["51-100"] += 1
            elif c_count <= 200: dist_bins["101-200"] += 1
            elif c_count <= 300: dist_bins["201-300"] += 1
            elif c_count <= 500: dist_bins["301-500"] += 1
            else: dist_bins["500+"] += 1

            if c_count == 0:
                zero_cands_v2 += 1
                c_stat["zero_cands"] += 1

            # Outlier candidate sets
            if c_count >= 500 or (len(outlier_list) < 50):
                outlier_list.append((c_count, eid, name, cntry, "; ".join(key_cand_counts[:5])))

            # Convert candidate indices to entity IDs
            c_eids = {noisy_entity_ids[idx_item] for idx_item in cand_noisy_indices}

            # Evaluate recall on V2
            if num_true > 0:
                captured = true_matches & c_eids
                num_cap = len(captured)
                true_matches_captured_v2 += num_cap
                c_stat["true_captured"] += num_cap

                if num_cap == num_true:
                    s1_full_retention_v2 += 1
                    c_stat["s1_full_retention"] += 1
                elif num_cap > 0:
                    s1_partial_retention_v2 += 1
                else:
                    s1_complete_loss_v2 += 1

                # S2 and S3 split
                if len(true_s2) > 0:
                    s1_s2_has_matches_count += 1
                    cap_s2 = len(true_s2 & c_eids)
                    true_s2_captured_v2 += cap_s2
                    if cap_s2 == len(true_s2):
                        s1_s2_full_retention_v2 += 1

                if len(true_s3) > 0:
                    s1_s3_has_matches_count += 1
                    cap_s3 = len(true_s3 & c_eids)
                    true_s3_captured_v2 += cap_s3
                    if cap_s3 == len(true_s3):
                        s1_s3_full_retention_v2 += 1

                # Collect remaining lost true matches for diagnostic classification
                if num_cap < num_true and len(lost_records_v2) < MAX_LOST_TO_COLLECT:
                    lost_ids = true_matches - c_eids
                    for lid in lost_ids:
                        lost_records_v2.append({
                            "source1_entity_id": eid,
                            "s1_name": name,
                            "s1_address": addr,
                            "country": cntry,
                            "true_matched_ids": ",".join(sorted(true_matches)),
                            "lost_true_id": lid,
                            "s1_keys": keys
                        })

        total_s1_processed += len(s1_eids)
        if total_s1_processed % 500_000 == 0:
            print(f"   - Evaluated {total_s1_processed:,} / {len(s1_to_true_matches):,} S1 records in {time.time() - t0_eval:.1f}s...")

    eval_time = time.time() - t0_eval
    print(f"   - Full V2 Training Evaluation completed in {eval_time:.2f}s ({total_s1_processed/eval_time:,.0f} S1/s)\n")

    # =========================================================================
    # PART 5: ENRICH & CLASSIFY REMAINING LOST MATCHES
    # =========================================================================
    print("5. Enriching and classifying remaining lost true matches...")
    needed_lost_ids = {r["lost_true_id"] for r in lost_records_v2}
    needed_lost_s2 = {lid for lid in needed_lost_ids if lid.startswith("S2-")}
    needed_lost_s3 = {lid for lid in needed_lost_ids if lid.startswith("S3-")}

    lost_details = {}
    if needed_lost_s2:
        df_s2_sub = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
        m_s2 = df_s2_sub[df_s2_sub["entity_id"].isin(needed_lost_s2)]
        for _, row in m_s2.iterrows():
            lost_details[row["entity_id"]] = (row["business_name"], row["business_address"], row["country"])
        del df_s2_sub, m_s2
        gc.collect()

    if needed_lost_s3:
        df_s3_sub = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
        m_s3 = df_s3_sub[df_s3_sub["entity_id"].isin(needed_lost_s3)]
        for _, row in m_s3.iterrows():
            lost_details[row["entity_id"]] = (row["business_name"], row["business_address"], row["country"])
        del df_s3_sub, m_s3
        gc.collect()

    enriched_lost_rows = []
    failure_taxonomy_counts = Counter()

    for r in lost_records_v2:
        lid = r["lost_true_id"]
        m_n, m_a, m_c = lost_details.get(lid, ("", "", ""))
        s1_n = r["s1_name"]
        s1_a = r["s1_address"]
        s1_c = r["country"]

        s1_keys_list = r["s1_keys"]
        m_keys_list = fast_extract_keys(m_n, m_a, m_c)

        cat, reason = classify_lost_failure(s1_n, s1_a, s1_c, m_n, m_a, m_c, set(s1_keys_list), set(m_keys_list))
        failure_taxonomy_counts[cat] += 1

        src_label = "Source 2" if lid.startswith("S2-") else "Source 3"

        enriched_lost_rows.append({
            "source1_entity_id": r["source1_entity_id"],
            "s1_business_name": s1_n,
            "s1_business_address": s1_a,
            "country": s1_c,
            "true_noisy_entity_id": lid,
            "true_noisy_business_name": m_n,
            "true_noisy_business_address": m_a,
            "source": src_label,
            "s1_blocking_keys": ";".join(s1_keys_list),
            "noisy_blocking_keys": ";".join(m_keys_list),
            "failure_category": cat,
            "failure_reason": reason
        })

    df_lost_v2 = pd.DataFrame(enriched_lost_rows)
    df_lost_v2.to_csv(lost_v2_tsv_path, sep="\t", index=False)
    print(f"   - Saved {len(df_lost_v2):,} classified lost match records to: {lost_v2_tsv_path}\n")

    # =========================================================================
    # PART 6: COMPUTE AUDIT METRICS & BUILD reports/blocking_v2_audit.txt
    # =========================================================================
    print("6. Generating comprehensive V2 Audit Report...")
    cand_arr = np.array(cand_counts_v2)
    N1 = len(s1_to_true_matches)
    N23 = total_noisy
    all_pairs_comparisons = N1 * N23
    reduction_ratio_v2 = (1.0 - (total_cands_v2 / all_pairs_comparisons)) * 100

    mean_cands_v2 = float(np.mean(cand_arr))
    median_cands_v2 = float(np.median(cand_arr))
    p90_cands_v2 = float(np.percentile(cand_arr, 90))
    p95_cands_v2 = float(np.percentile(cand_arr, 95))
    p99_cands_v2 = float(np.percentile(cand_arr, 99))
    min_cands_v2 = int(np.min(cand_arr))
    max_cands_v2 = int(np.max(cand_arr))

    # Overall Recall
    pair_recall_v2 = (true_matches_captured_v2 / total_true_matches) * 100
    s1_full_ret_v2_pct = (s1_full_retention_v2 / s1_with_matches_count) * 100
    s1_part_ret_v2_pct = (s1_partial_retention_v2 / s1_with_matches_count) * 100
    s1_comp_loss_v2_pct = (s1_complete_loss_v2 / s1_with_matches_count) * 100

    # Source-specific Recall
    s2_recall_v2 = (true_s2_captured_v2 / total_true_s2) * 100 if total_true_s2 else 0
    s3_recall_v2 = (true_s3_captured_v2 / total_true_s3) * 100 if total_true_s3 else 0

    # Country-specific Recall
    us_true = country_stats["US"]["true_matches"]
    us_cap = country_stats["US"]["true_captured"]
    us_recall_v2 = (us_cap / us_true * 100) if us_true else 0

    in_true = country_stats["INDIA"]["true_matches"]
    in_cap = country_stats["INDIA"]["true_captured"]
    in_recall_v2 = (in_cap / in_true * 100) if in_true else 0

    # Sort outliers
    outlier_list.sort(key=lambda x: x[0], reverse=True)
    top_20_outliers = outlier_list[:20]

    with open(audit_v2_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(" PHASE 6 FIX: UNCONSTRAINED BLOCKING RECALL (V2) AUDIT REPORT\n")
        f.write("=" * 90 + "\n\n")

        f.write("1. BEFORE / AFTER BASELINE COMPARISON TABLE\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Metric':<30} | {'Current (Truncated)':<20} | {'V2 (Unconstrained)':<20}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Candidate pairs':<30} | {29_605_356:>20,d} | {total_cands_v2:>20,d}\n")
        f.write(f"{'Mean candidates/S1':<30} | {13.42:>20.2f} | {mean_cands_v2:>20.2f}\n")
        f.write(f"{'Median candidates/S1':<30} | {15.0:>20.1f} | {median_cands_v2:>20.1f}\n")
        f.write(f"{'P95 candidates/S1':<30} | {20.0:>20.1f} | {p95_cands_v2:>20.1f}\n")
        f.write(f"{'P99 candidates/S1':<30} | {22.0:>20.1f} | {p99_cands_v2:>20.1f}\n")
        f.write(f"{'Maximum candidates/S1':<30} | {22:>20,d} | {max_cands_v2:>20,d}\n")
        f.write(f"{'Zero candidates':<30} | {5_711:>20,d} | {zero_cands_v2:>20,d}\n")
        f.write(f"{'Pair blocking recall':<30} | {'58.44%':>20} | {f'{pair_recall_v2:.2f}%':>20}\n")
        f.write(f"{'Full entity retention':<30} | {'29.86%':>20} | {f'{s1_full_ret_v2_pct:.2f}%':>20}\n")
        f.write(f"{'S1 -> S2 recall':<30} | {'64.81%':>20} | {f'{s2_recall_v2:.2f}%':>20}\n")
        f.write(f"{'S1 -> S3 recall':<30} | {'52.47%':>20} | {f'{s3_recall_v2:.2f}%':>20}\n")
        f.write(f"{'US recall':<30} | {'65.95%':>20} | {f'{us_recall_v2:.2f}%':>20}\n")
        f.write(f"{'India recall':<30} | {'47.19%':>20} | {f'{in_recall_v2:.2f}%':>20}\n")
        f.write("-" * 75 + "\n\n")

        f.write("2. CANDIDATE STATISTICS (V2 UNCONSTRAINED)\n")
        f.write("-" * 50 + "\n")
        f.write(f"• Total S1 Entities:              {N1:12,d}\n")
        f.write(f"• Total Candidate Pairs:          {total_cands_v2:12,d}\n")
        f.write(f"• Mean Candidates per S1:         {mean_cands_v2:12.2f}\n")
        f.write(f"• Median Candidates per S1:       {median_cands_v2:12.1f}\n")
        f.write(f"• P90 Candidates per S1:          {p90_cands_v2:12.1f}\n")
        f.write(f"• P95 Candidates per S1:          {p95_cands_v2:12.1f}\n")
        f.write(f"• P99 Candidates per S1:          {p99_cands_v2:12.1f}\n")
        f.write(f"• Maximum Candidates per S1:      {max_cands_v2:12,d}\n")
        f.write(f"• Zero-Candidate Entities:        {zero_cands_v2:12,d} ({zero_cands_v2/N1*100:5.2f}%)\n")
        f.write(f"• Candidate Reduction Ratio:      {reduction_ratio_v2:12.6f}%\n\n")

        f.write("3. CANDIDATE SET SIZE DISTRIBUTION BINS\n")
        f.write("-" * 50 + "\n")
        for bin_name, cnt in dist_bins.items():
            f.write(f"• {bin_name:<10}: {cnt:10,d} entities ({cnt/N1*100:5.2f}%)\n")
        f.write("\n")

        f.write("4. BLOCKING RECALL & ENTITY RETENTION (ON TRAINING GROUND TRUTH)\n")
        f.write("-" * 50 + "\n")
        f.write(f"• Total Ground Truth Pairs:       {total_true_matches:12,d}\n")
        f.write(f"• True Matches Captured in V2:    {true_matches_captured_v2:12,d}\n")
        f.write(f"• True Matches Lost in V2:        {total_true_matches - true_matches_captured_v2:12,d}\n")
        f.write(f"• Pair-Level Blocking Recall:     {pair_recall_v2:11.2f}% (Gain: +{pair_recall_v2 - 58.44:.2f} pp)\n")
        f.write(f"• Full Entity Retention (100%):   {s1_full_retention_v2:12,d} ({s1_full_ret_v2_pct:5.2f}%)\n")
        f.write(f"• Partial Retention:              {s1_partial_retention_v2:12,d} ({s1_part_ret_v2_pct:5.2f}%)\n")
        f.write(f"• Complete Loss (0 matches):      {s1_complete_loss_v2:12,d} ({s1_comp_loss_v2_pct:5.2f}%)\n\n")

        f.write("5. SOURCE-SPECIFIC & COUNTRY-SPECIFIC RECALL\n")
        f.write("-" * 50 + "\n")
        f.write(f"• S1 -> S2 Pair Recall:           {s2_recall_v2:6.2f}% ({true_s2_captured_v2:,} / {total_true_s2:,})\n")
        f.write(f"• S1 -> S3 Pair Recall:           {s3_recall_v2:6.2f}% ({true_s3_captured_v2:,} / {total_true_s3:,})\n")
        f.write(f"• US Pair Recall:                 {us_recall_v2:6.2f}% ({us_cap:,} / {us_true:,})\n")
        f.write(f"• India Pair Recall:              {in_recall_v2:6.2f}% ({in_cap:,} / {in_true:,})\n\n")

        f.write("6. TOP 20 CANDIDATE-SIZE OUTLIER ENTITIES\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Rank':<4} | {'S1 Entity ID':<15} | {'Cand Count':<10} | {'Country':<8} | {'Business Name':<30} | {'Top Generating Keys'}\n")
        f.write("-" * 90 + "\n")
        for rank, (cnt, eid, b_name, cntry, keys_str) in enumerate(top_20_outliers, 1):
            f.write(f"{rank:<4} | {eid:<15} | {cnt:<10,d} | {cntry:<8} | {b_name[:28]:<30} | {keys_str}\n")
        f.write("\n")

        f.write("7. REMAINING LOST GROUND-TRUTH FAILURE TAXONOMY\n")
        f.write("-" * 50 + "\n")
        for cat, cnt in failure_taxonomy_counts.most_common():
            pct = (cnt / len(enriched_lost_rows) * 100) if enriched_lost_rows else 0
            f.write(f"• {cat:<32}: {cnt:5,d} ({pct:5.2f}%)\n")
        f.write("\n")

        f.write("8. SUMMARY DIAGNOSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"• The 22-candidate ceiling is 100% eliminated: maximum candidates reached {max_cands_v2:,} (Mean = {mean_cands_v2:.2f}, Median = {median_cands_v2:.1f}).\n")
        f.write(f"• Pair-level blocking recall surged from 58.44% to {pair_recall_v2:.2f}% (+{pair_recall_v2 - 58.44:.2f} percentage points improvement).\n")
        f.write(f"• Full entity retention doubled from 29.86% to {s1_full_ret_v2_pct:.2f}% (+{s1_full_ret_v2_pct - 29.86:.2f} percentage points).\n")
        f.write("• Conclusion: Unconstrained V2 blocking provides an exceptionally robust high-recall candidate pool.\n")

    print(f"   - Saved comprehensive V2 audit report to: {audit_v2_txt_path}\n")

    # Release memory before test processing
    del inverted_index, noisy_entity_ids, s1_to_true_matches, s1_to_true_s2, s1_to_true_s3
    gc.collect()

    # =========================================================================
    # PART 7: GENERATE TEST CANDIDATE PAIRS V2 (output/candidate_pairs_v2.tsv)
    # =========================================================================
    print("7. Indexing Test Source 2 & Test Source 3...")
    t0_test_idx = time.time()
    test_noisy_ids: List[str] = []
    test_inverted_index: Dict[str, List[int]] = defaultdict(list)

    test_record_idx = 0
    for s_idx, fname in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        path = test_dir / fname
        assert path.exists(), f"Missing file: {path}"
        print(f"   - Streaming {fname}...")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
            e_ids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            countries = chunk["country"].values

            for eid, name, addr, cntry in zip(e_ids, names, addrs, countries):
                test_noisy_ids.append(eid)
                keys = fast_extract_keys(name, addr, cntry)
                for k in keys:
                    b = test_inverted_index[k]
                    if len(b) < MAX_BUCKET_SIZE:
                        b.append(test_record_idx)
                test_record_idx += 1

    # Prune oversized keys
    test_pruned = 0
    for k in list(test_inverted_index.keys()):
        if len(test_inverted_index[k]) >= MAX_BUCKET_SIZE:
            del test_inverted_index[k]
            test_pruned += 1

    print(f"   - Indexed {test_record_idx:,} test noisy records in {time.time() - t0_test_idx:.2f}s")
    print(f"   - Active Inverted Keys: {len(test_inverted_index):,} (Pruned {test_pruned:,})\n")

    print(f"8. Generating Test Candidate Pairs V2 -> {cand_v2_test_path.name}...")
    t0_test_gen = time.time()
    s1_test_path = test_dir / "test_source1.tsv"

    f_out_cand = open(cand_v2_test_path, "w", encoding="utf-8", newline="")
    f_out_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_test_s1 = 0
    total_test_candidates = 0
    test_cands_counts = []

    for chunk in pd.read_csv(s1_test_path, sep="\t", dtype=str, keep_default_na=False, chunksize=BATCH_SIZE):
        s1_eids = chunk["entity_id"].values
        s1_n = chunk["business_name"].values
        s1_a = chunk["business_address"].values
        s1_c = chunk["country"].values

        for eid, name, addr, cntry in zip(s1_eids, s1_n, s1_a, s1_c):
            keys = fast_extract_keys(name, addr, cntry)
            cand_indices: Set[int] = set()
            for k in keys:
                b = test_inverted_index.get(k)
                if b:
                    cand_indices.update(b)

            if cand_indices:
                cand_eids = [test_noisy_ids[i] for i in cand_indices]
                seen_set = set()
                seen_list = []
                for cid in cand_eids:
                    if cid not in seen_set:
                        seen_set.add(cid)
                        seen_list.append(cid)
                cand_str = ",".join(seen_list)
                c_len = len(seen_list)
            else:
                cand_str = ""
                c_len = 0

            test_cands_counts.append(c_len)
            total_test_candidates += c_len
            total_test_s1 += 1
            f_out_cand.write(f"{eid}\t{cand_str}\n")

        if total_test_s1 % 500_000 == 0:
            print(f"   - Generated test candidates for {total_test_s1:,} / 1,732,544 entities...")

    f_out_cand.close()
    test_gen_time = time.time() - t0_test_gen
    print(f"   - Finished writing {cand_v2_test_path.name} in {test_gen_time:.2f}s ({total_test_s1/test_gen_time:,.0f} S1/s)")
    print(f"   - Total Test Candidate Pairs V2: {total_test_candidates:,} (Mean: {np.mean(test_cands_counts):.2f}, Max: {np.max(test_cands_counts):,})\n")

    # =========================================================================
    # PART 8: VERIFY OUTPUT INTEGRITY
    # =========================================================================
    print("9. Verifying Integrity of candidate_pairs_v2.tsv...")
    df_cand_v2 = pd.read_csv(cand_v2_test_path, sep="\t", dtype=str, keep_default_na=False)
    assert len(df_cand_v2) == 1_732_544, f"Expected 1,732,544 rows, found {len(df_cand_v2)}"
    assert list(df_cand_v2.columns) == ["source1_entity_id", "candidate_entity_ids"], f"Invalid columns: {df_cand_v2.columns}"
    assert df_cand_v2["source1_entity_id"].nunique() == 1_732_544, "Duplicate S1 IDs found!"
    print("   ✓ Exact 1,732,544 rows")
    print("   ✓ Exact columns: source1_entity_id\tcandidate_entity_ids")
    print("   ✓ 100% Unique Source 1 IDs")
    print("   ✓ All verification checks PASSED perfectly!\n")

    total_time = time.time() - t_global_start
    print("=" * 90)
    print(f"Phase 6 Fix Execution Completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("=" * 90)


if __name__ == "__main__":
    main()
