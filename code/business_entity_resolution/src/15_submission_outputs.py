"""
Phase 16: Required Submission Outputs Pipeline
Amazon ML Challenge 2026 - Business Entity Resolution

Generates and strictly validates the two required submission outputs:
1. output/matching_results.tsv
   - Columns: source1_entity_id\tmatched_entity_ids
   - Format: S1-xxx\tS2-yyy,S3-zzz (Strict comma delimiter without spaces)
2. output/candidate_pairs.tsv
   - Columns: source1_entity_id\tcandidate_entity_ids
   - Format: S1-xxx\tS2-yyy,S3-zzz (Strict comma delimiter without spaces)

Strict Requirements:
1. Every test Source 1 ID appears exactly once.
2. No extra Source 1 IDs (exact 1,732,544 rows).
3. Empty match lists represented correctly.
4. No duplicate predicted IDs.
5. Every predicted ID exists in test Source 2 or Source 3.
6. Every predicted match must exist in that S1's candidate set.
7. candidate_pairs.tsv represents the FINAL candidate set immediately before ML inference.
8. Strictly tab-separated TSV format with unspaced comma-separated ID lists.
"""

import sys
import os
import re
import time
import shutil
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict, Counter

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
norm_module = importlib.import_module("04_normalization")
strip_legal_suffixes = norm_module.strip_legal_suffixes
standardize_address_tokens = norm_module.standardize_address_tokens
clean_domain_name = norm_module.clean_domain_name

# Fast compiled C regexes
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
    """Extracts high-precision multi-pass blocking keys identical to Phase 15."""
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


def main():
    t_start = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    test_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_results_path = output_dir / "matching_results.tsv"
    candidate_pairs_path = output_dir / "candidate_pairs.tsv"
    submission_src_path = base_dir / "dataset" / "submission.tsv"

    print("=" * 90)
    print(" PHASE 16: REQUIRED SUBMISSION OUTPUTS GENERATION")
    print("=" * 90)
    print(f"Base Directory:           {base_dir}")
    print(f"Output Directory:         {output_dir}")
    print(f"Matching Results Target:  {matching_results_path}")
    print(f"Candidate Pairs Target:   {candidate_pairs_path}\n")

    # Step 1: Reformat & Write output/matching_results.tsv & dataset/submission.tsv (Strict comma without space)
    print("1. Formatting output/matching_results.tsv and dataset/submission.tsv with strict unspaced commas...")
    assert submission_src_path.exists(), f"Missing Phase 15 submission: {submission_src_path}"

    with open(submission_src_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    with open(matching_results_path, "w", encoding="utf-8", newline="") as f_out, \
         open(submission_src_path, "w", encoding="utf-8", newline="") as f_sub:
        
        # Header
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        f_sub.write("source1_entity_id\tmatched_entity_ids\n")

        for line in lines[1:]:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if rest.strip():
                # Split on comma and strip whitespace from every ID
                clean_ids = [x.strip() for x in rest.split(",") if x.strip()]
                clean_str = ",".join(clean_ids)
            else:
                clean_str = ""
            f_out.write(f"{s1_id}\t{clean_str}\n")
            f_sub.write(f"{s1_id}\t{clean_str}\n")

    print(f"   - Successfully formatted {matching_results_path} ({matching_results_path.stat().st_size:,} bytes)\n")

    # Step 2: Index Test Source 2 & Test Source 3
    print("2. Building Multi-Pass Inverted Index over Test Source 2 & Source 3...")
    t0_idx = time.time()

    noisy_entity_ids: List[str] = []
    noisy_names: List[str] = []
    noisy_addrs: List[str] = []
    noisy_countries: List[str] = []
    valid_noisy_id_set: Set[str] = set()

    inverted_index: Dict[str, List[int]] = defaultdict(list)
    MAX_BUCKET_SIZE = 300

    noisy_record_idx = 0
    for s_idx, fname in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        path = test_dir / fname
        assert path.exists(), f"Missing test file: {path}"
        print(f"   - Streaming {fname}...")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
            e_ids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            countries = chunk["country"].values

            for eid, name, addr, cntry in zip(e_ids, names, addrs, countries):
                noisy_entity_ids.append(eid)
                noisy_names.append(name)
                noisy_addrs.append(addr)
                noisy_countries.append(cntry)
                valid_noisy_id_set.add(eid)

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
    print(f"   - Total Noisy Records Indexed: {noisy_record_idx:,} in {idx_time:.2f} s")
    print(f"   - Active Inverted Keys:        {len(inverted_index):,} (Pruned {pruned_keys:,} high-frequency keys)\n")

    # Step 3: Stream Test Source 1 and write output/candidate_pairs.tsv with strict comma delimiter
    print("3. Generating Final Candidate Sets and Writing output/candidate_pairs.tsv...")
    s1_path = test_dir / "test_source1.tsv"
    assert s1_path.exists(), f"Missing test Source 1 file: {s1_path}"

    cand_file = open(candidate_pairs_path, "w", encoding="utf-8", newline="")
    cand_file.write("source1_entity_id\tcandidate_entity_ids\n")

    s1_candidate_sets: Dict[str, Set[str]] = {}
    BATCH_SIZE = 100_000
    t0_gen = time.time()
    total_s1 = 0
    total_candidates = 0

    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=BATCH_SIZE):
        s1_eids = chunk["entity_id"].values
        s1_n = chunk["business_name"].values
        s1_a = chunk["business_address"].values
        s1_c = chunk["country"].values

        for eid, name, addr, cntry in zip(s1_eids, s1_n, s1_a, s1_c):
            keys = fast_extract_keys(name, addr, cntry)
            cand_noisy_indices: Set[int] = set()
            for k in keys:
                b = inverted_index.get(k)
                if b:
                    for idx_item in b:
                        cand_noisy_indices.add(idx_item)

            if cand_noisy_indices:
                cand_eids = [noisy_entity_ids[i] for i in cand_noisy_indices]
                # Preserve unique set
                seen_cands = []
                seen_cands_set = set()
                for cid in cand_eids:
                    if cid not in seen_cands_set:
                        seen_cands_set.add(cid)
                        seen_cands.append(cid)
                cand_str = ",".join(seen_cands)
                s1_candidate_sets[eid] = seen_cands_set
                total_candidates += len(seen_cands)
            else:
                cand_str = ""
                s1_candidate_sets[eid] = set()

            cand_file.write(f"{eid}\t{cand_str}\n")
            total_s1 += 1

        if total_s1 % 500_000 == 0:
            print(f"   - Generated candidates for {total_s1:,} / 1,732,544 S1 entities...")

    cand_file.close()
    gen_time = time.time() - t0_gen
    print(f"   - Finished writing {candidate_pairs_path.name}: {total_s1:,} entities, {total_candidates:,} candidates in {gen_time:.2f} s\n")

    # Step 4: Strict Submission Integrity & Consistency Checks
    print("=" * 90)
    print(" 4. COMPREHENSIVE PHASE 16 VALIDATION & INTEGRITY CHECKS")
    print("=" * 90)

    # Validation 1: Check matching_results.tsv
    print("• Validating output/matching_results.tsv:")
    df_match = pd.read_csv(matching_results_path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df_match.columns) == ["source1_entity_id", "matched_entity_ids"], f"Invalid columns: {df_match.columns}"
    assert len(df_match) == 1_732_544, f"Expected 1,732,544 rows, got {len(df_match)}"
    assert df_match["source1_entity_id"].nunique() == 1_732_544, "Duplicate S1 IDs in matching_results.tsv"
    assert not df_match["source1_entity_id"].isna().any(), "NaN found in matching_results.tsv"
    print(f"  ✓ Row Count: exact 1,732,544 rows")
    print(f"  ✓ Header: exact 'source1_entity_id\\tmatched_entity_ids'")
    print(f"  ✓ Uniqueness: exact 1,732,544 unique Source 1 IDs")

    # Validation 2: Check candidate_pairs.tsv
    print("\n• Validating output/candidate_pairs.tsv:")
    df_cand = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df_cand.columns) == ["source1_entity_id", "candidate_entity_ids"], f"Invalid columns: {df_cand.columns}"
    assert len(df_cand) == 1_732_544, f"Expected 1,732,544 rows, got {len(df_cand)}"
    assert df_cand["source1_entity_id"].nunique() == 1_732_544, "Duplicate S1 IDs in candidate_pairs.tsv"
    assert not df_cand["source1_entity_id"].isna().any(), "NaN found in candidate_pairs.tsv"
    print(f"  ✓ Row Count: exact 1,732,544 rows")
    print(f"  ✓ Header: exact 'source1_entity_id\\tcandidate_entity_ids'")
    print(f"  ✓ Uniqueness: exact 1,732,544 unique Source 1 IDs")

    # Validation 3: Check S1 ID Alignment
    print("\n• Validating S1 ID Alignment between files:")
    assert (df_match["source1_entity_id"] == df_cand["source1_entity_id"]).all(), "Mismatch in S1 ID sequence between files!"
    print(f"  ✓ 100% Exact 1-to-1 S1 ID Alignment")

    # Validation 4: Check No Duplicate IDs in match or candidate lists
    print("\n• Validating No Duplicate IDs in predicted match sets or candidate sets:")
    dup_matches = 0
    dup_cands = 0
    invalid_ids = 0
    subset_violations = 0

    match_vals = df_match["matched_entity_ids"].values
    cand_vals = df_cand["candidate_entity_ids"].values
    s1_vals = df_match["source1_entity_id"].values

    total_matches = 0
    total_singletons = 0
    total_single_match = 0
    total_multi_match = 0

    for s1_id, m_str, c_str in zip(s1_vals, match_vals, cand_vals):
        m_list = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []
        c_list = [x.strip() for x in c_str.split(",") if x.strip()] if c_str else []
        c_set = set(c_list)

        if len(m_list) != len(set(m_list)):
            dup_matches += 1
        if len(c_list) != len(c_set):
            dup_cands += 1

        for mid in m_list:
            if mid not in valid_noisy_id_set:
                invalid_ids += 1
            if mid not in c_set:
                subset_violations += 1

        m_len = len(m_list)
        total_matches += m_len
        if m_len == 0:
            total_singletons += 1
        elif m_len == 1:
            total_single_match += 1
        else:
            total_multi_match += 1

    print(f"  ✓ Duplicate Matched IDs per entity:       {dup_matches} (Must be 0)")
    print(f"  ✓ Duplicate Candidate IDs per entity:     {dup_cands} (Must be 0)")
    print(f"  ✓ Invalid Matched IDs (not in S2/S3):     {invalid_ids} (Must be 0)")
    print(f"  ✓ Subset Violations (match not in cand):  {subset_violations} (Must be 0)")
    assert dup_matches == 0, "Found duplicate matched IDs"
    assert dup_cands == 0, "Found duplicate candidate IDs"
    assert invalid_ids == 0, "Found invalid IDs not in S2/S3"
    assert subset_violations == 0, "Found matches not present in candidate set"

    total_time = time.time() - t_start
    print("\n" + "=" * 90)
    print(" ALL 9 PHASE 16 REQUIREMENTS VERIFIED & PASSED PERFECTLY!")
    print("=" * 90)
    print(f"Total Phase 16 Execution Time: {total_time:.2f} s")
    print(f"Matching Results Output:       {matching_results_path}")
    print(f"Candidate Pairs Output:        {candidate_pairs_path}")
    print(f"Total Matches Predicted:       {total_matches:,}")
    print(f"Singletons (0 matches):        {total_singletons:,} ({total_singletons / 1_732_544 * 100:.2f}%)")
    print(f"Single Match (1 match):        {total_single_match:,} ({total_single_match / 1_732_544 * 100:.2f}%)")
    print(f"Multi-Match (2+ matches):      {total_multi_match:,} ({total_multi_match / 1_732_544 * 100:.2f}%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
