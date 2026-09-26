"""
Phase 15: Final Test Inference Pipeline for Amazon ML Challenge 2026 - Business Entity Resolution
Executes the frozen end-to-end entity resolution pipeline on the test dataset.

Pipeline Sequence (Strictly Frozen):
1. Normalization: Multi-representation cleaning and standardized token representations.
2. Candidate Blocking: Multi-pass union indexing over Test Source 2 & Test Source 3 with bucket pruning.
3. Candidate Generation: Extracts final candidate set immediately prior to ML inference.
4. Feature Engineering: 44-dimensional pairwise comparison feature suite.
5. Model Scoring: Evaluates champion HistGradientBoosting GBDT model.
6. Decision Engine: Frozen optimal threshold (tau* = 0.840) with deduplication & singleton handling.
7. Submission Formatting: Outputs valid TSV predictions.
"""

import sys
import gc
import re
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
from rapidfuzz import fuzz

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import importlib
norm_module = importlib.import_module("04_normalization")
clean_str_input = norm_module.clean_str_input
strip_legal_suffixes = norm_module.strip_legal_suffixes
standardize_address_tokens = norm_module.standardize_address_tokens
get_sorted_tokens_str = norm_module.get_sorted_tokens_str
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


def fast_extract_keys(name_val: str, addr_val: str, country_val: str) -> List[str]:
    """Extracts high-precision multi-pass blocking keys."""
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


def get_char_ngrams(s: str, n: int = 3) -> Set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard_sim(set_a: Set[Any], set_b: Set[Any]) -> float:
    if not set_a or not set_b:
        return 0.0
    u = len(set_a | set_b)
    return len(set_a & set_b) / u if u > 0 else 0.0


def compute_44_pair_features(
    s1_name: str, s1_addr: str, s1_country: str,
    noisy_name: str, noisy_addr: str, noisy_country: str
) -> List[float]:
    """Computes exact 44-dimensional feature vector matching the champion GBDT model."""
    p_s1_n = RE_PUNCT.sub(" ", s1_name.lower()).strip()
    p_m_n = RE_PUNCT.sub(" ", noisy_name.lower()).strip()
    p_s1_a = RE_PUNCT.sub(" ", s1_addr.lower()).strip()
    p_m_a = RE_PUNCT.sub(" ", noisy_addr.lower()).strip()

    s1_c = s1_country.upper()
    m_c = noisy_country.upper()

    s1_n_toks = p_s1_n.split()
    m_n_toks = p_m_n.split()
    s1_a_toks = p_s1_a.split()
    m_a_toks = p_m_a.split()

    # 1. Name Features
    name_exact = 1.0 if (s1_name and noisy_name and s1_name == noisy_name) else 0.0
    name_clean_exact = 1.0 if (p_s1_n and p_m_n and p_s1_n == p_m_n) else 0.0
    name_fuzz = fuzz.ratio(p_s1_n, p_m_n) / 100.0 if (p_s1_n and p_m_n) else 0.0
    name_partial = fuzz.partial_ratio(p_s1_n, p_m_n) / 100.0 if (p_s1_n and p_m_n) else 0.0
    name_tok_sort = fuzz.token_sort_ratio(p_s1_n, p_m_n) / 100.0 if (p_s1_n and p_m_n) else 0.0
    name_tok_set = fuzz.token_set_ratio(p_s1_n, p_m_n) / 100.0 if (p_s1_n and p_m_n) else 0.0
    name_tok_jacc = jaccard_sim(set(s1_n_toks), set(m_n_toks))

    ng3_s1_n = get_char_ngrams(p_s1_n, 3)
    ng3_m_n = get_char_ngrams(p_m_n, 3)
    name_c3_jacc = jaccard_sim(ng3_s1_n, ng3_m_n)

    ng4_s1_n = get_char_ngrams(p_s1_n, 4)
    ng4_m_n = get_char_ngrams(p_m_n, 4)
    name_c4_jacc = jaccard_sim(ng4_s1_n, ng4_m_n)

    name_len_diff = float(abs(len(p_s1_n) - len(p_m_n)))
    name_len_ratio = (min(len(p_s1_n), len(p_m_n)) / max(len(p_s1_n), len(p_m_n))) if (p_s1_n and p_m_n) else 0.0
    name_tok_diff = float(abs(len(s1_n_toks) - len(m_n_toks)))

    s1_nl_toks = strip_legal_suffixes(s1_n_toks)
    m_nl_toks = strip_legal_suffixes(m_n_toks)
    s1_nl_str = " ".join(s1_nl_toks)
    m_nl_str = " ".join(m_nl_toks)
    name_nl_exact = 1.0 if (s1_nl_str and m_nl_str and s1_nl_str == m_nl_str) else 0.0
    name_nl_sort = fuzz.token_sort_ratio(s1_nl_str, m_nl_str) / 100.0 if (s1_nl_str and m_nl_str) else 0.0

    # 2. Address Features
    addr_missing = 1.0 if (not s1_addr or not noisy_addr) else 0.0
    addr_exact = 1.0 if (s1_addr and noisy_addr and s1_addr == noisy_addr) else 0.0
    addr_clean_exact = 1.0 if (p_s1_a and p_m_a and p_s1_a == p_m_a) else 0.0
    addr_fuzz = fuzz.ratio(p_s1_a, p_m_a) / 100.0 if (p_s1_a and p_m_a) else 0.0
    addr_partial = fuzz.partial_ratio(p_s1_a, p_m_a) / 100.0 if (p_s1_a and p_m_a) else 0.0
    addr_tok_sort = fuzz.token_sort_ratio(p_s1_a, p_m_a) / 100.0 if (p_s1_a and p_m_a) else 0.0
    addr_tok_set = fuzz.token_set_ratio(p_s1_a, p_m_a) / 100.0 if (p_s1_a and p_m_a) else 0.0
    addr_tok_jacc = jaccard_sim(set(s1_a_toks), set(m_a_toks))

    ng3_s1_a = get_char_ngrams(p_s1_a, 3)
    ng3_m_a = get_char_ngrams(p_m_a, 3)
    addr_c3_jacc = jaccard_sim(ng3_s1_a, ng3_m_a)

    ng4_s1_a = get_char_ngrams(p_s1_a, 4)
    ng4_m_a = get_char_ngrams(p_m_a, 4)
    addr_c4_jacc = jaccard_sim(ng4_s1_a, ng4_m_a)

    addr_len_diff = float(abs(len(p_s1_a) - len(p_m_a)))
    addr_len_ratio = (min(len(p_s1_a), len(p_m_a)) / max(len(p_s1_a), len(p_m_a))) if (p_s1_a and p_m_a) else 0.0
    addr_tok_diff = float(abs(len(s1_a_toks) - len(m_a_toks)))

    std_s1_a = get_sorted_tokens_str(standardize_address_tokens(s1_a_toks))
    std_m_a = get_sorted_tokens_str(standardize_address_tokens(m_a_toks))
    addr_std_exact = 1.0 if (std_s1_a and std_m_a and std_s1_a == std_m_a) else 0.0

    nums_s1 = set(RE_DIGITS.findall(p_s1_a))
    nums_m = set(RE_DIGITS.findall(p_m_a))
    addr_num_match = 1.0 if (nums_s1 and nums_m and bool(nums_s1 & nums_m)) else 0.0

    # Fast zip check
    addr_zip_match = 1.0 if (len(p_s1_a) >= 5 and len(p_m_a) >= 5 and any(t.isdigit() and len(t) in (5, 6) and t in p_m_a for t in s1_a_toks)) else 0.0

    # 3. Country Features
    cntry_exact = 1.0 if (s1_c and m_c and s1_c == m_c) else 0.0
    cntry_missing = 1.0 if (not s1_c or not m_c) else 0.0

    # 4. Cross Interactions
    cross_prod = name_tok_sort * addr_c3_jacc
    cross_mean = 0.5 * (name_tok_sort + addr_c3_jacc)
    cross_min = min(name_tok_sort, addr_c3_jacc)
    cross_max = max(name_tok_sort, addr_c3_jacc)
    denom = name_tok_sort + addr_c3_jacc
    cross_harm = (2.0 * name_tok_sort * addr_c3_jacc / denom) if denom > 0 else 0.0

    cross_exact_n_str_a = 1.0 if (name_clean_exact == 1.0 and addr_c3_jacc >= 0.70) else 0.0
    cross_str_n_wk_a = 1.0 if (name_tok_sort >= 0.85 and addr_c3_jacc < 0.40) else 0.0
    cross_str_a_wk_n = 1.0 if (addr_c3_jacc >= 0.85 and name_tok_sort < 0.40) else 0.0

    # 5. Four Forensic Features
    addr_num_mismatch = 1.0 if (len(nums_s1) > 0 and len(nums_m) > 0 and len(nums_s1 & nums_m) == 0) else 0.0

    s1_acr = "".join(t[0] for t in s1_n_toks if t[0].isalnum())
    m_acr = "".join(t[0] for t in m_n_toks if t[0].isalnum())
    s1_compact = "".join(c for c in p_s1_n if c.isalnum())
    m_compact = "".join(c for c in p_m_n if c.isalnum())
    name_acr_match = 1.0 if (s1_acr and s1_acr == m_compact) or (m_acr and m_acr == s1_compact) else 0.0

    pfx4_s1 = p_s1_n[:4]
    pfx4_m = p_m_n[:4]
    name_pfx4_match = 1.0 if (pfx4_s1 and pfx4_s1 == pfx4_m) else 0.0

    set_s1_a = set(s1_a_toks)
    set_m_a = set(m_a_toks)
    addr_containment = max(len(set_s1_a & set_m_a) / len(set_s1_a), len(set_s1_a & set_m_a) / len(set_m_a)) if (set_s1_a and set_m_a) else 0.0

    return [
        name_exact, name_clean_exact, name_fuzz, name_partial, name_tok_sort, name_tok_set,
        name_tok_jacc, name_c3_jacc, name_c4_jacc, name_len_diff, name_len_ratio, name_tok_diff,
        name_nl_exact, name_nl_sort, addr_missing, addr_exact, addr_clean_exact, addr_fuzz,
        addr_partial, addr_tok_sort, addr_tok_set, addr_tok_jacc, addr_c3_jacc, addr_c4_jacc,
        addr_len_diff, addr_len_ratio, addr_tok_diff, addr_std_exact, addr_num_match, addr_zip_match,
        cntry_exact, cntry_missing, cross_prod, cross_mean, cross_min, cross_max, cross_harm,
        cross_exact_n_str_a, cross_str_n_wk_a, cross_str_a_wk_n,
        addr_num_mismatch, name_acr_match, name_pfx4_match, addr_containment
    ]


def run_test_inference():
    t_start = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    models_dir = base_dir / "models"
    test_dir = base_dir / "dataset" / "test"
    output_submission_path = base_dir / "dataset" / "submission.tsv"
    report_file = base_dir / "final_test_inference_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 90)
    print(" PHASE 15: FINAL TEST INFERENCE PIPELINE")
    print("=" * 90)
    print(f"Project Base Directory:     {base_dir}")
    print(f"Test Directory:             {test_dir}")
    print(f"Target Submission Output:   {output_submission_path}")
    print(f"Report Output File:         {report_file}\n")

    # Step 1: Load Champion Model & Frozen Threshold
    model_bundle_path = models_dir / "best_model_controlled.joblib"
    if not model_bundle_path.exists():
        model_bundle_path = models_dir / "best_model.joblib"

    print(f"1. Loading Frozen Champion Model: {model_bundle_path.name}...")
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    tau = bundle.get("optimal_threshold", 0.840)

    print(f"   - Model Class:       {type(model).__name__}")
    print(f"   - Frozen Threshold:  {tau:.3f}")
    print(f"   - Feature Dimension: {len(feature_cols)}\n")

    # Step 2: Stream & Build Inverted Index over Test Source 2 & Test Source 3
    print("2. Building Multi-Pass Inverted Index over Test Source 2 & Source 3...")
    t0_idx = time.time()

    noisy_entity_ids: List[str] = []
    noisy_names: List[str] = []
    noisy_addrs: List[str] = []
    noisy_countries: List[str] = []

    # Inverted index: key -> list of noisy record indices
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

    # Step 3: Stream Test Source 1 in Batches, Generate Candidates, Score, & Format Submission
    print("3. Generating Final Candidates and Executing ML Inference on Test Source 1...")
    s1_path = test_dir / "test_source1.tsv"
    assert s1_path.exists(), f"Missing test Source 1 file: {s1_path}"

    out_file = open(output_submission_path, "w", encoding="utf-8", newline="")
    out_file.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1_processed = 0
    total_candidates_evaluated = 0
    total_matches_predicted = 0

    singleton_pred_count = 0
    single_match_pred_count = 0
    multi_match_pred_count = 0
    size_distribution = Counter()

    BATCH_SIZE = 50_000
    t0_inf = time.time()
    batch_num = 0

    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=BATCH_SIZE):
        batch_num += 1
        s1_eids = chunk["entity_id"].values
        s1_n = chunk["business_name"].values
        s1_a = chunk["business_address"].values
        s1_c = chunk["country"].values

        batch_pairs_features = []
        batch_pair_meta = []  # (s1_eid, noisy_eid)

        # Candidate Retrieval per S1 in batch (max 15 high-quality candidates per S1)
        for eid, name, addr, cntry in zip(s1_eids, s1_n, s1_a, s1_c):
            keys = fast_extract_keys(name, addr, cntry)
            cand_noisy_indices: Set[int] = set()
            for k in keys:
                b = inverted_index.get(k)
                if b:
                    for idx_item in b:
                        cand_noisy_indices.add(idx_item)

            if cand_noisy_indices:
                for n_idx in cand_noisy_indices:
                    n_name = noisy_names[n_idx]
                    n_addr = noisy_addrs[n_idx]
                    n_cntry = noisy_countries[n_idx]
                    n_eid = noisy_entity_ids[n_idx]

                    feats = compute_44_pair_features(name, addr, cntry, n_name, n_addr, n_cntry)
                    batch_pairs_features.append(feats)
                    batch_pair_meta.append((eid, n_eid))

        # Model Inference on Batch
        batch_decisions = defaultdict(list)

        if batch_pairs_features:
            X_batch = np.array(batch_pairs_features, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            total_candidates_evaluated += len(probs)

            # Group by S1 and Apply Decision Logic
            s1_cand_scores = defaultdict(list)
            for (s1_id, n_id), prob in zip(batch_pair_meta, probs):
                s1_cand_scores[s1_id].append((n_id, float(prob)))

            for s1_id, cands in s1_cand_scores.items():
                cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)
                selected = []
                seen = set()
                for nid, p in cands_sorted:
                    if p >= tau:
                        if nid not in seen:
                            seen.add(nid)
                            selected.append(nid)
                batch_decisions[s1_id] = selected

        # Write Batch Decisions to Output TSV
        for eid in s1_eids:
            matches = batch_decisions.get(eid, [])
            match_count = len(matches)
            total_matches_predicted += match_count
            size_distribution[match_count] += 1

            if match_count == 0:
                singleton_pred_count += 1
            elif match_count == 1:
                single_match_pred_count += 1
            else:
                multi_match_pred_count += 1

            match_str = ",".join(matches) if matches else ""
            out_file.write(f"{eid}\t{match_str}\n")

        total_s1_processed += len(s1_eids)
        elapsed_b = time.time() - t0_inf
        rate = total_s1_processed / elapsed_b if elapsed_b > 0 else 0
        print(f"   - Batch {batch_num:2d} | Processed: {total_s1_processed:,} / 1,732,544 ({total_s1_processed/1732544*100:5.1f}%) | "
              f"Cand Pairs: {total_candidates_evaluated:,} | Matches: {total_matches_predicted:,} | Rate: {rate:,.0f} S1/s")

    out_file.close()

    # Step 4: Final Validation & Integrity Checks on Submission TSV
    print("\n" + "=" * 90)
    print(" 4. FINAL SUBMISSION INTEGRITY VERIFICATION")
    print("=" * 90)
    df_sub = pd.read_csv(output_submission_path, sep="\t", dtype=str, keep_default_na=False)
    
    total_sub_rows = len(df_sub)
    print(f"• Total Rows in Submission File:    {total_sub_rows:,} (Expected: 1,732,544)")
    assert total_sub_rows == 1_732_544, f"Row count mismatch! Found {total_sub_rows}"

    print(f"• Null / NaN Entity IDs:            {df_sub['source1_entity_id'].isna().sum()} (Must be 0)")
    assert df_sub["source1_entity_id"].isna().sum() == 0

    unique_s1_count = df_sub["source1_entity_id"].nunique()
    print(f"• Unique Source 1 Entity IDs:       {unique_s1_count:,} (Must be 1,732,544)")
    assert unique_s1_count == 1_732_544

    # Check for duplicate predictions per row
    duplicate_pred_rows = 0
    invalid_format_rows = 0
    for _, row in df_sub.iterrows():
        raw_m = row["matched_entity_ids"].strip()
        if raw_m:
            items = [x.strip() for x in raw_m.split(",") if x.strip()]
            if len(items) != len(set(items)):
                duplicate_pred_rows += 1
            if any(not (x.startswith("S2-") or x.startswith("S3-")) for x in items):
                invalid_format_rows += 1

    print(f"• Rows with Duplicate Matched IDs:  {duplicate_pred_rows} (Must be 0)")
    print(f"• Rows with Invalid S2/S3 IDs:      {invalid_format_rows} (Must be 0)")
    assert duplicate_pred_rows == 0, "Submission contains duplicate matches within a row!"
    assert invalid_format_rows == 0, "Submission contains invalid match ID formats!"

    print("• All Submission Integrity Checks PASSED Perfectly!\n")

    # Step 5: Final Performance & Summary Report
    print("=" * 90)
    print(" 5. FINAL TEST INFERENCE SUMMARY STATISTICS")
    print("=" * 90)
    print(f"• Total Test S1 Entities:           {total_s1_processed:,}")
    print(f"• Total Noisy Records Evaluated:    {noisy_record_idx:,} (S2: 4,887,273 | S3: 5,082,316)")
    print(f"• Total Candidate Pairs Evaluated:  {total_candidates_evaluated:,} (Avg {total_candidates_evaluated/total_s1_processed:.2f} per S1)")
    print(f"• Total Matches Predicted:          {total_matches_predicted:,} (Avg {total_matches_predicted/total_s1_processed:.2f} per S1)")
    print(f"• Predicted Singletons (0 matches): {singleton_pred_count:,} ({singleton_pred_count/total_s1_processed*100:5.2f}%)")
    print(f"• Predicted Single Match (1 match): {single_match_pred_count:,} ({single_match_pred_count/total_s1_processed*100:5.2f}%)")
    print(f"• Predicted Multi-Match (2+ match): {multi_match_pred_count:,} ({multi_match_pred_count/total_s1_processed*100:5.2f}%)")

    print("\nPredicted Set Cardinality Distribution:")
    for size, cnt in sorted(size_distribution.items()):
        if size <= 10:
            print(f"  - {size:2d} matches: {cnt:7,d} entities ({cnt/total_s1_processed*100:5.2f}%)")
    larger_cnt = sum(cnt for size, cnt in size_distribution.items() if size > 10)
    if larger_cnt > 0:
        print(f"  - >10 matches: {larger_cnt:7,d} entities ({larger_cnt/total_s1_processed*100:5.2f}%)")

    total_pipeline_time = time.time() - t_start
    print("\n" + "=" * 90)
    print(f"Total Test Pipeline Duration: {total_pipeline_time:.2f} seconds ({total_pipeline_time/60:.2f} minutes)")
    print(f"Output Submission File:       {output_submission_path}")
    print(" PHASE 15 FINAL TEST INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 90)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_test_inference()
