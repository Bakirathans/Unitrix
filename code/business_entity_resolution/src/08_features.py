"""
Phase 8: Entity-Pair Feature Engineering for Amazon ML Challenge 2026 - Business Entity Resolution
This module extracts rich numerical comparison features for any candidate (Source 1, Noisy) pair.

Key Design Rules:
1. Symmetrical & Reusable: Identical feature extraction pipeline for Training, Validation, and Test/Inference.
2. Label Independent: No ground truth labels or targets are used in feature extraction.
3. Explicit Missing-Value Handling: Missing/empty fields mapped to 0.0 with dedicated indicator flags.
4. Comprehensive Feature Suite:
   - String similarity & edit distance (Levenshtein, Token Sort, Token Set, Partial Ratio)
   - Token & Character N-Gram Jaccard / Overlap
   - Structural & Length disparity features
   - Corporate legal suffix & Address abbreviation invariance features
   - Postal code and Street number exact matching
   - Cross-field non-linear interactions & asymmetry indicators
"""

import sys
import gc
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

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

clean_str_input = norm_module.clean_str_input
strip_legal_suffixes = norm_module.strip_legal_suffixes
standardize_address_tokens = norm_module.standardize_address_tokens
get_sorted_tokens_str = norm_module.get_sorted_tokens_str

# Fast compiled regexes
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_DIGITS = re.compile(r"\b\d+\b")
RE_ZIPCODE = re.compile(r"\b\d{5,6}\b")


def get_char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Extract set of character n-grams from a string."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def extract_pair_features(
    s1_name: Any,
    s1_addr: Any,
    s1_country: Any,
    noisy_name: Any,
    noisy_addr: Any,
    noisy_country: Any
) -> Dict[str, float]:
    """
    Extract a comprehensive 35+ numerical feature dictionary for a single entity pair.
    Fast, robust, missing-value resilient.
    """
    feats: Dict[str, float] = {}

    # String cleaning & normalization
    raw_s1_n = clean_str_input(s1_name)
    raw_m_n = clean_str_input(noisy_name)
    raw_s1_a = clean_str_input(s1_addr)
    raw_m_a = clean_str_input(noisy_addr)
    raw_s1_c = clean_str_input(s1_country).upper()
    raw_m_c = clean_str_input(noisy_country).upper()

    # Preprocessed lower/punct strings
    p_s1_n = RE_PUNCT.sub(" ", raw_s1_n.lower()).strip()
    p_m_n = RE_PUNCT.sub(" ", raw_m_n.lower()).strip()
    p_s1_a = RE_PUNCT.sub(" ", raw_s1_a.lower()).strip()
    p_m_a = RE_PUNCT.sub(" ", raw_m_a.lower()).strip()

    # Word tokens
    s1_n_toks = p_s1_n.split()
    m_n_toks = p_m_n.split()
    s1_a_toks = p_s1_a.split()
    m_a_toks = p_m_a.split()

    # =========================================================================
    # 1. NAME FEATURES
    # =========================================================================
    feats["name_exact_match"] = 1.0 if (raw_s1_n and raw_m_n and raw_s1_n == raw_m_n) else 0.0
    feats["name_clean_exact"] = 1.0 if (p_s1_n and p_m_n and p_s1_n == p_m_n) else 0.0
    
    if p_s1_n and p_m_n:
        feats["name_fuzz_ratio"] = fuzz.ratio(p_s1_n, p_m_n) / 100.0
        feats["name_partial_ratio"] = fuzz.partial_ratio(p_s1_n, p_m_n) / 100.0
        feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(p_s1_n, p_m_n) / 100.0
        feats["name_token_set_ratio"] = fuzz.token_set_ratio(p_s1_n, p_m_n) / 100.0
        feats["name_token_jaccard"] = jaccard_similarity(set(s1_n_toks), set(m_n_toks))
        
        # Char n-grams
        ng3_s1 = get_char_ngrams(p_s1_n, 3)
        ng3_m = get_char_ngrams(p_m_n, 3)
        feats["name_char_3gram_jaccard"] = jaccard_similarity(ng3_s1, ng3_m)

        ng4_s1 = get_char_ngrams(p_s1_n, 4)
        ng4_m = get_char_ngrams(p_m_n, 4)
        feats["name_char_4gram_jaccard"] = jaccard_similarity(ng4_s1, ng4_m)

        # Length disparity
        len_s1 = len(p_s1_n)
        len_m = len(p_m_n)
        feats["name_len_diff"] = float(abs(len_s1 - len_m))
        feats["name_len_ratio"] = min(len_s1, len_m) / max(len_s1, len_m) if max(len_s1, len_m) > 0 else 0.0
        feats["name_token_count_diff"] = float(abs(len(s1_n_toks) - len(m_n_toks)))

        # Suffix-stripped name similarity (Phase 3 finding: 21.3% legal suffix variation)
        no_leg_s1 = " ".join(strip_legal_suffixes(s1_n_toks))
        no_leg_m = " ".join(strip_legal_suffixes(m_n_toks))
        feats["name_no_legal_exact"] = 1.0 if (no_leg_s1 and no_leg_m and no_leg_s1 == no_leg_m) else 0.0
        feats["name_no_legal_token_sort"] = fuzz.token_sort_ratio(no_leg_s1, no_leg_m) / 100.0 if (no_leg_s1 and no_leg_m) else 0.0
    else:
        feats["name_fuzz_ratio"] = 0.0
        feats["name_partial_ratio"] = 0.0
        feats["name_token_sort_ratio"] = 0.0
        feats["name_token_set_ratio"] = 0.0
        feats["name_token_jaccard"] = 0.0
        feats["name_char_3gram_jaccard"] = 0.0
        feats["name_char_4gram_jaccard"] = 0.0
        feats["name_len_diff"] = 0.0
        feats["name_len_ratio"] = 0.0
        feats["name_token_count_diff"] = 0.0
        feats["name_no_legal_exact"] = 0.0
        feats["name_no_legal_token_sort"] = 0.0

    # =========================================================================
    # 2. ADDRESS FEATURES
    # =========================================================================
    addr_missing = 1.0 if (not p_m_a or not p_s1_a) else 0.0
    feats["addr_is_missing"] = addr_missing
    feats["addr_exact_match"] = 1.0 if (raw_s1_a and raw_m_a and raw_s1_a == raw_m_a) else 0.0
    feats["addr_clean_exact"] = 1.0 if (p_s1_a and p_m_a and p_s1_a == p_m_a) else 0.0

    if p_s1_a and p_m_a:
        feats["addr_fuzz_ratio"] = fuzz.ratio(p_s1_a, p_m_a) / 100.0
        feats["addr_partial_ratio"] = fuzz.partial_ratio(p_s1_a, p_m_a) / 100.0
        feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(p_s1_a, p_m_a) / 100.0
        feats["addr_token_set_ratio"] = fuzz.token_set_ratio(p_s1_a, p_m_a) / 100.0
        feats["addr_token_jaccard"] = jaccard_similarity(set(s1_a_toks), set(m_a_toks))

        # Char n-grams
        ng3_s1_a = get_char_ngrams(p_s1_a, 3)
        ng3_m_a = get_char_ngrams(p_m_a, 3)
        feats["addr_char_3gram_jaccard"] = jaccard_similarity(ng3_s1_a, ng3_m_a)

        ng4_s1_a = get_char_ngrams(p_s1_a, 4)
        ng4_m_a = get_char_ngrams(p_m_a, 4)
        feats["addr_char_4gram_jaccard"] = jaccard_similarity(ng4_s1_a, ng4_m_a)

        # Length disparity
        len_s1_a = len(p_s1_a)
        len_m_a = len(p_m_a)
        feats["addr_len_diff"] = float(abs(len_s1_a - len_m_a))
        feats["addr_len_ratio"] = min(len_s1_a, len_m_a) / max(len_s1_a, len_m_a) if max(len_s1_a, len_m_a) > 0 else 0.0
        feats["addr_token_count_diff"] = float(abs(len(s1_a_toks) - len(m_a_toks)))

        # Standardized sorted address match (Phase 4 invariant)
        std_s1 = get_sorted_tokens_str(standardize_address_tokens(s1_a_toks))
        std_m = get_sorted_tokens_str(standardize_address_tokens(m_a_toks))
        feats["addr_std_sorted_exact"] = 1.0 if (std_s1 and std_m and std_s1 == std_m) else 0.0

        # Street number exact match
        nums_s1 = RE_DIGITS.findall(p_s1_a)
        nums_m = RE_DIGITS.findall(p_m_a)
        if nums_s1 and nums_m:
            feats["addr_number_match"] = 1.0 if (nums_s1[0] == nums_m[0]) else 0.0
        else:
            feats["addr_number_match"] = 0.5  # Neutral indicator when numbers absent

        # Postal code match
        z_s1 = RE_ZIPCODE.findall(p_s1_a)
        z_m = RE_ZIPCODE.findall(p_m_a)
        if z_s1 and z_m:
            feats["addr_zip_match"] = 1.0 if (set(z_s1) & set(z_m)) else 0.0
        else:
            feats["addr_zip_match"] = 0.5  # Neutral indicator when zip absent
    else:
        feats["addr_fuzz_ratio"] = 0.0
        feats["addr_partial_ratio"] = 0.0
        feats["addr_token_sort_ratio"] = 0.0
        feats["addr_token_set_ratio"] = 0.0
        feats["addr_token_jaccard"] = 0.0
        feats["addr_char_3gram_jaccard"] = 0.0
        feats["addr_char_4gram_jaccard"] = 0.0
        feats["addr_len_diff"] = 0.0
        feats["addr_len_ratio"] = 0.0
        feats["addr_token_count_diff"] = 0.0
        feats["addr_std_sorted_exact"] = 0.0
        feats["addr_number_match"] = 0.0
        feats["addr_zip_match"] = 0.0

    # =========================================================================
    # 3. COUNTRY FEATURES
    # =========================================================================
    feats["country_exact_match"] = 1.0 if (raw_s1_c and raw_m_c and raw_s1_c == raw_m_c) else 0.0
    feats["country_is_missing"] = 1.0 if (not raw_s1_c or not raw_m_c or raw_m_c in ("NAN", "NULL", "NONE")) else 0.0

    # =========================================================================
    # 4. CROSS-FIELD & INTERACTION FEATURES
    # =========================================================================
    n_sim = feats["name_fuzz_ratio"]
    a_sim = feats["addr_fuzz_ratio"]

    feats["cross_name_addr_prod"] = n_sim * a_sim
    feats["cross_name_addr_mean"] = 0.5 * (n_sim + a_sim)
    feats["cross_name_addr_min"] = min(n_sim, a_sim)
    feats["cross_name_addr_max"] = max(n_sim, a_sim)
    feats["cross_name_addr_harm_mean"] = (2.0 * n_sim * a_sim) / (n_sim + a_sim + 1e-6)

    # Asymmetric informativeness indicators (Phase 3 findings)
    feats["cross_exact_name_strong_addr"] = 1.0 if (n_sim >= 0.95 and a_sim >= 0.70) else 0.0
    feats["cross_strong_name_weak_addr"] = 1.0 if (n_sim >= 0.85 and a_sim < 0.40) else 0.0
    feats["cross_strong_addr_weak_name"] = 1.0 if (a_sim >= 0.80 and n_sim < 0.45) else 0.0

    return feats


def extract_features_for_dataframe(df_pairs: pd.DataFrame, batch_size: int = 50000) -> pd.DataFrame:
    """
    Compute features across all pairs in a DataFrame in vectorized batches.
    Returns DataFrame containing original identifiers, labels, and all numerical features.
    """
    total = len(df_pairs)
    feature_rows: List[Dict[str, float]] = []
    
    t_start = time.time()
    s1_names = df_pairs["s1_name"].tolist()
    s1_addrs = df_pairs["s1_address"].tolist()
    s1_cntrys = df_pairs["s1_country"].tolist()
    m_names = df_pairs["noisy_name"].tolist()
    m_addrs = df_pairs["noisy_address"].tolist()
    m_cntrys = df_pairs["noisy_country"].tolist()

    for i in range(total):
        f = extract_pair_features(
            s1_names[i], s1_addrs[i], s1_cntrys[i],
            m_names[i], m_addrs[i], m_cntrys[i]
        )
        feature_rows.append(f)

    df_feats = pd.DataFrame(feature_rows)
    
    # Prepend identifier columns
    meta_cols = ["source1_entity_id", "noisy_entity_id"]
    if "label" in df_pairs.columns:
        meta_cols.append("label")
    if "is_hard_negative" in df_pairs.columns:
        meta_cols.append("is_hard_negative")

    for col in reversed(meta_cols):
        df_feats.insert(0, col, df_pairs[col].values)

    return df_feats


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


def run_feature_engineering_pipeline():
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[3]
    proc_dir = base_dir / "dataset" / "processed"
    report_file = base_dir / "feature_engineering_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 85)
    print(" PHASE 8: ENTITY-PAIR FEATURE ENGINEERING")
    print("=" * 85)
    print(f"Project Base Directory: {base_dir}")
    print(f"Data Directory:         {proc_dir}")
    print(f"Report Output File:     {report_file}\n")

    train_pairs_path = proc_dir / "train_pairs.parquet"
    val_pairs_path = proc_dir / "val_pairs.parquet"

    assert train_pairs_path.exists(), f"Train pairs file not found: {train_pairs_path}"
    assert val_pairs_path.exists(), f"Validation pairs file not found: {val_pairs_path}"

    print(f"1. Loading train pairs from {train_pairs_path.name}...")
    df_train_pairs = pd.read_parquet(train_pairs_path)
    print(f"   Loaded {len(df_train_pairs):,} training pairs.")

    print(f"2. Loading validation pairs from {val_pairs_path.name}...")
    df_val_pairs = pd.read_parquet(val_pairs_path)
    print(f"   Loaded {len(df_val_pairs):,} validation pairs.\n")

    train_feat_path = proc_dir / "train_features.parquet"
    val_feat_path = proc_dir / "val_features.parquet"
    train_sample_tsv = proc_dir / "train_features_sample.tsv"

    if train_feat_path.exists() and val_feat_path.exists():
        print(f"Loading existing feature matrices from {train_feat_path.name} and {val_feat_path.name}...")
        df_train_features = pd.read_parquet(train_feat_path)
        df_val_features = pd.read_parquet(val_feat_path)
    else:
        # Extract Features for Training Set
        print("3. Extracting 35+ engineered features for Training Pairs...")
        t_tr = time.time()
        df_train_features = extract_features_for_dataframe(df_train_pairs)
        elapsed_tr = time.time() - t_tr
        print(f"   Training features extracted in {elapsed_tr:.2f}s ({len(df_train_pairs)/elapsed_tr:.1f} pairs/sec).\n")

        # Extract Features for Validation Set
        print("4. Extracting engineered features for Validation Pairs...")
        t_val = time.time()
        df_val_features = extract_features_for_dataframe(df_val_pairs)
        elapsed_val = time.time() - t_val
        print(f"   Validation features extracted in {elapsed_val:.2f}s ({len(df_val_pairs)/elapsed_val:.1f} pairs/sec).\n")

        # Save to Parquet
        print("5. Persisting feature matrices to disk...")
        df_train_features.to_parquet(train_feat_path, index=False)
        df_val_features.to_parquet(val_feat_path, index=False)
        df_train_features.head(100).to_csv(train_sample_tsv, sep="\t", index=False)

        print(f"   Saved {len(df_train_features):,} rows to: {train_feat_path.name}")
        print(f"   Saved {len(df_val_features):,} rows to:   {val_feat_path.name}\n")

    # Step 6: Feature Distributions & Summary
    num_feature_cols = [c for c in df_train_features.columns if c not in ("source1_entity_id", "noisy_entity_id", "label", "is_hard_negative")]
    
    print("=" * 85)
    print(f" FEATURE SUITE DEFINITIONS & BASIC DISTRIBUTIONS (Total Features = {len(num_feature_cols)})")
    print("=" * 85)
    
    # Compute summary stats
    desc_df = df_train_features[num_feature_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc_df.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
    
    print(desc_df.to_string())

    print("\n" + "=" * 85)
    print(" FEATURE CORRELATION WITH TARGET LABEL (Top Correlated Features)")
    print("=" * 85)
    corrs = df_train_features[num_feature_cols].apply(lambda s: s.corr(df_train_features["label"])).dropna().sort_values(ascending=False)
    for feat_name, r_val in corrs.items():
        if pd.isna(r_val):
            continue
        bar = "#" * int(abs(r_val) * 30)
        print(f"• {feat_name:<32}: r = {r_val:6.4f}  {bar}")

    elapsed_total = time.time() - t0
    print("\n" + "=" * 85)
    print(f"Total Pipeline Time: {elapsed_total:.2f} seconds")
    print(" PHASE 8 ENTITY-PAIR FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 85)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_feature_engineering_pipeline()
