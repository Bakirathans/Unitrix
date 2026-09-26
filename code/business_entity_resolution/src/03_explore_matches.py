"""
Phase 3: Ground-Truth Match Forensics for Amazon ML Challenge 2026 - Business Entity Resolution
This script conducts an empirical forensic analysis of true matching pairs across Source 1,
Source 2, and Source 3 to discover the exact nature and frequency of real-world noise in
business names and business addresses.

No normalization rules are invented or applied here; this is purely exploratory and diagnostic.
"""

import sys
import gc
import re
import time
import random
import unicodedata
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
from rapidfuzz import fuzz, distance

# Non-Latin Unicode script regex detectors
RE_NON_LATIN = re.compile(r"[^\x00-\x7F]")
RE_TAMIL = re.compile(r"[\u0B80-\u0BFF]")
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
RE_TELUGU = re.compile(r"[\u0C00-\u0C7F]")
RE_KANNADA = re.compile(r"[\u0C80-\u0CFF]")
RE_MALAYALAM = re.compile(r"[\u0D00-\u0D7F]")
RE_BENGALI = re.compile(r"[\u0980-\u09FF]")
RE_ARABIC = re.compile(r"[\u0600-\u06FF]")
RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
RE_CJK = re.compile(r"[\u4E00-\u9FFF]")

RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_ZIPCODE = re.compile(r"\b\d{5,6}\b")
RE_DOMAIN = re.compile(r"\b[a-zA-Z0-9-]+\.(com|org|net|in|co|io|biz|info|us|gov|edu)\b", re.IGNORECASE)

LEGAL_TERMS = {
    "inc", "incorporated", "llc", "l.l.c.", "ltd", "limited", "corp", "corporation",
    "co", "company", "llp", "l.l.p.", "pvt", "private", "gmbh", "sa", "sarl", "bv",
    "plc", "lp", "pc", "pllc"
}

STATE_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "tamil nadu": "tn", "maharashtra": "mh", "karnataka": "ka", "delhi": "dl", "gujarat": "gj",
    "uttar pradesh": "up", "telangana": "ts", "andhra pradesh": "ap", "kerala": "kl", "west bengal": "wb"
}

ADDR_ABBR_PAIRS = [
    ("street", "st"), ("road", "rd"), ("avenue", "ave"), ("boulevard", "blvd"),
    ("highway", "hwy"), ("parkway", "pkwy"), ("suite", "ste"), ("apartment", "apt"),
    ("drive", "dr"), ("lane", "ln"), ("court", "ct"), ("circle", "cir"), ("floor", "fl")
]


def clean_str(val: Any) -> str:
    """Safely convert any value to string, treating NaN/None/null as empty string."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "null", "none", "<missing>"):
        return ""
    return s


def detect_script(text: str) -> str:
    """Identify the primary non-Latin script family if present."""
    if RE_TAMIL.search(text): return "Tamil"
    if RE_DEVANAGARI.search(text): return "Devanagari (Hindi/Marathi)"
    if RE_TELUGU.search(text): return "Telugu"
    if RE_KANNADA.search(text): return "Kannada"
    if RE_MALAYALAM.search(text): return "Malayalam"
    if RE_BENGALI.search(text): return "Bengali"
    if RE_ARABIC.search(text): return "Arabic"
    if RE_CYRILLIC.search(text): return "Cyrillic"
    if RE_CJK.search(text): return "CJK (Chinese/Japanese)"
    if RE_NON_LATIN.search(text): return "Other Non-Latin"
    return "Latin/English"


def strip_legal_suffixes(text: str) -> str:
    tokens = [t.strip(",.") for t in text.lower().split()]
    filtered = [t for t in tokens if t not in LEGAL_TERMS]
    return " ".join(filtered)


def normalize_punct_whitespace(text: str) -> str:
    t = RE_PUNCT.sub(" ", text)
    t = RE_WHITESPACE.sub(" ", t).strip()
    return t.lower()


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


def run_forensics(sample_size: int = 60000, random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    t0 = time.time()

    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "dataset" / "train"
    report_file = base_dir / "match_forensics_report.txt"

    tee = TeeLogger(report_file)
    sys.stdout = tee

    print("=" * 85)
    print(" PHASE 3: GROUND-TRUTH MATCH FORENSICS")
    print("=" * 85)
    print(f"Project Base Directory: {base_dir}")
    print(f"Report Output File:     {report_file}\n")

    # Step 1: Load Ground Truth and extract true matching pairs
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print("1. Loading ground-truth data...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    # Flatten ground truth into (s1_id, match_id, source_type) pairs
    print("   Extracting ground-truth match pairs...")
    all_pairs: List[Tuple[str, str, str]] = []
    for s1_id, matched_raw in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if pd.isna(matched_raw) or not str(matched_raw).strip():
            continue
        tokens = [t.strip() for t in str(matched_raw).split(",") if t.strip()]
        for m in tokens:
            src = "S2" if m.startswith("S2-") else ("S3" if m.startswith("S3-") else "Other")
            all_pairs.append((str(s1_id).strip(), m, src))

    total_gt_pairs = len(all_pairs)
    print(f"   Total Ground-Truth Positive Pairs in Dataset: {total_gt_pairs:,}")

    # Sample representative pairs for deep forensic analysis
    sampled_pairs = random.sample(all_pairs, min(sample_size, total_gt_pairs))
    print(f"   Sampled {len(sampled_pairs):,} true pairs for statistical match forensics (Seed: {random_seed}).\n")

    # Group required entity IDs
    needed_s1 = {p[0] for p in sampled_pairs}
    needed_s2 = {p[1] for p in sampled_pairs if p[2] == "S2"}
    needed_s3 = {p[1] for p in sampled_pairs if p[2] == "S3"}

    print(f"2. Fetching text attributes for sampled entities (S1: {len(needed_s1):,}, S2: {len(needed_s2):,}, S3: {len(needed_s3):,})...")
    
    print("   - Loading Source 1 records...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    s1_records = df_s1[df_s1["entity_id"].isin(needed_s1)].set_index("entity_id").to_dict(orient="index")
    del df_s1
    gc.collect()

    print("   - Loading Source 2 records...")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    s2_records = df_s2[df_s2["entity_id"].isin(needed_s2)].set_index("entity_id").to_dict(orient="index")
    del df_s2
    gc.collect()

    print("   - Loading Source 3 records...")
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
    s3_records = df_s3[df_s3["entity_id"].isin(needed_s3)].set_index("entity_id").to_dict(orient="index")
    del df_s3
    gc.collect()

    print("   Attribute retrieval complete.\n")

    # Forensic Counters
    N = len(sampled_pairs)

    # Business Name Metrics
    name_exact = 0
    name_case_diff = 0
    name_punct_diff = 0
    name_whitespace_diff = 0
    name_reordered_tokens = 0
    name_legal_diff = 0
    name_typo_slight = 0  # 80 <= ratio < 100
    name_typo_moderate = 0 # 50 <= ratio < 80
    name_low_sim = 0       # ratio < 50
    name_domain_in_noisy = 0
    name_translit = 0
    name_script_counts = Counter()

    # Address Metrics
    addr_exact = 0
    addr_missing = 0
    addr_punct_diff = 0
    addr_whitespace_diff = 0
    addr_reordered_tokens = 0
    addr_abbr_diff = 0
    addr_zip_exact = 0
    addr_zip_diff = 0
    addr_zip_missing = 0
    addr_state_expanded_abbr = 0
    addr_translit = 0
    addr_low_sim = 0
    addr_script_counts = Counter()

    # Informative comparison
    addr_more_informative = 0
    name_more_informative = 0

    # Detailed example collector
    examples_false_looking_true = []
    examples_dba_domain = []
    examples_translit = []
    examples_addr_reordered = []
    examples_addr_informative = []
    examples_name_informative = []

    print("3. Executing forensic comparisons on sampled true match pairs...")
    for s1_id, match_id, src in sampled_pairs:
        s1_row = s1_records.get(s1_id, {})
        m_row = s2_records.get(match_id, {}) if src == "S2" else s3_records.get(match_id, {})

        s1_name = clean_str(s1_row.get("business_name"))
        m_name = clean_str(m_row.get("business_name"))
        s1_addr = clean_str(s1_row.get("business_address"))
        m_addr = clean_str(m_row.get("business_address"))
        s1_country = clean_str(s1_row.get("country"))
        m_country = clean_str(m_row.get("country"))

        # --- BUSINESS NAME ANALYSIS ---
        if s1_name == m_name and s1_name != "":
            name_exact += 1
        else:
            if s1_name.lower() == m_name.lower():
                name_case_diff += 1
            
            p_s1 = normalize_punct_whitespace(s1_name)
            p_m = normalize_punct_whitespace(m_name)
            
            if p_s1 == p_m:
                if s1_name.replace(" ", "") == m_name.replace(" ", ""):
                    name_whitespace_diff += 1
                else:
                    name_punct_diff += 1
            else:
                # Check token reordering
                tokens_s1 = sorted(p_s1.split())
                tokens_m = sorted(p_m.split())
                if tokens_s1 == tokens_m and len(tokens_s1) > 1:
                    name_reordered_tokens += 1
                
                # Check legal suffix variation
                leg_s1 = strip_legal_suffixes(p_s1)
                leg_m = strip_legal_suffixes(p_m)
                if leg_s1 == leg_m and leg_s1 != "":
                    name_legal_diff += 1

        # Check domain style
        if RE_DOMAIN.search(m_name) and not RE_DOMAIN.search(s1_name):
            name_domain_in_noisy += 1
            if len(examples_dba_domain) < 5:
                examples_dba_domain.append((s1_id, match_id, s1_name, m_name, s1_addr, m_addr))

        # Check script / transliteration
        s1_script = detect_script(s1_name)
        m_script = detect_script(m_name)
        if m_script != "Latin/English" or s1_script != "Latin/English":
            name_translit += 1
            name_script_counts[f"{s1_script} -> {m_script}"] += 1
            if len(examples_translit) < 5:
                examples_translit.append((s1_id, match_id, s1_name, m_name, s1_addr, m_addr))

        # RapidFuzz similarity ratios
        name_ratio = fuzz.ratio(s1_name.lower(), m_name.lower())
        if name_ratio >= 80 and name_ratio < 100:
            name_typo_slight += 1
        elif name_ratio >= 50 and name_ratio < 80:
            name_typo_moderate += 1
        elif name_ratio < 50:
            name_low_sim += 1
            if len(examples_false_looking_true) < 5 and (s1_name and m_name):
                examples_false_looking_true.append((s1_id, match_id, s1_name, m_name, s1_addr, m_addr, name_ratio))

        # --- BUSINESS ADDRESS ANALYSIS ---
        if not m_addr or m_addr.lower() in ("nan", "null", ""):
            addr_missing += 1
        elif s1_addr == m_addr:
            addr_exact += 1
        else:
            p_s1_addr = normalize_punct_whitespace(s1_addr)
            p_m_addr = normalize_punct_whitespace(m_addr)

            if p_s1_addr == p_m_addr:
                if s1_addr.replace(" ", "") == m_addr.replace(" ", ""):
                    addr_whitespace_diff += 1
                else:
                    addr_punct_diff += 1
            else:
                tokens_s1_addr = sorted(p_s1_addr.split())
                tokens_m_addr = sorted(p_m_addr.split())
                if tokens_s1_addr == tokens_m_addr and len(tokens_s1_addr) > 2:
                    addr_reordered_tokens += 1
                    if len(examples_addr_reordered) < 5:
                        examples_addr_reordered.append((s1_id, match_id, s1_addr, m_addr))

            # Check address abbreviations (st vs street, etc.)
            for long_v, short_v in ADDR_ABBR_PAIRS:
                if (long_v in p_s1_addr and short_v in p_m_addr) or (short_v in p_s1_addr and long_v in p_m_addr):
                    addr_abbr_diff += 1
                    break

            # State expansion / abbreviation (california vs ca)
            for full_state, ab_state in STATE_MAP.items():
                if (full_state in p_s1_addr and f" {ab_state} " in f" {p_m_addr} ") or \
                   (f" {ab_state} " in f" {p_s1_addr} " and full_state in p_m_addr):
                    addr_state_expanded_abbr += 1
                    break

            # Postal code check
            z1 = RE_ZIPCODE.findall(s1_addr)
            z2 = RE_ZIPCODE.findall(m_addr)
            if z1 and z2:
                if set(z1) & set(z2):
                    addr_zip_exact += 1
                else:
                    addr_zip_diff += 1
            else:
                addr_zip_missing += 1

            # Address script
            addr_s1_script = detect_script(s1_addr)
            addr_m_script = detect_script(m_addr)
            if addr_m_script != "Latin/English" or addr_s1_script != "Latin/English":
                addr_translit += 1
                addr_script_counts[f"{addr_s1_script} -> {addr_m_script}"] += 1

        addr_ratio = fuzz.ratio(s1_addr.lower(), m_addr.lower()) if (s1_addr and m_addr) else 0

        # Compare informativeness
        if name_ratio < 40 and addr_ratio >= 75:
            addr_more_informative += 1
            if len(examples_addr_informative) < 4:
                examples_addr_informative.append((s1_id, match_id, s1_name, m_name, s1_addr, m_addr, name_ratio, addr_ratio))
        elif name_ratio >= 85 and addr_ratio < 40:
            name_more_informative += 1
            if len(examples_name_informative) < 4:
                examples_name_informative.append((s1_id, match_id, s1_name, m_name, s1_addr, m_addr, name_ratio, addr_ratio))

    print("   Forensics processing completed successfully.\n")

    # Step 4: Inspect Homonyms / Similar Names Across Distinct S1 Entities
    print("4. Inspecting similar / identical names belonging to DIFFERENT entities in Source 1...")
    s1_all_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    s1_name_counts = s1_all_df["business_name"].dropna().value_counts()
    frequent_s1_names = s1_name_counts[s1_name_counts > 1]
    
    homonym_examples = []
    for h_name in frequent_s1_names.head(5).index:
        rows = s1_all_df[s1_all_df["business_name"] == h_name].head(3)
        homonym_examples.append((h_name, rows[["entity_id", "business_name", "business_address", "country"]].to_dict(orient="records")))

    del s1_all_df
    gc.collect()

    # --- PRINT FORENSIC REPORT ---
    print("=" * 85)
    print(" BUSINESS NAME NOISE FORENSIC STATISTICS (N = {:,})".format(N))
    print("=" * 85)
    print(f"1.  Exact Identical Name:                   {name_exact:7,d} ({name_exact/N*100:6.2f}%)")
    print(f"2.  Case Difference Only:                   {name_case_diff:7,d} ({name_case_diff/N*100:6.2f}%)")
    print(f"3.  Punctuation Difference Only:            {name_punct_diff:7,d} ({name_punct_diff/N*100:6.2f}%)")
    print(f"4.  Whitespace Difference Only:             {name_whitespace_diff:7,d} ({name_whitespace_diff/N*100:6.2f}%)")
    print(f"5.  Token Permutation / Reordering:         {name_reordered_tokens:7,d} ({name_reordered_tokens/N*100:6.2f}%)")
    print(f"6.  Legal Suffix Variation (LLC/Inc/Ltd):   {name_legal_diff:7,d} ({name_legal_diff/N*100:6.2f}%)")
    print(f"7.  Slight Typos (Fuzz Ratio 80-99%):       {name_typo_slight:7,d} ({name_typo_slight/N*100:6.2f}%)")
    print(f"8.  Moderate Divergence (Fuzz Ratio 50-79%):{name_typo_moderate:7,d} ({name_typo_moderate/N*100:6.2f}%)")
    print(f"9.  Severe Divergence (Fuzz Ratio < 50%):   {name_low_sim:7,d} ({name_low_sim/N*100:6.2f}%)")
    print(f"10. Web Domain as Business Name (.com/.in): {name_domain_in_noisy:7,d} ({name_domain_in_noisy/N*100:6.2f}%)")
    print(f"11. Multilingual / Script Transliteration:  {name_translit:7,d} ({name_translit/N*100:6.2f}%)")
    print("    Script Breakdown:")
    for sc, count in name_script_counts.most_common(6):
        print(f"      - {sc}: {count:,} ({count/N*100:.2f}%)")

    print("\n" + "=" * 85)
    print(" BUSINESS ADDRESS NOISE FORENSIC STATISTICS (N = {:,})".format(N))
    print("=" * 85)
    print(f"1.  Exact Identical Address:                {addr_exact:7,d} ({addr_exact/N*100:6.2f}%)")
    print(f"2.  Missing / Null Address in Noisy Match:  {addr_missing:7,d} ({addr_missing/N*100:6.2f}%)")
    print(f"3.  Punctuation Difference:                 {addr_punct_diff:7,d} ({addr_punct_diff/N*100:6.2f}%)")
    print(f"4.  Whitespace Difference:                  {addr_whitespace_diff:7,d} ({addr_whitespace_diff/N*100:6.2f}%)")
    print(f"5.  Component Reordering (City-first/etc):  {addr_reordered_tokens:7,d} ({addr_reordered_tokens/N*100:6.2f}%)")
    print(f"6.  Common Address Abbreviations (St/Ave):  {addr_abbr_diff:7,d} ({addr_abbr_diff/N*100:6.2f}%)")
    print(f"7.  State Abbrev vs Full Name (NY/New York):{addr_state_expanded_abbr:7,d} ({addr_state_expanded_abbr/N*100:6.2f}%)")
    print(f"8.  Postal Code Matches Exactly:            {addr_zip_exact:7,d} ({addr_zip_exact/N*100:6.2f}%)")
    print(f"9.  Postal Code Differs / Missing:          {addr_zip_diff + addr_zip_missing:7,d} ({(addr_zip_diff + addr_zip_missing)/N*100:6.2f}%)")
    print(f"10. Multilingual / Transliterated Address:  {addr_translit:7,d} ({addr_translit/N*100:6.2f}%)")

    print("\n" + "=" * 85)
    print(" INFORMATIVENESS ASYMMETRY (N = {:,})".format(N))
    print("=" * 85)
    print(f"• Cases where Address is more informative than Name: {addr_more_informative:6,d} ({addr_more_informative/N*100:.2f}%)")
    print(f"  (Low name similarity <40%, but high address similarity >=75%)")
    print(f"• Cases where Name is more informative than Address: {name_more_informative:6,d} ({name_more_informative/N*100:.2f}%)")
    print(f"  (High name similarity >=85%, but address is missing or severely corrupted <40%)")

    # Concrete Examples
    print("\n" + "=" * 85)
    print(" FORENSIC EVIDENCE & ANOMALY EXAMPLES")
    print("=" * 85)

    print("\n[A] False-Looking True Matches (Name Levenshtein Ratio < 40%):")
    for idx, (s1_id, m_id, s1_n, m_n, s1_a, m_a, ratio) in enumerate(examples_false_looking_true, 1):
        print(f"  Example A{idx} [Sim: {ratio:.1f}%]:")
        print(f"    S1 [{s1_id}]:   Name: '{s1_n}' | Addr: '{s1_a}'")
        print(f"    Match [{m_id}]: Name: '{m_n}' | Addr: '{m_a}'")

    print("\n[B] Domain-Style and DBA Name Variations:")
    for idx, (s1_id, m_id, s1_n, m_n, s1_a, m_a) in enumerate(examples_dba_domain, 1):
        print(f"  Example B{idx}:")
        print(f"    S1 [{s1_id}]:   Name: '{s1_n}' | Addr: '{s1_a}'")
        print(f"    Match [{m_id}]: Name: '{m_n}' | Addr: '{m_a}'")

    print("\n[C] Multilingual Transliteration (Tamil / Hindi / Regional Scripts):")
    for idx, (s1_id, m_id, s1_n, m_n, s1_a, m_a) in enumerate(examples_translit, 1):
        print(f"  Example C{idx}:")
        print(f"    S1 [{s1_id}]:   Name: '{s1_n}' | Addr: '{s1_a}'")
        print(f"    Match [{m_id}]: Name: '{m_n}' | Addr: '{m_a}'")

    print("\n[D] Address Component Reordering (e.g., City/State First):")
    for idx, (s1_id, m_id, s1_a, m_a) in enumerate(examples_addr_reordered, 1):
        print(f"  Example D{idx}:")
        print(f"    S1 [{s1_id}]:   '{s1_a}'")
        print(f"    Match [{m_id}]: '{m_a}'")

    print("\n[E] Address is Crucial (Address matches, but Name is DBA/Domain/Obfuscated):")
    for idx, (s1_id, m_id, s1_n, m_n, s1_a, m_a, n_r, a_r) in enumerate(examples_addr_informative, 1):
        print(f"  Example E{idx} [Name Sim: {n_r:.1f}%, Addr Sim: {a_r:.1f}%]:")
        print(f"    S1 [{s1_id}]:   Name: '{s1_n}' | Addr: '{s1_a}'")
        print(f"    Match [{m_id}]: Name: '{m_n}' | Addr: '{m_a}'")

    print("\n[F] Name is Crucial (Name matches, but Address is Missing/Empty/Corrupted):")
    for idx, (s1_id, m_id, s1_n, m_n, s1_a, m_a, n_r, a_r) in enumerate(examples_name_informative, 1):
        print(f"  Example F{idx} [Name Sim: {n_r:.1f}%, Addr Sim: {a_r:.1f}%]:")
        print(f"    S1 [{s1_id}]:   Name: '{s1_n}' | Addr: '{s1_a}'")
        print(f"    Match [{m_id}]: Name: '{m_n}' | Addr: '{m_a}'")

    print("\n[G] Homonyms: Identical Names Belonging to DISTINCT S1 Entities:")
    print(f"    Found {len(frequent_s1_names):,} distinct names appearing multiple times across different S1 IDs.")
    for idx, (h_name, recs) in enumerate(homonym_examples, 1):
        print(f"  Homonym G{idx}: Name '{h_name}'")
        for r in recs:
            print(f"    - ID: {r['entity_id']} | Address: {r['business_address']} | Country: {r['country']}")

    elapsed = time.time() - t0
    print("\n" + "=" * 85)
    print(f"Total Elapsed Time: {elapsed:.2f} seconds")
    print(" PHASE 3 GROUND-TRUTH MATCH FORENSICS COMPLETED SUCCESSFULLY")
    print("=" * 85)

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    run_forensics(sample_size=60000)
