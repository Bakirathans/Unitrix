"""
Phase 4: Robust Normalization for Amazon ML Challenge 2026 - Business Entity Resolution
This module provides multi-representation normalization pipelines for business names,
business addresses, and countries.

Core Principles:
1. NEVER mutate or destroy original raw fields (business_name, business_address, country).
2. Generate MULTIPLE complementary representations (clean, alphanumeric, tokenized, sorted-tokens,
   legal-stripped, standardized-abbreviation, domain-stripped) to empower downstream blocking & matching.
3. Open-set Country handling (no hardcoded two-country assumption).
4. Full Unicode & Non-Latin Script Safety (preserves Indic, Cyrillic, CJK, Arabic, combining marks).
5. Safe Missing-Value Resilience.
"""

import sys
import re
import unicodedata
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np


RE_WHITESPACE = re.compile(r"\s+")
RE_DOMAIN_SUFFIX = re.compile(r"\.(com|org|net|in|co|io|biz|info|us|gov|edu|ai|app|tech)\b", re.IGNORECASE)
RE_WEB_PREFIX = re.compile(r"^(https?:\/\/)?(www\.)?", re.IGNORECASE)

# Legal designations discovered in Phase 3 match forensics
LEGAL_SUFFIX_SET: Set[str] = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "llp", "pvt", "private", "gmbh", "sa", "sarl", "bv",
    "plc", "lp", "pc", "pllc", "holdings", "group", "services", "center",
    "enterprises", "associates", "consultants", "solutions", "international", "intl"
}

# Standard address abbreviations discovered in Phase 3
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "room": "rm",
    "terrace": "ter",
    "place": "pl",
    "square": "sq",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "township": "twp",
    "post": "po",
    "box": "bx",
}


def clean_str_input(val: Any) -> str:
    """Safely convert any raw value to string, treating NaN, None, null, etc. as empty string."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "null", "none", "<missing>", "n/a"):
        return ""
    return s


def normalize_unicode(text: str) -> str:
    """
    Apply Unicode NFC normalization to preserve combining marks with base characters
    across Indic scripts (Tamil, Devanagari, etc.), Latin accents, and global alphabets.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Strip leading/trailing spaces and collapse multiple consecutive whitespace to a single space."""
    if not text:
        return ""
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_punctuation(text: str) -> str:
    """
    Replace punctuation and symbols with spaces while converting '&' to 'and'.
    Preserves all letters, digits, and combining marks (matras, viramas, diacritics).
    """
    if not text:
        return ""
    t = text.replace("&", " and ")
    chars = []
    for ch in t:
        cat = unicodedata.category(ch)
        # Replace Punctuation (P*) and Symbols (S*) with spaces; keep Letters (L*), Numbers (N*), Marks (M*)
        if cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        else:
            chars.append(ch)
    return normalize_whitespace("".join(chars))


def get_alphanumeric_compact(text: str) -> str:
    """
    Extract a compact alphanumeric representation (no spaces, punctuation, or symbols).
    Preserves letters, digits, and combining marks.
    """
    if not text:
        return ""
    chars = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if not (cat.startswith("P") or cat.startswith("S") or cat.startswith("Z")):
            chars.append(ch)
    return "".join(chars)


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into clean, non-empty word tokens."""
    if not text:
        return []
    return [t for t in text.split() if t]


def get_sorted_tokens_str(tokens: List[str]) -> str:
    """
    Sort tokens alphabetically and join with single space.
    Crucial for matching permuted names and reordered addresses.
    """
    if not tokens:
        return ""
    return " ".join(sorted(tokens))


def strip_legal_suffixes(tokens: List[str]) -> List[str]:
    """
    Filter out common corporate/legal suffix tokens.
    Returns the core distinct business identity tokens.
    """
    if not tokens:
        return []
    filtered = [t for t in tokens if t.lower() not in LEGAL_SUFFIX_SET]
    # If stripping removed everything (e.g. name was literally just 'Enterprises Inc'), keep original tokens
    return filtered if filtered else tokens


def clean_domain_name(text: str) -> str:
    """
    Phase 3 Finding: 5.05% of noisy matches are web domains (e.g. 'lopezinnovativenetwork.com').
    Transforms web domain syntax into natural readable words.
    """
    if not text:
        return ""
    t = RE_WEB_PREFIX.sub("", text)
    t = RE_DOMAIN_SUFFIX.sub("", t)
    return t


def standardize_address_tokens(tokens: List[str]) -> List[str]:
    """
    Phase 3 Finding: 46.72% of addresses differ by abbreviations (e.g. 'st' vs 'street').
    Maps common street type tokens to canonical abbreviated forms.
    """
    if not tokens:
        return []
    return [ADDRESS_ABBREVIATIONS.get(t.lower(), t.lower()) for t in tokens]


def normalize_country(country_val: Any) -> str:
    """
    Normalize country as an open-set string.
    Trims whitespace, handles missing values safely, and standardizes casing.
    Does NOT hardcode to only US and India.
    """
    cleaned = clean_str_input(country_val)
    if not cleaned:
        return ""
    return normalize_whitespace(cleaned.upper())


# =========================================================================================
# STRUCTURED MULTI-REPRESENTATION DATA CLASSES
# =========================================================================================

@dataclass
class NormalizedName:
    original: str
    is_empty: bool
    clean: str
    alphanumeric: str
    tokens: List[str]
    sorted_tokens: str
    no_legal: str
    no_legal_tokens: List[str]
    no_legal_sorted: str
    domain_clean: str
    domain_clean_tokens: List[str]


@dataclass
class NormalizedAddress:
    original: str
    is_empty: bool
    clean: str
    alphanumeric: str
    tokens: List[str]
    sorted_tokens: str
    standardized_tokens: List[str]
    standardized_sorted: str


@dataclass
class NormalizedEntityRecord:
    entity_id: str
    # Guaranteed original fields
    original_business_name: str
    original_business_address: str
    original_country: str
    # Multi-representation normalized structures
    name: NormalizedName
    address: NormalizedAddress
    country: str


# =========================================================================================
# BUILDER FUNCTIONS
# =========================================================================================

def build_name_representations(raw_name: Any) -> NormalizedName:
    """Construct all multi-representation features for a business name."""
    raw_str = "" if raw_name is None or pd.isna(raw_name) else str(raw_name)
    cleaned_input = clean_str_input(raw_name)
    
    if not cleaned_input:
        return NormalizedName(
            original=raw_str,
            is_empty=True,
            clean="",
            alphanumeric="",
            tokens=[],
            sorted_tokens="",
            no_legal="",
            no_legal_tokens=[],
            no_legal_sorted="",
            domain_clean="",
            domain_clean_tokens=[]
        )

    # 1. Unicode NFC & lowercase
    u_norm = normalize_unicode(cleaned_input).lower()
    
    # 2. Domain-cleaned version
    d_clean = clean_domain_name(u_norm)
    d_clean_punct = normalize_punctuation(d_clean)
    d_tokens = tokenize(d_clean_punct)

    # 3. Standard punctuation & whitespace cleaned
    p_clean = normalize_punctuation(u_norm)
    tokens = tokenize(p_clean)
    sorted_toks = get_sorted_tokens_str(tokens)
    alphanumeric = get_alphanumeric_compact(p_clean)

    # 4. Legal suffix stripped
    no_legal_toks = strip_legal_suffixes(tokens)
    no_legal_str = " ".join(no_legal_toks)
    no_legal_sorted = get_sorted_tokens_str(no_legal_toks)

    return NormalizedName(
        original=raw_str,
        is_empty=False,
        clean=p_clean,
        alphanumeric=alphanumeric,
        tokens=tokens,
        sorted_tokens=sorted_toks,
        no_legal=no_legal_str,
        no_legal_tokens=no_legal_toks,
        no_legal_sorted=no_legal_sorted,
        domain_clean=d_clean_punct,
        domain_clean_tokens=d_tokens
    )


def build_address_representations(raw_address: Any) -> NormalizedAddress:
    """Construct all multi-representation features for a business address."""
    raw_str = "" if raw_address is None or pd.isna(raw_address) else str(raw_address)
    cleaned_input = clean_str_input(raw_address)

    if not cleaned_input:
        return NormalizedAddress(
            original=raw_str,
            is_empty=True,
            clean="",
            alphanumeric="",
            tokens=[],
            sorted_tokens="",
            standardized_tokens=[],
            standardized_sorted=""
        )

    # 1. Unicode NFC & lowercase
    u_norm = normalize_unicode(cleaned_input).lower()

    # 2. Punctuation & whitespace cleaned
    p_clean = normalize_punctuation(u_norm)
    tokens = tokenize(p_clean)
    sorted_toks = get_sorted_tokens_str(tokens)
    alphanumeric = get_alphanumeric_compact(p_clean)

    # 3. Standardized address abbreviations
    std_tokens = standardize_address_tokens(tokens)
    std_sorted = get_sorted_tokens_str(std_tokens)

    return NormalizedAddress(
        original=raw_str,
        is_empty=False,
        clean=p_clean,
        alphanumeric=alphanumeric,
        tokens=tokens,
        sorted_tokens=sorted_toks,
        standardized_tokens=std_tokens,
        standardized_sorted=std_sorted
    )


def normalize_record(
    entity_id: Any,
    business_name: Any,
    business_address: Any,
    country: Any
) -> NormalizedEntityRecord:
    """
    Main entry point for single-record normalization.
    Preserves original values in their raw form and produces all multi-representations.
    """
    eid_str = str(entity_id).strip() if entity_id is not None and not pd.isna(entity_id) else ""
    orig_name = "" if business_name is None or pd.isna(business_name) else str(business_name)
    orig_addr = "" if business_address is None or pd.isna(business_address) else str(business_address)
    orig_country = "" if country is None or pd.isna(country) else str(country)

    norm_name = build_name_representations(business_name)
    norm_addr = build_address_representations(business_address)
    norm_country = normalize_country(country)

    return NormalizedEntityRecord(
        entity_id=eid_str,
        original_business_name=orig_name,
        original_business_address=orig_addr,
        original_country=orig_country,
        name=norm_name,
        address=norm_addr,
        country=norm_country
    )


def normalize_dataframe(df: pd.DataFrame) -> List[NormalizedEntityRecord]:
    """Normalize an entire DataFrame of entities efficiently while strictly preserving original data."""
    records = []
    # Identify column names flexibly
    id_col = "entity_id" if "entity_id" in df.columns else ("source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0])
    name_col = "business_name" if "business_name" in df.columns else None
    addr_col = "business_address" if "business_address" in df.columns else None
    country_col = "country" if "country" in df.columns else None

    for _, row in df.iterrows():
        eid = row[id_col]
        b_name = row[name_col] if name_col else ""
        b_addr = row[addr_col] if addr_col else ""
        b_cntry = row[country_col] if country_col else ""
        records.append(normalize_record(eid, b_name, b_addr, b_cntry))

    return records


# =========================================================================================
# COMPREHENSIVE VERIFICATION TESTS & EXAMPLES
# =========================================================================================

def run_tests_and_demonstration():
    print("=" * 85)
    print(" PHASE 4: ROBUST NORMALIZATION - VERIFICATION & TESTS")
    print("=" * 85)

    test_cases = [
        # Case 1: Legal Suffixes & Punctuation
        {
            "id": "T-1",
            "name": "Acme Industrial Technologies, Inc.",
            "addr": "100 South Main Street, Suite #400, Springfield, IL",
            "country": "US",
            "desc": "Legal Suffixes, Directionals, Address Abbreviations"
        },
        # Case 2: Scrambled / Reordered Address (Phase 3 finding)
        {
            "id": "T-2",
            "name": "Quality Total Runway LLC",
            "addr": "Springfield, IL, Suite 400, 100 S Main St",
            "country": "United States",
            "desc": "Reordered Address Components & Abbreviated Tokens"
        },
        # Case 3: Domain Name as Business Name (Phase 3 finding)
        {
            "id": "T-3",
            "name": "https://www.lopezinnovativenetwork.com",
            "addr": "1005 Delmas St, Houston, TX",
            "country": "US",
            "desc": "Domain Name Stripping and Parsing"
        },
        # Case 4: Multilingual / Non-Latin Unicode (Tamil) (Phase 3 finding)
        {
            "id": "T-4",
            "name": "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
            "addr": "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு",
            "country": "India",
            "desc": "Tamil Script Name & Multilingual Address Preservation"
        },
        # Case 5: Non-Latin Unicode (Devanagari / Hindi) (Phase 3 finding)
        {
            "id": "T-5",
            "name": "सनराइज इंफ्रा प्राइवेट लिमिटेड",
            "addr": "B-57 Okhla Industrial Area Phase-1, South Delhi, Delhi",
            "country": "India",
            "desc": "Devanagari Script Name & Address Preservation"
        },
        # Case 6: Open-set International Country (Germany / Brazil / Japan)
        {
            "id": "T-6",
            "name": "Müller & Schmidt Maschinenbau GmbH",
            "addr": "Industriestraße 12, 80331 München",
            "country": "Germany",
            "desc": "Open-Set Country & German Umlauts"
        },
        # Case 7: Missing / Null Fields Resilience
        {
            "id": "T-7",
            "name": np.nan,
            "addr": None,
            "country": "<MISSING>",
            "desc": "Missing / Null / NaN Resilience"
        }
    ]

    for tc in test_cases:
        print(f"\n--- Test Case {tc['id']}: {tc['desc']} ---")
        orig_name_input = tc["name"]
        orig_addr_input = tc["addr"]
        orig_cntry_input = tc["country"]

        record = normalize_record(tc["id"], orig_name_input, orig_addr_input, orig_cntry_input)

        # 1. Verify Original Immutability
        print(f"  [RAW INPUTS PRESERVED]")
        print(f"    - Original Name:    {repr(record.original_business_name)}")
        print(f"    - Original Address: {repr(record.original_business_address)}")
        print(f"    - Original Country: {repr(record.original_country)}")

        # 2. Multi-representation Name Features
        print(f"  [NORMALIZED NAME REPRESENTATIONS]")
        print(f"    - Clean:            '{record.name.clean}'")
        print(f"    - Alphanumeric:     '{record.name.alphanumeric}'")
        print(f"    - Sorted Tokens:    '{record.name.sorted_tokens}'")
        print(f"    - No Legal Suffix:  '{record.name.no_legal}'")
        print(f"    - No Legal Sorted:  '{record.name.no_legal_sorted}'")
        if record.name.domain_clean != record.name.clean:
            print(f"    - Domain Cleaned:   '{record.name.domain_clean}'")

        # 3. Multi-representation Address Features
        print(f"  [NORMALIZED ADDRESS REPRESENTATIONS]")
        print(f"    - Clean:            '{record.address.clean}'")
        print(f"    - Alphanumeric:     '{record.address.alphanumeric}'")
        print(f"    - Sorted Tokens:    '{record.address.sorted_tokens}'")
        print(f"    - Std Abbr Sorted:  '{record.address.standardized_sorted}'")

        # 4. Open-set Country
        print(f"  [NORMALIZED COUNTRY (OPEN-SET)]")
        print(f"    - Country:          '{record.country}'")

        # Assertions to ensure invariants hold
        assert record.entity_id == tc["id"]
        if pd.isna(orig_name_input) or orig_name_input is None:
            assert record.name.is_empty is True
        else:
            assert record.original_business_name == str(orig_name_input)
        if pd.isna(orig_addr_input) or orig_addr_input is None:
            assert record.address.is_empty is True

    # Check pair invariant: T-1 and T-2 addresses match under standardized_sorted!
    r1 = normalize_record(test_cases[0]["id"], test_cases[0]["name"], test_cases[0]["addr"], test_cases[0]["country"])
    r2 = normalize_record(test_cases[1]["id"], test_cases[1]["name"], test_cases[1]["addr"], test_cases[1]["country"])
    
    print("\n" + "=" * 85)
    print(" INVARIANT & EQUIVALENCE VALIDATION")
    print("=" * 85)
    print(f"Raw Address 1: '{r1.original_business_address}'")
    print(f"Raw Address 2: '{r2.original_business_address}'")
    print(f"Raw Address Equality:                  {r1.original_business_address == r2.original_business_address}")
    print(f"Normalized Standardized Sorted Match:  {r1.address.standardized_sorted == r2.address.standardized_sorted}")
    assert r1.address.standardized_sorted == r2.address.standardized_sorted, "Reordered address tokens must resolve to equal sorted standardized representations!"

    print("\n" + "=" * 85)
    print(" ALL NORMALIZATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 85)


if __name__ == "__main__":
    run_tests_and_demonstration()
