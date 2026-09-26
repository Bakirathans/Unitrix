"""
Verification script for high-speed multi-evidence candidate ranking on Ground Truth
"""
import time
import math
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

print("Loading GT...")
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
s1_to_gt = {}
total_gt_matches = 0
for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
    if m_str.strip():
        toks = set(m_str.strip().split(","))
        s1_to_gt[s1_id] = toks
        total_gt_matches += len(toks)
    else:
        s1_to_gt[s1_id] = set()

del gt_df
print(f"Loaded {len(s1_to_gt):,} S1 GT mappings ({total_gt_matches:,} total true matches).")

# Stream index 1M sample of S2/S3
from run_blocking_v2_experiment import fast_extract_keys

KEY_WEIGHTS = {
    "alpha": 1.20,
    "dom6": 1.10,
    "2tok": 1.00,
    "pfx8": 0.90,
    "pfx6": 0.75,
    "anum": 0.70,
    "tok": 0.50
}

def get_key_weight(k: str) -> float:
    # Key format: COUNTRY|type:value
    if "|" in k and ":" in k:
        k_type = k.split("|")[1].split(":")[0]
        return KEY_WEIGHTS.get(k_type, 0.50)
    return 0.50

print("Verification test ready.")
