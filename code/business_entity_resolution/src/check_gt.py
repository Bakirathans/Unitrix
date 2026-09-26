"""
Test script to check ground truth parsing and dataset stats.
"""
import time
import pandas as pd
from collections import defaultdict, Counter

t0 = time.time()
gt_path = "dataset/train/train_ground_truth.tsv"
print("Loading GT...")
df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
print(f"Loaded {len(df_gt):,} rows in {time.time() - t0:.2f}s")

total_s1 = len(df_gt)
s1_to_gt = {}
gt_match_counts = Counter()
s2_matches = 0
s3_matches = 0
total_matches = 0
singleton_count = 0

for s1_id, m_str in zip(df_gt["source1_entity_id"], df_gt["matched_entity_ids"]):
    m_str = m_str.strip()
    if m_str:
        tokens = [t.strip() for t in m_str.split(",") if t.strip()]
        s1_to_gt[s1_id] = set(tokens)
        match_len = len(tokens)
        gt_match_counts[match_len] += 1
        total_matches += match_len
        for t in tokens:
            if t.startswith("S2-"):
                s2_matches += 1
            elif t.startswith("S3-"):
                s3_matches += 1
    else:
        s1_to_gt[s1_id] = set()
        singleton_count += 1
        gt_match_counts[0] += 1

print(f"Total S1 entities: {total_s1:,}")
print(f"Singletons (0 true matches): {singleton_count:,} ({singleton_count/total_s1*100:.2f}%)")
print(f"S1 with >=1 matches: {len(s1_to_gt) - singleton_count:,} ({(len(s1_to_gt) - singleton_count)/total_s1*100:.2f}%)")
print(f"Total True Match Pairs: {total_matches:,}")
print(f"  - S1 -> S2 matches: {s2_matches:,} ({s2_matches/total_matches*100:.2f}%)")
print(f"  - S1 -> S3 matches: {s3_matches:,} ({s3_matches/total_matches*100:.2f}%)")
print("Top Match Count distribution:", sorted(gt_match_counts.items())[:10])
