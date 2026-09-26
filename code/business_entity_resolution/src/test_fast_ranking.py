"""
Fast Key-Weighted & Multi-Evidence Candidate Ranking Benchmark
"""
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

print("Testing vector/hash accumulator ranking speed...")
t0 = time.time()
# Simulate 10,000 S1 records with 100 candidates each
cand_counts = []
for _ in range(50_000):
    # Simulated 5 keys with buckets
    cand_scores = defaultdict(float)
    # 5 buckets of 20 items
    for w in [1.0, 0.9, 0.8, 0.5, 0.3]:
        for idx in range(25):
            cand_scores[idx] += w
    
    # Sort top 50
    top_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:50]
    cand_counts.append(len(top_cands))

elapsed = time.time() - t0
print(f"50,000 S1 records ranked in {elapsed:.3f}s ({50_000/elapsed:,.0f} S1/sec)")
print(f"Projected time for all 2,206,821 S1 entities: {2_206_821 / (50_000/elapsed):.1f} seconds!")
