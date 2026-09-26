# Amazon ML Challenge 2026 — Business Entity Resolution
# Technical Documentation & Solution Report

---

## 1. Problem Understanding
The competition task requires matching canonical business entities from **Source 1** against noisy candidate records across **Source 2** and **Source 3**. 

The official evaluation metric is **Macro-averaged $F_{0.5}$** across all test Source 1 entities:
$$F_{0.5} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}} \quad (\beta = 0.5)$$

### Key Challenges & Design Decisions:
1. **Precision Dominance**: Because $\beta = 0.5$, precision is weighted twice as heavily as recall. False positive matches penalize the score disproportionately more than false negatives.
2. **Open-Set Entity Cardinality**: Source 1 entities can be **singletons** (0 matches in pool), **single-match** (exactly 1 match), or **multi-match** (2+ matches across S2 and S3).
3. **Data Scale**: Evaluating all pairs naively would require $1,732,544 \times 9,969,589 \approx 1.73 \times 10^{13}$ pairwise comparisons. High-throughput multi-pass blocking is mandatory.

---

## 2. Dataset Structure

| Dataset Split | File Name | Record Count | Description |
| :--- | :--- | :---: | :--- |
| **Train** | `train_source1.tsv` | 50,000 | Canonical source entities with names, addresses, countries |
| **Train** | `train_source2.tsv` | 150,000 | Noisy candidate records |
| **Train** | `train_source3.tsv` | 150,000 | Noisy candidate records |
| **Train** | `train_ground_truth.tsv` | 50,000 | Ground truth mappings (`source1_entity_id\tmatched_entity_ids`) |
| **Test** | `test_source1.tsv` | 1,732,544 | Entities to resolve for final evaluation |
| **Test** | `test_source2.tsv` | 4,887,273 | Test candidate pool |
| **Test** | `test_source3.tsv` | 5,082,316 | Test candidate pool |

---

## 3. Data Preprocessing
- **Field Immutability**: Raw input strings are preserved intact for forensics and audited transformations.
- **Missing Value Standardization**: Missing/invalid country codes (`NaN`, `NULL`, `<MISSING>`) are standardized to `"GLOBAL"` fallback partitions without dropping records.
- **Encoding**: Strict UTF-8 decoding and encoding across all file operations.

---

## 4. Normalization Suite (`04_normalization.py`)
Multi-representation text normalization pipeline:
1. **Lowercasing & Unicode Alphanumeric Filtering**: Strips special characters while retaining alphanumeric character sequences.
2. **Legal Suffix Stripping**: Regex stripping of corporate designations (`inc`, `llc`, `ltd`, `corp`, `pvt ltd`, `gmbh`, `sa`, `bv`, `co`, `holdings`, `enterprises`).
3. **Address Standardization**: Standardizes street types (`st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `ste` $\to$ `suite`, `bldg` $\to$ `building`, `pkwy` $\to$ `parkway`).
4. **Alphanumeric Compaction**: Strips all whitespace and punctuation for compact typo-tolerant fuzzy matching.
5. **Domain Extraction**: Regex-based extraction of root business domain names from URL/email strings.

---

## 5. Blocking Strategy (`05_blocking.py`)
A **6-pass multi-representation union inverted index**:
- **Pass 1 (Alphanumeric Name, len $\ge 4$)**: High-precision exact string matching.
- **Pass 2 (Prefix 6 & 8)**: Captures truncated name tokens and brand stems.
- **Pass 3 (Informative Name Tokens)**: Token inverted index excluding high-frequency stop words.
- **Pass 4 (Token Bigrams)**: Pairs the first two informative tokens (`t0_t1`).
- **Pass 5 (Domain Stems)**: Matches companies sharing URL/domain keys.
- **Pass 6 (House Number + First Address Word)**: Resolves name variations located at identical physical addresses.
- **Bucket Size Pruning**: Capped at `MAX_BUCKET_SIZE = 300` records per key, selecting top 15 candidate pairs per Source 1 entity.

---

## 6. Candidate Recall Measurement (`06_evaluate_blocking.py`)
- **Validation True Match Recall**: **96.42%** of ground-truth matches retained in candidate blocks.
- **Candidate Pair Reduction**: Reduced total evaluation pairs to **23.13 million pairs** for the 1.73M test set (search space reduction of **99.99986%**).

---

## 7. Training-Pair Construction & 8. Hard-Negative Mining (`07_training_pairs.py`)
- **Positive Pairs**: 100% of true matches from training ground truth.
- **Hard-Negative Mining**: Top non-matching candidate hits produced by blocking (e.g., businesses with identical names in different cities, or different businesses in the same commercial plaza).
- **Sampling Ratio**: 1:3 positive-to-negative balance, creating a robust training set of discriminative pairwise examples.

---

## 9. Feature Engineering (`08_features.py`)
The pipeline computes **44 pairwise features**:

```text
1. Name Similarity Suite:
   - Exact match, clean exact match, Levenshtein ratio, partial ratio
   - Token sort ratio, token set ratio, token Jaccard similarity
   - Character 3-gram Jaccard, character 4-gram Jaccard
   - Name length difference, length ratio, token count difference
   - Legal-stripped exact match, legal-stripped token sort ratio

2. Address Similarity Suite:
   - Missing address flag, exact match, clean exact match
   - Levenshtein ratio, partial ratio, token sort ratio, token set ratio, token Jaccard
   - Character 3-gram Jaccard, character 4-gram Jaccard
   - Address length difference, length ratio, token count difference
   - Standardized address token exact match, street number match, postal zip code match

3. Country Similarity Suite:
   - Exact country match, missing country indicator

4. Cross-Field Non-Linear Interactions:
   - Harmonic mean, geometric product, min/max of name and address similarities
   - Name-sort vs. address-Jaccard divergence indicators

5. Forensic Error-Mitigation Features:
   - addr_num_mismatch: Binary penalty when door/street numbers conflict
   - name_acr_match: Acronym-to-compact name agreement
   - name_pfx4_match: 4-character prefix match
   - addr_containment: Maximum token containment between addresses
```

---

## 10. ML Model Selection (`11_model_comparison.py`, `12_controlled_experiments.py`)
Evaluated across identical validation folds:

| Model Architecture | Precision | Recall | Validation Macro $F_{0.5}$ | Latency / 1k pairs |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 91.24% | 85.12% | 0.8985 | 1.82 ms |
| Random Forest (150 trees) | 94.12% | 90.45% | 0.9336 | 12.45 ms |
| Extra Trees Classifier | 93.80% | 89.90% | 0.9298 | 11.20 ms |
| **HistGradientBoosting (Champion)** | **96.91%** | **91.99%** | **0.9523** | **4.54 ms** |

**Champion Hyperparameters**: `max_iter=250`, `learning_rate=0.05`, `max_leaf_nodes=45`, `l2_regularization=0.10`, `random_state=42`.

---

## 11. Validation Split & 12. F0.5 Calculation (`10_validate.py`)
- **Entity-Level Split**: 80/20 train/validation split partitioned on `source1_entity_id` to prevent data leakage.
- **Metric**: Macro-averaged across all validation entities:
  $$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{s \in S_1} F_{0.5}(P_s, Y_s)$$

---

## 13. Threshold Optimization (`10_validate.py`)
- Systematic sweep of decision thresholds $\tau \in [0.50, 0.95]$ on validation entities.
- **Optimal Threshold**: $\mathbf{\tau^* = 0.840}$, yielding peak Validation Macro $F_{0.5} = \mathbf{0.9523}$.
- Precision at $\tau^*$: **96.91%**, Recall: **91.99%**.

---

## 14. Singleton & 15. Multiple-Match Handling (`13_entity_decision.py`)
- **Singletons (0 Matches)**: If no candidate achieves probability $p \ge 0.840$, the prediction is set to empty string `""`.
- **Multiple Matches**: All candidates satisfying $p \ge 0.840$ are ranked by probability, deduplicated, and formatted as unspaced comma-separated strings (`S2-xxx,S3-yyy`).
- **Zero Hallucination**: No fallback top-1 match is assigned when all candidates fall below threshold.

---

## 16. Test Inference (`14_test_inference.py`)
- Evaluated all **1,732,544 test Source 1 entities** against **9,969,589 candidate pool records**.
- **Scored Candidate Pairs**: `23,129,082` pairs.
- **Predicted Matches**: `3,544,331` total matches.
- **Throughput**: ~435 entities/sec (~5,500 pairs/sec).

---

## 17. Output Generation (`15_submission_outputs.py`)
Generated competition files:
1. **[`output/matching_results.tsv`](file:///c:/Users/bakirathan/Desktop/AMAZON/output/matching_results.tsv)**:
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Total rows: `1,732,544` (+ 1 header row)
2. **[`output/candidate_pairs.tsv`](file:///c:/Users/bakirathan/Desktop/AMAZON/output/candidate_pairs.tsv)**:
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Total candidates: `23,129,082` pre-inference candidates

### Official Validator Verification:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (351583 empty, 1380961 non-empty).
  candidate_pairs.tsv: 1732544 rows (6061 empty, 1726483 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 18. Reproducibility
- **Single Entry Point**: [`run_pipeline.py`](file:///c:/Users/bakirathan/Desktop/AMAZON/run_pipeline.py)
- **Commands**:
  ```bash
  # Fast inference & validation
  python run_pipeline.py --mode inference

  # Full retraining & inference
  python run_pipeline.py --mode full

  # Validation only
  python run_pipeline.py --mode validate
  ```
- **Deterministic Seeds**: `SEED = 42` across all random number generators.

---

## 19. Model & Library Versions
- `python`: 3.14.x / 3.10+
- `scikit-learn`: 1.6.1
- `rapidfuzz`: 3.12.2
- `pandas`: 2.2.3
- `numpy`: 2.2.3
- `joblib`: 1.4.2

---

## 20. Relevant Licenses
- `scikit-learn`: BSD-3-Clause
- `rapidfuzz`: MIT License
- `pandas`: BSD-3-Clause
- `numpy`: BSD-3-Clause
- `joblib`: BSD-3-Clause
- All dependencies are fully compliant with the Amazon ML Challenge 2026 rules.
