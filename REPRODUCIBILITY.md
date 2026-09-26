# Amazon ML Challenge 2026 — Business Entity Resolution
## Reproducible Pipeline & Submission Guide

This document details the end-to-end architecture, instructions for full reproduction, project structure, and validation protocols for the **Business Entity Resolution** task.

---

## 1. Project Overview & Architecture

* **Objective**: Match canonical Source 1 entities against noisy candidate pools in Source 2 and Source 3, maximizing Macro-averaged $F_{0.5}$:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
* **Core Model**: Champion **HistGradientBoostingClassifier** (GBDT with 250 trees, max leaf nodes 45, learning rate 0.05, $L_2 = 0.10$).
* **Decision Engine**: Frozen optimal threshold $\tau^* = 0.840$, deduplication, singleton (0-match) and multi-match preservation.
* **Licensing**: 100% compliant with competition rules (Standard library, `scikit-learn`, `rapidfuzz`, `pandas`, `numpy`, `joblib`).

---

## 2. Directory Structure

```text
AMAZON/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   ├── test/
│   │   ├── test_source1.tsv
│   │   ├── test_source2.tsv
│   │   └── test_source3.tsv
│   └── submission.tsv                  <- Verified full submission TSV
├── models/
│   ├── best_model_controlled.joblib    <- Champion GBDT model (44 features, tau=0.840)
│   └── baseline_logreg.joblib          <- Baseline model
├── output/
│   ├── matching_results.tsv            <- Final competition matching output (Leaderboard)
│   └── candidate_pairs.tsv             <- Final pre-inference candidate set
├── code/
│   └── business_entity_resolution/
│       └── src/
│           ├── 01_profile_data.py
│           ├── 02_ground_truth_index.py
│           ├── 03_match_forensics.py
│           ├── 04_normalization.py
│           ├── 05_blocking.py
│           ├── 06_evaluate_blocking.py
│           ├── 07_training_pairs.py
│           ├── 08_features.py
│           ├── 09_train_model.py
│           ├── 10_validate.py
│           ├── 11_model_comparison.py
│           ├── 11_error_analysis.py
│           ├── 12_controlled_experiments.py
│           ├── 12_validate_outputs.py
│           ├── 13_entity_decision.py
│           ├── 14_test_inference.py
│           └── 15_submission_outputs.py
├── utils/
│   └── validate_submission.py          <- Official competition submission validator
├── run_pipeline.py                     <- Single reproducible master entry point
├── REPRODUCIBILITY.md
└── final_test_inference_report.txt
```

---

## 3. How to Reproduce

### Prerequisites
Install standard compliant Python dependencies:
```bash
pip install pandas numpy scikit-learn rapidfuzz joblib
```

### Execution Modes

#### Mode 1: Fast Reproducible Test Inference & Output Generation (Recommended)
Executes multi-pass candidate blocking, extracts 44 pairwise features, runs champion GBDT model scoring, generates both required outputs, and executes all validators:
```bash
python run_pipeline.py --mode inference
```

#### Mode 2: Full End-to-End Retraining & Inference
Rebuilds hard-negative training pairs from scratch, trains the GBDT model with deterministic `SEED=42`, optimizes threshold, runs test inference, and generates submission files:
```bash
python run_pipeline.py --mode full
```

#### Mode 3: Validate Existing Outputs
Runs the local 12-point validation suite and the official competition validator:
```bash
python run_pipeline.py --mode validate
```

---

## 4. Official Validation Command

To run the official competition validator directly:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### Official Validator Verification Output:
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

## 5. Submission Integrity Metrics

* **Total Test Source 1 Entities**: `1,732,544`
* **Total Pool Candidates**: `9,969,589`
* **Total Evaluated Candidates**: `23,129,082` (Average `13.35` per S1)
* **Total Matches Predicted**: `3,544,331` (Average `2.05` per S1)
* **Singletons (0 matches / empty string)**: `351,583` (`20.29%`)
* **Single Matches (1 match)**: `375,391` (`21.67%`)
* **Multi-Matches (2+ matches)**: `1,005,570` (`58.04%`)
* **Deterministic Seeds**: All splits, hard-negative sampling, and tree building use `SEED = 42`.
* **External Data**: Zero external data or API calls utilized.
