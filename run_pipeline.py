"""
Amazon ML Challenge 2026 - Business Entity Resolution
Single Reproducible Pipeline Entry Point

Usage:
    python run_pipeline.py --mode full
        Runs the complete end-to-end pipeline:
        1. Feature engineering & Training pairs generation
        2. Champion GBDT Model training & Hyperparameter optimization (SEED=42)
        3. Threshold validation (tau* = 0.840)
        4. Test inference over test_source1, test_source2, test_source3
        5. Generates output/matching_results.tsv & output/candidate_pairs.tsv
        6. Executes official validation suite

    python run_pipeline.py --mode inference
        Fast reproducible inference using the frozen champion model bundle:
        1. Multi-pass candidate blocking over test_source2 & test_source3
        2. 44-dimensional pairwise feature vectorization
        3. Model scoring & entity-level decision logic (tau* = 0.840)
        4. Generates output/matching_results.tsv & output/candidate_pairs.tsv
        5. Executes official competition validation suite

    python run_pipeline.py --mode validate
        Executes local and official validation suites on existing output files.
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_step(description: str, cmd: list, cwd: Path):
    """Executes a pipeline step with clean formatted logging."""
    print("\n" + "=" * 90)
    print(f" >>> RUNNING: {description}")
    print(f"     Command: {' '.join(str(c) for c in cmd)}")
    print("=" * 90)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {result.returncode}: {description}")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] Completed in {elapsed:.2f} s: {description}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Reproducible Pipeline Entry Point"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "inference", "validate"],
        default="inference",
        help="Pipeline execution mode: 'full' (train + inference), 'inference' (fast test inference), 'validate' (validate existing outputs)."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Path to directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to directory where submission outputs will be saved"
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        default=True,
        help="Enable full memory ID-existence check during official validation."
    )

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "code" / "business_entity_resolution" / "src"

    python_exe = sys.executable

    print("*" * 90)
    print(" AMAZON ML CHALLENGE 2026 - BUSINESS ENTITY RESOLUTION PIPELINE")
    print("*" * 90)
    print(f"Execution Mode:    {args.mode}")
    print(f"Base Directory:    {base_dir}")
    print(f"Source Directory:  {src_dir}")
    print(f"Test Directory:    {base_dir / args.test_dir}")
    print(f"Output Directory:  {base_dir / args.output_dir}\n")

    t_pipeline_start = time.time()

    # Step A: Full Training Workflow (if mode == 'full')
    if args.mode == "full":
        # 1. Generate Training Pairs & Feature Suite
        run_step(
            "Phase 7 & 8: Generate Hard-Negative Training Pairs & Feature Matrices",
            [python_exe, str(src_dir / "07_training_pairs.py")],
            base_dir
        )
        run_step(
            "Phase 8: Extract 44-Dimensional Feature Vectors",
            [python_exe, str(src_dir / "08_features.py")],
            base_dir
        )
        # 2. Model Training & Controlled Experiments
        run_step(
            "Phase 13: Train Champion HistGradientBoosting Model (Deterministic SEED=42)",
            [python_exe, str(src_dir / "12_controlled_experiments.py")],
            base_dir
        )
        # 3. Threshold Optimization
        run_step(
            "Phase 10: F0.5 Validation Threshold Optimization",
            [python_exe, str(src_dir / "10_validate.py")],
            base_dir
        )

    # Step B: Test Inference & Submission Generation (if mode in ['full', 'inference'])
    if args.mode in ["full", "inference"]:
        # 1. Run Final Test Inference (Scoring & Entity Decision Engine)
        run_step(
            "Phase 15: Final Test Inference (Multi-Pass Blocking & Champion GBDT Scoring)",
            [python_exe, str(src_dir / "14_test_inference.py")],
            base_dir
        )
        # 2. Generate Final Required Submission Outputs (TSV with unspaced comma format)
        run_step(
            "Phase 16: Generate Required Output TSVs (matching_results.tsv & candidate_pairs.tsv)",
            [python_exe, str(src_dir / "15_submission_outputs.py")],
            base_dir
        )

    # Step C: Validation (All modes)
    # 1. Local 12-Point Output Validation Suite
    run_step(
        "Phase 17: Local Output Validation Suite (12 Comprehensive Integrity Checks)",
        [python_exe, str(src_dir / "12_validate_outputs.py")],
        base_dir
    )

    # 2. Official Competition Submission Validator
    validator_cmd = [
        python_exe,
        str(base_dir / "utils" / "validate_submission.py"),
        "--matching", str(base_dir / args.output_dir / "matching_results.tsv"),
        "--candidate", str(base_dir / args.output_dir / "candidate_pairs.tsv"),
        "--test-dir", str(base_dir / args.test_dir)
    ]
    if args.check_ids:
        validator_cmd.append("--check-ids")

    run_step(
        "Phase 18: Official Competition Submission Validator",
        validator_cmd,
        base_dir
    )

    total_duration = time.time() - t_pipeline_start
    print("=" * 90)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Total Duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
    print(f" Verified Matching Results: {base_dir / args.output_dir / 'matching_results.tsv'}")
    print(f" Verified Candidate Pairs:  {base_dir / args.output_dir / 'candidate_pairs.tsv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
