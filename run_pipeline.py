#!/usr/bin/env python3
"""
run_pipeline.py
================
Master pipeline runner for IsoSci-150.
Runs all 7 stages in sequence, with resume support and cost tracking.

Usage:
    # Full pipeline
    python run_pipeline.py

    # Start from a specific stage (skips earlier stages)
    python run_pipeline.py --from-stage 4

    # Run only specific stages
    python run_pipeline.py --stages 1 2

    # Quick test mode (tiny subsets, mock API calls if no key set)
    python run_pipeline.py --test-mode

    # Skip existing outputs (resume after interruption)
    python run_pipeline.py --resume

Environment variables:
    OPENROUTER_API_KEY    Required for all API-calling stages
    HF_TOKEN              Required only for --push-to-hub in stage 7

Estimated wall time (with full API budget):
    Stage 1:  ~30 min  (seed generation)
    Stage 2:  ~3–4 hr  (pair generation, 150 × 3 candidates)
    Stage 3:  ~2–3 hr  (automated verification with 3 judges)
    Stage 4:  ~8–12 hr (12 models × 3 benchmarks + IsoSci)
    Stage 5:  ~2 min   (analysis, local compute only)
    Stage 6:  ~2 min   (figures, local compute only)
    Stage 7:  ~1 min   (dataset packaging)
    TOTAL:    ~14–20 hr of actual API time
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOGS_DIR, RESULTS_DIR, VERIFIED_DIR, PAIRS_DIR, RAW_DIR,
    OUTPUTS_DIR, ANALYSIS_DIR,
)
from src.api_client import cost_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")


# ── Stage registry ─────────────────────────────────────────────────────────

STAGES = {
    1: {
        "name":    "Seed collection",
        "output":  RAW_DIR / "seed_problems.json",
        "module":  "src.stage1_seed_collection",
        "fn":      "run",
        "kwargs":  {},
        "desc":    "Collect/generate seed problems from benchmarks + LLM",
    },
    2: {
        "name":    "Pair generation",
        "output":  PAIRS_DIR / "candidate_pairs.json",
        "module":  "src.stage2_pair_generation",
        "fn":      "run",
        "kwargs":  {},
        "desc":    "LLM-generate isomorphic partner problems",
    },
    3: {
        "name":    "Verification",
        "output":  VERIFIED_DIR / "verified_pairs.json",
        "module":  "src.stage3_verification",
        "fn":      "run",
        "kwargs":  {"mode": "auto", "target_n": None},
        "desc":    "Auto-verify pairs with 3 LLM judges",
    },
    4: {
        "name":    "Model evaluation",
        "output":  RESULTS_DIR / "all_results.json",
        "module":  "src.stage4_evaluation",
        "fn":      "run",
        "kwargs":  {"benchmarks": ["isosci", "gpqa", "scibench", "mmlu"]},
        "desc":    "Evaluate all 12 models on all benchmarks",
    },
    5: {
        "name":    "Analysis",
        "output":  ANALYSIS_DIR / "summary_for_paper.json",
        "module":  "src.stage5_analysis",
        "fn":      "run",
        "kwargs":  {},
        "desc":    "Compute domain-asymmetric gains + decoupling + stats",
    },
    6: {
        "name":    "Figures & tables",
        "output":  OUTPUTS_DIR / "figure_1_domain_asymmetry.pdf",
        "module":  "src.stage6_figures",
        "fn":      "run",
        "kwargs":  {},
        "desc":    "Generate paper figures and LaTeX tables",
    },
    7: {
        "name":    "Dataset release",
        "output":  OUTPUTS_DIR / "isosci150" / "test.json",
        "module":  "src.stage7_dataset_release",
        "fn":      "run",
        "kwargs":  {},
        "desc":    "Package dataset for HuggingFace release",
    },
}

TEST_OVERRIDES = {
    # In test mode, use tiny limits to verify the pipeline runs end-to-end
    1: {"n_per_domain": 5, "synthetic_only": True},
    2: {"limit": 3},
    3: {"target_n": None},
    4: {"limit": 5, "benchmarks": ["isosci"]},
}


# ── Pre-flight checks ─────────────────────────────────────────────────────

def preflight(stages_to_run: list[int]):
    """Check prerequisites before running."""
    errors = []

    if any(s <= 4 for s in stages_to_run):
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            errors.append("OPENROUTER_API_KEY not set. Export it before running.")

    if errors:
        for e in errors:
            logger.error(f"PREFLIGHT FAIL: {e}")
        sys.exit(1)

    logger.info("Preflight checks passed.")


# ── Stage runner ──────────────────────────────────────────────────────────

def run_stage(stage_num: int, test_mode: bool = False, resume: bool = False) -> bool:
    """
    Run a single pipeline stage.
    Returns True if successful (or skipped), False on error.
    """
    stage = STAGES[stage_num]
    name = stage["name"]
    output = stage["output"]

    logger.info(f"\n{'='*70}")
    logger.info(f"STAGE {stage_num}: {name}")
    logger.info(f"  {stage['desc']}")
    logger.info(f"  Output: {output}")
    logger.info(f"{'='*70}")

    # Resume: skip if output already exists
    if resume and output.exists():
        logger.info(f"  → SKIPPED (output exists, --resume mode)")
        return True

    # Import and run
    import importlib
    kwargs = dict(stage["kwargs"])
    if test_mode and stage_num in TEST_OVERRIDES:
        kwargs.update(TEST_OVERRIDES[stage_num])
        logger.info(f"  [TEST MODE] kwargs overridden: {TEST_OVERRIDES[stage_num]}")

    t0 = time.time()
    try:
        mod = importlib.import_module(stage["module"])
        fn = getattr(mod, stage["fn"])
        fn(**kwargs)
        elapsed = time.time() - t0
        logger.info(f"  → DONE in {elapsed/60:.1f} min")
        return True
    except SystemExit as e:
        logger.error(f"  → Stage {stage_num} exited with code {e.code}")
        return False
    except Exception as e:
        import traceback
        logger.error(f"  → FAILED: {e}")
        logger.error(traceback.format_exc())
        return False


# ── Progress report ────────────────────────────────────────────────────────

def print_progress():
    """Print current state of all pipeline outputs."""
    print("\n" + "="*70)
    print("PIPELINE STATUS")
    print("="*70)
    for num, stage in STAGES.items():
        exists = stage["output"].exists()
        status = "✓ done" if exists else "○ pending"
        print(f"  Stage {num}: {stage['name']:<25} {status}")

    # Cost so far
    summary = cost_tracker.summary()
    print(f"\nAPI calls so far: {summary['total_calls']}")
    print(f"Tokens used:      {summary['total_tokens']:,}")
    print("="*70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IsoSci-150 pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--from-stage", type=int, default=1, metavar="N",
                        help="Start from stage N (default: 1)")
    parser.add_argument("--to-stage", type=int, default=7, metavar="N",
                        help="Stop after stage N (default: 7)")
    parser.add_argument("--stages", type=int, nargs="+", metavar="N",
                        help="Run only specific stages (overrides --from/--to)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use tiny limits to test pipeline end-to-end")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose output already exists")
    parser.add_argument("--status", action="store_true",
                        help="Print pipeline status and exit")
    args = parser.parse_args()

    if args.status:
        print_progress()
        sys.exit(0)

    # Determine which stages to run
    if args.stages:
        stages_to_run = sorted(args.stages)
    else:
        stages_to_run = list(range(args.from_stage, args.to_stage + 1))

    logger.info(f"Stages to run: {stages_to_run}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info(f"Resume mode: {args.resume}")

    preflight(stages_to_run)

    # Run pipeline
    t_start = time.time()
    failed_stages = []
    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            logger.error(f"Unknown stage: {stage_num}")
            continue
        success = run_stage(stage_num, test_mode=args.test_mode, resume=args.resume)
        if not success:
            failed_stages.append(stage_num)
            logger.error(f"Stage {stage_num} failed. Stopping pipeline.")
            break

    # Final report
    elapsed = time.time() - t_start
    print_progress()
    logger.info(f"Total wall time: {elapsed/3600:.2f} hr")
    logger.info(f"API cost summary: {cost_tracker.summary()}")

    if failed_stages:
        logger.error(f"Pipeline FAILED at stage(s): {failed_stages}")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully.")
        logger.info(f"Outputs in: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
