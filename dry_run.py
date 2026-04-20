#!/usr/bin/env python3
"""
dry_run.py
===========
Runs the complete pipeline with MOCK API responses — no real API calls,
no real cost. Useful for:
  - Verifying your environment is set up correctly
  - Testing pipeline logic before spending money
  - CI/CD validation

Generates synthetic-but-plausible data at each stage so all downstream
stages (analysis, figures, tables) run on realistic-shaped data.

Usage:
    python dry_run.py                        # run all stages
    python dry_run.py --from-stage 5        # only analysis + figures + release
    python dry_run.py --n-pairs 20          # smaller dataset
"""

import argparse
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dry_run")

from config import (
    DOMAINS, DOMAIN_MAPPINGS, MODEL_PAIRS,
    RAW_DIR, PAIRS_DIR, VERIFIED_DIR, RESULTS_DIR, ANALYSIS_DIR,
)


# ── Mock data generators ──────────────────────────────────────────────────

def mock_seed_problems(n_per_domain: int = 30) -> list[dict]:
    """Generate realistic-looking seed problems."""
    BANK = {
        "physics":      [
            ("A 5.0 kg object moves at 4.0 m/s. What is its kinetic energy?",          "40.0 J",   "KE=½mv²"),
            ("A spring (k=200 N/m) compressed 0.10 m. Stored energy?",                "1.00 J",   "E=½kx²"),
            ("2.0 mol ideal gas at 300 K in 49.2 L. Pressure?",                       "0.996 atm","PV=nRT"),
            ("A wave has frequency 440 Hz, wavelength 0.78 m. Speed?",                 "343 m/s",  "v=fλ"),
            ("A 3.0 kg ball falls 10 m. Potential energy lost?",                       "294 J",    "PE=mgh"),
        ],
        "chemistry":    [
            ("pH of 0.10 M acetic acid (Ka=1.8e-5)?",                                 "2.87",     "pH=½(pKa-log c)"),
            ("Grams in 2.5 mol NaCl (MW=58.44 g/mol)?",                               "146.1 g",  "mass=mol×MW"),
            ("Molarity of 10.0 g NaOH (MW=40) in 250 mL?",                            "1.00 M",   "M=mol/L"),
            ("Volume of 2.0 M HCl to neutralise 0.050 mol NaOH?",                     "25.0 mL",  "n=C×V"),
            ("50 kJ/mol released at 298 K with ΔS=100 J/K. ΔG?",                      "20.2 kJ",  "ΔG=ΔH-TΔS"),
        ],
        "biology":      [
            ("Bacteria double every 20 min, start 100 cells. After 2 h?",              "6400",     "N=N₀×2^(t/td)"),
            ("Allele freq p=0.6. Hardy-Weinberg homozygous dominant freq?",             "0.36",     "p²"),
            ("200 amino acids. Approximate molecular weight?",                          "22 kDa",   "~110 Da/AA"),
            ("Enzyme Km=2 mM, [S]=8 mM. Fraction of Vmax?",                            "0.80",     "Michaelis-Menten"),
            ("Mutation rate 1e-6/base/generation. Expected in 1e8 bases?",              "100",      "μ×L"),
        ],
        "earth_science": [
            ("K-Ar: 3 half-lives elapsed, t½=1.25 Ga. Rock age?",                     "3.75 Ga",  "t=n×t½"),
            ("P-wave travels 500 km at 8 km/s. Travel time?",                          "62.5 s",   "t=d/v"),
            ("Ocean depth 4000 m, density 1025 kg/m³. Pressure?",                      "40.2 MPa", "P=ρgh"),
            ("Geothermal gradient 30°C/km, surface 15°C. Temp at 5 km?",               "165°C",    "T=T₀+g×d"),
            ("CO₂ rises 2 ppm/yr. Rise over 50 years?",                                "100 ppm",  "linear"),
        ],
    }
    rng = random.Random(42)
    seeds = []
    for domain in DOMAINS:
        bank = BANK[domain]
        for i in range(n_per_domain):
            q, a, formula = bank[i % len(bank)]
            seeds.append({
                "id":             str(uuid.uuid4()),
                "source":         "synthetic_mock",
                "domain":         domain,
                "question":       q,
                "answer":         a,
                "difficulty":     "college",
                "structure_type": "formula_recall_and_substitute",
                "metadata": {
                    "formula_used":    formula,
                    "solution_steps":  ["recall formula", "substitute values", "compute"],
                    "estimated_steps": 3,
                },
            })
    logger.info(f"Generated {len(seeds)} mock seeds ({n_per_domain}/domain)")
    return seeds

def mock_candidate_pairs(seeds: list[dict], n_per_mapping: int = 8) -> list[dict]:
    """Generate mock candidate pairs from seeds."""
    rng = random.Random(123)
    pairs = []
    for src_domain, tgt_domain in DOMAIN_MAPPINGS:
        src_seeds = [s for s in seeds if s["domain"] == src_domain]
        rng.shuffle(src_seeds)
        for seed in src_seeds[:n_per_mapping]:
            # Generate 2 mock candidates
            for _ in range(2):
                pairs.append({
                    "pair_id":       str(uuid.uuid4()),
                    "mapping":       f"{src_domain}_to_{tgt_domain}",
                    "source_domain": src_domain,
                    "target_domain": tgt_domain,
                    "source":        seed,
                    "target": {
                        "id":            str(uuid.uuid4()),
                        "question":      f"Mock {tgt_domain} problem isomorphic to: {seed['question'][:60]}…",
                        "answer":        f"{rng.uniform(0.1, 100):.2f} units",
                        "domain":        tgt_domain,
                        "structure_type": seed.get("structure_type", "formula_recall_and_substitute"),
                        "metadata": {
                            "formula_used":              f"Mock {tgt_domain} formula",
                            "solution_steps":            ["recall formula", "substitute", "compute"],
                            "isomorphism_justification": "Both require formula recall + substitution + computation",
                            "sub_topic":                 f"Mock {tgt_domain} sub-topic",
                        },
                    },
                    "verification_status": "pending",
                    "verification_scores": {},
                })
    logger.info(f"Generated {len(pairs)} mock candidate pairs")
    return pairs


def mock_verified_pairs(candidates: list[dict], accept_rate: float = 0.75) -> list[dict]:
    """Mock verification — accept ~75% of candidates."""
    rng = random.Random(456)
    verified = []
    for pair in candidates:
        if rng.random() < accept_rate:
            scores = {c: rng.uniform(3.8, 5.0)
                      for c in ["logical_equivalence", "domain_independence",
                                "difficulty_parity", "self_containment"]}
            scores["overall"] = sum(scores.values()) / 4
            pair = dict(pair)
            pair["verification_status"]  = "auto_accepted"
            pair["verification_scores"]  = {
                "avg": scores,
                "individual": [{"judge_model": "mock", **scores}],
                "rejection_reasons": [],
            }
            verified.append(pair)
    logger.info(f"Mock verification: {len(candidates)} → {len(verified)} accepted")
    return verified


def mock_eval_results(verified_pairs: list[dict]) -> list[dict]:
    """
    Generate mock evaluation results for all 12 models.
    Reasoning models score higher on physics-sourced, lower on biology-sourced.
    Standard models score uniformly lower.
    """
    rng = random.Random(789)
    results = []

    def base_accuracy(model: str, domain: str, is_reasoning: bool) -> float:
        """Simulate domain-asymmetric gains."""
        comp_intensity = {
            "physics":      0.5,
            "chemistry":    0.3,
            "biology":      0.1,
            "earth_science":0.1,
        }.get(domain, 0.2)

        base = 0.55 + rng.gauss(0, 0.03)
        if is_reasoning:
            # Reasoning models benefit more on computation-heavy domains
            gain = 0.05 + 0.30 * comp_intensity + rng.gauss(0, 0.02)
        else:
            gain = 0.0
        return min(0.95, max(0.20, base + gain))

    # Build flat item list from verified pairs
    items = []
    for pair in verified_pairs:
        for role in ("source", "target"):
            prob = pair[role]
            items.append({
                "item_id":       f"{pair['pair_id']}_{role}",
                "pair_id":       pair["pair_id"],
                "role":          role,
                "mapping":       pair["mapping"],
                "domain":        prob["domain"],
                "question":      prob["question"],
                "answer":        prob["answer"],
                "answer_type":   "numeric",
                "benchmark":     "isosci",
                "structure_type": prob.get("structure_type", "unknown"),
            })

    # Add mock existing benchmark items
    for bench, n in [("gpqa", 198), ("scibench", 200), ("mmlu_stem", 400)]:
        for i in range(n):
            domain = rng.choice(DOMAINS)
            items.append({
                "item_id":    f"{bench}_{i}",
                "pair_id":    None,
                "domain":     domain,
                "question":   f"Mock {bench} question {i}",
                "answer":     "A" if bench in ("gpqa", "mmlu_stem") else f"{rng.uniform(1,100):.2f}",
                "answer_type":"mcq" if bench in ("gpqa", "mmlu_stem") else "numeric",
                "benchmark":  bench,
            })

    # Evaluate each model
    for reasoning_model, standard_model, label in MODEL_PAIRS:
        for model, is_reasoning in [(reasoning_model, True), (standard_model, False)]:
            short = model.split("/")[-1]
            for item in items:
                domain = item["domain"]
                acc = base_accuracy(model, domain, is_reasoning)
                correct = rng.random() < acc
                prompt_tok = rng.randint(300, 500)
                comp_tok = rng.randint(800, 4500) if is_reasoning else rng.randint(400, 1000)
                results.append({
                    **item,
                    "model":             model,
                    "response":          f"[Mock response from {short}] **Final Answer:** {item['answer']}",
                    "extracted_answer":  item["answer"] if correct else f"wrong_{rng.randint(0,99)}",
                    "gold_answer":       item["answer"],
                    "correct":           correct,
                    "prompt_tokens":     prompt_tok,
                    "completion_tokens": comp_tok,
                    "total_tokens":      prompt_tok + comp_tok,
                    "latency_s":         rng.uniform(0.5, 8.0),
                    "error":             None,
                })

    logger.info(f"Generated {len(results)} mock eval results")
    return results


# ── Full dry run ──────────────────────────────────────────────────────────

def run_dry_run(from_stage: int = 1, n_pairs_per_mapping: int = 4):
    """
    Run all pipeline stages with mock data.
    No real API calls are made.
    """
    logger.info("="*60)
    logger.info("DRY RUN MODE — no real API calls")
    logger.info("="*60)

    # Stage 1: Seeds
    if from_stage <= 1:
        logger.info("\n[Stage 1] Generating mock seeds…")
        seeds = mock_seed_problems(n_per_domain=30)
        seed_path = RAW_DIR / "seed_problems.json"
        with open(seed_path, "w") as f:
            json.dump(seeds, f, indent=2)
        logger.info(f"  → {len(seeds)} seeds saved to {seed_path}")
    else:
        with open(RAW_DIR / "seed_problems.json") as f:
            seeds = json.load(f)

    # Stage 2: Pairs
    if from_stage <= 2:
        logger.info("\n[Stage 2] Generating mock candidate pairs…")
        candidates = mock_candidate_pairs(seeds, n_per_mapping=n_pairs_per_mapping)
        cand_path = PAIRS_DIR / "candidate_pairs.json"
        with open(cand_path, "w") as f:
            json.dump(candidates, f, indent=2)
        logger.info(f"  → {len(candidates)} candidates saved")
    else:
        with open(PAIRS_DIR / "candidate_pairs.json") as f:
            candidates = json.load(f)

    # Stage 3: Verification
    if from_stage <= 3:
        logger.info("\n[Stage 3] Mock verification…")
        verified = mock_verified_pairs(candidates, accept_rate=0.75)
        # Trim to balanced target
        from collections import defaultdict
        by_mapping = defaultdict(list)
        for p in verified:
            by_mapping[p["mapping"]].append(p)
        final = []
        for pairs in by_mapping.values():
            final.extend(pairs[:n_pairs_per_mapping])
        vp = VERIFIED_DIR / "verified_pairs.json"
        with open(vp, "w") as f:
            json.dump(final, f, indent=2)
        logger.info(f"  → {len(final)} verified pairs saved")
        verified = final
    else:
        with open(VERIFIED_DIR / "verified_pairs.json") as f:
            verified = json.load(f)

    # Stage 4: Evaluation
    if from_stage <= 4:
        logger.info("\n[Stage 4] Generating mock evaluation results…")
        results = mock_eval_results(verified)
        rp = RESULTS_DIR / "all_results.json"
        with open(rp, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  → {len(results)} result records saved")

    # Stage 5: Analysis (real code, mock data)
    if from_stage <= 5:
        logger.info("\n[Stage 5] Running analysis on mock data…")
        from src.stage5_analysis import run as analysis_run
        summary = analysis_run()
        logger.info("  → Analysis complete")

    # Stage 6: Figures (real code, mock data)
    if from_stage <= 6:
        logger.info("\n[Stage 6] Generating figures from mock data…")
        from src.stage6_figures import run as figures_run
        figures_run()
        logger.info("  → Figures generated")

    # Stage 7: Dataset release
    if from_stage <= 7:
        logger.info("\n[Stage 7] Packaging mock dataset…")
        from src.stage7_dataset_release import run as release_run
        release_run()
        logger.info("  → Dataset packaged")

    logger.info("\n" + "="*60)
    logger.info("DRY RUN COMPLETE — all stages passed")
    logger.info("Check outputs/ for figures and tables")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry-run pipeline with mock data")
    parser.add_argument("--from-stage", type=int, default=1,
                        help="Start from stage N (default: 1)")
    parser.add_argument("--n-pairs", type=int, default=4,
                        help="Pairs per mapping in mock data (default: 4, for speed)")
    args = parser.parse_args()
    run_dry_run(from_stage=args.from_stage, n_pairs_per_mapping=args.n_pairs)
