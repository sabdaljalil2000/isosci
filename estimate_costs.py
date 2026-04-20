#!/usr/bin/env python3
"""
estimate_costs.py
==================
Pre-run cost estimator. Run this BEFORE the full pipeline to get a
realistic breakdown of API spend per stage and per model.

Usage:
    python estimate_costs.py
    python estimate_costs.py --pairs 100        # estimate for 100 pairs instead of 150
    python estimate_costs.py --models o1 r1     # only include specific models
    python estimate_costs.py --conservative     # use upper-bound token estimates
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_PAIRS, DOMAINS, PAIRS_PER_MAPPING, DOMAIN_MAPPINGS

# ── Token estimates (per call) ─────────────────────────────────────────────
# These are empirically calibrated estimates. --conservative doubles them.

ESTIMATES = {
    "stage1_seed_gen": {
        "desc":           "Synthetic seed generation (stage 1)",
        "model":          "anthropic/claude-sonnet-4-5",
        "calls":          24,       # 4 domains × 6 structure types = 24 batches
        "prompt_tok":     400,
        "completion_tok": 2000,
    },
    "stage2_pair_gen": {
        "desc":           "Isomorphic pair generation (stage 2)",
        "model":          "anthropic/claude-sonnet-4-5",
        "calls_per_pair": 1,
        "prompt_tok":     600,
        "completion_tok": 1500,
    },
    "stage3_verification": {
        "desc":           "Pair verification — 3 judges × each candidate (stage 3)",
        "judges": [
            "anthropic/claude-sonnet-4-5",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat",
        ],
        "calls_per_pair": 3,        # one per judge
        "prompt_tok":     700,
        "completion_tok": 200,
    },
    "stage4_isosci": {
        "desc":           "IsoSci-150 evaluation — all models (stage 4)",
        "items_per_model": None,    # filled in below = 2 × n_pairs
        "prompt_tok_standard":   400,
        "completion_tok_standard":800,
        "prompt_tok_reasoning":  400,
        "completion_tok_reasoning": 4000,   # reasoning models ~5× longer chains
    },
    "stage4_existing": {
        "desc":           "GPQA + SciBench + MMLU-STEM evaluation (stage 4)",
        "items_per_model": {
            "gpqa":      198,
            "scibench":  695,
            "mmlu_stem": 1000,      # capped at 1000 for budget
        },
        "prompt_tok_standard":   350,
        "completion_tok_standard":700,
        "prompt_tok_reasoning":  350,
        "completion_tok_reasoning": 3500,
    },
}

# Cost per 1M tokens (USD) — update periodically
COST_PER_1M = {
    "anthropic/claude-sonnet-4-5":             {"in": 3.0,   "out": 15.0},
    "openai/o1-mini":                          {"in": 3.0,   "out": 12.0},
    "openai/gpt-4o-mini":                      {"in": 0.15,  "out": 0.60},
    "deepseek/deepseek-r1":                    {"in": 0.55,  "out": 2.19},
    "deepseek/deepseek-chat":                  {"in": 0.27,  "out": 1.10},
    "deepseek/deepseek-r1-distill-llama-70b":  {"in": 0.23,  "out": 0.69},
    "meta-llama/llama-3.1-70b-instruct":       {"in": 0.12,  "out": 0.30},
    "qwen/qwq-32b":                            {"in": 0.15,  "out": 0.60},
    "qwen/qwen-2.5-72b-instruct":              {"in": 0.35,  "out": 0.40},
    "deepseek/deepseek-r1-distill-qwen-7b":    {"in": 0.10,  "out": 0.20},
    "qwen/qwen-2.5-7b-instruct":              {"in": 0.10,  "out": 0.20},
}

def cost_usd(model: str, prompt_tok: int, completion_tok: int) -> float:
    r = COST_PER_1M.get(model, {"in": 1.0, "out": 3.0})
    return (prompt_tok * r["in"] + completion_tok * r["out"]) / 1_000_000

def fmt(n: float) -> str:
    return f"${n:.2f}"

def fmt_tok(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


def estimate(n_pairs: int = 150, conservative: bool = False,
             model_filter: list[str] = None):

    mult = 2.0 if conservative else 1.0
    # Candidate pairs = pairs_per_mapping × mappings × candidates_per_seed
    n_candidates = n_pairs * 3      # 3 candidates generated per target pair
    n_mappings   = len(DOMAIN_MAPPINGS)

    stage_costs = {}
    total = 0.0

    # ── STAGE 1 ───────────────────────────────────────────────────────────
    e = ESTIMATES["stage1_seed_gen"]
    c = cost_usd(e["model"],
                 e["calls"] * e["prompt_tok"] * mult,
                 e["calls"] * e["completion_tok"] * mult)
    stage_costs["Stage 1 — Seed generation"] = {
        "cost": c,
        "calls": e["calls"],
        "model": e["model"],
        "note": f"{e['calls']} batches (4 domains × 6 structure types)",
    }
    total += c

    # ── STAGE 2 ───────────────────────────────────────────────────────────
    e = ESTIMATES["stage2_pair_gen"]
    # We generate candidates for each mapping: seeds_per_mapping ≈ n_pairs/n_mappings
    seeds_used = n_pairs   # one seed per final pair, but we overshoot by 30%
    seeds_total = int(seeds_used / 0.70)
    calls = seeds_total
    c = cost_usd(e["model"],
                 calls * e["prompt_tok"] * mult,
                 calls * e["completion_tok"] * mult)
    stage_costs["Stage 2 — Pair generation"] = {
        "cost": c,
        "calls": calls,
        "model": e["model"],
        "note": f"~{seeds_total} seed→partner calls (targeting {n_pairs} pairs after ~30% rejection)",
    }
    total += c

    # ── STAGE 3 ───────────────────────────────────────────────────────────
    e = ESTIMATES["stage3_verification"]
    judge_costs = {}
    stage3_total = 0.0
    for judge in e["judges"]:
        c = cost_usd(judge,
                     seeds_total * e["prompt_tok"] * mult,
                     seeds_total * e["completion_tok"] * mult)
        judge_costs[judge.split("/")[-1]] = c
        stage3_total += c
    stage_costs["Stage 3 — Verification (3 judges)"] = {
        "cost": stage3_total,
        "calls": seeds_total * 3,
        "by_judge": judge_costs,
        "note": f"{seeds_total} candidates × 3 judges each",
    }
    total += stage3_total

    # ── STAGE 4 ───────────────────────────────────────────────────────────
    stage4_total = 0.0
    stage4_by_model = {}
    isosci_items_per_model = n_pairs * 2  # source + target
    existing_items = sum(ESTIMATES["stage4_existing"]["items_per_model"].values())
    total_items = isosci_items_per_model + existing_items

    for reasoning_model, standard_model, label in MODEL_PAIRS:
        if model_filter and not any(f in reasoning_model or f in standard_model
                                    for f in model_filter):
            continue
        # Reasoning model
        r_c = cost_usd(
            reasoning_model,
            total_items * ESTIMATES["stage4_isosci"]["prompt_tok_reasoning"] * mult,
            total_items * ESTIMATES["stage4_isosci"]["completion_tok_reasoning"] * mult,
        )
        # Standard model
        s_c = cost_usd(
            standard_model,
            total_items * ESTIMATES["stage4_isosci"]["prompt_tok_standard"] * mult,
            total_items * ESTIMATES["stage4_isosci"]["completion_tok_standard"] * mult,
        )
        stage4_by_model[label] = {
            "reasoning": {"model": reasoning_model, "cost": r_c},
            "standard":  {"model": standard_model,  "cost": s_c},
            "pair_total": r_c + s_c,
            "items_per_model": total_items,
        }
        stage4_total += r_c + s_c

    stage_costs["Stage 4 — Model evaluation"] = {
        "cost": stage4_total,
        "by_model_pair": stage4_by_model,
        "note": (f"{total_items} items/model "
                 f"({isosci_items_per_model} IsoSci + {existing_items} existing benchmarks) "
                 f"× {len(MODEL_PAIRS)} pairs × 2 models"),
    }
    total += stage4_total

    # ── STAGES 5–7 ─────────────────────────────────────────────────────────
    for stage in ["Stage 5 — Analysis", "Stage 6 — Figures", "Stage 7 — Dataset release"]:
        stage_costs[stage] = {"cost": 0.0, "note": "Local compute only, no API calls"}

    # ── Print report ───────────────────────────────────────────────────────
    width = 72
    print("\n" + "="*width)
    print(f"  IsoSci-150 Pipeline Cost Estimate")
    print(f"  Pairs: {n_pairs}  |  Conservative: {conservative}")
    print("="*width)

    for stage_name, info in stage_costs.items():
        cost = info["cost"]
        note = info.get("note", "")
        print(f"\n  {stage_name}")
        print(f"  {'─'*60}")
        if "by_model_pair" in info:
            for pair_label, pair_info in info["by_model_pair"].items():
                r = pair_info["reasoning"]
                s = pair_info["standard"]
                print(f"    {pair_label:<35} {fmt(pair_info['pair_total']):>8}")
                print(f"      reasoning ({r['model'].split('/')[-1]:<25}) {fmt(r['cost']):>8}")
                print(f"      standard  ({s['model'].split('/')[-1]:<25}) {fmt(s['cost']):>8}")
        elif "by_judge" in info:
            for judge, jcost in info["by_judge"].items():
                print(f"    {judge:<40} {fmt(jcost):>8}")
        print(f"  {'Subtotal':.<50} {fmt(cost):>8}")
        if note:
            print(f"  Note: {note}")

    print(f"\n{'='*width}")
    print(f"  {'TOTAL ESTIMATED COST':<50} {fmt(total):>8}")
    print(f"  {'(with conservative ×2 multiplier)' if not conservative else '(this IS the conservative estimate)':<50} "
          f"  {'→ ' + fmt(total*2) if not conservative else ''}")
    print(f"{'='*width}")

    print(f"\n  Budget check (your budget: $1,000–5,000):")
    budget_lo, budget_hi = 1000, 5000
    status_lo = "✓ within" if total <= budget_lo else "✗ over"
    status_hi = "✓ within" if total <= budget_hi else "✗ over"
    print(f"    Base estimate ({fmt(total)}) vs $1k lower bound: {status_lo}")
    print(f"    Base estimate ({fmt(total)}) vs $5k upper bound: {status_hi}")
    conservative_total = total * 2
    print(f"    Conservative ({fmt(conservative_total)}) vs $5k upper bound: "
          f"{'✓ within' if conservative_total <= budget_hi else '✗ over'}")

    print(f"\n  Cost-reduction options:")
    cheaper_pairs = int(n_pairs * 0.67)
    print(f"    • Reduce to {cheaper_pairs} pairs → est. {fmt(total * 0.67)} base")
    print(f"    • Drop GPQA+SciBench+MMLU (IsoSci only) → saves "
          f"~{fmt(stage4_total * existing_items / total_items * 0.5)}")
    print(f"    • Use only 3 model pairs (drop 2 smallest) → saves "
          f"~{fmt(stage4_total * 2/5)}")
    print(f"    • Use gpt-4o-mini as sole verifier (not 3 judges) → saves "
          f"~{fmt(stage_costs['Stage 3 — Verification (3 judges)']['cost'] * 0.7)}")
    print()

    return {
        "total_usd":     total,
        "conservative":  total * 2,
        "stage_costs":   {k: v["cost"] for k, v in stage_costs.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate API costs before running pipeline")
    parser.add_argument("--pairs", type=int, default=150,
                        help="Target number of pairs (default: 150)")
    parser.add_argument("--conservative", action="store_true",
                        help="Use 2× token multiplier for upper-bound estimate")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models (substring match)")
    args = parser.parse_args()
    estimate(n_pairs=args.pairs, conservative=args.conservative,
             model_filter=args.models)
