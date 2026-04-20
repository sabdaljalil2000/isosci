from typing import Optional, Union, Tuple, List, Dict
"""
src/stage5_analysis.py
=======================
Stage 5 — Compute all paper findings from evaluation results:

  (A) Domain-asymmetric gains (Table 1 in paper)
      ∆acc = reasoning_acc − standard_acc, per domain
      Spearman ρ between ∆acc and domain computation intensity

  (B) IsoSci-150 decoupling analysis (Table 2)
      Domain gap δ = acc(source) − acc(target) per pair
      % of gains that are domain-dependent vs. structure-invariant

  (C) Accuracy per benchmark per model (Tables for appendix)

  (D) Bootstrap confidence intervals + McNemar's test

Output: analysis/domain_asymmetry.json
        analysis/isosci_decoupling.json
        analysis/full_accuracy_table.json
        analysis/statistical_tests.json
        analysis/summary_for_paper.json  ← the one you fill tables from

Usage:
    python src/stage5_analysis.py
"""

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_PAIRS, DOMAINS, DOMAIN_PROFILES,
    RESULTS_DIR, ANALYSIS_DIR, LOGS_DIR,
    BOOTSTRAP_SAMPLES, CONFIDENCE_LEVEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage5.log"),
    ],
)
logger = logging.getLogger("stage5")


# ── Data loading ──────────────────────────────────────────────────────────

def load_results() -> List[dict]:
    path = RESULTS_DIR / "all_results.json"
    if not path.exists():
        logger.error(f"Results file not found: {path}. Run stage4 first.")
        sys.exit(1)
    with open(path) as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} result records")
    return results


def model_key(result: dict) -> str:
    """
    Unique display key for a model run.
    Toggle-mode models get a thinking_on / thinking_off suffix
    so qwen3-8b with reasoning=True != qwen3-8b with reasoning=False.
    """
    model = result["model"]
    flag  = result.get("reasoning_flag", None)
    if flag is True:
        return f"{model} [thinking=ON]"
    elif flag is False:
        return f"{model} [thinking=OFF]"
    return model


def index_results(results: List[dict]) -> dict:
    """
    Index results by (model_key, benchmark, domain).
    Toggle-mode models are indexed separately via their thinking=ON/OFF suffix.
    Returns nested dict: {model_key: {benchmark: {domain: [result_dicts]}}}
    """
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        idx[model_key(r)][r["benchmark"]][r["domain"]].append(r)
    return idx


# ── Accuracy utilities ────────────────────────────────────────────────────

def accuracy(records: List[dict]) -> float:
    if not records:
        return float("nan")
    return sum(1 for r in records if r.get("correct")) / len(records)


def bootstrap_ci(records: List[dict], n_boot: int = BOOTSTRAP_SAMPLES,
                 alpha: float = 1 - CONFIDENCE_LEVEL) -> Tuple[float, float]:
    """Bootstrap 95% CI for accuracy."""
    if not records:
        return (float("nan"), float("nan"))
    correct = [1 if r.get("correct") else 0 for r in records]
    n = len(correct)
    boot_accs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(correct, size=n, replace=True)
        boot_accs.append(sample.mean())
    lo = float(np.percentile(boot_accs, 100 * alpha / 2))
    hi = float(np.percentile(boot_accs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def mcnemar_test(records_a: List[dict], records_b: List[dict]) -> dict:
    """
    McNemar's test between two models on the same items.
    Requires same item_ids in both record sets.
    """
    # Align by item_id
    a_by_id = {r["item_id"]: r.get("correct", False) for r in records_a}
    b_by_id = {r["item_id"]: r.get("correct", False) for r in records_b}
    common = set(a_by_id) & set(b_by_id)
    if len(common) < 10:
        return {"error": f"Insufficient common items: {len(common)}"}

    b01 = sum(1 for i in common if not a_by_id[i] and b_by_id[i])   # A wrong, B right
    b10 = sum(1 for i in common if a_by_id[i] and not b_by_id[i])   # A right, B wrong
    n_discordant = b01 + b10
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_common": len(common)}
    # With continuity correction
    statistic = (abs(b01 - b10) - 1) ** 2 / n_discordant
    p_value = float(1 - stats.chi2.cdf(statistic, df=1))
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "b01": b01,
        "b10": b10,
        "n_common": len(common),
        "significant": p_value < 0.05,
    }


# ── Analysis A: Domain-asymmetric gains ──────────────────────────────────

def compute_domain_asymmetric_gains(idx: dict) -> dict:
    """
    For each model pair (reasoning vs standard), compute ∆acc per domain
    across all benchmarks combined.
    Returns table + Spearman correlation.
    """
    logger.info("\nAnalysis A: Domain-asymmetric gains")
    rows = []

    for pair in MODEL_PAIRS:
        reasoning_model, standard_model, label = pair[0], pair[1], pair[2]
        r_extra = pair[3] if len(pair) > 3 else {}
        s_extra = pair[4] if len(pair) > 4 else {}
        # Build the idx keys (same logic as model_key() but from config extras)
        r_flag = r_extra.get("reasoning", {}).get("enabled", None)
        s_flag = s_extra.get("reasoning", {}).get("enabled", None)
        r_key = f"{reasoning_model} [thinking=ON]"  if r_flag is True  else                 f"{reasoning_model} [thinking=OFF]" if r_flag is False else reasoning_model
        s_key = f"{standard_model} [thinking=ON]"  if s_flag is True  else                 f"{standard_model} [thinking=OFF]" if s_flag is False else standard_model

        if r_key not in idx or s_key not in idx:
            logger.warning(f"Missing results for pair: {label} (r_key={r_key}, s_key={s_key})")
            continue

        row = {"pair": label, "reasoning": r_key, "standard": s_key}

        for domain in DOMAINS:
            # Collect across all benchmarks
            r_records = []
            s_records = []
            for bench in idx[r_key]:
                r_records += idx[r_key][bench].get(domain, [])
            for bench in idx[s_key]:
                s_records += idx[s_key][bench].get(domain, [])

            r_acc = accuracy(r_records)
            s_acc = accuracy(s_records)
            delta = r_acc - s_acc if not (math.isnan(r_acc) or math.isnan(s_acc)) else float("nan")

            r_ci = bootstrap_ci(r_records)
            s_ci = bootstrap_ci(s_records)

            row[domain] = {
                "reasoning_acc": round(r_acc * 100, 1),
                "standard_acc":  round(s_acc * 100, 1),
                "delta_acc":     round(delta * 100, 1) if not math.isnan(delta) else None,
                "r_ci_95":       [round(r_ci[0]*100,1), round(r_ci[1]*100,1)],
                "s_ci_95":       [round(s_ci[0]*100,1), round(s_ci[1]*100,1)],
                "n_reasoning":   len(r_records),
                "n_standard":    len(s_records),
            }

        rows.append(row)

    # Average ∆acc per domain
    avg_deltas = {}
    for domain in DOMAINS:
        deltas = [row[domain]["delta_acc"] for row in rows
                  if domain in row and row[domain]["delta_acc"] is not None]
        avg_deltas[domain] = round(sum(deltas) / len(deltas), 1) if deltas else None

    # Spearman correlation: avg ∆acc vs computation intensity
    comp_intensities = [DOMAIN_PROFILES[d]["C"] for d in DOMAINS]
    delta_vals = [avg_deltas.get(d) for d in DOMAINS]
    valid = [(c, d) for c, d in zip(comp_intensities, delta_vals) if d is not None]
    if len(valid) >= 3:
        x, y = zip(*valid)
        rho, p_val = stats.spearmanr(x, y)
        spearman = {"rho": round(float(rho), 3), "p_value": round(float(p_val), 4)}
    else:
        spearman = {"error": "insufficient data"}

    result = {
        "model_pair_rows":   rows,
        "average_deltas":    avg_deltas,
        "spearman_rho":      spearman,
        "domain_profiles":   {d: DOMAIN_PROFILES[d] for d in DOMAINS},
    }

    logger.info(f"Average ∆acc per domain: {avg_deltas}")
    logger.info(f"Spearman ρ (∆acc vs computation intensity): {spearman}")
    return result


# ── Analysis B: IsoSci-150 decoupling ────────────────────────────────────

def compute_isosci_decoupling(idx: dict) -> dict:
    """
    For each model pair, compute:
      - Accuracy on source vs target problems
      - Domain gap δ = acc(source) − acc(target) per pair
      - % of reasoning model gains that are domain-dependent
    """
    logger.info("\nAnalysis B: IsoSci-150 knowledge-reasoning decoupling")

    def get_isosci_records(model: str, role: str) -> List[dict]:
        records = idx.get(model, {}).get("isosci", {})
        # Flatten all domains for this role
        flat = []
        for domain_records in records.values():
            flat += [r for r in domain_records if r.get("role") == role]
        return flat

    rows = []
    pair_level_gaps = []

    for _pair in MODEL_PAIRS:
        reasoning_model, standard_model, label = _pair[0], _pair[1], _pair[2]
        _r_extra = _pair[3] if len(_pair) > 3 else {}
        _s_extra = _pair[4] if len(_pair) > 4 else {}
        _r_flag = _r_extra.get("reasoning", {}).get("enabled", None)
        _s_flag = _s_extra.get("reasoning", {}).get("enabled", None)
        reasoning_model = (f"{reasoning_model} [thinking=ON]" if _r_flag is True else
                          f"{reasoning_model} [thinking=OFF]" if _r_flag is False else reasoning_model)
        standard_model  = (f"{standard_model} [thinking=ON]" if _s_flag is True else
                          f"{standard_model} [thinking=OFF]" if _s_flag is False else standard_model)
        # Source accuracy (physics/computation-heavy problems)
        r_source = get_isosci_records(reasoning_model, "source")
        s_source = get_isosci_records(standard_model, "source")
        r_target = get_isosci_records(reasoning_model, "target")
        s_target = get_isosci_records(standard_model, "target")

        if not r_source:
            logger.warning(f"No IsoSci results for {reasoning_model}")
            continue

        # Per-pair domain gap
        r_by_pair = {r["pair_id"]: r.get("correct", False) for r in r_source + r_target}
        s_by_pair = {r["pair_id"]: r.get("correct", False) for r in s_source + s_target}

        # For each pair: did reasoning model improve on source but not target?
        source_pairs = {r["pair_id"] for r in r_source}
        target_pairs = {r["pair_id"] for r in r_target}
        common_pairs = source_pairs & target_pairs

        n_knowledge_dependent = 0    # R improved on source, not target
        n_structure_invariant = 0    # R improved on both
        n_neither = 0

        for pid in common_pairs:
            r_src_correct = next((r.get("correct") for r in r_source if r["pair_id"] == pid), None)
            r_tgt_correct = next((r.get("correct") for r in r_target if r["pair_id"] == pid), None)
            s_src_correct = next((r.get("correct") for r in s_source if r["pair_id"] == pid), None)
            s_tgt_correct = next((r.get("correct") for r in s_target if r["pair_id"] == pid), None)

            if None in (r_src_correct, r_tgt_correct, s_src_correct, s_tgt_correct):
                continue

            r_improved_src = r_src_correct and not s_src_correct
            r_improved_tgt = r_tgt_correct and not s_tgt_correct

            if r_improved_src and not r_improved_tgt:
                n_knowledge_dependent += 1
            elif r_improved_src and r_improved_tgt:
                n_structure_invariant += 1
            else:
                n_neither += 1

        total_improvements = n_knowledge_dependent + n_structure_invariant
        pct_knowledge = (n_knowledge_dependent / total_improvements * 100
                         if total_improvements > 0 else float("nan"))
        pct_structure = (n_structure_invariant / total_improvements * 100
                         if total_improvements > 0 else float("nan"))

        # Overall accuracies
        r_acc_src = accuracy(r_source)
        s_acc_src = accuracy(s_source)
        r_acc_tgt = accuracy(r_target)
        s_acc_tgt = accuracy(s_target)

        row = {
            "pair": label,
            "source_acc_reasoning": round(r_acc_src * 100, 1),
            "source_acc_standard":  round(s_acc_src * 100, 1),
            "source_delta":         round((r_acc_src - s_acc_src) * 100, 1),
            "target_acc_reasoning": round(r_acc_tgt * 100, 1),
            "target_acc_standard":  round(s_acc_tgt * 100, 1),
            "target_delta":         round((r_acc_tgt - s_acc_tgt) * 100, 1),
            "domain_gap_delta":     round(((r_acc_src - s_acc_src) - (r_acc_tgt - s_acc_tgt)) * 100, 1),
            "n_knowledge_dependent_gains": n_knowledge_dependent,
            "n_structure_invariant_gains": n_structure_invariant,
            "pct_gains_knowledge_dependent": round(pct_knowledge, 1),
            "pct_gains_structure_invariant": round(pct_structure, 1),
            "n_common_pairs": len(common_pairs),
        }
        rows.append(row)

    # Average across model pairs
    def avg_field(field):
        vals = [r[field] for r in rows if r.get(field) is not None
                and not (isinstance(r[field], float) and math.isnan(r[field]))]
        return round(sum(vals) / len(vals), 1) if vals else None

    avg_pct_knowledge = avg_field("pct_gains_knowledge_dependent")

    result = {
        "model_pair_rows":              rows,
        "avg_pct_knowledge_dependent":  avg_pct_knowledge,
        "interpretation": (
            f"On average, {avg_pct_knowledge}% of reasoning model gains on IsoSci-150 "
            f"are domain-dependent (knowledge retrieval), not structure-invariant (reasoning)."
            if avg_pct_knowledge else "Insufficient data"
        ),
    }

    logger.info(f"Avg % knowledge-dependent gains: {avg_pct_knowledge}%")
    return result


# ── Analysis C: Full accuracy table ──────────────────────────────────────

def compute_full_accuracy_table(idx: dict) -> dict:
    """Accuracy for every (model, benchmark, domain) combination."""
    logger.info("\nAnalysis C: Full accuracy table")
    table = {}
    benchmarks = set()
    for model in idx:
        for bench in idx[model]:
            benchmarks.add(bench)

    for model in idx:
        table[model] = {}
        for bench in sorted(benchmarks):
            table[model][bench] = {"overall": None, "by_domain": {}}
            all_records = []
            for domain in DOMAINS:
                records = idx[model].get(bench, {}).get(domain, [])
                acc = accuracy(records)
                ci = bootstrap_ci(records)
                table[model][bench]["by_domain"][domain] = {
                    "accuracy": round(acc * 100, 1) if not math.isnan(acc) else None,
                    "ci_95":    [round(ci[0]*100,1), round(ci[1]*100,1)],
                    "n":        len(records),
                }
                all_records += records
            # Overall
            ov = accuracy(all_records)
            table[model][bench]["overall"] = round(ov * 100, 1) if not math.isnan(ov) else None
            table[model][bench]["n_total"] = len(all_records)

    return table


# ── Analysis D: McNemar's tests ───────────────────────────────────────────

def compute_statistical_tests(idx: dict) -> dict:
    """McNemar's tests between reasoning and standard models for each pair."""
    logger.info("\nAnalysis D: Statistical significance tests")
    tests = {}
    for _pair in MODEL_PAIRS:
        reasoning_model, standard_model, label = _pair[0], _pair[1], _pair[2]
        _r_extra = _pair[3] if len(_pair) > 3 else {}
        _s_extra = _pair[4] if len(_pair) > 4 else {}
        _r_flag = _r_extra.get("reasoning", {}).get("enabled", None)
        _s_flag = _s_extra.get("reasoning", {}).get("enabled", None)
        reasoning_model = (f"{reasoning_model} [thinking=ON]" if _r_flag is True else
                          f"{reasoning_model} [thinking=OFF]" if _r_flag is False else reasoning_model)
        standard_model  = (f"{standard_model} [thinking=ON]" if _s_flag is True else
                          f"{standard_model} [thinking=OFF]" if _s_flag is False else standard_model)
        if reasoning_model not in idx or standard_model not in idx:
            continue
        tests[label] = {}
        for bench in set(idx[reasoning_model]) | set(idx[standard_model]):
            r_all = []
            s_all = []
            for domain in DOMAINS:
                r_all += idx[reasoning_model].get(bench, {}).get(domain, [])
                s_all += idx[standard_model].get(bench, {}).get(domain, [])
            if r_all and s_all:
                tests[label][bench] = mcnemar_test(r_all, s_all)
    return tests


# ── Paper summary ─────────────────────────────────────────────────────────

def build_paper_summary(domain_asym: dict, decoupling: dict, full_table: dict) -> dict:
    """
    Build the headline numbers for the paper.
    This is what goes directly into the paper tables.
    """
    summary = {
        "headline_findings": [],
        "table1_domain_asymmetric_gains": domain_asym["model_pair_rows"],
        "table1_average_deltas": domain_asym["average_deltas"],
        "table1_spearman": domain_asym["spearman_rho"],
        "table2_decoupling": decoupling["model_pair_rows"],
        "table2_avg_pct_knowledge": decoupling["avg_pct_knowledge_dependent"],
        "full_accuracy_table": full_table,
    }

    # Auto-generate headline sentences
    avg_deltas = domain_asym.get("average_deltas", {})
    if avg_deltas.get("physics") and avg_deltas.get("biology"):
        summary["headline_findings"].append(
            f"Reasoning models gain {avg_deltas['physics']}pp on physics "
            f"but only {avg_deltas['biology']}pp on biology (domain-asymmetric gains)."
        )
    pct_k = decoupling.get("avg_pct_knowledge_dependent")
    if pct_k:
        summary["headline_findings"].append(
            f"{pct_k}% of reasoning model gains on IsoSci-150 are knowledge-dependent, "
            f"not structure-invariant (reasoning-knowledge decoupling)."
        )
    rho = domain_asym.get("spearman_rho", {})
    if "rho" in rho:
        summary["headline_findings"].append(
            f"Spearman ρ = {rho['rho']} (p = {rho['p_value']}) between ∆acc and "
            f"domain computation intensity confirms the asymmetry is systematic."
        )

    return summary


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    results = load_results()
    idx = index_results(results)

    domain_asym  = compute_domain_asymmetric_gains(idx)
    decoupling   = compute_isosci_decoupling(idx)
    full_table   = compute_full_accuracy_table(idx)
    stat_tests   = compute_statistical_tests(idx)
    paper_summary = build_paper_summary(domain_asym, decoupling, full_table)

    # Save all
    files = {
        ANALYSIS_DIR / "domain_asymmetry.json":    domain_asym,
        ANALYSIS_DIR / "isosci_decoupling.json":   decoupling,
        ANALYSIS_DIR / "full_accuracy_table.json": full_table,
        ANALYSIS_DIR / "statistical_tests.json":   stat_tests,
        ANALYSIS_DIR / "summary_for_paper.json":   paper_summary,
    }
    for path, data in files.items():
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved: {path}")

    logger.info("\n" + "="*60)
    logger.info("PAPER HEADLINE FINDINGS:")
    for finding in paper_summary["headline_findings"]:
        logger.info(f"  • {finding}")
    logger.info("="*60)

    return paper_summary


if __name__ == "__main__":
    run()
