#!/usr/bin/env python3
"""
compare_human_vs_llm.py
========================
Compares human annotator verdicts against LLM judge scores for the
same 50 sampled pairs. Produces:
  - Confusion matrix (LLM accepted vs human accepted)
  - Per-criterion score correlation
  - Precision / recall of LLM judges using human as gold standard
  - Summary table for the paper

Usage:
    python compare_human_vs_llm.py \
        --human1  annotation/annotation_sheet_annotator1_filled.csv \
        --human2  annotation/annotation_sheet_annotator2_filled.csv \
        --sampled annotation/sampled_pairs.json \
        --output  annotation/comparison_report.txt
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ── Loaders ───────────────────────────────────────────────────────────────

def load_human_csv(path: str) -> dict:
    """Returns dict: pair_id -> annotation dict."""
    annotations = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["pair_id"].strip()
            try:
                annotations[pid] = {
                    "logical_equivalence": float(row["logical_equivalence"]) if row["logical_equivalence"].strip() else None,
                    "domain_independence":  float(row["domain_independence"])  if row["domain_independence"].strip()  else None,
                    "difficulty_parity":    float(row["difficulty_parity"])    if row["difficulty_parity"].strip()    else None,
                    "self_containment":     float(row["self_containment"])     if row["self_containment"].strip()     else None,
                    "verdict":              row["verdict"].strip().upper(),
                    "failure_modes":        [fm.strip() for fm in row.get("failure_modes","").split(",") if fm.strip()],
                    "notes":                row.get("notes","").strip(),
                    "pair_idx":             int(row["pair_idx"]),
                }
            except (ValueError, KeyError):
                continue
    return annotations


def load_sampled_pairs(path: str) -> dict:
    """Returns dict: pair_id -> pair dict (with LLM scores)."""
    with open(path) as f:
        pairs = json.load(f)
    return {p["pair_id"]: p for p in pairs}


# ── LLM verdict extraction ────────────────────────────────────────────────

LLM_ACCEPT_THRESHOLD = 3.5   # same threshold used in Stage 3

def llm_verdict(pair: dict) -> str:
    """Derive ACCEPT/REJECT from LLM judge average scores."""
    scores = pair.get("verification_scores", {}).get("avg", {})
    if not scores:
        return "UNKNOWN"
    criteria = ["logical_equivalence", "domain_independence",
                "difficulty_parity", "self_containment"]
    vals = [scores.get(c) for c in criteria]
    if any(v is None for v in vals):
        return "UNKNOWN"
    if all(v >= LLM_ACCEPT_THRESHOLD for v in vals):
        return "ACCEPT"
    return "REJECT"


def llm_scores(pair: dict) -> dict:
    """Return the averaged LLM judge scores dict."""
    return pair.get("verification_scores", {}).get("avg", {})


# ── Agreement statistics ──────────────────────────────────────────────────

def cohens_kappa(labels_a: list, labels_b: list) -> float:
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan")
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    po = agree / n
    classes = set(labels_a) | set(labels_b)
    pe = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in classes)
    return (po - pe) / (1 - pe) if pe < 1.0 else 1.0


def pearson_r(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num / den if den > 0 else float("nan")


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# ── Main analysis ─────────────────────────────────────────────────────────

def run(human1_path, human2_path, sampled_path, output_path):
    h1 = load_human_csv(human1_path)
    h2 = load_human_csv(human2_path)
    pairs = load_sampled_pairs(sampled_path)

    # Find common pair IDs across all three sources
    common_ids = sorted(set(h1) & set(h2) & set(pairs))
    n = len(common_ids)
    print(f"Common pairs across human annotations and LLM scores: {n}")

    if n == 0:
        print("ERROR: No common pair IDs found.")
        print("  Human1 IDs (sample):", list(h1.keys())[:3])
        print("  Pairs IDs (sample):",  list(pairs.keys())[:3])
        return

    # ── Build aligned lists ───────────────────────────────────────────────
    llm_verdicts = []
    h1_verdicts  = []
    h2_verdicts  = []
    # Gold = majority of h1 + h2 (if they agree → gold; if not → ambiguous)
    gold_verdicts = []

    criteria = ["logical_equivalence", "domain_independence",
                "difficulty_parity",   "self_containment"]
    llm_scores_by_c  = defaultdict(list)
    h1_scores_by_c   = defaultdict(list)
    h2_scores_by_c   = defaultdict(list)

    disagreed_pairs = []   # pairs where h1 != h2

    for pid in common_ids:
        lv = llm_verdict(pairs[pid])
        v1 = h1[pid]["verdict"]
        v2 = h2[pid]["verdict"]

        # Collapse MARGINAL into REJECT for binary analysis
        lv_bin = "ACCEPT" if lv == "ACCEPT" else "REJECT"
        v1_bin = "ACCEPT" if v1 == "ACCEPT" else "REJECT"
        v2_bin = "ACCEPT" if v2 == "ACCEPT" else "REJECT"

        llm_verdicts.append(lv_bin)
        h1_verdicts.append(v1_bin)
        h2_verdicts.append(v2_bin)

        # Gold = both humans agree; else ambiguous
        if v1_bin == v2_bin:
            gold_verdicts.append(v1_bin)
        else:
            gold_verdicts.append("AMBIGUOUS")
            disagreed_pairs.append(pid)

        ls = llm_scores(pairs[pid])
        for c in criteria:
            lc = ls.get(c)
            h1c = h1[pid].get(c)
            h2c = h2[pid].get(c)
            if lc is not None: llm_scores_by_c[c].append(lc)
            if h1c is not None: h1_scores_by_c[c].append(h1c)
            if h2c is not None: h2_scores_by_c[c].append(h2c)

    # ── Confusion matrix: LLM vs gold ─────────────────────────────────────
    # Only on pairs where humans agreed
    agreed_ids = [pid for pid in common_ids
                  if h1[pid]["verdict"] != "AMBIGUOUS"
                  and h2[pid]["verdict"] != "AMBIGUOUS"
                  and (("ACCEPT" if h1[pid]["verdict"]=="ACCEPT" else "REJECT") ==
                       ("ACCEPT" if h2[pid]["verdict"]=="ACCEPT" else "REJECT"))]

    tp = fp = tn = fn = 0
    for pid in agreed_ids:
        lv = "ACCEPT" if llm_verdict(pairs[pid]) == "ACCEPT" else "REJECT"
        gold = "ACCEPT" if h1[pid]["verdict"] == "ACCEPT" else "REJECT"
        if   lv == "ACCEPT" and gold == "ACCEPT": tp += 1
        elif lv == "ACCEPT" and gold == "REJECT":  fp += 1
        elif lv == "REJECT" and gold == "ACCEPT":  fn += 1
        else:                                       tn += 1

    n_agreed = len(agreed_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    accuracy  = (tp + tn) / n_agreed if n_agreed > 0 else float("nan")
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else float("nan")

    # ── Kappa: LLM vs each human ──────────────────────────────────────────
    kappa_llm_h1 = cohens_kappa(llm_verdicts, h1_verdicts)
    kappa_llm_h2 = cohens_kappa(llm_verdicts, h2_verdicts)
    kappa_h1_h2  = cohens_kappa(h1_verdicts, h2_verdicts)

    # ── Score correlations ────────────────────────────────────────────────
    score_corrs = {}
    for c in criteria:
        ls = llm_scores_by_c[c]
        h1s = h1_scores_by_c[c]
        h2s = h2_scores_by_c[c]
        n_c = min(len(ls), len(h1s), len(h2s))
        if n_c > 1:
            r_h1 = pearson_r(ls[:n_c], h1s[:n_c])
            r_h2 = pearson_r(ls[:n_c], h2s[:n_c])
            score_corrs[c] = {
                "r_llm_h1": r_h1,
                "r_llm_h2": r_h2,
                "mean_llm": mean(ls),
                "mean_h1":  mean(h1s),
                "mean_h2":  mean(h2s),
            }

    # ── Failure mode analysis on rejected pairs ───────────────────────────
    # For pairs LLM accepted but humans rejected — what did humans say was wrong?
    llm_accepted_human_rejected = []
    for pid in agreed_ids:
        lv = llm_verdict(pairs[pid])
        gold = "ACCEPT" if h1[pid]["verdict"] == "ACCEPT" else "REJECT"
        if lv == "ACCEPT" and gold == "REJECT":
            fms = list(set(h1[pid]["failure_modes"] + h2[pid]["failure_modes"]))
            llm_accepted_human_rejected.append({
                "pair_id": pid,
                "failure_modes": fms,
                "h1_scores": {c: h1[pid].get(c) for c in criteria},
                "h2_scores": {c: h2[pid].get(c) for c in criteria},
                "llm_scores": llm_scores(pairs[pid]),
            })

    fm_on_false_positives = Counter()
    for p in llm_accepted_human_rejected:
        for fm in p["failure_modes"]:
            fm_on_false_positives[fm] += 1

    # ── Build report ──────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("LLM JUDGE vs HUMAN ANNOTATION COMPARISON REPORT")
    lines.append("=" * 65)
    lines.append(f"Total pairs in sample:              {n}")
    lines.append(f"Pairs where humans agreed:          {n_agreed}  ({n_agreed/n*100:.0f}%)")
    lines.append(f"Pairs where humans disagreed:       {len(disagreed_pairs)}  ({len(disagreed_pairs)/n*100:.0f}%)")

    lines.append("\n--- ACCEPTANCE RATES ---")
    lines.append(f"LLM judges accepted:                {llm_verdicts.count('ACCEPT')}/{n}  ({llm_verdicts.count('ACCEPT')/n*100:.1f}%)")
    lines.append(f"Annotator 1 accepted:               {h1_verdicts.count('ACCEPT')}/{n}  ({h1_verdicts.count('ACCEPT')/n*100:.1f}%)")
    lines.append(f"Annotator 2 accepted:               {h2_verdicts.count('ACCEPT')}/{n}  ({h2_verdicts.count('ACCEPT')/n*100:.1f}%)")

    lines.append("\n--- LLM JUDGE PERFORMANCE (gold = human consensus) ---")
    lines.append(f"N pairs with human consensus:       {n_agreed}")
    lines.append(f"True Positives  (LLM✓ Human✓):     {tp}")
    lines.append(f"False Positives (LLM✓ Human✗):     {fp}  ← LLM over-accepted")
    lines.append(f"False Negatives (LLM✗ Human✓):     {fn}  ← LLM over-rejected")
    lines.append(f"True Negatives  (LLM✗ Human✗):     {tn}")
    lines.append(f"Precision:                          {precision:.3f}  (of LLM accepts, how many are valid)")
    lines.append(f"Recall:                             {recall:.3f}  (of valid pairs, how many LLM found)")
    lines.append(f"F1:                                 {f1:.3f}")
    lines.append(f"Accuracy:                           {accuracy:.3f}")

    lines.append("\n--- INTER-RATER AGREEMENT (Cohen's κ) ---")
    lines.append(f"LLM vs Annotator 1:                 κ = {kappa_llm_h1:.3f}")
    lines.append(f"LLM vs Annotator 2:                 κ = {kappa_llm_h2:.3f}")
    lines.append(f"Annotator 1 vs Annotator 2:         κ = {kappa_h1_h2:.3f}  (human agreement)")

    lines.append("\n--- CRITERION SCORES (mean, 1–5 scale) ---")
    lines.append(f"{'Criterion':<28} {'LLM':>6} {'Human1':>8} {'Human2':>8} {'r(LLM,H1)':>11} {'r(LLM,H2)':>11}")
    lines.append("-" * 75)
    for c in criteria:
        if c in score_corrs:
            sc = score_corrs[c]
            lines.append(
                f"{c:<28} {sc['mean_llm']:>6.2f} {sc['mean_h1']:>8.2f} {sc['mean_h2']:>8.2f}"
                f" {sc['r_llm_h1']:>11.3f} {sc['r_llm_h2']:>11.3f}"
            )

    lines.append(f"\n--- FAILURE MODES ON LLM FALSE POSITIVES (n={fp}) ---")
    lines.append("(Pairs LLM accepted that humans rejected — what went wrong?)")
    if fm_on_false_positives:
        for fm, count in fm_on_false_positives.most_common():
            lines.append(f"  {fm:<35} {count}")
    else:
        lines.append("  (no false positives — LLM and humans fully agree on accepts)")

    lines.append("\n--- INTERPRETATION ---")
    if precision >= 0.80:
        lines.append("  ✓ LLM precision ≥ 0.80: most LLM-accepted pairs are valid.")
        lines.append("    The benchmark findings are unlikely to be invalidated by LLM noise.")
    elif precision >= 0.65:
        lines.append("  ~ LLM precision 0.65–0.80: moderate false positive rate.")
        lines.append("    Recommend filtering to human-consensus subset for key claims,")
        lines.append("    or reporting p_know on the human-validated subset separately.")
    else:
        lines.append("  ✗ LLM precision < 0.65: high false positive rate.")
        lines.append("    The LLM-verified pairs contain too many invalid pairs to support")
        lines.append("    strong causal claims. Use human-consensus subset for all key results.")

    if recall >= 0.80:
        lines.append("  ✓ LLM recall ≥ 0.80: most valid pairs were found by LLM judges.")
    else:
        lines.append(f"  ~ LLM recall = {recall:.2f}: some valid pairs were rejected by LLM judges,")
        lines.append("    but this is a conservative bias (does not inflate false discovery rate).")

    lines.append("=" * 65)

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")

    # ── Save human-consensus subset for reanalysis ────────────────────────
    consensus_accept_ids = set(
        pid for pid in agreed_ids
        if ("ACCEPT" if h1[pid]["verdict"]=="ACCEPT" else "REJECT") == "ACCEPT"
    )
    print(f"\nHuman-consensus ACCEPT pairs: {len(consensus_accept_ids)}")
    print("These pair IDs can be used to filter verified_pairs.json for")
    print("a conservative reanalysis of p_know on the validated subset.")

    # Save the consensus IDs for use in reanalysis
    consensus_path = Path(output_path).parent / "human_consensus_accept_ids.json" if output_path else Path("human_consensus_accept_ids.json")
    with open(consensus_path, "w") as f:
        json.dump({
            "consensus_accept_ids": sorted(consensus_accept_ids),
            "n_consensus_accept": len(consensus_accept_ids),
            "n_agreed_pairs": n_agreed,
            "n_total_sampled": n,
            "llm_precision": precision,
            "llm_recall": recall,
            "llm_f1": f1,
            "kappa_llm_h1": kappa_llm_h1,
            "kappa_llm_h2": kappa_llm_h2,
            "kappa_h1_h2": kappa_h1_h2,
        }, f, indent=2)
    print(f"Consensus IDs saved to: {consensus_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human1",   required=True, help="Filled CSV from annotator 1")
    parser.add_argument("--human2",   required=True, help="Filled CSV from annotator 2")
    parser.add_argument("--sampled",  required=True, help="sampled_pairs.json from generate_annotation_sheet.py")
    parser.add_argument("--output",   default="annotation/comparison_report.txt")
    args = parser.parse_args()
    run(args.human1, args.human2, args.sampled, args.output)
