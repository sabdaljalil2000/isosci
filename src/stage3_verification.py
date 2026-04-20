from typing import Optional, Union, Tuple, List, Dict
"""
src/stage3_verification.py
===========================
Stage 3 — Automated + human-assisted verification of candidate pairs.

Two verification modes:
  (A) LLM-automated: three LLM "judges" score each pair on 4 criteria
  (B) Human annotation UI: generates annotation sheets (CSV) for expert review

Scoring criteria (each 1-5):
  1. Logical equivalence   — same reasoning structure?
  2. Domain independence   — knowledge is non-overlapping?
  3. Difficulty parity     — comparable challenge?
  4. Self-containment      — problem is complete without external info?

A pair is ACCEPTED if:
  - Auto score ≥ 3.5/5 on all 4 criteria (avg across 3 judges)
  - OR human score ≥ 4/5 on all 4 criteria (overrides auto)

Computes inter-annotator agreement (Cohen's κ and Krippendorff's α).

Output: data/verified/verified_pairs.json     — accepted pairs
        data/verified/rejected_pairs.json     — rejected pairs
        data/verified/annotation_sheet.csv    — for human annotators
        data/verified/agreement_stats.json    — κ and α statistics

Usage:
    python src/stage3_verification.py [--mode auto|human|both] [--pilot N]
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GENERATION_MODEL, PAIRS_DIR, VERIFIED_DIR, LOGS_DIR, PILOT_PAIRS,
)
from src.api_client import make_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage3.log"),
    ],
)
logger = logging.getLogger("stage3")

# Acceptance thresholds
AUTO_ACCEPT_THRESHOLD  = 3.5   # avg score across judges
HUMAN_ACCEPT_THRESHOLD = 4.0   # avg score for human-verified pairs


# ── Verification prompts ───────────────────────────────────────────────────

VERIFIER_SYSTEM = """You are an expert science educator and benchmark quality reviewer.
You evaluate pairs of science problems for structural isomorphism — whether they share
identical logical structure but require different domain knowledge.

Score each criterion from 1 to 5:
1 = completely fails the criterion
3 = partially meets the criterion
5 = perfectly meets the criterion

Respond with valid JSON only."""

VERIFIER_TEMPLATE = """Evaluate this isomorphic problem pair:

=== SOURCE PROBLEM ({source_domain}) ===
{source_question}
ANSWER: {source_answer}

=== TARGET PROBLEM ({target_domain}) ===
{target_question}
ANSWER: {target_answer}

=== CLAIMED ISOMORPHISM ===
Structure type: {structure_type}
Justification: {justification}

Score this pair on 4 criteria (1-5 each):

1. LOGICAL EQUIVALENCE (1-5): Do both problems require the same reasoning procedure?
   Same number of steps? Same types of operations (recall, substitute, compute)?
   
2. DOMAIN INDEPENDENCE (1-5): Does the knowledge needed for one problem give NO advantage
   for the other? Are the formulas/facts genuinely from different domains?
   
3. DIFFICULTY PARITY (1-5): Would a strong undergraduate student find both problems
   equally challenging? Are they at the same level of complexity?
   
4. SELF-CONTAINMENT (1-5): Is each problem fully specified? Can it be solved with
   only the information given plus standard domain knowledge?

Also provide:
- A flag if the target answer seems incorrect (boolean)
- A brief reason if you reject (score < 3 on any criterion)

Return JSON:
{{
  "logical_equivalence": <1-5>,
  "domain_independence": <1-5>,
  "difficulty_parity": <1-5>,
  "self_containment": <1-5>,
  "answer_seems_correct": true/false,
  "rejection_reason": "reason or null",
  "overall_assessment": "brief 1-2 sentence assessment"
}}"""


# ── Judge ensemble ─────────────────────────────────────────────────────────

JUDGE_MODELS = [
    "anthropic/claude-sonnet-4-5",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
]


def score_pair_with_judge(client, pair: dict, judge_model: str) -> Optional[dict]:
    """Score a single pair with one judge model. Returns score dict or None on failure."""
    src = pair["source"]
    tgt = pair["target"]
    meta = tgt.get("metadata", {})

    prompt = VERIFIER_TEMPLATE.format(
        source_domain=src["domain"].replace("_", " "),
        source_question=src["question"],
        source_answer=src["answer"],
        target_domain=tgt["domain"].replace("_", " "),
        target_question=tgt["question"],
        target_answer=tgt["answer"],
        structure_type=src.get("structure_type", "unknown"),
        justification=meta.get("isomorphism_justification", "not provided"),
    )
    try:
        result = client.call_json(
            judge_model,
            [{"role": "user", "content": prompt}],
            system=VERIFIER_SYSTEM,
            temperature=0.1,
            max_tokens=512,
        )
        # Validate required keys
        required = ["logical_equivalence", "domain_independence",
                    "difficulty_parity", "self_containment"]
        for k in required:
            if k not in result:
                return None
            result[k] = float(result[k])
        result["judge_model"] = judge_model
        return result
    except Exception as e:
        logger.warning(f"Judge {judge_model} failed on pair {pair['pair_id'][:8]}: {e}")
        return None


def auto_verify_pair(client, pair: dict) -> dict:
    """Run all 3 judges on a pair. Returns updated pair with scores and decision."""
    scores_list = []
    for judge_model in JUDGE_MODELS:
        score = score_pair_with_judge(client, pair, judge_model)
        if score:
            scores_list.append(score)
        time.sleep(0.3)

    if not scores_list:
        pair["verification_status"] = "failed"
        pair["verification_scores"] = {}
        return pair

    # Average across judges
    criteria = ["logical_equivalence", "domain_independence",
                "difficulty_parity", "self_containment"]
    avg_scores = {}
    for c in criteria:
        vals = [s[c] for s in scores_list if c in s]
        avg_scores[c] = sum(vals) / len(vals) if vals else 0.0

    avg_scores["overall"] = sum(avg_scores[c] for c in criteria) / len(criteria)

    # Decision
    all_above_threshold = all(avg_scores[c] >= AUTO_ACCEPT_THRESHOLD for c in criteria)
    any_answer_wrong = any(not s.get("answer_seems_correct", True) for s in scores_list)

    if all_above_threshold and not any_answer_wrong:
        status = "auto_accepted"
    else:
        status = "auto_rejected"

    pair["verification_status"] = status
    pair["verification_scores"] = {
        "avg": avg_scores,
        "individual": scores_list,
        "rejection_reasons": [s.get("rejection_reason") for s in scores_list
                              if s.get("rejection_reason")],
    }
    return pair


# ── Human annotation sheet ─────────────────────────────────────────────────

def generate_annotation_sheet(pairs: List[dict], output_path: Path, n: int = None):
    """
    Generate a CSV annotation sheet for human expert review.
    Annotators fill in scores 1-5 for each criterion.
    """
    if n:
        pairs = pairs[:n]

    fieldnames = [
        "pair_id", "mapping", "source_domain", "target_domain",
        "source_question", "source_answer",
        "target_question", "target_answer",
        "structure_type", "isomorphism_justification",
        # Annotator fills these in:
        "logical_equivalence_score",    # 1-5
        "domain_independence_score",    # 1-5
        "difficulty_parity_score",      # 1-5
        "self_containment_score",       # 1-5
        "accept_reject",                # ACCEPT / REJECT
        "annotator_notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            src = pair["source"]
            tgt = pair["target"]
            meta = tgt.get("metadata", {})
            writer.writerow({
                "pair_id":                    pair["pair_id"],
                "mapping":                    pair["mapping"],
                "source_domain":              pair["source_domain"],
                "target_domain":              pair["target_domain"],
                "source_question":            src["question"],
                "source_answer":              src["answer"],
                "target_question":            tgt["question"],
                "target_answer":              tgt["answer"],
                "structure_type":             src.get("structure_type", ""),
                "isomorphism_justification":  meta.get("isomorphism_justification", ""),
                # Leave blank for annotator
                "logical_equivalence_score":  "",
                "domain_independence_score":  "",
                "difficulty_parity_score":    "",
                "self_containment_score":     "",
                "accept_reject":              "",
                "annotator_notes":            "",
            })

    logger.info(f"Annotation sheet saved to {output_path} ({len(pairs)} pairs)")


def load_human_annotations(csv_path: Path) -> Dict[str, dict]:
    """Load completed annotation sheet. Returns dict pair_id → scores."""
    annotations = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"]
            try:
                annotations[pair_id] = {
                    "logical_equivalence":  float(row["logical_equivalence_score"] or 0),
                    "domain_independence":  float(row["domain_independence_score"] or 0),
                    "difficulty_parity":    float(row["difficulty_parity_score"] or 0),
                    "self_containment":     float(row["self_containment_score"] or 0),
                    "accept_reject":        row.get("accept_reject", "").strip().upper(),
                    "notes":                row.get("annotator_notes", ""),
                }
            except (ValueError, KeyError):
                continue
    logger.info(f"Loaded human annotations for {len(annotations)} pairs")
    return annotations


# ── Inter-annotator agreement ──────────────────────────────────────────────

def compute_agreement(annotations_by_annotator: Dict[str, dict]) -> dict:
    """
    Compute Cohen's κ (pairwise) and Krippendorff's α (multi-annotator).
    annotations_by_annotator: {annotator_id: {pair_id: binary_decision (0/1)}}
    """
    annotators = list(annotations_by_annotator.keys())
    if len(annotators) < 2:
        return {"error": "Need at least 2 annotators"}

    # Get common pair IDs
    all_pair_ids = set.intersection(*[set(d.keys()) for d in annotations_by_annotator.values()])
    pair_ids = sorted(all_pair_ids)
    logger.info(f"Computing agreement on {len(pair_ids)} common pairs, {len(annotators)} annotators")

    results = {}

    # Pairwise Cohen's κ
    kappas = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a1 = annotators[i]
            a2 = annotators[j]
            d1 = annotations_by_annotator[a1]
            d2 = annotations_by_annotator[a2]
            ratings_1 = [d1[pid] for pid in pair_ids]
            ratings_2 = [d2[pid] for pid in pair_ids]
            k = _cohens_kappa(ratings_1, ratings_2)
            kappas.append(k)
            results[f"kappa_{a1}_vs_{a2}"] = k

    results["mean_kappa"] = sum(kappas) / len(kappas) if kappas else 0.0
    results["n_pairs"] = len(pair_ids)
    results["n_annotators"] = len(annotators)

    # Krippendorff's α (using krippendorff package if available)
    try:
        import krippendorff
        data = [[annotations_by_annotator[a].get(pid, None) for pid in pair_ids]
                for a in annotators]
        alpha = krippendorff.alpha(data, level_of_measurement="nominal")
        results["krippendorff_alpha"] = alpha
    except ImportError:
        results["krippendorff_alpha"] = None
        logger.warning("krippendorff package not installed; skipping α")
    except Exception as e:
        results["krippendorff_alpha_error"] = str(e)

    return results


def _cohens_kappa(r1: list, r2: list) -> float:
    """Compute Cohen's κ for two binary rating lists."""
    if len(r1) != len(r2) or len(r1) == 0:
        return 0.0
    n = len(r1)
    agree = sum(1 for a, b in zip(r1, r2) if a == b)
    po = agree / n  # observed agreement

    # Expected agreement
    classes = set(r1) | set(r2)
    pe = sum(
        (r1.count(c) / n) * (r2.count(c) / n)
        for c in classes
    )
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ── Main ──────────────────────────────────────────────────────────────────

def run(mode: str = "auto", pilot_n: int = PILOT_PAIRS,
        human_csv: Path = None, target_n: int = None):
    cand_path = PAIRS_DIR / "candidate_pairs.json"
    if not cand_path.exists():
        logger.error(f"Candidate pairs not found at {cand_path}. Run stage2 first.")
        sys.exit(1)

    with open(cand_path) as f:
        candidates = json.load(f)
    logger.info(f"Loaded {len(candidates)} candidate pairs")

    client = make_client()

    # ── Auto verification ──
    if mode in ("auto", "both"):
        unscored = [p for p in candidates
                    if not p.get("verification_scores", {}).get("avg")]
        already  = len(candidates) - len(unscored)
        logger.info(f"\nAuto verification: {len(unscored)} to score, {already} already scored (skipping)")

        verified = []
        for i, pair in enumerate(unscored):
            logger.info(f"  Verifying pair {i+1}/{len(unscored)} ({pair['mapping']})")
            pair = auto_verify_pair(client, pair)
            verified.append(pair)

            # Save scores back to candidate_pairs.json every 10 pairs
            if (i + 1) % 10 == 0:
                _save_verified(verified, prefix="interim")
                # Persist scores into candidate file so progress is never lost
                _flush_scores_to_candidates(cand_path, verified)

        # Final flush
        _flush_scores_to_candidates(cand_path, verified)

        # Merge newly scored with already-scored
        scored_ids = {p["pair_id"] for p in verified}
        candidates = verified + [p for p in candidates if p["pair_id"] not in scored_ids]

    # ── Generate human annotation sheet ──
    if mode in ("human", "both"):
        sheet_path = VERIFIED_DIR / "annotation_sheet.csv"
        generate_annotation_sheet(candidates, sheet_path, n=pilot_n)
        logger.info(f"Annotation sheet generated: {sheet_path}")
        logger.info("→ Have annotators fill in scores, then re-run with --mode load-human")

    # ── Load completed human annotations ──
    if mode == "load-human" and human_csv:
        annotations = load_human_annotations(Path(human_csv))
        for pair in candidates:
            pid = pair["pair_id"]
            if pid in annotations:
                ann = annotations[pid]
                criteria = ["logical_equivalence", "domain_independence",
                            "difficulty_parity", "self_containment"]
                avg = sum(ann[c] for c in criteria) / len(criteria)
                all_ok = all(ann[c] >= HUMAN_ACCEPT_THRESHOLD for c in criteria)
                human_decision = ann["accept_reject"]

                if human_decision == "ACCEPT" or (all_ok and human_decision != "REJECT"):
                    pair["verification_status"] = "human_accepted"
                else:
                    pair["verification_status"] = "human_rejected"

                pair["verification_scores"]["human"] = ann

    # ── Split accepted / rejected ──
    accepted = [p for p in candidates
                if p["verification_status"] in ("auto_accepted", "human_accepted")]
    rejected = [p for p in candidates
                if p["verification_status"] in ("auto_rejected", "human_rejected", "failed")]

    logger.info(f"\nVerification results:")
    logger.info(f"  Accepted: {len(accepted)}")
    logger.info(f"  Rejected: {len(rejected)}")
    logger.info(f"  Acceptance rate: {len(accepted)/len(candidates)*100:.1f}%")

    # ── Trim to target_n (balanced across mappings) ──
    if target_n is None:
        from config import PAIRS_PER_MAPPING, DOMAIN_MAPPINGS
        target_n = PAIRS_PER_MAPPING * len(DOMAIN_MAPPINGS)
        logger.info(f"  Target: {target_n} pairs ({PAIRS_PER_MAPPING} per mapping)")
    if len(accepted) > target_n:
        accepted = _balance_mappings(accepted, target_n)
        logger.info(f"  Trimmed to {len(accepted)} (balanced across mappings)")

    # ── Save ──
    _save_verified(accepted, prefix="verified")
    with open(VERIFIED_DIR / "rejected_pairs.json", "w") as f:
        json.dump(rejected, f, indent=2)

    # ── Summary stats ──
    mapping_counts = {}
    for p in accepted:
        m = p["mapping"]
        mapping_counts[m] = mapping_counts.get(m, 0) + 1

    logger.info("\nAccepted pairs by mapping:")
    for m, c in sorted(mapping_counts.items()):
        logger.info(f"  {m}: {c}")

    return accepted


def _save_verified(pairs: List[dict], prefix: str = "verified"):
    path = VERIFIED_DIR / f"{prefix}_pairs.json"
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved {len(pairs)} pairs to {path}")


def _flush_scores_to_candidates(cand_path: Path, scored_pairs: list):
    """Write verification scores back into candidate_pairs.json immediately."""
    try:
        candidates = json.load(open(cand_path))
        score_map = {p["pair_id"]: p for p in scored_pairs}
        for c in candidates:
            pid = c.get("pair_id")
            if pid in score_map:
                c["verification_scores"] = score_map[pid]["verification_scores"]
                c["verification_status"] = score_map[pid]["verification_status"]
        with open(cand_path, "w") as f:
            json.dump(candidates, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not flush scores to candidate file: {e}")


def _balance_mappings(accepted: List[dict], target_n: int) -> List[dict]:
    """Select target_n pairs balanced across mappings."""
    from collections import defaultdict
    by_mapping = defaultdict(list)
    for p in accepted:
        by_mapping[p["mapping"]].append(p)
    n_mappings = len(by_mapping)
    per_mapping = target_n // n_mappings
    result = []
    for pairs in by_mapping.values():
        result.extend(pairs[:per_mapping])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Verify isomorphic pairs")
    parser.add_argument("--mode", choices=["auto", "human", "both", "load-human"],
                        default="auto")
    parser.add_argument("--pilot", type=int, default=PILOT_PAIRS,
                        help="Number of pairs for human annotation pilot")
    parser.add_argument("--human-csv", type=str, default=None,
                        help="Path to completed annotation CSV (for load-human mode)")
    parser.add_argument("--target-n", type=int, default=150,
                        help="Target number of verified pairs")
    args = parser.parse_args()
    run(
        mode=args.mode,
        pilot_n=args.pilot,
        human_csv=args.human_csv,
        target_n=args.target_n,
    )