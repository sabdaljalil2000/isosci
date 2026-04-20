from typing import Optional, Union, Tuple, List, Dict
"""
src/stage2_pair_generation.py
==============================
Stage 2 — For each seed problem, generate CANDIDATES_PER_SEED isomorphic
partner problems in a different domain using the LLM.

For each (source_domain, target_domain) mapping, picks seeds from the
source domain and generates partners in the target domain.

Output: data/pairs/candidate_pairs.json
        — list of CandidatePair dicts (unverified)

Usage:
    python src/stage2_pair_generation.py [--mapping physics chemistry] [--limit N]
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DOMAIN_MAPPINGS, PAIRS_PER_MAPPING, CANDIDATES_PER_SEED,
    GENERATION_MODEL, RAW_DIR, PAIRS_DIR, LOGS_DIR,
)
from src.api_client import make_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage2.log"),
    ],
)
logger = logging.getLogger("stage2")


# ── Prompts ───────────────────────────────────────────────────────────────

PAIR_GEN_SYSTEM = """You are an expert in multiple scientific disciplines with deep knowledge of
physics, chemistry, biology, and earth science.

Your task is to create ISOMORPHIC science problems — problems that share identical logical and
mathematical structure but require different domain knowledge.

"Isomorphic" means:
- Same number and type of reasoning steps
- Same mathematical operations (if any)
- Same solution procedure (e.g., recall formula → substitute values → compute)
- BUT different domain facts, constants, formulas, and named entities
- A student solving one problem gains NO advantage on the other from domain knowledge

You must respond with valid JSON only — no markdown, no preamble."""

PAIR_GEN_TEMPLATE = """I have a source problem from {source_domain} with the following structure:

SOURCE PROBLEM:
{question}

CORRECT ANSWER: {answer}

REASONING STRUCTURE:
- Structure type: {structure_type}
- Key formula/principle used: {formula}
- Solution steps: {steps}

Your task: Generate {n} isomorphic partner problems in {target_domain}.

Each partner must:
1. Use a DIFFERENT formula/principle from {target_domain} (not the same concept in a different name)
2. Have IDENTICAL logical structure: {structure_type}
3. Have the same number of solution steps ({n_steps} steps)
4. Be solvable at the same difficulty level (college undergraduate)
5. Be completely self-contained — all needed values given
6. Have a unique, unambiguous correct answer

Return a JSON array of {n} objects, each with:
{{
  "question": "complete problem text with all given values",
  "answer": "correct answer with units",
  "formula_used": "the specific formula or law used",
  "solution_steps": ["step 1", "step 2", ...],
  "domain_knowledge_required": ["fact 1", "fact 2"],
  "isomorphism_justification": "explain in one sentence why this has identical structure to the source",
  "sub_topic": "specific sub-topic within {target_domain}"
}}"""


# ── Pair generation ───────────────────────────────────────────────────────

def generate_partner(client, seed: dict, target_domain: str, n: int) -> List[dict]:
    """Generate n candidate partner problems for a seed in target_domain."""
    meta = seed.get("metadata", {})
    steps = meta.get("solution_steps", [])
    n_steps = meta.get("estimated_steps", len(steps) or 3)
    formula = meta.get("formula_used", "unknown")

    prompt = PAIR_GEN_TEMPLATE.format(
        source_domain=seed["domain"].replace("_", " "),
        question=seed["question"],
        answer=seed["answer"],
        structure_type=seed.get("structure_type", "formula_recall_and_substitute"),
        formula=formula,
        steps=", ".join(steps) if steps else "not specified",
        n_steps=n_steps,
        target_domain=target_domain.replace("_", " "),
        n=n,
    )

    try:
        items = client.call_json(
            GENERATION_MODEL,
            [{"role": "user", "content": prompt}],
            system=PAIR_GEN_SYSTEM,
            temperature=0.7,
            max_tokens=6000,
        )
        if isinstance(items, dict):
            # Handle {"candidates": [...]} or {"partners": [...]}
            items = items.get("candidates", items.get("partners", items.get("problems", [])))
        if not isinstance(items, list):
            logger.warning(f"Unexpected response type: {type(items)}")
            return []

        partners = []
        for item in items:
            if not isinstance(item, dict):
                continue
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if not q or not a:
                continue
            partners.append({
                "id":            str(uuid.uuid4()),
                "question":      q,
                "answer":        a,
                "domain":        target_domain,
                "structure_type": seed.get("structure_type", "formula_recall_and_substitute"),
                "metadata": {
                    "formula_used":               item.get("formula_used", ""),
                    "solution_steps":             item.get("solution_steps", []),
                    "domain_knowledge_required":  item.get("domain_knowledge_required", []),
                    "isomorphism_justification":  item.get("isomorphism_justification", ""),
                    "sub_topic":                  item.get("sub_topic", ""),
                },
            })
        return partners

    except Exception as e:
        logger.error(f"Partner generation failed for seed {seed['id']}: {e}")
        return []


def build_candidate_pairs(
    seeds: List[dict],
    mapping: Tuple[str, str],
    n_pairs: int,
    candidates_per_seed: int,
    client,
) -> List[dict]:
    """
    For a (source, target) domain mapping, build candidate pairs.
    Returns list of CandidatePair dicts.
    """
    source_domain, target_domain = mapping
    source_seeds = [s for s in seeds if s["domain"] == source_domain]

    if len(source_seeds) < n_pairs:
        logger.warning(
            f"Only {len(source_seeds)} seeds for {source_domain}, "
            f"need {n_pairs}. Using all available."
        )

    # Shuffle for variety
    import random
    random.shuffle(source_seeds)
    selected_seeds = source_seeds[:n_pairs]

    candidate_pairs = []
    for i, seed in enumerate(selected_seeds):
        logger.info(
            f"  [{source_domain}→{target_domain}] "
            f"Seed {i+1}/{len(selected_seeds)}: {seed['id'][:8]}…"
        )
        partners = generate_partner(client, seed, target_domain, candidates_per_seed)
        for partner in partners:
            candidate_pairs.append({
                "pair_id":       str(uuid.uuid4()),
                "mapping":       f"{source_domain}_to_{target_domain}",
                "source_domain": source_domain,
                "target_domain": target_domain,
                "source": {
                    "id":             seed["id"],
                    "question":       seed["question"],
                    "answer":         seed["answer"],
                    "domain":         seed["domain"],
                    "structure_type": seed.get("structure_type", "unknown"),
                    "source_bench":   seed.get("source", "unknown"),
                    "metadata":       seed.get("metadata", {}),
                },
                "target": partner,
                "verification_status": "pending",
                "verification_scores": {},
            })
        time.sleep(0.5)  # gentle rate limiting

    logger.info(f"  → Generated {len(candidate_pairs)} candidate pairs for {source_domain}→{target_domain}")
    return candidate_pairs


# ── Main ──────────────────────────────────────────────────────────────────

def run(
    mapping_filter: Optional[Tuple[str, str]] = None,
    limit: Optional[int] = None,
):
    seed_path = RAW_DIR / "seed_problems.json"
    if not seed_path.exists():
        logger.error(f"Seed file not found: {seed_path}. Run stage1 first.")
        sys.exit(1)

    with open(seed_path) as f:
        seeds = json.load(f)
    logger.info(f"Loaded {len(seeds)} seeds.")

    client = make_client()
    all_pairs: List[dict] = []

    mappings = DOMAIN_MAPPINGS
    if mapping_filter:
        mappings = [m for m in mappings if set(m) == set(mapping_filter)]
        if not mappings:
            logger.error(f"No matching mapping for {mapping_filter}")
            sys.exit(1)

    for mapping in mappings:
        src, tgt = mapping
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing mapping: {src} → {tgt}")
        logger.info(f"{'='*60}")

        n = min(PAIRS_PER_MAPPING, limit or PAIRS_PER_MAPPING)
        pairs = build_candidate_pairs(
            seeds=seeds,
            mapping=mapping,
            n_pairs=n,
            candidates_per_seed=CANDIDATES_PER_SEED,
            client=client,
        )
        all_pairs.extend(pairs)

        # Save incrementally (in case of interruption)
        interim_path = PAIRS_DIR / f"candidates_{src}_to_{tgt}.json"
        with open(interim_path, "w") as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"Saved {len(pairs)} pairs to {interim_path}")

    # Merge all
    output_path = PAIRS_DIR / "candidate_pairs.json"
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    logger.info(f"\nTotal candidate pairs generated: {len(all_pairs)}")
    logger.info(f"Saved to {output_path}")

    # Summary
    from collections import Counter
    mapping_counts = Counter(p["mapping"] for p in all_pairs)
    for m, c in sorted(mapping_counts.items()):
        logger.info(f"  {m}: {c} candidates")

    return all_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Generate isomorphic pair candidates")
    parser.add_argument("--mapping", nargs=2, metavar=("SOURCE", "TARGET"),
                        help="Only process this specific mapping (e.g., --mapping physics chemistry)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max pairs per mapping (default: PAIRS_PER_MAPPING from config)")
    args = parser.parse_args()
    run(
        mapping_filter=tuple(args.mapping) if args.mapping else None,
        limit=args.limit,
    )
