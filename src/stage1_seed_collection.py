from typing import Optional, Union
"""
src/stage1_seed_collection.py
==============================
Stage 1 — Collect seed problems from:
  (a) Existing benchmarks: GPQA Diamond, SciBench, MMLU-STEM
      (if CSVs/JSONs are present in data/raw/)
  (b) Synthetic LLM generation: fills gaps to reach target per-domain count

Output: data/raw/seed_problems.json
        — list of SeedProblem dicts, each with domain, structure_type,
          question, answer, source, and a manually checkable difficulty flag.

Usage:
    python src/stage1_seed_collection.py [--synthetic-only] [--n-per-domain N]
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
    DOMAINS, DOMAIN_PROFILES, STRUCTURE_TYPES,
    GENERATION_MODEL, RAW_DIR, LOGS_DIR,
    EXPECTED_REJECTION, PAIRS_PER_MAPPING, DOMAIN_MAPPINGS,
)
from src.api_client import make_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage1.log"),
    ],
)
logger = logging.getLogger("stage1")

# Seeds needed per domain to cover all mappings with rejection headroom
# Each domain appears in 3 mappings, need 25 pairs each → 75 seeds per domain
# With 30% rejection → 75 / 0.70 ≈ 108 seeds per domain
SEEDS_PER_DOMAIN = 108


# ── Benchmark loaders ─────────────────────────────────────────────────────

def load_gpqa(filepath: Path) -> list[dict]:
    """Load GPQA Diamond CSV. Expected columns: Question, Answer, Explanation, Subdomain."""
    if not filepath.exists():
        logger.warning(f"GPQA file not found at {filepath}. Skipping.")
        return []
    import csv
    seeds = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subdomain = row.get("Subdomain", "").lower()
            domain = _gpqa_subdomain_to_domain(subdomain)
            if domain is None:
                continue
            seeds.append({
                "id":             str(uuid.uuid4()),
                "source":         "gpqa_diamond",
                "domain":         domain,
                "question":       row.get("Question", "").strip(),
                "answer":         row.get("Correct Answer", row.get("Answer", "")).strip(),
                "difficulty":     "graduate",
                "structure_type": "unknown",  # will be classified in stage2
                "metadata":       {"subdomain": subdomain},
            })
    logger.info(f"Loaded {len(seeds)} GPQA seeds.")
    return seeds


def _gpqa_subdomain_to_domain(subdomain: str) -> Optional[str]:
    mapping = {
        "physics": "physics", "quantum": "physics", "mechanics": "physics",
        "electro": "physics", "optics": "physics", "thermo": "physics",
        "chemistry": "chemistry", "organic": "chemistry", "inorganic": "chemistry",
        "biochem": "chemistry", "physical chem": "chemistry",
        "biology": "biology", "cell": "biology", "molecular": "biology",
        "genetics": "biology", "ecology": "biology", "evolution": "biology",
        "earth": "earth_science", "geology": "earth_science",
        "atmospheric": "earth_science", "ocean": "earth_science",
    }
    for key, dom in mapping.items():
        if key in subdomain:
            return dom
    return None


def load_scibench(filepath: Path) -> list[dict]:
    """Load SciBench JSON."""
    if not filepath.exists():
        logger.warning(f"SciBench file not found at {filepath}. Skipping.")
        return []
    with open(filepath) as f:
        data = json.load(f)
    seeds = []
    for item in data:
        domain = item.get("subject", "").lower()
        if "phys" in domain:
            domain = "physics"
        elif "chem" in domain:
            domain = "chemistry"
        else:
            continue
        seeds.append({
            "id":             str(uuid.uuid4()),
            "source":         "scibench",
            "domain":         domain,
            "question":       item.get("problem", "").strip(),
            "answer":         str(item.get("answer", "")).strip(),
            "difficulty":     "college",
            "structure_type": "formula_recall_and_substitute",
            "metadata":       {"unit": item.get("unit", "")},
        })
    logger.info(f"Loaded {len(seeds)} SciBench seeds.")
    return seeds


def load_mmlu_stem(filepath: Path) -> list[dict]:
    """Load MMLU-STEM JSON (dict of subject → list of [question, A, B, C, D, answer])."""
    if not filepath.exists():
        logger.warning(f"MMLU-STEM file not found at {filepath}. Skipping.")
        return []
    with open(filepath) as f:
        data = json.load(f)
    seeds = []
    for subject, items in data.items():
        domain = _mmlu_subject_to_domain(subject)
        if domain is None:
            continue
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            q, a, b, c, d, ans = item[:6]
            choices = {"A": a, "B": b, "C": c, "D": d}
            seeds.append({
                "id":             str(uuid.uuid4()),
                "source":         "mmlu_stem",
                "domain":         domain,
                "question":       f"{q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}",
                "answer":         f"{ans}) {choices.get(ans, '')}",
                "difficulty":     "mixed",
                "structure_type": "unknown",
                "metadata":       {"subject": subject},
            })
    logger.info(f"Loaded {len(seeds)} MMLU-STEM seeds.")
    return seeds


def _mmlu_subject_to_domain(subject: str) -> Optional[str]:
    s = subject.lower()
    if any(k in s for k in ["physics", "astronomy"]):               return "physics"
    if any(k in s for k in ["chemistry"]):                          return "chemistry"
    if any(k in s for k in ["biology", "anatomy", "nutrition"]):    return "biology"
    if any(k in s for k in ["earth", "geology", "environmental"]):  return "earth_science"
    return None


# ── Synthetic seed generation ──────────────────────────────────────────────

SEED_GENERATION_SYSTEM = """You are an expert science educator creating evaluation problems.
Your task is to generate clear, well-defined science problems suitable for a benchmark dataset.
Each problem must:
1. Be solvable at college or advanced undergraduate level
2. Have a single unambiguous correct answer
3. Require a clear, identifiable reasoning procedure
4. Be self-contained (all needed information is in the problem)
Respond only with valid JSON — no preamble or explanation."""

SEED_GENERATION_TEMPLATE = """Generate {n} distinct {structure_type} problems in {domain}.

Structure type definition:
- formula_recall_and_substitute: student must recall a specific law/formula, substitute given values, compute result
- unit_conversion_chain: multi-step unit conversion requiring tracking of units throughout
- conservation_law_application: identify and apply a conservation principle (energy, mass, charge, etc.)
- proportional_reasoning: use ratios or scaling relationships to find an unknown quantity
- two_step_causal_chain: qualitative reasoning where A leads to B leads to C (no computation required)

Requirements:
- All numerical values must be given in the problem
- Difficulty: college undergraduate
- Vary the sub-topics within {domain}
- Each problem must be solvable in 3-5 reasoning steps

Return a JSON array of objects, each with:
{{
  "question": "full problem text",
  "answer": "correct answer with units if applicable",
  "solution_steps": ["step 1 description", "step 2 description", ...],
  "formula_used": "name of the key formula or principle",
  "sub_topic": "specific topic within {domain}",
  "estimated_steps": <integer 3-5>
}}"""


def generate_synthetic_seeds(client, domain: str, structure_type: str, n: int) -> list[dict]:
    """Generate n synthetic seed problems for a given domain and structure type."""
    prompt = SEED_GENERATION_TEMPLATE.format(
        n=n, domain=domain.replace("_", " "), structure_type=structure_type
    )
    logger.info(f"Generating {n} synthetic seeds: {domain} / {structure_type}")
    try:
        items = client.call_json(
            GENERATION_MODEL,
            [{"role": "user", "content": prompt}],
            system=SEED_GENERATION_SYSTEM,
            temperature=0.7,
            max_tokens=6000,
        )
        # Handle both list and {"problems": [...]} responses
        if isinstance(items, dict):
            items = items.get("problems", items.get("questions", []))
        seeds = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("question") or not item.get("answer"):
                continue
            seeds.append({
                "id":             str(uuid.uuid4()),
                "source":         "synthetic_llm",
                "domain":         domain,
                "question":       item["question"].strip(),
                "answer":         str(item["answer"]).strip(),
                "difficulty":     "college",
                "structure_type": structure_type,
                "metadata": {
                    "formula_used":    item.get("formula_used", ""),
                    "sub_topic":       item.get("sub_topic", ""),
                    "solution_steps":  item.get("solution_steps", []),
                    "estimated_steps": item.get("estimated_steps", 0),
                },
            })
        logger.info(f"  → Got {len(seeds)} valid seeds")
        return seeds
    except Exception as e:
        logger.error(f"Synthetic generation failed: {e}")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────

def deduplicate(seeds: list[dict], similarity_threshold: int = 40) -> list[dict]:
    """
    Simple token-overlap deduplication.
    Removes seeds whose question shares >similarity_threshold% tokens with an earlier one.
    """
    seen_tokens: list[set] = []
    unique = []
    for seed in seeds:
        tokens = set(seed["question"].lower().split())
        is_dup = False
        for prev_tokens in seen_tokens:
            if len(tokens) == 0:
                is_dup = True
                break
            overlap = len(tokens & prev_tokens) / len(tokens | prev_tokens) * 100
            if overlap > similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            seen_tokens.append(tokens)
            unique.append(seed)
    logger.info(f"Deduplication: {len(seeds)} → {len(unique)} seeds")
    return unique


# ── Main ──────────────────────────────────────────────────────────────────

def run(synthetic_only: bool = False, n_per_domain: int = SEEDS_PER_DOMAIN):
    from config import GPQA_FILE, SCIBENCH_FILE, MMLU_STEM_FILE
    client = make_client()
    all_seeds: list[dict] = []

    # ── Load existing benchmarks ──
    if not synthetic_only:
        all_seeds += load_gpqa(GPQA_FILE)
        all_seeds += load_scibench(SCIBENCH_FILE)
        all_seeds += load_mmlu_stem(MMLU_STEM_FILE)

    # ── Count per domain ──
    domain_counts = {d: sum(1 for s in all_seeds if s["domain"] == d) for d in DOMAINS}
    logger.info(f"Benchmark seeds per domain: {domain_counts}")

    # ── Fill gaps with synthetic generation ──
    for domain in DOMAINS:
        needed = max(0, n_per_domain - domain_counts[domain])
        if needed == 0:
            continue
        logger.info(f"Need {needed} more seeds for {domain}")
        # Distribute across structure types, capped at 10 per call to avoid truncation
        MAX_BATCH = 10
        per_type = max(1, needed // len(STRUCTURE_TYPES))
        remainder = needed - per_type * len(STRUCTURE_TYPES)
        for i, stype in enumerate(STRUCTURE_TYPES):
            n = per_type + (1 if i < remainder else 0)
            if n <= 0:
                continue
            while n > 0:
                batch_n = min(n, MAX_BATCH)
                batch = generate_synthetic_seeds(client, domain, stype, batch_n)
                all_seeds += batch
                n -= batch_n
                time.sleep(1)

    # ── Deduplicate ──
    all_seeds = deduplicate(all_seeds)

    # ── Report ──
    domain_counts = {d: sum(1 for s in all_seeds if s["domain"] == d) for d in DOMAINS}
    logger.info(f"Final seed counts: {domain_counts}")
    logger.info(f"Total seeds: {len(all_seeds)}")

    # ── Save ──
    output_path = RAW_DIR / "seed_problems.json"
    with open(output_path, "w") as f:
        json.dump(all_seeds, f, indent=2)
    logger.info(f"Saved {len(all_seeds)} seeds to {output_path}")
    return all_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Collect seed problems")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip benchmark loading, use only synthetic generation")
    parser.add_argument("--n-per-domain", type=int, default=SEEDS_PER_DOMAIN,
                        help=f"Target seeds per domain (default: {SEEDS_PER_DOMAIN})")
    args = parser.parse_args()
    run(synthetic_only=args.synthetic_only, n_per_domain=args.n_per_domain)
