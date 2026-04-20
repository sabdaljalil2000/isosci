from typing import Optional, Union, Tuple, List, Dict
"""
src/stage4_evaluation.py
=========================
Stage 4 — Evaluate all 12 models (6 reasoning/standard pairs) on:
  (A) IsoSci-150 verified pairs
  (B) Existing benchmarks: GPQA Diamond, SciBench, MMLU-STEM

For each problem, records: model response, extracted answer,
correctness (auto-graded), token counts, latency.

Output: data/results/eval_{model_slug}_{benchmark}.json
        data/results/all_results.json  — merged

Usage:
    python src/stage4_evaluation.py [--benchmark isosci|gpqa|scibench|mmlu|all]
                                    [--model MODEL_ID] [--limit N]
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ALL_MODELS, RESULTS_DIR, VERIFIED_DIR, RAW_DIR, LOGS_DIR,
    EVAL_TEMPERATURE, EVAL_MAX_TOKENS, EVAL_CONCURRENT,
    NUMERIC_TOLERANCE, ZERO_SHOT_COT_PROMPT,
    GPQA_FILE, SCIBENCH_FILE, MMLU_STEM_FILE,
    DOMAIN_PROFILES,
)
from src.api_client import make_client, cost_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage4.log"),
    ],
)
logger = logging.getLogger("stage4")


# ── Answer extraction ─────────────────────────────────────────────────────

def extract_final_answer(response: str) -> str:
    """
    Extract the final answer from a model response.
    Handles plain text, LaTeX boxed, dollar-sign math, and bare numbers.
    """
    if not response or not response.strip():
        return ""

    # 1. Explicit final answer marker
    explicit_patterns = [
        r"[*][*]Final Answer[:[*]]*[*]?\s*(.+?)(?:\n|$)",
        r"Final Answer[:\s]+(.+?)(?:\n|$)",
        r"[\\]boxed[{]([^}]+)[}]",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val and val not in ("$$", "$"):
                return val

    # 2. Number inside display math $$ ... $$
    m = re.search(r'[$][$]\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?[^$\n]*?)\s*[$][$]', response)
    if m:
        return m.group(1).strip()

    # 3. Therefore / The answer is
    for pat in [
        r"Therefore[,\s]+(?:the answer is\s+)?(.+?)(?:\n|$)",
        r"The answer is[:\s]+(.+?)(?:\n|$)",
    ]:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val:
                return val

    # 4. Last line containing a number
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        m = re.search(r'([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)', line)
        if m:
            return m.group(1)

    return lines[-1] if lines else ""

def is_correct_mcq(predicted: str, gold: str) -> bool:
    """Check multiple-choice answer correctness."""
    pred = predicted.strip().upper()
    gold = gold.strip().upper()
    # Match letter only
    pred_letter = re.search(r'^([A-D])', pred)
    gold_letter = re.search(r'^([A-D])', gold)
    if pred_letter and gold_letter:
        return pred_letter.group(1) == gold_letter.group(1)
    return pred == gold


def is_correct_numeric(predicted: str, gold: str, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    """Check numeric answer correctness within tolerance."""
    def extract_number(s: str) -> Optional[float]:
        # Remove units, keep number and sign
        s = re.sub(r'[^\d.\-e+]', ' ', s).strip()
        # Handle scientific notation
        nums = re.findall(r'-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?', s)
        for n in nums:
            try:
                return float(n)
            except ValueError:
                continue
        return None

    pred_num = extract_number(predicted)
    gold_num = extract_number(gold)
    if pred_num is None or gold_num is None:
        return predicted.strip().lower() == gold.strip().lower()
    if gold_num == 0:
        return abs(pred_num) < 1e-10
    return abs(pred_num - gold_num) / abs(gold_num) <= tolerance


def grade_response(predicted: str, gold: str, answer_type: str = "mcq",
                   choices: dict = None, answer_text: str = None) -> bool:
    """Grade a model response against the gold answer.
    For MCQ, tries letter match first, then full-text match against choices.
    """
    extracted = extract_final_answer(predicted)
    if answer_type == "mcq":
        # Try direct letter match first
        if is_correct_mcq(extracted, gold):
            return True
        # If model wrote out the answer text instead of a letter,
        # check if extracted text matches the correct answer text
        if answer_text and extracted.strip().lower() == answer_text.strip().lower():
            return True
        # Also check against all choices text to find which letter it matches
        if choices:
            for letter, text in choices.items():
                if extracted.strip().lower() == text.strip().lower():
                    return letter == gold
        return False
    elif answer_type == "numeric":
        return is_correct_numeric(extracted, gold)
    else:
        return extracted.strip().lower() == gold.strip().lower()


# ── Benchmark loaders ─────────────────────────────────────────────────────

def load_isosci(filepath: Path) -> List[dict]:
    """Load IsoSci-150 verified pairs as flat list of eval items."""
    with open(filepath) as f:
        pairs = json.load(f)
    items = []
    for pair in pairs:
        for role in ("source", "target"):
            prob = pair[role]
            items.append({
                "item_id":      f"{pair['pair_id']}_{role}",
                "pair_id":      pair["pair_id"],
                "role":         role,
                "mapping":      pair["mapping"],
                "domain":       prob["domain"],
                "question":     prob["question"],
                "answer":       prob["answer"],
                "answer_type":  "numeric",
                "benchmark":    "isosci",
                "structure_type": prob.get("structure_type", "unknown"),
            })
    logger.info(f"Loaded {len(items)} IsoSci eval items ({len(pairs)} pairs × 2)")
    return items


def load_gpqa_eval(filepath: Path) -> List[dict]:
    if not filepath.exists():
        logger.warning(f"GPQA file not found: {filepath}")
        return []
    import csv, random
    items = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            subdomain = row.get("Subdomain", "").lower()
            from src.stage1_seed_collection import _gpqa_subdomain_to_domain
            domain = _gpqa_subdomain_to_domain(subdomain) or "unknown"

            # Build proper 4-choice MCQ: shuffle correct + 3 incorrect answers
            correct  = row.get("Correct Answer", "").strip()
            wrong1   = row.get("Incorrect Answer 1", "").strip()
            wrong2   = row.get("Incorrect Answer 2", "").strip()
            wrong3   = row.get("Incorrect Answer 3", "").strip()

            options  = [correct, wrong1, wrong2, wrong3]
            random.seed(i)           # deterministic shuffle per question
            random.shuffle(options)
            correct_letter = "ABCD"[options.index(correct)]

            q_text = row.get("Question", "").strip() + "\n"
            for letter, opt in zip("ABCD", options):
                q_text += f"{letter}) {opt}\n"

            items.append({
                "item_id":      f"gpqa_{i}",
                "pair_id":      None,
                "domain":       domain,
                "question":     q_text.strip(),
                "answer":       correct_letter,       # letter A/B/C/D
                "answer_text":  correct,              # full text of correct answer
                "choices":      dict(zip("ABCD", options)),  # {A: text, B: text, ...}
                "answer_type":  "mcq",
                "benchmark":    "gpqa",
            })
    logger.info(f"Loaded {len(items)} GPQA eval items")
    return items


def load_scibench_eval(filepath: Path) -> List[dict]:
    if not filepath.exists():
        logger.warning(f"SciBench file not found: {filepath}")
        return []
    with open(filepath) as f:
        data = json.load(f)
    items = []
    for i, item in enumerate(data):
        domain_raw = item.get("subject", "").lower()
        domain = "physics" if "phys" in domain_raw else "chemistry"
        items.append({
            "item_id":    f"scibench_{i}",
            "pair_id":    None,
            "domain":     domain,
            "question":   item.get("problem", "").strip(),
            "answer":     str(item.get("answer", "")).strip(),
            "answer_type": "numeric",
            "benchmark":  "scibench",
        })
    logger.info(f"Loaded {len(items)} SciBench eval items")
    return items


def load_mmlu_eval(filepath: Path, max_per_domain: int = 250) -> List[dict]:
    if not filepath.exists():
        logger.warning(f"MMLU-STEM file not found: {filepath}")
        return []
    with open(filepath) as f:
        data = json.load(f)
    from src.stage1_seed_collection import _mmlu_subject_to_domain
    items = []
    domain_counts = {}
    for subject, examples in data.items():
        domain = _mmlu_subject_to_domain(subject)
        if not domain:
            continue
        if domain_counts.get(domain, 0) >= max_per_domain:
            continue
        for i, ex in enumerate(examples):
            if domain_counts.get(domain, 0) >= max_per_domain:
                break
            if not isinstance(ex, (list, tuple)) or len(ex) < 6:
                continue
            q, a, b, c, d, ans = ex[:6]
            question = (f"{q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}")
            items.append({
                "item_id":    f"mmlu_{subject}_{i}",
                "pair_id":    None,
                "domain":     domain,
                "question":   question,
                "answer":     ans,
                "answer_type": "mcq",
                "benchmark":  "mmlu_stem",
            })
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    logger.info(f"Loaded {len(items)} MMLU-STEM eval items")
    return items


# ── Single model evaluation ────────────────────────────────────────────────

def evaluate_single(client, model: str, item: dict,
                    reasoning_flag: Optional[bool] = None) -> dict:
    """Evaluate one model on one item. Returns result dict."""
    messages = [
        {"role": "user", "content": f"{ZERO_SHOT_COT_PROMPT}\n\n{item['question']}"}
    ]
    try:
        resp = client.call_model(
            model=model,
            messages=messages,
            temperature=EVAL_TEMPERATURE,
            max_tokens=EVAL_MAX_TOKENS,
            reasoning_enabled=reasoning_flag,
        )
        predicted = resp.content
        correct = grade_response(
            predicted,
            item["answer"],
            item.get("answer_type", "mcq"),
            choices=item.get("choices"),
            answer_text=item.get("answer_text"),
        )
        extracted = extract_final_answer(predicted)
        return {
            **item,
            "model":            model,
            "reasoning_flag":   reasoning_flag,
            "response":         predicted,
            "extracted_answer": extracted,
            "gold_answer":      item["answer"],
            "correct":          correct,
            "prompt_tokens":    resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens":     resp.total_tokens,
            "latency_s":        resp.latency_s,
            "error":            None,
        }
    except Exception as e:
        logger.error(f"Model {model} failed on {item['item_id']}: {e}")
        return {
            **item,
            "model":            model,
            "response":         None,
            "extracted_answer": None,
            "gold_answer":      item["answer"],
            "correct":          False,
            "prompt_tokens":    0,
            "completion_tokens": 0,
            "total_tokens":     0,
            "latency_s":        0.0,
            "error":            str(e),
        }


def evaluate_model_on_benchmark(
    client, model: str, items: List[dict], benchmark: str,
    limit: int = None, reasoning_flag: Optional[bool] = None
) -> List[dict]:
    """Evaluate one model on all items in a benchmark. Returns list of result dicts."""
    if limit:
        items = items[:limit]

    results = []
    # Include reasoning flag in slug so toggle/non-toggle results are separate files
    flag_suffix = "_thinking_on" if reasoning_flag is True else "_thinking_off" if reasoning_flag is False else ""
    slug = model.replace("/", "_") + flag_suffix
    final_path = RESULTS_DIR / f"eval_{slug}_{benchmark}.json"
    resume_path = RESULTS_DIR / f"eval_{slug}_{benchmark}_partial.json"
    done_ids = set()

    # If final output already exists, skip entirely
    if final_path.exists():
        with open(final_path) as f:
            results = json.load(f)
        acc = sum(1 for r in results if r.get("correct")) / len(results) * 100 if results else 0
        logger.info(f"SKIPPING {model} on {benchmark} — already complete ({len(results)} items, acc={acc:.1f}%)")
        return results

    # Resume from partial if exists
    if resume_path.exists():
        with open(resume_path) as f:
            partial = json.load(f)
        results = partial
        done_ids = {r["item_id"] for r in results}
        logger.info(f"Resuming {model} on {benchmark}: {len(done_ids)} already done")

    remaining = [item for item in items if item["item_id"] not in done_ids]
    logger.info(f"Evaluating {model} on {benchmark}: {len(remaining)} items remaining")

    for i, item in enumerate(remaining):
        result = evaluate_single(client, model, item, reasoning_flag=reasoning_flag)
        results.append(result)

        if (i + 1) % 10 == 0:
            # Save partial
            with open(resume_path, "w") as f:
                json.dump(results, f, indent=2)
            correct_so_far = sum(1 for r in results if r.get("correct"))
            logger.info(
                f"  [{model}|{benchmark}] {i+1}/{len(remaining)} | "
                f"Acc: {correct_so_far/len(results)*100:.1f}%"
            )
        time.sleep(0.2)  # gentle pacing

    # Save final
    out_path = RESULTS_DIR / f"eval_{slug}_{benchmark}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if resume_path.exists():
        resume_path.unlink()

    acc = sum(1 for r in results if r.get("correct")) / len(results) * 100 if results else 0
    logger.info(f"  DONE: {model} on {benchmark} — Accuracy: {acc:.1f}%")
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def run(benchmarks: List[str] = None, model_filter: str = None, limit: int = None):
    client = make_client()
    benchmarks = benchmarks or ["isosci", "gpqa", "scibench", "mmlu"]

    # Load benchmark items
    benchmark_items = {}
    if "isosci" in benchmarks:
        vp = VERIFIED_DIR / "verified_pairs.json"
        if vp.exists():
            benchmark_items["isosci"] = load_isosci(vp)
        else:
            logger.warning("verified_pairs.json not found — skipping IsoSci")

    if "gpqa" in benchmarks:
        benchmark_items["gpqa"] = load_gpqa_eval(GPQA_FILE)
    if "scibench" in benchmarks:
        benchmark_items["scibench"] = load_scibench_eval(SCIBENCH_FILE)
    if "mmlu" in benchmarks:
        benchmark_items["mmlu_stem"] = load_mmlu_eval(MMLU_STEM_FILE)

    from config import ALL_MODELS_WITH_FLAGS
    model_flags = ALL_MODELS_WITH_FLAGS
    if model_filter:
        model_flags = [(m, f) for m, f in model_flags if model_filter.lower() in m.lower()]
        logger.info(f"Filtered to models: {[m for m,f in model_flags]}")

    all_results = []
    for model, reasoning_flag in model_flags:
        flag_str = f" [reasoning={'ON' if reasoning_flag else 'OFF'}]" if reasoning_flag is not None else ""
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model}{flag_str}")
        logger.info(f"{'='*60}")
        for bench_name, items in benchmark_items.items():
            if not items:
                continue
            results = evaluate_model_on_benchmark(
                client, model, items, bench_name,
                limit=limit, reasoning_flag=reasoning_flag
            )
            all_results.extend(results)

    # Save merged
    merged_path = RESULTS_DIR / "all_results.json"
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {merged_path}")
    logger.info(f"API usage summary: {cost_tracker.summary()}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Model evaluation")
    parser.add_argument("--benchmark", nargs="+",
                        choices=["isosci", "gpqa", "scibench", "mmlu", "all"],
                        default=["all"])
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to specific model (substring match)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max items per benchmark (for testing)")
    args = parser.parse_args()
    benches = args.benchmark
    if "all" in benches:
        benches = ["isosci", "gpqa", "scibench", "mmlu"]
    run(benchmarks=benches, model_filter=args.model, limit=args.limit)
