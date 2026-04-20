from typing import Optional, Union, Tuple, List, Dict
"""
src/stage7_dataset_release.py
==============================
Stage 7 — Prepare the IsoSci-150 dataset for HuggingFace Hub release.

Generates:
  - outputs/isosci150/   (HuggingFace-ready dataset directory)
    ├── dataset_dict.json
    ├── train.json           (80 pairs — for few-shot / classifier training)
    ├── test.json            (70 pairs — held-out for benchmark evaluation)
    ├── README.md            (dataset card)
    └── evaluation/
        ├── eval_script.py   (standalone evaluation script)
        └── baseline_results.json

Usage:
    python src/stage7_dataset_release.py [--push-to-hub]
    
    # To push: set HF_TOKEN env var, then:
    python src/stage7_dataset_release.py --push-to-hub --repo-id your-org/isosci-150
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VERIFIED_DIR, OUTPUTS_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage7.log"),
    ],
)
logger = logging.getLogger("stage7")

HF_DIR = OUTPUTS_DIR / "isosci150"
HF_DIR.mkdir(exist_ok=True)
(HF_DIR / "evaluation").mkdir(exist_ok=True)


# ── Dataset card ──────────────────────────────────────────────────────────

README_TEMPLATE = """---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - science
  - reasoning
  - benchmark
  - isomorphic
  - physics
  - chemistry
  - biology
  - earth-science
size_categories:
  - n<1K
---

# IsoSci-150: Isomorphic Cross-Domain Science Problem Pairs

## Overview

IsoSci-150 is a benchmark of **150 isomorphic problem pairs** designed to cleanly separate 
reasoning ability from domain knowledge retrieval in large language model evaluation.

Each pair consists of two problems:
- **Source problem**: from one scientific domain (e.g., physics)
- **Target problem**: from a different domain (e.g., chemistry) with identical logical structure

The key property: if a model solves the source but fails the target, the bottleneck is 
**domain knowledge**, not **reasoning ability**. This enables direct attribution of 
performance gaps.

## Motivation

Standard science benchmarks (GPQA, SciBench, MMLU-STEM) confound reasoning and knowledge:
it is impossible to determine whether a model fails because it cannot reason or because it 
lacks the relevant domain fact. IsoSci-150 solves this by holding reasoning structure constant.

## Dataset Structure

```
isosci150/
├── train.json    (80 pairs — for few-shot / in-context learning)
├── test.json     (70 pairs — held-out evaluation)
```

Each entry has the following fields:

```json
{
  "pair_id": "uuid",
  "mapping": "physics_to_chemistry",
  "source_domain": "physics",
  "target_domain": "chemistry",
  "source": {
    "question": "A 2.0 mol sample of ideal gas...",
    "answer": "0.996 atm",
    "structure_type": "formula_recall_and_substitute",
    "domain": "physics"
  },
  "target": {
    "question": "What is the pH of a 0.1 M acetic acid solution...",
    "answer": "2.87",
    "structure_type": "formula_recall_and_substitute",
    "domain": "chemistry"
  },
  "verification_status": "auto_accepted",
  "verification_scores": { ... }
}
```

## Domain Mappings

| Mapping | N pairs |
|---------|---------|
| Physics → Chemistry | 25 |
| Physics → Biology | 25 |
| Physics → Earth Science | 25 |
| Chemistry → Biology | 25 |
| Chemistry → Earth Science | 25 |
| Biology → Earth Science | 25 |

## Evaluation

Use the provided `evaluation/eval_script.py`:

```bash
python evaluation/eval_script.py \\
    --model openai/gpt-4o \\
    --split test \\
    --api-key YOUR_OPENROUTER_KEY
```

## Metrics

- **Source accuracy**: accuracy on source domain problems
- **Target accuracy**: accuracy on target domain problems  
- **Domain gap δ**: source_accuracy − target_accuracy
- **Knowledge-dependent gain ratio**: fraction of gains that are domain-specific

## Citation

```bibtex
@dataset{isosci150_2025,
  title={IsoSci-150: Isomorphic Cross-Domain Science Problem Pairs for 
         Evaluating Reasoning vs. Knowledge Retrieval in LLMs},
  author={[Author names]},
  year={2025},
  url={https://huggingface.co/datasets/[your-org]/isosci-150},
  license={CC-BY-4.0}
}
```

## License

CC-BY-4.0. Please cite the associated NeurIPS 2025 paper if you use this dataset.
"""


# ── Standalone eval script ────────────────────────────────────────────────

EVAL_SCRIPT = '''#!/usr/bin/env python3
"""
IsoSci-150 Standalone Evaluation Script
=========================================
Evaluates any model via OpenRouter on the IsoSci-150 test set.

Usage:
    python eval_script.py --model openai/gpt-4o --split test --api-key YOUR_KEY
    
    # Or with environment variable:
    export OPENROUTER_API_KEY=your_key
    python eval_script.py --model openai/gpt-4o
"""

import argparse, json, re, time
from pathlib import Path
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_TEMPLATE = (
    "Think step by step. Show your reasoning clearly. "
    "Provide your final answer at the end in the format: "
    "**Final Answer:** <your answer>\\n\\n{question}"
)

def call_model(api_key, model, question, max_tokens=2048, temperature=0.0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://isosci-benchmark.github.io",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2: raise
            time.sleep(5)

def extract_answer(text):
    m = re.search(r\'\\*\\*Final Answer[:\\*]*\\*?\\s*(.+?)(?:\\n|$)\', text, re.I)
    if m: return m.group(1).strip()
    nums = re.findall(r\'-?[\\d]+\\.?[\\d]*(?:[eE][+-]?\\d+)?\', text)
    return nums[-1] if nums else text.strip().split("\\n")[-1]

def is_correct(pred, gold):
    def to_num(s):
        nums = re.findall(r\'-?[\\d]+\\.?[\\d]*(?:[eE][+-]?\\d+)?\', re.sub(r\'[^\\d.\\-e+]\', \' \', s))
        try: return float(nums[0]) if nums else None
        except: return None
    p, g = to_num(pred), to_num(gold)
    if p is not None and g is not None and g != 0:
        return abs(p - g) / abs(g) <= 0.02
    return pred.strip().upper()[:1] == gold.strip().upper()[:1]

def evaluate(model, api_key, split="test"):
    data_path = Path(__file__).parent.parent / f"{split}.json"
    with open(data_path) as f: pairs = json.load(f)
    
    results = {"model": model, "split": split, "pairs": []}
    src_correct = tgt_correct = 0
    
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair[\'mapping\']}", end="", flush=True)
        
        src_resp = call_model(api_key, model, pair["source"]["question"])
        src_pred = extract_answer(src_resp)
        src_ok = is_correct(src_pred, pair["source"]["answer"])
        
        tgt_resp = call_model(api_key, model, pair["target"]["question"])
        tgt_pred = extract_answer(tgt_resp)
        tgt_ok = is_correct(tgt_pred, pair["target"]["answer"])
        
        src_correct += src_ok; tgt_correct += tgt_ok
        results["pairs"].append({
            "pair_id": pair["pair_id"],
            "source_correct": src_ok,
            "target_correct": tgt_ok,
            "source_pred": src_pred,
            "target_pred": tgt_pred,
        })
        print(f" src={'✓' if src_ok else '✗'} tgt={'✓' if tgt_ok else '✗'}")
        time.sleep(0.3)
    
    n = len(pairs)
    results["source_accuracy"] = round(src_correct / n * 100, 1)
    results["target_accuracy"] = round(tgt_correct / n * 100, 1)
    results["domain_gap_delta"] = round((src_correct - tgt_correct) / n * 100, 1)
    
    out = f"results_{model.replace(\'/\', \'_\')}_{split}.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\\nSource acc: {results[\'source_accuracy\']}%")
    print(f"Target acc: {results[\'target_accuracy\']}%")
    print(f"Domain gap δ: {results[\'domain_gap_delta\']}pp")
    print(f"Results saved to {out}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    import os
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key: raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")
    evaluate(args.model, key, args.split)
'''


# ── Main ──────────────────────────────────────────────────────────────────

def run(push_to_hub: bool = False, repo_id: str = None):
    vp = VERIFIED_DIR / "verified_pairs.json"
    if not vp.exists():
        logger.error("verified_pairs.json not found. Run stage3 first.")
        sys.exit(1)

    with open(vp) as f:
        pairs = json.load(f)

    logger.info(f"Loaded {len(pairs)} verified pairs")

    # Strip verbose internal metadata for release (keep essential fields only)
    def clean_pair(p: dict) -> dict:
        return {
            "pair_id":          p["pair_id"],
            "mapping":          p["mapping"],
            "source_domain":    p["source_domain"],
            "target_domain":    p["target_domain"],
            "source": {
                "question":      p["source"]["question"],
                "answer":        p["source"]["answer"],
                "structure_type":p["source"].get("structure_type", "unknown"),
                "domain":        p["source"]["domain"],
                "source_bench":  p["source"].get("source_bench", ""),
            },
            "target": {
                "question":      p["target"]["question"],
                "answer":        p["target"]["answer"],
                "structure_type":p["target"].get("structure_type", "unknown"),
                "domain":        p["target"]["domain"],
            },
            "verification_status": p.get("verification_status", ""),
            "verification_scores": {
                "avg": p.get("verification_scores", {}).get("avg", {}),
            },
        }

    cleaned = [clean_pair(p) for p in pairs]

    # Train/test split (80/70 or 80/20 if fewer pairs)
    import random
    random.seed(42)
    random.shuffle(cleaned)
    n_train = int(len(cleaned) * 0.53)   # ~80 of 150
    train_pairs = cleaned[:n_train]
    test_pairs  = cleaned[n_train:]

    # Save
    with open(HF_DIR / "train.json", "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(HF_DIR / "test.json", "w") as f:
        json.dump(test_pairs, f, indent=2)

    # Dataset card
    (HF_DIR / "README.md").write_text(README_TEMPLATE)

    # Eval script
    (HF_DIR / "evaluation" / "eval_script.py").write_text(EVAL_SCRIPT)

    # Dataset metadata
    from collections import Counter
    mapping_counts = Counter(p["mapping"] for p in cleaned)
    meta = {
        "n_total_pairs":  len(cleaned),
        "n_train_pairs":  len(train_pairs),
        "n_test_pairs":   len(test_pairs),
        "mapping_counts": dict(mapping_counts),
        "license":        "CC-BY-4.0",
        "version":        "1.0.0",
        "splits": {
            "train": "train.json",
            "test":  "test.json",
        },
    }
    with open(HF_DIR / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Dataset prepared at {HF_DIR}")
    logger.info(f"  Train: {len(train_pairs)} pairs")
    logger.info(f"  Test:  {len(test_pairs)} pairs")

    # Optional: push to HuggingFace Hub
    if push_to_hub:
        if not repo_id:
            logger.error("--repo-id required for push-to-hub")
            sys.exit(1)
        try:
            from datasets import Dataset, DatasetDict
            import os
            hf_token = os.environ.get("HF_TOKEN", "")
            if not hf_token:
                raise ValueError("HF_TOKEN env var not set")

            train_ds = Dataset.from_list(train_pairs)
            test_ds  = Dataset.from_list(test_pairs)
            ds_dict  = DatasetDict({"train": train_ds, "test": test_ds})
            ds_dict.push_to_hub(repo_id, token=hf_token)
            logger.info(f"Dataset pushed to HuggingFace Hub: {repo_id}")
        except ImportError:
            logger.error("Install `datasets` package to push to HuggingFace Hub")
        except Exception as e:
            logger.error(f"HuggingFace push failed: {e}")

    return HF_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: Prepare dataset for HuggingFace")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="HuggingFace repo ID, e.g. your-org/isosci-150")
    args = parser.parse_args()
    run(push_to_hub=args.push_to_hub, repo_id=args.repo_id)
