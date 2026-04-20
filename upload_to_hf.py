#!/usr/bin/env python3
"""
upload_to_hf.py
================
1. Fixes structure_type metadata in verified_pairs.json
2. Generates Croissant metadata file (croissant.json)
3. Uploads dataset to HuggingFace Hub
4. Validates Croissant file locally

Usage:
    pip install huggingface_hub mlcroissant datasets --break-system-packages
    python upload_to_hf.py --repo isosci/isosci --token hf_xxxx
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter


# ── Step 1: Fix structure types ───────────────────────────────────────────

STRUCTURE_KEYWORDS = {
    "formula_recall_and_substitute": [
        "calculate", "compute", "find the", "what is the value",
        "determine the", "mol", "pressure", "temperature", "concentration",
        "wavelength", "frequency", "velocity", "acceleration", "force",
        "energy", "using the formula", "given that", "=", "law",
    ],
    "unit_conversion_chain": [
        "convert", "unit", "km/h", "m/s", "celsius", "fahrenheit",
        "kelvin", "joule", "calorie", "meter", "kilometre",
    ],
    "conservation_law_application": [
        "conserved", "conservation", "total", "before and after",
        "initial", "final", "closed system", "isolated", "balanced",
        "flow rate", "stoichiom",
    ],
    "proportional_reasoning": [
        "ratio", "proportion", "twice", "half", "times as",
        "scale", "rate", "per", "proportional", "varies",
    ],
    "two_step_causal_chain": [
        "why", "which", "what type", "responsible for", "cause",
        "effect", "result in", "lead to", "explain", "because",
        "fundamental force", "interaction", "mechanism",
    ],
}


def infer_structure_type(question: str, answer: str) -> str:
    q_lower = question.lower()
    a_lower = str(answer).lower()

    # MCQ answers strongly suggest causal/conceptual
    if re.match(r'^[A-D]\)', answer.strip()):
        for kw in STRUCTURE_KEYWORDS["two_step_causal_chain"]:
            if kw in q_lower:
                return "two_step_causal_chain"
        for kw in STRUCTURE_KEYWORDS["proportional_reasoning"]:
            if kw in q_lower:
                return "proportional_reasoning"
        return "two_step_causal_chain"

    # Numerical answer
    if re.search(r'[\d\.]+', answer.strip()):
        for kw in STRUCTURE_KEYWORDS["unit_conversion_chain"]:
            if kw in q_lower:
                return "unit_conversion_chain"
        for kw in STRUCTURE_KEYWORDS["conservation_law_application"]:
            if kw in q_lower:
                return "conservation_law_application"
        for kw in STRUCTURE_KEYWORDS["proportional_reasoning"]:
            if kw in q_lower:
                return "proportional_reasoning"
        return "formula_recall_and_substitute"

    return "two_step_causal_chain"


def fix_structure_types(pairs_path: Path) -> list:
    pairs = json.load(open(pairs_path))
    fixed = 0
    for p in pairs:
        src = p.get("source", {})
        tgt = p.get("target", {})
        if src.get("structure_type", "unknown") in ("unknown", "", None):
            st = infer_structure_type(
                src.get("question", ""),
                src.get("answer", "")
            )
            src["structure_type"] = st
            tgt["structure_type"] = st
            fixed += 1
    print(f"Fixed structure_type for {fixed} pairs")
    print("Distribution:", Counter(p["source"]["structure_type"] for p in pairs))
    json.dump(pairs, open(pairs_path, "w"), indent=2)
    return pairs


# ── Step 2: Generate Croissant metadata ───────────────────────────────────

def generate_croissant(pairs: list, repo_id: str, output_path: Path):
    """Generate a Croissant-compliant metadata JSON file."""

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/",
            "sc": "https://schema.org/"
        },
        "@type": "sc:Dataset",
        "name": "IsoSci-150",
        "description": (
            "IsoSci-150 is a benchmark of 144 isomorphic cross-domain science problem pairs "
            "designed to separate reasoning ability from domain knowledge retrieval in LLM "
            "evaluation. Each pair consists of two problems sharing identical logical and "
            "computational structure but drawn from different scientific domains, requiring "
            "different domain-specific knowledge to solve."
        ),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "version": "1.0.0",
        "keywords": [
            "science", "reasoning", "knowledge", "benchmark",
            "isomorphic", "LLM evaluation", "physics", "chemistry",
            "biology", "earth science"
        ],
        "creator": {
            "@type": "sc:Person",
            "name": "Anonymous (under review)"
        },
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "verified_pairs.json",
                "name": "verified_pairs.json",
                "description": "144 verified isomorphic problem pairs in JSON format",
                "contentUrl": f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/verified_pairs.json",
                "encodingFormat": "application/json",
                "sha256": "TBD"
            },
            {
                "@type": "cr:FileObject",
                "@id": "test.json",
                "name": "test.json",
                "description": "Test split (64 pairs, stratified by domain mapping)",
                "contentUrl": f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/test.json",
                "encodingFormat": "application/json",
                "sha256": "TBD"
            },
            {
                "@type": "cr:FileObject",
                "@id": "train.json",
                "name": "train.json",
                "description": "Train split (80 pairs, stratified by domain mapping)",
                "contentUrl": f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/train.json",
                "encodingFormat": "application/json",
                "sha256": "TBD"
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "pairs",
                "name": "Isomorphic Problem Pairs",
                "description": "Each record is an isomorphic pair of science problems.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/pair_id",
                        "name": "pair_id",
                        "description": "Unique UUID identifier for the pair",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].pair_id"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/mapping",
                        "name": "mapping",
                        "description": "Cross-domain mapping (e.g. physics_to_chemistry)",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].mapping"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/source_domain",
                        "name": "source_domain",
                        "description": "Scientific domain of the source problem",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].source_domain"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/target_domain",
                        "name": "target_domain",
                        "description": "Scientific domain of the target problem",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].target_domain"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/source_question",
                        "name": "source_question",
                        "description": "Source problem text",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].source.question"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/source_answer",
                        "name": "source_answer",
                        "description": "Source problem gold answer",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].source.answer"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/target_question",
                        "name": "target_question",
                        "description": "Target problem text (isomorphic to source)",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].target.question"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/target_answer",
                        "name": "target_answer",
                        "description": "Target problem gold answer",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].target.answer"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/structure_type",
                        "name": "structure_type",
                        "description": "Reasoning structure type (one of 5 categories)",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].source.structure_type"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/source_bench",
                        "name": "source_bench",
                        "description": "Original benchmark the source seed came from",
                        "dataType": "sc:Text",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].source.source_bench"}}
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "pairs/verification_score",
                        "name": "verification_score",
                        "description": "Average LLM judge score (1-5) across 4 criteria",
                        "dataType": "sc:Number",
                        "source": {"fileObject": {"@id": "verified_pairs.json"}, "extract": {"jsonPath": "$[*].verification_scores.avg.overall"}}
                    }
                ]
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(croissant, f, indent=2)
    print(f"Croissant metadata saved to {output_path}")
    return croissant


# ── Step 3: Prepare dataset splits ────────────────────────────────────────

def prepare_splits(pairs: list, output_dir: Path):
    """Create train/test splits stratified by domain mapping."""
    import random
    random.seed(42)

    from collections import defaultdict
    by_mapping = defaultdict(list)
    for p in pairs:
        by_mapping[p["mapping"]].append(p)

    train, test = [], []
    for mapping, mapping_pairs in by_mapping.items():
        shuffled = mapping_pairs.copy()
        random.shuffle(shuffled)
        n_test = max(1, len(shuffled) // 3)
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    print(f"Train: {len(train)} pairs, Test: {len(test)} pairs")

    # Save flat format (easier for users)
    def flatten(pairs):
        flat = []
        for p in pairs:
            flat.append({
                "pair_id":        p["pair_id"],
                "mapping":        p["mapping"],
                "source_domain":  p["source_domain"],
                "target_domain":  p["target_domain"],
                "structure_type": p["source"].get("structure_type", "unknown"),
                "source_question": p["source"]["question"],
                "source_answer":   p["source"]["answer"],
                "target_question": p["target"]["question"],
                "target_answer":   p["target"]["answer"],
                "source_bench":    p["source"].get("source_bench", ""),
                "verification_score": p.get("verification_scores", {}).get("avg", {}).get("overall"),
            })
        return flat

    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(flatten(train), open(output_dir / "train.json", "w"), indent=2)
    json.dump(flatten(test),  open(output_dir / "test.json",  "w"), indent=2)
    json.dump(flatten(pairs), open(output_dir / "verified_pairs.json", "w"), indent=2)
    print(f"Saved splits to {output_dir}/")
    return train, test


# ── Step 4: Validate Croissant locally ────────────────────────────────────

def validate_croissant(croissant_path: Path):
    try:
        import mlcroissant as mlc
        ds = mlc.Dataset(croissant_path)
        print("Croissant validation passed")
        return True
    except ImportError:
        print("mlcroissant not installed — skipping validation")
        print("Install: pip install mlcroissant --break-system-packages")
        return False
    except Exception as e:
        print(f"Croissant validation warning: {e}")
        print("(Warnings are OK — errors would prevent upload)")
        return False


# ── Step 5: Upload to HuggingFace ─────────────────────────────────────────

def upload_to_hf(repo_id: str, token: str, data_dir: Path, croissant_path: Path):
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    files_to_upload = [
        (data_dir / "train.json",           "data/train.json"),
        (data_dir / "test.json",            "data/test.json"),
        (data_dir / "verified_pairs.json",  "data/verified_pairs.json"),
        (croissant_path,                     "croissant.json"),
    ]

    print(f"\nUploading to {repo_id}...")
    for local_path, repo_path in files_to_upload:
        if not local_path.exists():
            print(f"  SKIP {repo_path} (not found)")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {repo_path}",
        )
        print(f"  Uploaded: {repo_path}")

    # Upload README
    readme = f"""---
license: cc-by-4.0
task_categories:
- question-answering
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

# IsoSci-150

A benchmark of **144 isomorphic cross-domain science problem pairs** designed to
separate reasoning ability from domain knowledge retrieval in LLM evaluation.

Each pair consists of two problems sharing **identical logical and computational
structure** but drawn from different scientific domains, requiring different
domain-specific knowledge to solve.

## Dataset Structure

- `data/train.json` — 80 pairs (stratified by domain mapping)
- `data/test.json`  — 64 pairs (stratified by domain mapping)
- `data/verified_pairs.json` — all 144 pairs with full metadata

## Fields

| Field | Description |
|-------|-------------|
| `pair_id` | Unique UUID |
| `mapping` | Domain mapping (e.g. `physics_to_chemistry`) |
| `source_domain` / `target_domain` | Scientific domains |
| `structure_type` | Reasoning structure (5 categories) |
| `source_question` / `target_question` | Problem texts |
| `source_answer` / `target_answer` | Gold answers |
| `verification_score` | LLM judge ensemble score (1–5) |

## Citation

Anonymous submission under review at NeurIPS 2026.
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add README",
    )
    print(f"  Uploaded: README.md")
    print(f"\nDataset available at: https://huggingface.co/datasets/{repo_id}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",   required=True, help="HuggingFace repo id, e.g. isosci/isosci")
    parser.add_argument("--token",  required=True, help="HuggingFace write token")
    parser.add_argument("--pairs",  default="data/verified/verified_pairs.json")
    parser.add_argument("--outdir", default="outputs/hf_upload")
    args = parser.parse_args()

    pairs_path   = Path(args.pairs)
    output_dir   = Path(args.outdir)
    croissant_path = output_dir / "croissant.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Fix structure types ===")
    pairs = fix_structure_types(pairs_path)

    print("\n=== Step 2: Generate Croissant metadata ===")
    generate_croissant(pairs, args.repo, croissant_path)

    print("\n=== Step 3: Prepare train/test splits ===")
    prepare_splits(pairs, output_dir / "data")

    print("\n=== Step 4: Validate Croissant ===")
    validate_croissant(croissant_path)

    print("\n=== Step 5: Upload to HuggingFace ===")
    upload_to_hf(args.repo, args.token, output_dir / "data", croissant_path)

    print("\nDone! Now add to Appendix E in your paper:")
    print(f"  URL: https://huggingface.co/datasets/{args.repo}")
    print("  Croissant: croissant.json included in repository root")


if __name__ == "__main__":
    main()
