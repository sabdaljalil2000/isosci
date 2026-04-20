#!/usr/bin/env python3
"""
create_scibench.py
===================
Downloads SciBench from xw27/scibench on HuggingFace and saves it
as scibench.json in the format expected by the IsoSci pipeline.

Usage:
    python create_scibench.py
    python create_scibench.py --output /path/to/isosci/data/raw/scibench.json

Requirements:
    pip install datasets
"""

import argparse
import json
from pathlib import Path


# Map source field values → pipeline domain
PHYSICS_SOURCES   = {"fund", "analytic", "class_mech", "modern", "stat_therm",
                     "circuit", "electmag", "physics"}
CHEMISTRY_SOURCES = {"atkins", "chemmc", "matter", "quan", "thermo", "chemistry"}
SKIP_SOURCES      = {"calculus", "diff", "math"}


def download_scibench(output_path: str = "scibench.json"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        return

    print("Downloading xw27/scibench (default config)...")
    try:
        ds = load_dataset("xw27/scibench", "default", split="train")
    except Exception:
        try:
            ds = load_dataset("xw27/scibench", split="train")
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return

    print(f"Total rows: {len(ds)}")
    print(f"Fields: {ds.column_names}")

    sources = set(str(item.get("source", "")).strip() for item in ds)
    print(f"Unique sources found: {sorted(sources)}\n")

    result = []
    skipped = 0

    for item in ds:
        source = str(item.get("source", "")).strip().lower()

        if source in SKIP_SOURCES:
            skipped += 1
            continue

        if source in PHYSICS_SOURCES or "phys" in source:
            domain = "physics"
        elif source in CHEMISTRY_SOURCES or "chem" in source:
            domain = "chemistry"
        else:
            domain = "physics"  # safe default for unlabelled STEM

        problem = (item.get("problem_text")
                   or item.get("problem")
                   or item.get("question")
                   or "").strip()

        answer = str(item.get("answer_number")
                     or item.get("answer")
                     or item.get("solution")
                     or "").strip()

        unit = str(item.get("unit") or item.get("answer_unit") or "").strip()

        if not problem or not answer:
            skipped += 1
            continue

        result.append({
            "problem": problem,
            "answer":  answer,
            "unit":    unit,
            "subject": domain,
            "source":  source,
        })

    print(f"Kept:    {len(result)} questions")
    print(f"Skipped: {skipped}")

    by_domain = {}
    for item in result:
        d = item["subject"]
        by_domain[d] = by_domain.get(d, 0) + 1
    print("By domain:")
    for d, n in sorted(by_domain.items()):
        print(f"  {d}: {n}")

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    size = Path(output_path).stat().st_size / 1024
    print(f"Done. File size: {size:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="scibench.json",
        help="Output path (default: scibench.json in current directory)"
    )
    args = parser.parse_args()
    download_scibench(args.output)
