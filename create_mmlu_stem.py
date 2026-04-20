#!/usr/bin/env python3
"""
create_mmlu_stem.py
====================
Downloads MMLU STEM subjects from HuggingFace and saves them
as mmlu_stem.json in the format expected by the IsoSci pipeline.

Usage:
    python create_mmlu_stem.py
    python create_mmlu_stem.py --output /path/to/isosci/data/raw/mmlu_stem.json

Requirements:
    pip install datasets huggingface_hub
"""

import argparse
import json
from pathlib import Path

STEM_SUBJECTS = [
    "high_school_physics",
    "college_physics",
    "conceptual_physics",
    "astronomy",
    "high_school_chemistry",
    "college_chemistry",
    "high_school_biology",
    "college_biology",
    "anatomy",
    "medical_genetics",
    "high_school_earth_science",
    "earth_sciences",  # fallback name used in some versions
]

def download_mmlu_stem(output_path: str = "mmlu_stem.json"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        return

    result = {}
    total = 0

    for subject in STEM_SUBJECTS:
        print(f"Downloading {subject}...", end=" ", flush=True)
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            rows = []
            for item in ds:
                choices = item["choices"]          # list of 4 strings
                answer_idx = item["answer"]        # 0-3
                answer_letter = "ABCD"[answer_idx]
                rows.append([
                    item["question"],
                    choices[0],   # A
                    choices[1],   # B
                    choices[2],   # C
                    choices[3],   # D
                    answer_letter
                ])
            result[subject] = rows
            total += len(rows)
            print(f"{len(rows)} questions")
        except Exception as e:
            print(f"SKIPPED ({e})")
            continue

    print(f"\nTotal: {total} questions across {len(result)} subjects")
    print(f"Saving to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done. File size: {Path(output_path).stat().st_size / 1024:.0f} KB")
    print("\nSubject breakdown:")
    for subject, rows in result.items():
        print(f"  {subject}: {len(rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", 
        default="mmlu_stem.json",
        help="Output path (default: mmlu_stem.json in current directory)"
    )
    args = parser.parse_args()
    download_mmlu_stem(args.output)
