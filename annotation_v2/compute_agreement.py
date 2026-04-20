#!/usr/bin/env python3
"""
compute_agreement.py
=====================
Computes inter-annotator agreement from two filled annotation CSVs.
Reports: % valid, Cohen's kappa, failure mode breakdown.

Usage:
    python compute_agreement.py annotator1.csv annotator2.csv
"""
import csv, sys, math
from collections import Counter

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def kappa(r1, r2):
    """Cohen's kappa for binary ACCEPT vs non-ACCEPT."""
    assert len(r1) == len(r2)
    n = len(r1)
    agree = sum(a == b for a, b in zip(r1, r2))
    po = agree / n
    classes = set(r1) | set(r2)
    pe = sum((r1.count(c)/n) * (r2.count(c)/n) for c in classes)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_agreement.py annotator1.csv annotator2.csv")
        sys.exit(1)

    a1 = load(sys.argv[1])
    a2 = load(sys.argv[2])
    assert len(a1) == len(a2), f"Different number of rows: {len(a1)} vs {len(a2)}"
    n = len(a1)

    verdicts1 = [r["verdict"].strip().upper() for r in a1]
    verdicts2 = [r["verdict"].strip().upper() for r in a2]

    # % valid (ACCEPT) per annotator
    pct_accept1 = verdicts1.count("ACCEPT") / n * 100
    pct_accept2 = verdicts2.count("ACCEPT") / n * 100

    # Binary: ACCEPT vs non-ACCEPT for kappa
    bin1 = ["A" if v == "ACCEPT" else "R" for v in verdicts1]
    bin2 = ["A" if v == "ACCEPT" else "R" for v in verdicts2]
    k = kappa(bin1, bin2)

    # Agreement on exact verdict
    exact_agree = sum(a == b for a, b in zip(verdicts1, verdicts2)) / n * 100

    # Failure modes
    all_fms = []
    for row in a1 + a2:
        if row.get("failure_modes"):
            all_fms.extend([fm.strip() for fm in row["failure_modes"].split(",") if fm.strip()])
    fm_counts = Counter(all_fms)

    # Criteria scores
    criteria = ["logical_equivalence", "domain_independence", "difficulty_parity", "self_containment"]
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 60)
    print(f"N pairs annotated: {n}")
    print(f"\nAnnotator 1 — ACCEPT: {pct_accept1:.1f}%  REJECT/MARGINAL: {100-pct_accept1:.1f}%")
    print(f"Annotator 2 — ACCEPT: {pct_accept2:.1f}%  REJECT/MARGINAL: {100-pct_accept2:.1f}%")
    print(f"\nCohen's kappa (ACCEPT vs non-ACCEPT): {k:.3f}")
    print(f"Exact verdict agreement: {exact_agree:.1f}%")
    print(f"\nMean criterion scores:")
    for c in criteria:
        scores1 = [float(r[c]) for r in a1 if r[c]]
        scores2 = [float(r[c]) for r in a2 if r[c]]
        if scores1 and scores2:
            print(f"  {c:<25} A1={sum(scores1)/len(scores1):.2f}  A2={sum(scores2)/len(scores2):.2f}")
    if fm_counts:
        print(f"\nFailure modes (both annotators combined):")
        for fm, count in fm_counts.most_common():
            print(f"  {fm:<30} {count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
