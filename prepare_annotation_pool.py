#!/usr/bin/env python3
"""
prepare_annotation_pool.py
===========================
Merges verified_pairs.json, rejected_pairs.json, and interim_pairs.json
into a single pool with tier labels, ready for stratified annotation sampling.

Run this BEFORE generate_annotation_sheet.py:

    python prepare_annotation_pool.py
    python generate_annotation_sheet.py \
        --pairs data/verified/all_pairs_with_tiers.json \
        --n 50 --seed 42 --tier-split 20 15 15 \
        --output-dir annotation_v2/
"""

import json
from collections import Counter
from pathlib import Path

VERIFIED_DIR = Path("data/verified")

def load_if_exists(path):
    if path.exists():
        data = json.load(open(path))
        print(f"  Loaded {len(data)} pairs from {path.name}")
        return data
    print(f"  Not found: {path.name}")
    return []

def assign_tier(pair):
    """Assign tier from verification status and scores."""
    status = pair.get("verification_status", "")
    avg    = pair.get("verification_scores", {}).get("avg", {})
    score  = avg.get("overall")

    # Status-based (most reliable)
    if "accepted" in status:
        return "accepted"
    if "rejected" in status or "failed" in status:
        if score is not None and score >= 3.0:
            return "borderline"
        return "rejected"

    # Score-based fallback
    if score is not None:
        if score >= 3.5:  return "accepted"
        if score >= 3.0:  return "borderline"
        return "rejected"

    return "rejected"  # conservative default

def main():
    print("Loading pairs from data/verified/...")

    verified = load_if_exists(VERIFIED_DIR / "verified_pairs.json")
    rejected = load_if_exists(VERIFIED_DIR / "rejected_pairs.json")
    interim  = load_if_exists(VERIFIED_DIR / "interim_pairs.json")

    pool = []
    seen_ids = set()

    for p in verified + rejected + interim:
        pid = p.get("pair_id")
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        # Assign tier and score
        p["_annotation_tier"]     = assign_tier(p)
        avg = p.get("verification_scores", {}).get("avg", {})
        score = avg.get("overall")
        p["_llm_overall_score"]   = round(score, 2) if score else None

        pool.append(p)

    tiers = Counter(p["_annotation_tier"] for p in pool)
    print(f"\nTotal pairs in pool: {len(pool)}")
    print("Tier breakdown:")
    for t, n in sorted(tiers.items()):
        print(f"  {t:<12}: {n}")

    output = VERIFIED_DIR / "all_pairs_with_tiers.json"
    json.dump(pool, open(output, "w"), indent=2)
    print(f"\nSaved: {output}")

    # Recommend tier-split based on available pairs
    n_acc  = tiers.get("accepted", 0)
    n_bor  = tiers.get("borderline", 0)
    n_rej  = tiers.get("rejected", 0)

    target_acc = min(20, n_acc)
    target_bor = min(15, n_bor)
    target_rej = min(15, n_rej)
    total = target_acc + target_bor + target_rej

    print(f"\nRecommended command:")
    print(f"  python generate_annotation_sheet.py \\")
    print(f"      --pairs data/verified/all_pairs_with_tiers.json \\")
    print(f"      --n {total} --seed 42 \\")
    print(f"      --tier-split {target_acc} {target_bor} {target_rej} \\")
    print(f"      --output-dir annotation_v2/")

if __name__ == "__main__":
    main()
