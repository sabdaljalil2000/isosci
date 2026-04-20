#!/usr/bin/env python3
"""
restore_scores.py
==================
Merges verification scores from verified_pairs.json and rejected_pairs.json
back into candidate_pairs.json so stage 3 can skip already-scored pairs.

Run this ONCE before rerunning stage 3:
    python restore_scores.py
"""
import json
from pathlib import Path

PAIRS_DIR   = Path("data/pairs")
VERIFIED_DIR = Path("data/verified")

def main():
    cand_path     = PAIRS_DIR / "candidate_pairs.json"
    verified_path = VERIFIED_DIR / "verified_pairs.json"
    rejected_path = VERIFIED_DIR / "rejected_pairs.json"
    interim_path  = VERIFIED_DIR / "interim_pairs.json"

    candidates = json.load(open(cand_path))
    print(f"Candidates: {len(candidates)}")

    # Build lookup: pair_id -> scored pair from any source
    scored = {}
    for path, label in [(verified_path, "verified"),
                        (rejected_path, "rejected"),
                        (interim_path,  "interim")]:
        if path.exists():
            pairs = json.load(open(path))
            for p in pairs:
                pid = p.get("pair_id")
                if pid and p.get("verification_scores", {}).get("avg"):
                    scored[pid] = p
            print(f"  Loaded {len(pairs)} from {label} ({path.name})")

    print(f"Total scored pairs found: {len(scored)}")

    # Merge scores back into candidates
    updated = 0
    for c in candidates:
        pid = c.get("pair_id")
        if pid in scored:
            c["verification_scores"] = scored[pid]["verification_scores"]
            c["verification_status"] = scored[pid]["verification_status"]
            updated += 1

    print(f"Updated {updated}/{len(candidates)} candidates with existing scores")
    print(f"Still unscored: {len(candidates) - updated}")

    # Save back
    json.dump(candidates, open(cand_path, "w"), indent=2)
    print(f"Saved to {cand_path}")

    if len(candidates) - updated > 0:
        print(f"\n{len(candidates) - updated} pairs still need scoring — stage 3 will score only these.")
    else:
        print("\nAll pairs already scored — stage 3 will just apply the cap, no API calls.")

if __name__ == "__main__":
    main()
