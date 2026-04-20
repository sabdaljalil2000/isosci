#!/usr/bin/env python3
"""
generate_annotation_sheet.py
==============================
Samples 50 pairs from IsoSci-150 and generates:
  1. annotation_sheet.csv       — one row per pair, annotators fill in scores
  2. annotation_booklet.html    — readable version for annotators (easier than CSV)
  3. annotation_instructions.md — guidelines to send alongside

Usage:
    python generate_annotation_sheet.py \
        --pairs data/verified/verified_pairs.json \
        --n 50 \
        --seed 42 \
        --output-dir annotation/

The HTML booklet is the recommended format to send annotators —
it's easier to read than a spreadsheet and works in any browser.
"""

import argparse
import csv
import json
import random
from pathlib import Path


# ── Scoring rubric ────────────────────────────────────────────────────────

CRITERIA = {
    "logical_equivalence": {
        "label": "Logical Equivalence",
        "question": "Do both problems require the same sequence of reasoning steps?",
        "rubric": {
            1: "Completely different procedure — one has extra steps, different operations, or different logic structure",
            2: "Mostly different — superficial similarity but the reasoning paths diverge significantly",
            3: "Partially equivalent — similar procedure but one problem has meaningfully more/fewer steps or different operation types",
            4: "Mostly equivalent — same structure with minor differences in complexity or step count",
            5: "Perfectly equivalent — identical number and type of reasoning steps, same operations (recall → substitute → compute, etc.)",
        },
    },
    "domain_independence": {
        "label": "Domain Independence",
        "question": "Does the knowledge needed for one problem give zero advantage on the other?",
        "rubric": {
            1: "Heavily overlapping — solving one essentially teaches you the other (same formula in different notation, closely related laws)",
            2: "Significant overlap — partial knowledge transfer would help substantially",
            3: "Partial overlap — some conceptual similarity, but different formulas/facts required",
            4: "Mostly independent — different domain knowledge, minor conceptual adjacency",
            5: "Completely independent — different formulas, constants, and domain facts; knowing one gives zero advantage on the other",
        },
    },
    "difficulty_parity": {
        "label": "Difficulty Parity",
        "question": "Are both problems equally challenging for a strong undergraduate student?",
        "rubric": {
            1: "Very unequal — one is trivially easy, the other is much harder (>2 difficulty levels apart)",
            2: "Moderately unequal — noticeable difference in difficulty",
            3: "Slightly unequal — one is marginally harder due to formula complexity or numeric conditioning",
            4: "Nearly equal — comparable difficulty with minor differences",
            5: "Perfectly equal — a student with equivalent domain knowledge would find them equally challenging",
        },
    },
    "self_containment": {
        "label": "Self-Containment",
        "question": "Can each problem be solved using only the information given plus standard undergraduate knowledge?",
        "rubric": {
            1: "Neither problem is self-contained — both require external information not provided",
            2: "One problem is missing key information",
            3: "Borderline — one problem assumes non-standard knowledge that might not be universally known",
            4: "Mostly self-contained — minor assumption about what 'standard' means",
            5: "Fully self-contained — all needed values, constants, and context are provided in the problem text",
        },
    },
}

FAILURE_MODES = [
    "formula_overlap",          # same formula in different notation (e.g., ideal gas law ≈ van der Waals)
    "difficulty_mismatch",      # one problem requires significantly more steps
    "ambiguous_answer",         # correct answer is unclear or multiple valid answers exist
    "missing_information",      # problem cannot be solved with given data
    "trivial_target",           # target problem is too simple / doesn't require recall
    "not_undergraduate_level",  # problem is below/above stated difficulty
    "structural_inequivalence", # procedure is not actually the same
    "other",
]


# ── HTML template ─────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IsoSci-150 Annotation Booklet</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 2rem;
         color: #222; line-height: 1.6; background: #fafafa; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
  h2 {{ font-size: 1.1rem; color: #555; margin-top: 0; }}
  .pair {{ background: white; border: 1px solid #ddd; border-radius: 8px;
           padding: 1.5rem; margin-bottom: 2.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }}
  .pair-header {{ display: flex; justify-content: space-between; align-items: baseline;
                  border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem; }}
  .pair-num {{ font-size: 1.2rem; font-weight: bold; color: #333; }}
  .mapping {{ font-size: 0.85rem; background: #e8f0fe; color: #1a56db;
              padding: 2px 8px; border-radius: 12px; }}
  .problem-block {{ margin-bottom: 1rem; }}
  .problem-label {{ font-size: 0.75rem; font-weight: bold; text-transform: uppercase;
                    letter-spacing: 0.05em; color: #666; margin-bottom: 0.25rem; }}
  .problem-box {{ background: #f8f9fa; border-left: 3px solid #6c757d;
                  padding: 0.75rem 1rem; border-radius: 0 4px 4px 0; font-size: 0.95rem; }}
  .source-box {{ border-left-color: #2563eb; }}
  .target-box {{ border-left-color: #059669; }}
  .answer {{ font-size: 0.85rem; color: #555; margin-top: 0.35rem; }}
  .structure-tag {{ font-size: 0.8rem; background: #fef3c7; color: #92400e;
                    padding: 2px 8px; border-radius: 4px; display: inline-block; margin-bottom: 1rem; }}
  .score-table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }}
  .score-table th {{ background: #f3f4f6; text-align: left; padding: 0.5rem;
                     border: 1px solid #e5e7eb; }}
  .score-table td {{ padding: 0.5rem; border: 1px solid #e5e7eb; }}
  .score-input {{ width: 60px; text-align: center; border: 1px solid #d1d5db;
                  border-radius: 4px; padding: 4px; }}
  .verdict {{ margin-top: 1rem; }}
  .verdict label {{ font-weight: bold; margin-right: 1.5rem; }}
  .verdict input {{ margin-right: 0.3rem; }}
  .notes-input {{ width: 100%; border: 1px solid #d1d5db; border-radius: 4px;
                  padding: 0.5rem; margin-top: 0.5rem; font-size: 0.9rem;
                  font-family: inherit; resize: vertical; min-height: 60px; }}
  .failure-modes {{ margin-top: 0.75rem; font-size: 0.85rem; }}
  .failure-modes label {{ margin-right: 1rem; }}
  .toc {{ background: white; border: 1px solid #ddd; border-radius: 8px;
          padding: 1rem 1.5rem; margin-bottom: 2rem; }}
  .toc a {{ color: #2563eb; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  .tier-badge {{ font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: bold; }}
  .tier-accepted   {{ background: #dcfce7; color: #166534; }}
  .tier-borderline {{ background: #fef9c3; color: #854d0e; }}
  .tier-rejected   {{ background: #fee2e2; color: #991b1b; }}
  @media print {{
    .pair {{ page-break-inside: avoid; }}
    body {{ background: white; }}
  }}
</style>
</head>
<body>

<h1>IsoSci-150 Human Annotation Booklet</h1>
<p><strong>Annotator:</strong> ______________________ &nbsp;&nbsp;
   <strong>Date:</strong> ______________________</p>
<p>Please score each pair on the four criteria below using the 1–5 scale.
See the attached instructions for rubric details. For each pair, also record
an overall ACCEPT / REJECT verdict and check any applicable failure modes.</p>

<div class="toc">
<strong>Quick navigation:</strong><br>
{toc}
</div>

{pairs_html}

</body>
</html>"""

PAIR_TEMPLATE = """<div class="pair" id="pair-{idx}">
  <div class="pair-header">
    <span class="pair-num">Pair {idx} of {total}</span>
    <span class="mapping">{mapping}</span>
    <span class="tier-badge tier-{tier}">{tier_label}</span>
  </div>
  <div class="structure-tag">Structure: {structure_type} &amp;nbsp;|&amp;nbsp; LLM avg score: {llm_score}/5</div>

  <div class="problem-block">
    <div class="problem-label">Source problem ({source_domain})</div>
    <div class="problem-box source-box">{source_question}</div>
    <div class="answer">&#x2713; Answer: <em>{source_answer}</em></div>
  </div>

  <div class="problem-block">
    <div class="problem-label">Target problem ({target_domain})</div>
    <div class="problem-box target-box">{target_question}</div>
    <div class="answer">&#x2713; Answer: <em>{target_answer}</em></div>
  </div>

  <table class="score-table">
    <tr>
      <th>Criterion</th>
      <th>Question</th>
      <th>Score (1–5)</th>
    </tr>
    <tr>
      <td><strong>Logical Equivalence</strong></td>
      <td>Same reasoning steps and operations?</td>
      <td><input class="score-input" type="number" min="1" max="5"
          name="p{idx}_logical_equivalence" placeholder="1–5"></td>
    </tr>
    <tr>
      <td><strong>Domain Independence</strong></td>
      <td>Knowledge is non-overlapping?</td>
      <td><input class="score-input" type="number" min="1" max="5"
          name="p{idx}_domain_independence" placeholder="1–5"></td>
    </tr>
    <tr>
      <td><strong>Difficulty Parity</strong></td>
      <td>Equally challenging for undergrad?</td>
      <td><input class="score-input" type="number" min="1" max="5"
          name="p{idx}_difficulty_parity" placeholder="1–5"></td>
    </tr>
    <tr>
      <td><strong>Self-Containment</strong></td>
      <td>All info needed is provided?</td>
      <td><input class="score-input" type="number" min="1" max="5"
          name="p{idx}_self_containment" placeholder="1–5"></td>
    </tr>
  </table>

  <div class="verdict">
    <strong>Overall verdict:</strong><br>
    <label><input type="radio" name="p{idx}_verdict" value="ACCEPT"> ACCEPT
      (all scores ≥ 4)</label>
    <label><input type="radio" name="p{idx}_verdict" value="MARGINAL"> MARGINAL
      (any score = 3)</label>
    <label><input type="radio" name="p{idx}_verdict" value="REJECT"> REJECT
      (any score ≤ 2)</label>
  </div>

  <div class="failure-modes">
    <strong>Failure modes (check all that apply, only if MARGINAL or REJECT):</strong><br>
    <label><input type="checkbox" name="p{idx}_fm_formula_overlap"> Formula overlap</label>
    <label><input type="checkbox" name="p{idx}_fm_difficulty_mismatch"> Difficulty mismatch</label>
    <label><input type="checkbox" name="p{idx}_fm_ambiguous_answer"> Ambiguous answer</label>
    <label><input type="checkbox" name="p{idx}_fm_missing_information"> Missing information</label>
    <label><input type="checkbox" name="p{idx}_fm_trivial_target"> Trivial target</label>
    <label><input type="checkbox" name="p{idx}_fm_structural_inequivalence"> Structural inequivalence</label>
    <label><input type="checkbox" name="p{idx}_fm_other"> Other</label>
  </div>

  <div>
    <strong>Notes (optional — explain any failure modes or borderline scores):</strong>
    <textarea class="notes-input" name="p{idx}_notes"
              placeholder="Explain any scores of 3 or below, or note anything unusual..."></textarea>
  </div>
</div>"""


# ── Instructions markdown ─────────────────────────────────────────────────

INSTRUCTIONS = """# IsoSci-150 Annotation Instructions

Thank you for helping validate the IsoSci-150 benchmark.
Please read these instructions carefully before starting.

---

## What you are evaluating

Each item is an **isomorphic problem pair**: two science problems from different
domains that are claimed to share identical logical structure but require different
domain knowledge. Your job is to judge whether this claim holds.

**Example of a valid pair:**
- Source (Physics): "A 2.0 mol sample of ideal gas at 300 K occupies 49.2 L.
  What is the pressure?" (requires PV=nRT)
- Target (Chemistry): "A weak acid solution has [HA]=0.10 M and Ka=1.8×10⁻⁵.
  What is the pH?" (requires pH = -log√(Ka·C))

Both require: recall formula → substitute values → compute. But different formulas.
This is a **valid pair**.

---

## Scoring criteria (1–5 scale)

### 1. Logical Equivalence
Are the reasoning procedures identical?

| Score | Meaning |
|-------|---------|
| 5 | Identical steps: same number, same types (recall→substitute→compute) |
| 4 | Nearly identical, minor differences in complexity |
| 3 | Similar structure but one problem has extra/different steps |
| 2 | Mostly different procedures |
| 1 | Completely different reasoning required |

### 2. Domain Independence
Does knowing one problem's solution help you solve the other?

| Score | Meaning |
|-------|---------|
| 5 | Completely independent — different formulas, facts, constants |
| 4 | Mostly independent — minor conceptual adjacency |
| 3 | Some overlap — partial transfer would help |
| 2 | Significant overlap |
| 1 | Nearly identical knowledge required |

**Watch for:** same formula in different notation (e.g., ideal gas law ≈ van der Waals
at low pressure), closely related conservation laws across domains.

### 3. Difficulty Parity
Would a strong undergraduate find them equally challenging?

| Score | Meaning |
|-------|---------|
| 5 | Equal difficulty |
| 4 | Minor difference |
| 3 | One noticeably harder (more complex formula, worse numeric conditioning) |
| 2 | Moderately unequal |
| 1 | Very unequal (one trivial, one hard) |

### 4. Self-Containment
Can each problem be solved with only the given information + standard undergrad knowledge?

| Score | Meaning |
|-------|---------|
| 5 | Fully self-contained |
| 4 | Mostly self-contained |
| 3 | Minor assumption required |
| 2 | One problem missing key information |
| 1 | Neither problem is self-contained |

---

## Overall verdict

- **ACCEPT**: All four scores are ≥ 4. The pair is a valid isomorphic pair.
- **MARGINAL**: Any score is exactly 3. The pair has a minor flaw but could be included
  with caveats.
- **REJECT**: Any score is ≤ 2. The pair fails the isomorphism criterion.

---

## Failure modes

If you select MARGINAL or REJECT, please check the applicable failure modes:

- **Formula overlap**: The two formulas are too similar (e.g., related conservation laws)
- **Difficulty mismatch**: One problem is significantly harder
- **Ambiguous answer**: The correct answer is unclear or there are multiple valid answers
- **Missing information**: A problem cannot be solved with the given data
- **Trivial target**: The target problem is too simple — doesn't genuinely require recall
- **Structural inequivalence**: The solution procedure is not actually the same
- **Other**: Anything else — please explain in the notes field

---

## Practical notes

- Aim for ~5–8 minutes per pair. If you are unsure, write a note and move on.
- You do not need domain expertise in both fields — judge the **structure**, not whether
  you could solve the problem yourself.
- Score independently; do not look at the other annotator's scores until you are done.
- For the answers provided: assume they are correct. You are evaluating the pair
  structure, not checking the math.

---

## Submitting your annotations

When complete, please return:
1. The filled-in CSV file (easier for analysis), **or**
2. A screenshot/printout of the HTML booklet with scores filled in

Thank you!
"""


# ── CSV generation ────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "pair_idx", "pair_id", "mapping", "source_domain", "target_domain",
    "structure_type",
    "source_question", "source_answer",
    "target_question", "target_answer",
    # Annotator fills these:
    "logical_equivalence",    # 1-5
    "domain_independence",    # 1-5
    "difficulty_parity",      # 1-5
    "self_containment",       # 1-5
    "verdict",                # ACCEPT / MARGINAL / REJECT
    "failure_modes",          # comma-separated from the list above
    "notes",
]


# ── Main ──────────────────────────────────────────────────────────────────

def get_overall_score(pair: dict):
    """Extract the overall average LLM judge score for a pair."""
    avg = pair.get("verification_scores", {}).get("avg", {})
    if avg.get("overall") is not None:
        return avg["overall"]
    # Compute from individual criteria if overall not stored
    criteria = ["logical_equivalence", "domain_independence",
                "difficulty_parity", "self_containment"]
    vals = [avg.get(c) for c in criteria if avg.get(c) is not None]
    return sum(vals) / len(vals) if vals else None


def assign_tier(pair: dict) -> str:
    """Assign a pair to accepted / borderline / rejected tier."""
    score = get_overall_score(pair)
    status = pair.get("verification_status", "")
    if score is None:
        # Fall back to status field
        if "accepted" in status:  return "accepted"
        if "rejected" in status:  return "rejected"
        return "unknown"
    if score >= 3.5:   return "accepted"
    if score >= 3.0:   return "borderline"
    return "rejected"


def sample_pairs(pairs, n, seed, tier_split=(20, 15, 15)):
    """
    Sample n pairs stratified by verification tier AND domain mapping.

    tier_split = (n_accepted, n_borderline, n_rejected)
    Default 20/15/15 = 50 total:
      - 20 accepted:   measures LLM precision (false positive rate)
      - 15 borderline: measures threshold calibration
      - 15 rejected:   measures LLM recall (false negative rate)

    This design lets you compute the full confusion matrix.
    """
    from collections import defaultdict

    n_accepted, n_borderline, n_rejected = tier_split
    assert n_accepted + n_borderline + n_rejected == n,         f"tier_split must sum to n ({n})"

    # Separate into tiers
    tiers = {"accepted": [], "borderline": [], "rejected": []}
    for p in pairs:
        t = assign_tier(p)
        if t in tiers:
            tiers[t].append(p)

    print(f"\nTier distribution in source file:")
    for tier, ps in tiers.items():
        print(f"  {tier}: {len(ps)} pairs")

    rng = random.Random(seed)
    sampled = []

    for tier, target_n in [("accepted", n_accepted),
                            ("borderline", n_borderline),
                            ("rejected", n_rejected)]:
        tier_pairs = tiers[tier]
        if len(tier_pairs) == 0:
            print(f"  WARNING: No {tier} pairs found — skipping this tier")
            continue

        # Within tier, stratify by domain mapping
        by_mapping = defaultdict(list)
        for p in tier_pairs:
            by_mapping[p["mapping"]].append(p)

        n_mappings = len(by_mapping)
        per_mapping = target_n // n_mappings
        remainder   = target_n - per_mapping * n_mappings

        tier_sampled = []
        for i, (mapping, mapping_pairs) in enumerate(sorted(by_mapping.items())):
            take = per_mapping + (1 if i < remainder else 0)
            take = min(take, len(mapping_pairs))
            tier_sampled.extend(rng.sample(mapping_pairs, take))

        # Tag each pair with its tier for annotator context
        for p in tier_sampled:
            p["_annotation_tier"] = tier
            p["_llm_overall_score"] = round(get_overall_score(p) or 0, 2)

        sampled.extend(tier_sampled)
        print(f"  Sampled {len(tier_sampled)}/{target_n} from {tier} tier")

    rng.shuffle(sampled)
    return sampled[:n]


def generate_csv(pairs: list[dict], output_path: Path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for i, pair in enumerate(pairs, 1):
            src = pair["source"]
            tgt = pair["target"]
            writer.writerow({
                "pair_idx":          i,
                "pair_id":           pair["pair_id"],
                "mapping":           pair["mapping"],
                "source_domain":     pair["source_domain"],
                "target_domain":     pair["target_domain"],
                "structure_type":    src.get("structure_type", "unknown"),
                "source_question":   src["question"],
                "source_answer":     src["answer"],
                "target_question":   tgt["question"],
                "target_answer":     tgt["answer"],
                # Leave blank for annotator
                "logical_equivalence": "",
                "domain_independence": "",
                "difficulty_parity":   "",
                "self_containment":    "",
                "verdict":             "",
                "failure_modes":       "",
                "notes":               "",
            })
    print(f"  CSV saved: {output_path}")


def generate_html(pairs: list[dict], output_path: Path):
    toc_items = []
    pair_blocks = []
    total = len(pairs)

    for i, pair in enumerate(pairs, 1):
        src = pair["source"]
        tgt = pair["target"]
        mapping_label = pair["mapping"].replace("_", " ").replace(" to ", " → ")

        toc_items.append(
            f'<a href="#pair-{i}">Pair {i}</a> '
            f'<span style="color:#888;font-size:0.85em">({mapping_label})</span>'
        )

        tier = pair.get("_annotation_tier", "unknown")
        tier_labels = {"accepted": "LLM ACCEPTED", "borderline": "LLM BORDERLINE",
                       "rejected": "LLM REJECTED", "unknown": "UNKNOWN"}
        block = PAIR_TEMPLATE.format(
            idx=i,
            total=total,
            mapping=mapping_label,
            structure_type=src.get("structure_type", "unknown").replace("_", " "),
            source_domain=pair["source_domain"].replace("_", " ").title(),
            target_domain=pair["target_domain"].replace("_", " ").title(),
            source_question=src["question"].replace("<", "&lt;").replace(">", "&gt;"),
            source_answer=src["answer"].replace("<", "&lt;").replace(">", "&gt;"),
            target_question=tgt["question"].replace("<", "&lt;").replace(">", "&gt;"),
            target_answer=tgt["answer"].replace("<", "&lt;").replace(">", "&gt;"),
            tier=tier,
            tier_label=tier_labels.get(tier, tier.upper()),
            llm_score=pair.get("_llm_overall_score", "N/A"),
        )
        pair_blocks.append(block)

    html = HTML_TEMPLATE.format(
        toc=" &nbsp;|&nbsp; ".join(toc_items),
        pairs_html="\n".join(pair_blocks),
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML booklet saved: {output_path}")


def generate_instructions(output_path: Path):
    output_path.write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"  Instructions saved: {output_path}")


def generate_analysis_template(output_path: Path):
    """Generate a Python script to compute agreement stats from filled CSVs."""
    script = '''#!/usr/bin/env python3
"""
compute_agreement.py
=====================
Computes inter-annotator agreement from two filled annotation CSVs.
Reports: % valid, Cohen\'s kappa, failure mode breakdown.

Usage:
    python compute_agreement.py annotator1.csv annotator2.csv
"""
import csv, sys, math
from collections import Counter

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def kappa(r1, r2):
    """Cohen\'s kappa for binary ACCEPT vs non-ACCEPT."""
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
    print(f"\\nAnnotator 1 — ACCEPT: {pct_accept1:.1f}%  REJECT/MARGINAL: {100-pct_accept1:.1f}%")
    print(f"Annotator 2 — ACCEPT: {pct_accept2:.1f}%  REJECT/MARGINAL: {100-pct_accept2:.1f}%")
    print(f"\\nCohen\'s kappa (ACCEPT vs non-ACCEPT): {k:.3f}")
    print(f"Exact verdict agreement: {exact_agree:.1f}%")
    print(f"\\nMean criterion scores:")
    for c in criteria:
        scores1 = [float(r[c]) for r in a1 if r[c]]
        scores2 = [float(r[c]) for r in a2 if r[c]]
        if scores1 and scores2:
            print(f"  {c:<25} A1={sum(scores1)/len(scores1):.2f}  A2={sum(scores2)/len(scores2):.2f}")
    if fm_counts:
        print(f"\\nFailure modes (both annotators combined):")
        for fm, count in fm_counts.most_common():
            print(f"  {fm:<30} {count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    output_path.write_text(script, encoding="utf-8")
    print(f"  Agreement script saved: {output_path}")


def run(pairs_path, n, seed, output_dir, tier_split=(20, 15, 15)):
    pairs_path  = Path(pairs_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pairs_path) as f:
        all_pairs = json.load(f)
    print(f"Loaded {len(all_pairs)} pairs from {pairs_path.name}.")

    sampled = sample_pairs(all_pairs, n, seed, tier_split=tier_split)
    print(f"\nSampled {len(sampled)} pairs total (seed={seed}).")

    # Mapping breakdown
    from collections import Counter
    mapping_counts = Counter(p["mapping"] for p in sampled)
    print("By mapping:")
    for m, c in sorted(mapping_counts.items()):
        print(f"  {m}: {c}")

    # Save sampled pairs JSON (for reference)
    sampled_path = output_dir / "sampled_pairs.json"
    with open(sampled_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"  Sampled pairs JSON: {sampled_path}")

    # Generate outputs
    generate_csv(sampled, output_dir / "annotation_sheet_annotator1.csv")
    generate_csv(sampled, output_dir / "annotation_sheet_annotator2.csv")
    generate_html(sampled, output_dir / "annotation_booklet.html")
    generate_instructions(output_dir / "annotation_instructions.md")
    generate_analysis_template(output_dir / "compute_agreement.py")

    print(f"\nDone. Send annotators:")
    print(f"  1. annotation_booklet.html  (recommended — readable in browser)")
    print(f"  2. annotation_sheet_annotatorN.csv  (for filling in)")
    print(f"  3. annotation_instructions.md")
    print(f"\nAfter both annotators return CSVs:")
    print(f"  python compute_agreement.py annotator1.csv annotator2.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IsoSci annotation materials")
    parser.add_argument("--pairs",
        default="data/pairs/candidate_pairs.json",
        help="Path to candidate_pairs.json (has all tiers) or verified_pairs.json")
    parser.add_argument("--n",          type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output-dir", default="annotation")
    parser.add_argument("--tier-split", nargs=3, type=int, default=[20, 15, 15],
        metavar=("N_ACCEPT", "N_BORDER", "N_REJECT"),
        help="How many pairs per tier (default: 20 accepted, 15 borderline, 15 rejected)")
    args = parser.parse_args()
    run(args.pairs, args.n, args.seed, args.output_dir,
        tier_split=tuple(args.tier_split))