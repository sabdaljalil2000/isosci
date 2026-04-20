#!/usr/bin/env python3
"""
extract_qualitative_examples.py
================================
Finds the best real examples from evaluation results to illustrate
the paper's three findings. Outputs LaTeX-formatted examples.

Four example types sought:
  Ex1: Reasoning-on correct on source, wrong on target (K-dependent gain)
  Ex2: Both configs wrong on target, right on source (pure knowledge bottleneck)
  Ex3: Reasoning-on correct, reasoning-off wrong on SAME problem (reasoning helps)
  Ex4: Both correct on source, both wrong on target (sharpest knowledge gap)

Usage:
    python extract_qualitative_examples.py \
        --results data/results/all_results.json \
        --pairs   data/verified/verified_pairs.json \
        --output  outputs/qualitative_examples.tex
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# ── Model display names ───────────────────────────────────────────────────
DISPLAY = {
    "openai/o3-mini":                            "o3-mini (R)",
    "openai/gpt-4o-mini-2024-07-18":            "GPT-4o-mini (S)",
    "qwen/qwen3-32b:nitro [thinking=ON]":        "Qwen3-32B thinking=ON (R)",
    "qwen/qwen3-32b:nitro [thinking=OFF]":       "Qwen3-32B thinking=OFF (S)",
    "google/gemini-2.0-flash-001 [thinking=ON]": "Gemini Flash thinking=ON (R)",
    "google/gemini-2.0-flash-001 [thinking=OFF]":"Gemini Flash thinking=OFF (S)",
}

MODEL_PAIRS_KEYS = [
    ("openai/o3-mini",
     "openai/gpt-4o-mini-2024-07-18"),
    ("qwen/qwen3-32b:nitro [thinking=ON]",
     "qwen/qwen3-32b:nitro [thinking=OFF]"),
    ("google/gemini-2.0-flash-001 [thinking=ON]",
     "google/gemini-2.0-flash-001 [thinking=OFF]"),
]


def model_key(r: dict) -> str:
    flag = r.get("reasoning_flag")
    if flag is True:  return r["model"] + " [thinking=ON]"
    if flag is False: return r["model"] + " [thinking=OFF]"
    return r["model"]


def truncate(text: str, max_chars: int = 400) -> str:
    """Truncate response to a readable length, keeping the end."""
    if not text:
        return "[empty]"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Keep the final answer part
    return "...\n" + text[-max_chars:]


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&",  "\\&"),
        ("%",  "\\%"),
        ("$",  "\\$"),
        ("#",  "\\#"),
        ("_",  "\\_"),
        ("{",  "\\{"),
        ("}",  "\\}"),
        ("~",  "\\textasciitilde{}"),
        ("^",  "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_response_for_display(response: str, max_chars: int = 350) -> str:
    """Clean and truncate a model response for display in paper."""
    if not response:
        return "[no response]"
    # Remove excessive whitespace
    response = re.sub(r'\n{3,}', '\n\n', response.strip())
    # Truncate keeping the end (where the final answer is)
    if len(response) > max_chars:
        response = "[...] " + response[-max_chars:]
    return response


# ── Index builder ─────────────────────────────────────────────────────────

def build_index(results: list, pairs: list) -> dict:
    """
    Build index: pair_id -> {model_key -> {source: result, target: result}}
    """
    # Index results
    result_idx = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r.get("benchmark") != "isosci":
            continue
        mk = model_key(r)
        pid = r.get("pair_id")
        role = r.get("role")
        if pid and role:
            result_idx[pid][mk][role] = r

    # Build pair metadata index
    pair_meta = {p["pair_id"]: p for p in pairs}

    return result_idx, pair_meta


# ── Example finders ───────────────────────────────────────────────────────

def find_example_1(result_idx, pair_meta, r_key, s_key):
    """
    Ex1: Reasoning correct on source AND wrong on target.
         Standard wrong on source OR wrong on target.
         (Classic K-dependent gain)
    """
    candidates = []
    for pid, models in result_idx.items():
        r = models.get(r_key, {})
        s = models.get(s_key, {})
        r_src = r.get("source", {})
        r_tgt = r.get("target", {})
        s_src = s.get("source", {})
        s_tgt = s.get("target", {})
        if not all([r_src, r_tgt, s_src, s_tgt]):
            continue
        # Reasoning correct on source, wrong on target
        if (r_src.get("correct") and not r_tgt.get("correct") and
                not s_src.get("correct")):
            score = 1
            # Prefer pairs with clear responses
            if r_src.get("response") and r_tgt.get("response"):
                score += 1
            candidates.append((score, pid, r_src, r_tgt, s_src, s_tgt))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0] if candidates else None


def find_example_2(result_idx, pair_meta, r_key, s_key):
    """
    Ex2: Both correct on source, BOTH wrong on target.
         (Pure knowledge bottleneck — neither model has the target domain knowledge)
    """
    candidates = []
    for pid, models in result_idx.items():
        r = models.get(r_key, {})
        s = models.get(s_key, {})
        r_src = r.get("source", {})
        r_tgt = r.get("target", {})
        s_src = s.get("source", {})
        s_tgt = s.get("target", {})
        if not all([r_src, r_tgt, s_src, s_tgt]):
            continue
        if (r_src.get("correct") and s_src.get("correct") and
                not r_tgt.get("correct") and not s_tgt.get("correct")):
            candidates.append((1, pid, r_src, r_tgt, s_src, s_tgt))
    return candidates[0] if candidates else None


def find_example_3(result_idx, pair_meta, r_key, s_key):
    """
    Ex3: Reasoning correct, standard wrong on the SAME problem (source).
         (Shows reasoning mode CAN help — but only on source domain)
    """
    candidates = []
    for pid, models in result_idx.items():
        r = models.get(r_key, {})
        s = models.get(s_key, {})
        r_src = r.get("source", {})
        s_src = s.get("source", {})
        if not all([r_src, s_src]):
            continue
        if r_src.get("correct") and not s_src.get("correct"):
            if r_src.get("response"):
                candidates.append((1, pid, r_src, s_src))
    return candidates[0] if candidates else None


def find_example_4(result_idx, pair_meta, r_key, s_key):
    """
    Ex4: Structure-invariant gain — reasoning correct on BOTH source and target.
         (Shows structure-invariant gains do exist, just rarely: 4.8%)
    """
    candidates = []
    for pid, models in result_idx.items():
        r = models.get(r_key, {})
        s = models.get(s_key, {})
        r_src = r.get("source", {})
        r_tgt = r.get("target", {})
        s_src = s.get("source", {})
        s_tgt = s.get("target", {})
        if not all([r_src, r_tgt, s_src, s_tgt]):
            continue
        # Reasoning improves on BOTH (structure-invariant)
        if (r_src.get("correct") and r_tgt.get("correct") and
                not s_src.get("correct") and not s_tgt.get("correct")):
            candidates.append((1, pid, r_src, r_tgt, s_src, s_tgt))
    return candidates[0] if candidates else None


# ── LaTeX formatter ────────────────────────────────────────────────────────

def format_example_box(
    example_num: int,
    title: str,
    finding: str,
    pair_meta: dict,
    pair_id: str,
    items: list,   # list of (label, result_dict, show_response)
    highlight_color: str = "blue",
) -> str:
    """Format a single qualitative example as a LaTeX tcolorbox."""
    meta = pair_meta.get(pair_id, {})
    src_domain = meta.get("source_domain", "").replace("_", " ").title()
    tgt_domain = meta.get("target_domain", "").replace("_", " ").title()
    mapping = f"{src_domain} $\\to$ {tgt_domain}"
    structure = meta.get("source", {}).get("structure_type", "").replace("_", " ")

    src_q = escape_latex(meta.get("source", {}).get("question", ""))[:300]
    tgt_q = escape_latex(meta.get("target", {}).get("question", ""))[:300]
    src_a = escape_latex(meta.get("source", {}).get("answer", ""))
    tgt_a = escape_latex(meta.get("target", {}).get("answer", ""))

    lines = []
    lines.append(f"\\begin{{tcolorbox}}[")
    lines.append(f"  colback=gray!5, colframe=gray!40,")
    lines.append(f"  title={{\\textbf{{Example {example_num}: {title}}}}},")
    lines.append(f"  fonttitle=\\small\\bfseries, breakable]")
    lines.append(f"")
    lines.append(f"\\textbf{{Finding:}} \\textit{{{escape_latex(finding)}}}")
    lines.append(f"\\\\[4pt]")
    lines.append(f"\\textbf{{Mapping:}} {mapping} \\quad")
    lines.append(f"\\textbf{{Structure:}} \\texttt{{{escape_latex(structure)}}}")
    lines.append(f"")
    lines.append(f"\\vspace{{6pt}}")
    lines.append(f"\\textbf{{Source problem ({src_domain}):}}\\\\")
    lines.append(f"\\begin{{quote}}\\small {src_q}\\\\")
    lines.append(f"\\textit{{Gold answer: {src_a}}}\\end{{quote}}")
    lines.append(f"")
    lines.append(f"\\textbf{{Target problem ({tgt_domain}):}}\\\\")
    lines.append(f"\\begin{{quote}}\\small {tgt_q}\\\\")
    lines.append(f"\\textit{{Gold answer: {tgt_a}}}\\end{{quote}}")
    lines.append(f"")
    lines.append(f"\\vspace{{4pt}}")

    for label, result, show_resp in items:
        correct = result.get("correct", False)
        extracted = escape_latex(result.get("extracted_answer", "") or "")
        tick = "\\textcolor{darkgreen}{\\cmark}" if correct else "\\textcolor{red}{\\xmark}"
        lines.append(f"\\textbf{{{escape_latex(label)}}} {tick} "
                     f"Extracted: \\texttt{{{extracted[:80]}}}")
        if show_resp and result.get("response"):
            resp = clean_response_for_display(result["response"], max_chars=280)
            resp_escaped = escape_latex(resp)
            lines.append(f"\\begin{{quote}}\\scriptsize\\ttfamily {resp_escaped}\\end{{quote}}")
        lines.append(f"\\\\[-2pt]")

    lines.append(f"\\end{{tcolorbox}}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def run(results_path: str, pairs_path: str, output_path: str):
    results = json.load(open(results_path))
    pairs   = json.load(open(pairs_path))

    result_idx, pair_meta = build_index(results, pairs)

    print(f"Loaded {len(results)} results, {len(pairs)} pairs")
    print(f"IsoSci pairs in index: {len(result_idx)}")

    all_examples = []

    for r_key, s_key in MODEL_PAIRS_KEYS:
        r_display = DISPLAY.get(r_key, r_key)
        s_display = DISPLAY.get(s_key, s_key)
        pair_label = f"{r_key.split('/')[1].split(':')[0]} pair"

        print(f"\nSearching {r_display} vs {s_display}...")

        # Example 1: K-dependent gain
        ex1 = find_example_1(result_idx, pair_meta, r_key, s_key)
        if ex1:
            _, pid, r_src, r_tgt, s_src, s_tgt = ex1
            print(f"  Ex1 found: pair {pid[:8]}")
            items = [
                (r_display + " on source", r_src, False),
                (r_display + " on target", r_tgt, True),
                (s_display + " on source", s_src, False),
                (s_display + " on target", s_tgt, False),
            ]
            tex = format_example_box(
                len(all_examples) + 1,
                "Knowledge-Dependent Gain",
                f"{r_display} solves the source problem but fails the isomorphic "
                f"target — same reasoning structure, different domain knowledge.",
                pair_meta, pid, items
            )
            all_examples.append(("ex1", tex))

        # Example 2: Pure knowledge bottleneck
        ex2 = find_example_2(result_idx, pair_meta, r_key, s_key)
        if ex2:
            _, pid, r_src, r_tgt, s_src, s_tgt = ex2
            print(f"  Ex2 found: pair {pid[:8]}")
            items = [
                (r_display + " on source", r_src, False),
                (s_display + " on source", s_src, False),
                (r_display + " on target", r_tgt, True),
                (s_display + " on target", s_tgt, True),
            ]
            tex = format_example_box(
                len(all_examples) + 1,
                "Pure Knowledge Bottleneck",
                "Both models solve the source problem correctly but fail the "
                "isomorphic target — neither possesses the target domain knowledge.",
                pair_meta, pid, items
            )
            all_examples.append(("ex2", tex))

        # Example 3: Reasoning mode helps (on source)
        ex3 = find_example_3(result_idx, pair_meta, r_key, s_key)
        if ex3:
            _, pid, r_src, s_src = ex3
            print(f"  Ex3 found: pair {pid[:8]}")
            items = [
                (r_display, r_src, True),
                (s_display, s_src, True),
            ]
            tex = format_example_box(
                len(all_examples) + 1,
                "Reasoning Mode Helps (Source Domain Only)",
                f"{r_display} solves the source problem correctly while "
                f"{s_display} fails — extended reasoning aids knowledge retrieval "
                f"for the source domain formula.",
                pair_meta, pid, items
            )
            all_examples.append(("ex3", tex))

        # Example 4: Structure-invariant gain (rare — 4.8%)
        ex4 = find_example_4(result_idx, pair_meta, r_key, s_key)
        if ex4:
            _, pid, r_src, r_tgt, s_src, s_tgt = ex4
            print(f"  Ex4 found: pair {pid[:8]}")
            items = [
                (r_display + " on source", r_src, False),
                (r_display + " on target", r_tgt, False),
                (s_display + " on source", s_src, False),
                (s_display + " on target", s_tgt, False),
            ]
            tex = format_example_box(
                len(all_examples) + 1,
                "Structure-Invariant Gain (Rare: 4.8\\% of gains)",
                f"{r_display} solves both source and target correctly while "
                f"{s_display} fails both — one of the few cases where reasoning "
                f"improvement transfers across domain knowledge.",
                pair_meta, pid, items
            )
            all_examples.append(("ex4", tex))

        # Stop after finding one of each type across all pairs
        if len(all_examples) >= 4:
            break

    if not all_examples:
        print("No examples found — check that results contain IsoSci data")
        return

    # Build full LaTeX output
    header = r"""% ── Qualitative Examples (Appendix) ─────────────────────────────────────
% Requires: tcolorbox, xcolor, pifont packages
% \definecolor{darkgreen}{RGB}{0,128,0} in preamble

\section{Qualitative Examples}
\label{app:examples}

The following examples are drawn directly from model responses on \isosci{}
pairs and illustrate the three findings reported in Section~\ref{sec:results}.
Each box shows the source and target problems, the gold answers, and the
extracted model responses. \cmark{} = correct; \xmark{} = incorrect.

"""
    footer = r"""
% ─────────────────────────────────────────────────────────────────────────
"""

    full_tex = header + "\n\\vspace{8pt}\n".join(tex for _, tex in all_examples) + footer

    Path(output_path).write_text(full_tex, encoding="utf-8")
    print(f"\nSaved {len(all_examples)} examples to {output_path}")
    print("\nTo include in paper, add to appendix:")
    print(r"  \input{outputs/qualitative_examples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/all_results.json")
    parser.add_argument("--pairs",   default="data/verified/verified_pairs.json")
    parser.add_argument("--output",  default="outputs/qualitative_examples.tex")
    args = parser.parse_args()
    run(args.results, args.pairs, args.output)
