"""
src/stage8_qualitative_examples.py
====================================
Stage 8 — Extract and format the most compelling qualitative examples
for paper Figure 4. Finds the clearest cases of:

  (A) Reasoning model SUCCEEDS on physics, FAILS on isomorphic biology pair
      → Shows knowledge bottleneck
  (B) Standard model SUCCEEDS on both (easier physics problem)
      → Contrast case  
  (C) Cross-domain asymmetry: same model, same logical structure, different outcome

Outputs:
  analysis/qualitative_examples.json   — raw examples with full responses
  outputs/figure_4_qualitative.tex     — LaTeX figure with annotated chains
  outputs/figure_4_qualitative.md      — Markdown version for paper draft

Usage:
    python src/stage8_qualitative_examples.py [--n-examples N]
"""

import json
import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR, ANALYSIS_DIR, OUTPUTS_DIR, LOGS_DIR, MODEL_PAIRS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage8.log"),
    ],
)
logger = logging.getLogger("stage8")


# ── Example selection ─────────────────────────────────────────────────────

def find_best_examples(results: list[dict], n_per_type: int = 2) -> dict:
    """
    Find the most illustrative examples for each case type.
    
    Returns dict with keys:
      'reasoning_wins_physics_loses_biology'  — Case A (knowledge bottleneck)
      'both_succeed'                          — Case B (pure reasoning)
      'both_fail'                             — Case C (knowledge + reasoning gap)
      'reasoning_reversal'                    — reasoning model correct at step N but then wrong
    """
    # Index results by (model, pair_id, role)
    by_key = {}
    for r in results:
        if r.get("benchmark") == "isosci" and r.get("pair_id"):
            key = (r["model"], r["pair_id"], r.get("role"))
            by_key[key] = r

    # Get all unique pair_ids
    pair_ids = set(r["pair_id"] for r in results
                   if r.get("benchmark") == "isosci" and r.get("pair_id"))

    examples = {
        "reasoning_wins_source_loses_target": [],
        "both_succeed_reasoning":             [],
        "standard_wins_reasoning_fails":      [],
        "domain_asymmetry_clearest":          [],
    }

    for reasoning_model, standard_model, label in MODEL_PAIRS:
        for pair_id in pair_ids:
            r_src = by_key.get((reasoning_model, pair_id, "source"))
            r_tgt = by_key.get((reasoning_model, pair_id, "target"))
            s_src = by_key.get((standard_model, pair_id, "source"))
            s_tgt = by_key.get((standard_model, pair_id, "target"))

            if not all([r_src, r_tgt, s_src, s_tgt]):
                continue

            # Case A: Reasoning wins on source (physics), loses on target (biology)
            # This is the most compelling case for the paper
            if (r_src.get("correct") and not r_tgt.get("correct") and
                    not s_src.get("correct")):
                entry = _build_example(
                    pair_id, label, "reasoning_wins_source_loses_target",
                    r_src, r_tgt, s_src, s_tgt,
                    caption=(
                        f"Reasoning model ({label.split('/')[0]}) solves the "
                        f"{r_src['domain']} (source) problem but fails the "
                        f"isomorphic {r_tgt['domain']} (target) problem — "
                        f"same logical structure, different knowledge required. "
                        f"Standard model fails both."
                    )
                )
                examples["reasoning_wins_source_loses_target"].append(entry)

            # Case B: Both models solve both problems (reasoning is sufficient)
            elif (r_src.get("correct") and r_tgt.get("correct") and
                  s_src.get("correct") and s_tgt.get("correct")):
                entry = _build_example(
                    pair_id, label, "both_succeed",
                    r_src, r_tgt, s_src, s_tgt,
                    caption="Both models solve both problems — reasoning structure sufficient for both domains."
                )
                examples["both_succeed_reasoning"].append(entry)

            # Case C: Standard wins, reasoning fails (over-reasoning or error)
            elif (s_src.get("correct") and not r_src.get("correct")):
                entry = _build_example(
                    pair_id, label, "standard_wins_reasoning_fails",
                    r_src, r_tgt, s_src, s_tgt,
                    caption=(
                        f"Standard model solves the {s_src['domain']} problem "
                        f"that the reasoning model fails — "
                        f"potential over-reasoning or introduced error."
                    )
                )
                examples["standard_wins_reasoning_fails"].append(entry)

    # Score each example by interestingness and trim to n_per_type
    for key in examples:
        scored = sorted(examples[key], key=lambda x: x.get("interest_score", 0), reverse=True)
        examples[key] = scored[:n_per_type]

    total = sum(len(v) for v in examples.values())
    logger.info(f"Found {total} qualitative examples across {len(examples)} types")
    for k, v in examples.items():
        logger.info(f"  {k}: {len(v)}")
    return examples


def _build_example(pair_id, model_pair_label, example_type,
                   r_src, r_tgt, s_src, s_tgt, caption) -> dict:
    """Build a structured example dict with interest score."""
    # Interest score: prefer shorter reasoning chains (easier to print)
    # and high token count difference between reasoning/standard
    r_tok = r_src.get("completion_tokens", 0) + r_tgt.get("completion_tokens", 0)
    s_tok = s_src.get("completion_tokens", 0) + s_tgt.get("completion_tokens", 0)
    tok_ratio = r_tok / max(s_tok, 1)

    # Prefer pairs where the domain contrast is physics vs biology (most striking)
    domain_score = 1.0
    if r_src.get("domain") == "physics" and r_tgt.get("domain") in ("biology", "earth_science"):
        domain_score = 2.0

    interest_score = domain_score * min(tok_ratio, 5.0)

    return {
        "pair_id":           pair_id,
        "model_pair":        model_pair_label,
        "example_type":      example_type,
        "caption":           caption,
        "interest_score":    interest_score,
        "token_ratio_r_to_s": round(tok_ratio, 2),
        "source": {
            "domain":            r_src.get("domain"),
            "question":          r_src.get("question", ""),
            "gold_answer":       r_src.get("gold_answer", ""),
            "reasoning_response":_trim_response(r_src.get("response", "")),
            "reasoning_correct": r_src.get("correct"),
            "standard_response": _trim_response(s_src.get("response", "")),
            "standard_correct":  s_src.get("correct"),
        },
        "target": {
            "domain":            r_tgt.get("domain"),
            "question":          r_tgt.get("question", ""),
            "gold_answer":       r_tgt.get("gold_answer", ""),
            "reasoning_response":_trim_response(r_tgt.get("response", "")),
            "reasoning_correct": r_tgt.get("correct"),
            "standard_response": _trim_response(s_tgt.get("response", "")),
            "standard_correct":  s_tgt.get("correct"),
        },
    }


def _trim_response(response: str, max_chars: int = 1200) -> str:
    """Trim long model responses to a printable length, keeping final answer."""
    if not response:
        return ""
    if len(response) <= max_chars:
        return response
    # Keep first 600 chars + last 400 chars (to preserve final answer)
    head = response[:600]
    tail = response[-400:]
    return head + "\n\n[...reasoning truncated...]\n\n" + tail


# ── LaTeX output ──────────────────────────────────────────────────────────

def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
        ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
        ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def format_latex_figure(examples: dict) -> str:
    """
    Generate LaTeX for Figure 4 — qualitative examples.
    Uses tcolorbox for nicely styled response boxes.
    """
    preamble = r"""% Requires: \usepackage{tcolorbox, xcolor}
% Colors: reasoning model = blue tint, standard = gray tint, correct = green, wrong = red

\definecolor{reasoningbg}{RGB}{235,242,250}
\definecolor{standardbg}{RGB}{245,245,245}
\definecolor{correctborder}{RGB}{40,167,69}
\definecolor{wrongborder}{RGB}{220,53,69}

\newcommand{\correctmark}{\textcolor{correctborder}{\textbf{✓}}}
\newcommand{\wrongmark}{\textcolor{wrongborder}{\textbf{✗}}}
"""

    sections = [preamble, r"\begin{figure*}[t]",
                r"\centering",
                r"\caption{\textbf{Figure 4: Qualitative examples from IsoSci-150.} "
                r"Each panel shows a source–target isomorphic pair with reasoning "
                r"and standard model responses. Correct answers marked \correctmark, "
                r"incorrect \wrongmark.}",
                r"\label{fig:qualitative}"]

    for example_type, ex_list in examples.items():
        for ex in ex_list:
            if not ex:
                continue

            label = example_type.replace("_", " ").title()
            src = ex["source"]
            tgt = ex["target"]

            sections.append(f"\n% --- {label} ---")
            sections.append(r"\begin{tcolorbox}[title={" + _latex_escape(ex["caption"]) +
                             r"}, colback=white, fonttitle=\small\bfseries]")

            for role_label, role_data in [("Source problem", src), ("Target problem", tgt)]:
                r_ok = role_data.get("reasoning_correct")
                s_ok = role_data.get("standard_correct")
                r_mark = r"\correctmark" if r_ok else r"\wrongmark"
                s_mark = r"\correctmark" if s_ok else r"\wrongmark"

                sections.append(
                    r"\textbf{" + _latex_escape(role_label) +
                    r" (" + _latex_escape(role_data.get("domain", "")) + r")} "
                )
                sections.append(
                    r"\textit{" + _latex_escape(role_data.get("question", "")[:200]) + r"}"
                )
                sections.append(
                    r"\textbf{Gold answer:} \texttt{" +
                    _latex_escape(str(role_data.get("gold_answer", ""))) + r"}"
                )
                sections.append(
                    r"Reasoning model " + r_mark + r" | Standard model " + s_mark
                )
            sections.append(r"\end{tcolorbox}")

    sections += [r"\end{figure*}"]
    return "\n".join(sections)


# ── Markdown output ───────────────────────────────────────────────────────

def format_markdown(examples: dict) -> str:
    """Generate readable Markdown for paper draft / supplement."""
    lines = ["# Figure 4: Qualitative Examples from IsoSci-150\n"]

    for example_type, ex_list in examples.items():
        if not ex_list:
            continue
        type_label = example_type.replace("_", " ").title()
        lines.append(f"\n## {type_label}\n")

        for i, ex in enumerate(ex_list):
            lines.append(f"### Example {i+1} — {ex.get('model_pair', '')}\n")
            lines.append(f"_{ex.get('caption', '')}_\n")

            for role, role_data in [("**Source problem**", ex["source"]),
                                    ("**Target problem**", ex["target"])]:
                domain = role_data.get("domain", "").replace("_", " ").title()
                lines.append(f"\n{role} ({domain})\n")
                lines.append(f"**Question:** {role_data.get('question', '')}\n")
                lines.append(f"**Gold answer:** `{role_data.get('gold_answer', '')}`\n")

                r_ok = role_data.get("reasoning_correct")
                s_ok = role_data.get("standard_correct")
                lines.append(f"- Reasoning model: {'✓ correct' if r_ok else '✗ incorrect'}")
                lines.append(f"- Standard model: {'✓ correct' if s_ok else '✗ incorrect'}\n")

                r_resp = role_data.get("reasoning_response", "")
                if r_resp:
                    lines.append(f"<details><summary>Reasoning model response (click to expand)</summary>\n")
                    lines.append(f"```\n{r_resp[:800]}\n```\n</details>\n")

                s_resp = role_data.get("standard_response", "")
                if s_resp:
                    lines.append(f"<details><summary>Standard model response</summary>\n")
                    lines.append(f"```\n{s_resp[:400]}\n```\n</details>\n")

            lines.append("\n---\n")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def run(n_per_type: int = 2):
    rp = RESULTS_DIR / "all_results.json"
    if not rp.exists():
        logger.error(f"all_results.json not found. Run stage4 first.")
        sys.exit(1)
    with open(rp) as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} result records")

    examples = find_best_examples(results, n_per_type=n_per_type)

    # Save raw examples
    ex_path = ANALYSIS_DIR / "qualitative_examples.json"
    with open(ex_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Saved qualitative examples: {ex_path}")

    # LaTeX figure
    latex = format_latex_figure(examples)
    latex_path = OUTPUTS_DIR / "figure_4_qualitative.tex"
    latex_path.write_text(latex)
    logger.info(f"Saved LaTeX figure: {latex_path}")

    # Markdown
    md = format_markdown(examples)
    md_path = OUTPUTS_DIR / "figure_4_qualitative.md"
    md_path.write_text(md)
    logger.info(f"Saved Markdown figure: {md_path}")

    return examples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=2,
                        help="Examples per type (default: 2)")
    args = parser.parse_args()
    run(n_per_type=args.n_examples)
