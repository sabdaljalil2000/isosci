from typing import Optional, Union, Tuple, List, Dict
"""
src/stage6_figures.py
======================
Stage 6 — Generate all paper figures and formatted tables.

Figures:
  Figure 1: Domain-asymmetric gains bar chart (∆acc per domain, per model pair)
  Figure 2: IsoSci decoupling scatter (source ∆ vs target ∆ per model pair)
  Figure 3: Accuracy heatmap (all models × all domains × benchmarks)
  Figure 4: Spearman correlation plot (∆acc vs computation intensity)

Tables (LaTeX):
  Table 1: Domain-stratified accuracy deltas
  Table 2: IsoSci-150 decoupling results
  Table A1: Full accuracy results (appendix)

Output: outputs/figure_{1,2,3,4}.pdf
        outputs/table_{1,2,a1}.tex
        outputs/paper_tables.json   (machine-readable)

Usage:
    python src/stage6_figures.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ANALYSIS_DIR, OUTPUTS_DIR, LOGS_DIR, DOMAINS, MODEL_PAIRS, DOMAIN_PROFILES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "stage6.log"),
    ],
)
logger = logging.getLogger("stage6")

# ── Display name mapping ─────────────────────────────────────────────────
# Maps internal model keys (from analysis JSON) → short paper-friendly names
MODEL_DISPLAY_NAMES = {
    # Traditional pairs
    "openai/o3-mini":                              "o3-mini",
    "openai/gpt-4o-mini-2024-07-18":              "gpt-4o-mini",
    "qwen/qwq-32b":                                "QwQ-32B",
    "qwen/qwen-2.5-72b-instruct":                 "Qwen2.5-72B",
    # Toggle pairs — thinking ON (reasoning)
    "qwen/qwen3-8b [thinking=ON]":                  "Qwen3-8B (think=ON)",
    "qwen/qwen3-4b [thinking=ON]":                  "Qwen3-4B (think=ON)",
    "qwen/qwen3-32b:nitro [thinking=ON]":           "Qwen3-32B (think=ON)",
    "google/gemini-2.5-flash-lite [thinking=ON]":   "Gemini-Flash-Lite (think=ON)",
    "google/gemini-2.0-flash-001 [thinking=ON]":    "Gemini-Flash (think=ON)",
    # Toggle pairs — thinking OFF (standard)
    "qwen/qwen3-8b [thinking=OFF]":                 "Qwen3-8B (think=OFF)",
    "qwen/qwen3-4b [thinking=OFF]":                 "Qwen3-4B (think=OFF)",
    "qwen/qwen3-32b:nitro [thinking=OFF]":          "Qwen3-32B (think=OFF)",
    "google/gemini-2.5-flash-lite [thinking=OFF]":  "Gemini-Flash-Lite (think=OFF)",
    "google/gemini-2.0-flash-001 [thinking=OFF]":   "Gemini-Flash (think=OFF)",
    # Pair labels used in tables
    "o3-mini / GPT-4o-mini":                        "o3-mini / GPT-4o-mini",
    "QwQ-32B / Qwen2.5-72B":                        "QwQ-32B / Qwen2.5-72B",
    "Qwen3-8B thinking-on vs off":                  "Qwen3-8B think on/off",
    "Qwen3-4B thinking-on vs off":                  "Qwen3-4B think on/off",
    "Qwen3-32B thinking-on vs off":                 "Qwen3-32B think on/off",
    "Gemini-Flash-Lite thinking-on vs off":         "Gemini-Flash-Lite think on/off",
    "Gemini-2.5-Flash-001 thinking-on vs off":      "Gemini-Flash think on/off",
    "Gemini-2.5-Flash-Lite thinking-on vs off":     "Gemini-Flash-Lite think on/off",
}

def display_name(key: str) -> str:
    """Return short display name for a model key, falling back to last segment."""
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    # Fallback: strip org prefix and truncate
    name = key.split("/")[-1] if "/" in key else key
    return name[:22]


# ── Style ─────────────────────────────────────────────────────────────────
COLORS = {
    "reasoning": "#4C72B0",
    "standard":  "#DD8452",
    "delta":     "#55A868",
    "physics":   "#4C72B0",
    "chemistry": "#DD8452",
    "biology":   "#55A868",
    "earth_science": "#C44E52",
}
DOMAIN_LABELS = {
    "physics":      "Physics",
    "chemistry":    "Chemistry",
    "biology":      "Biology",
    "earth_science":"Earth Sci.",
}
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     10,
    "axes.titlesize":12,
    "axes.labelsize":11,
    "legend.fontsize":9,
    "figure.dpi":   150,
})


# ── Load analysis results ─────────────────────────────────────────────────

def load_analysis() -> dict:
    summary_path = ANALYSIS_DIR / "summary_for_paper.json"
    if not summary_path.exists():
        logger.error(f"summary_for_paper.json not found. Run stage5 first.")
        sys.exit(1)
    with open(summary_path) as f:
        return json.load(f)


# ── Figure 1: Domain-asymmetric gains ─────────────────────────────────────

def figure1_domain_asymmetric_gains(summary: dict):
    """Bar chart of average ∆acc per domain across model pairs, with CI."""
    avg_deltas = summary.get("table1_average_deltas", {})
    rows = summary.get("table1_domain_asymmetric_gains", [])

    fig, (ax_main, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 1: Domain-Asymmetric Gains of Reasoning Models", fontsize=13, fontweight="bold")

    # Left: grouped bar — one group per domain, one bar per model pair
    domain_list = DOMAINS
    n_pairs = len(rows)
    x = np.arange(len(domain_list))
    width = 0.12
    offsets = np.linspace(-(n_pairs-1)/2, (n_pairs-1)/2, n_pairs) * width

    pair_colors = plt.cm.tab10(np.linspace(0, 0.6, n_pairs))
    for i, (row, offset, color) in enumerate(zip(rows, offsets, pair_colors)):
        deltas = [row.get(d, {}).get("delta_acc", 0) or 0 for d in domain_list]
        label = row["pair"].split("/")[0].strip()
        ax_main.bar(x + offset, deltas, width, label=label, color=color, alpha=0.85, edgecolor="white")

    ax_main.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels([DOMAIN_LABELS[d] for d in domain_list])
    ax_main.set_ylabel("∆ Accuracy (reasoning − standard), pp")
    ax_main.set_title("Per-pair accuracy delta by domain")
    ax_main.legend(fontsize=7, ncol=2)
    ax_main.grid(axis="y", alpha=0.3)

    # Average delta line overlay
    avg_vals = [avg_deltas.get(d, 0) or 0 for d in domain_list]
    ax_main.plot(x, avg_vals, "D-", color="black", linewidth=2,
                 markersize=6, label="Average ∆", zorder=10)
    ax_main.legend(fontsize=7, ncol=2)

    # Right: Spearman plot — ∆acc vs computation intensity
    comp_intensities = [DOMAIN_PROFILES[d]["C"] for d in domain_list]
    ax_scatter.scatter(comp_intensities, avg_vals,
                       s=120, zorder=5,
                       c=[COLORS.get(d, "gray") for d in domain_list])
    for d, ci, delta in zip(domain_list, comp_intensities, avg_vals):
        ax_scatter.annotate(
            DOMAIN_LABELS[d],
            (ci, delta),
            textcoords="offset points", xytext=(6, 3), fontsize=9,
        )

    # Fit line
    if len(comp_intensities) >= 3:
        m, b = np.polyfit(comp_intensities, avg_vals, 1)
        xs = np.linspace(min(comp_intensities)-0.05, max(comp_intensities)+0.05, 50)
        ax_scatter.plot(xs, m*xs+b, "--", color="gray", alpha=0.7)

    spearman = summary.get("table1_spearman", {})
    rho_text = (f"Spearman ρ = {spearman.get('rho', 'N/A')}, "
                f"p = {spearman.get('p_value', 'N/A')}")
    ax_scatter.set_xlabel("Domain computation intensity (C)")
    ax_scatter.set_ylabel("Average ∆ Accuracy (pp)")
    ax_scatter.set_title(f"∆acc vs computation intensity\n{rho_text}")
    ax_scatter.grid(alpha=0.3)
    ax_scatter.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = OUTPUTS_DIR / "figure_1_domain_asymmetry.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 1: {out}")


# ── Figure 2: IsoSci decoupling ────────────────────────────────────────────

def figure2_isosci_decoupling(summary: dict):
    """Scatter of source ∆ vs target ∆ for each model pair, with diagonal reference."""
    rows = summary.get("table2_decoupling", [])
    if not rows:
        logger.warning("No IsoSci decoupling data for Figure 2")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 2: IsoSci-150 Reasoning–Knowledge Decoupling", fontsize=13, fontweight="bold")

    # Left: source ∆ vs target ∆ scatter
    src_deltas = [r["source_delta"] for r in rows]
    tgt_deltas = [r["target_delta"] for r in rows]
    labels     = [r["pair"].split("/")[0].strip() for r in rows]

    ax1.scatter(src_deltas, tgt_deltas, s=100, zorder=5)
    for x, y, lbl in zip(src_deltas, tgt_deltas, labels):
        ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Diagonal: if gains were purely structural, points would fall on y=x
    lo = min(min(src_deltas), min(tgt_deltas)) - 2
    hi = max(max(src_deltas), max(tgt_deltas)) + 2
    ax1.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, label="y=x (pure reasoning)")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("∆ Accuracy on source domain problems (pp)")
    ax1.set_ylabel("∆ Accuracy on target domain problems (pp)")
    ax1.set_title("Source vs target delta\n(below diagonal = knowledge-dependent gains)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: stacked bar of knowledge-dependent vs structure-invariant gains
    pair_labels = [r["pair"].split("/")[0].strip() for r in rows]
    pct_k = [r.get("pct_gains_knowledge_dependent", 0) or 0 for r in rows]
    pct_s = [r.get("pct_gains_structure_invariant", 0) or 0 for r in rows]

    x = np.arange(len(pair_labels))
    ax2.bar(x, pct_k, label="Knowledge-dependent gains", color=COLORS["physics"], alpha=0.85)
    ax2.bar(x, pct_s, bottom=pct_k, label="Structure-invariant gains", color=COLORS["biology"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("% of reasoning model gains")
    ax2.set_title("Attribution of reasoning model gains")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 110)

    # Add avg annotation
    avg_k = summary.get("table2_avg_pct_knowledge")
    if avg_k:
        ax2.axhline(avg_k, color="black", linestyle=":", linewidth=1.5)
        ax2.text(len(pair_labels) - 0.5, avg_k + 2, f"Avg: {avg_k}%", fontsize=8, ha="right")

    plt.tight_layout()
    out = OUTPUTS_DIR / "figure_2_isosci_decoupling.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 2: {out}")


# ── Figure 3: Accuracy heatmap ─────────────────────────────────────────────

def figure3_accuracy_heatmap(summary: dict):
    """Heatmap of accuracy across all models × domains."""
    full_table = summary.get("full_accuracy_table", {})
    if not full_table:
        logger.warning("No full accuracy table for Figure 3")
        return

    # Build matrix: rows=models, cols=domains (overall)
    model_labels = list(full_table.keys())
    domain_cols  = DOMAINS + ["overall"]

    matrix = []
    for model in model_labels:
        row_vals = []
        # Aggregate across all benchmarks
        combined = {}
        for bench in full_table[model].values():
            for domain in DOMAINS:
                acc = bench.get("by_domain", {}).get(domain, {}).get("accuracy")
                if acc is not None:
                    combined[domain] = combined.get(domain, []) + [acc]
            ov = bench.get("overall")
            if ov is not None:
                combined["overall"] = combined.get("overall", []) + [ov]

        for col in domain_cols:
            vals = combined.get(col, [])
            row_vals.append(sum(vals)/len(vals) if vals else float("nan"))
        matrix.append(row_vals)

    matrix_np = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(model_labels) * 0.4 + 1)))
    im = ax.imshow(matrix_np, aspect="auto", cmap="RdYlGn", vmin=20, vmax=90)

    # Labels — use display_name() to get proper short names including thinking suffix
    short_labels = []
    for m in model_labels:
        name = display_name(m)
        # Ensure thinking ON/OFF is visible
        if "[thinking=ON]" in m:
            name = name + " (think=ON)" if "think" not in name.lower() else name
        elif "[thinking=OFF]" in m:
            name = name + " (think=OFF)" if "think" not in name.lower() else name
        short_labels.append(name)
    col_labels = [DOMAIN_LABELS.get(d, d) for d in DOMAINS] + ["Overall"]

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(domain_cols)):
            val = matrix_np[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if 30 < val < 75 else "white")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title(
        "Figure 3: Model Accuracy Heatmap Across Domains"
        "(averaged across IsoSci-150, GPQA Diamond, MMLU-STEM, SciBench)",
        fontweight="bold", fontsize=11)
    plt.tight_layout()
    out = OUTPUTS_DIR / "figure_3_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 3: {out}")


# ── LaTeX table generation ────────────────────────────────────────────────

def latex_table1(summary: dict) -> str:
    """Generate LaTeX for Table 1: Domain-Asymmetric Gains."""
    rows = summary.get("table1_domain_asymmetric_gains", [])
    avg  = summary.get("table1_average_deltas", {})

    col_spec = "l" + "r" * len(DOMAINS) + "r"
    header_domains = " & ".join(f"\\textbf{{{DOMAIN_LABELS[d]}}}" for d in DOMAINS)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Domain-stratified accuracy deltas ($\Delta_{\text{acc}}$, reasoning $-$ standard, in pp). "
        r"95\% CIs in parentheses.}",
        r"\label{tab:domain_asymmetry}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\textbf{{Model Pair}} & {header_domains} & \\textbf{{Overall}} \\\\",
        r"\midrule",
    ]

    for row in rows:
        cells = [row["pair"].replace("_", r"\_")]
        for domain in DOMAINS:
            d = row.get(domain, {})
            delta = d.get("delta_acc")
            ci = d.get("r_ci_95", [None, None])
            if delta is not None:
                cells.append(f"{delta:+.1f}")
            else:
                cells.append("—")
        # Overall (average across domains)
        domain_deltas = [row.get(d, {}).get("delta_acc") for d in DOMAINS]
        valid = [x for x in domain_deltas if x is not None]
        overall = f"{sum(valid)/len(valid):+.1f}" if valid else "—"
        cells.append(overall)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\midrule",
        r"\textit{Average $\Delta$} & " + " & ".join(
            f"{avg.get(d, 0):+.1f}" if avg.get(d) is not None else "—"
            for d in DOMAINS
        ) + " & — \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_table2(summary: dict) -> str:
    """Generate LaTeX for Table 2: IsoSci-150 Decoupling."""
    rows = summary.get("table2_decoupling", [])
    avg_k = summary.get("table2_avg_pct_knowledge", "N/A")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{IsoSci-150 reasoning–knowledge decoupling. "
        r"Source $\Delta$ = gain on computation-heavy (source) problems. "
        r"Target $\Delta$ = gain on knowledge-heavy (target) problems. "
        r"Knowledge-dep. = \% of gains that are domain-dependent.}",
        r"\label{tab:isosci_decoupling}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Model Pair} & \textbf{Source $\Delta$} & \textbf{Target $\Delta$} "
        r"& \textbf{Domain gap} & \textbf{Know.-dep. (\%)} \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['pair']} & "
            f"{row.get('source_delta', '—'):+.1f} & "
            f"{row.get('target_delta', '—'):+.1f} & "
            f"{row.get('domain_gap_delta', '—'):+.1f} & "
            f"{row.get('pct_gains_knowledge_dependent', '—'):.1f} \\\\"
        )

    lines += [
        r"\midrule",
        f"\\textit{{Average}} & — & — & — & {avg_k:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    summary = load_analysis()

    figure1_domain_asymmetric_gains(summary)
    figure2_isosci_decoupling(summary)
    figure3_accuracy_heatmap(summary)

    # LaTeX tables
    t1 = latex_table1(summary)
    t2 = latex_table2(summary)

    (OUTPUTS_DIR / "table_1_domain_asymmetry.tex").write_text(t1)
    (OUTPUTS_DIR / "table_2_isosci_decoupling.tex").write_text(t2)
    logger.info("LaTeX tables saved.")

    # Print tables to console for quick inspection
    print("\n" + "="*60 + "\nTABLE 1 (LaTeX preview):\n" + "="*60)
    print(t1)
    print("\n" + "="*60 + "\nTABLE 2 (LaTeX preview):\n" + "="*60)
    print(t2)


if __name__ == "__main__":
    run()
