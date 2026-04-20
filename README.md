# IsoSci-150 Research Pipeline

> **Full A-to-Z pipeline** for the IsoSci-150 NeurIPS D&B submission:
> dataset construction → model evaluation → analysis → figures → HuggingFace release.

---

## Quick Start

```bash
# 1. Clone and set up
cd isosci
python3 -m venv isosci_env
source isosci_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Set your OpenRouter API key
export OPENROUTER_API_KEY=your_key_here

# 3. Run the full pipeline
python run_pipeline.py

# 4. Or test end-to-end with tiny data first (recommended)
python run_pipeline.py --test-mode
```

---

## Project Structure

```
isosci/
├── run_pipeline.py              ← Master runner (start here)
├── config.py                   ← All settings in one place
├── requirements.txt
│
├── src/
│   ├── api_client.py           ← OpenRouter client + cost tracker
│   ├── stage1_seed_collection.py   ← Collect/generate seed problems
│   ├── stage2_pair_generation.py   ← LLM-generate isomorphic partners
│   ├── stage3_verification.py      ← 3-judge auto-verification
│   ├── stage4_evaluation.py        ← 12-model benchmark evaluation
│   ├── stage5_analysis.py          ← Domain-asymmetric gains + decoupling
│   ├── stage6_figures.py           ← Paper figures + LaTeX tables
│   └── stage7_dataset_release.py   ← HuggingFace packaging
│
├── data/
│   ├── raw/                    ← Seed problems + benchmark files
│   ├── pairs/                  ← Candidate pairs (before verification)
│   ├── verified/               ← IsoSci-150 final dataset
│   └── results/                ← Model evaluation results
│
├── analysis/                   ← JSON analysis outputs
├── outputs/                    ← Figures, LaTeX, HuggingFace package
└── logs/                       ← Per-stage logs
```

---

## Pipeline Stages

| Stage | Name | Wall Time | API Cost |
|-------|------|-----------|----------|
| 1 | Seed collection | ~30 min | ~$20 |
| 2 | Pair generation | ~3–4 hr | ~$150 |
| 3 | Verification | ~2–3 hr | ~$80 |
| 4 | Model evaluation | ~8–12 hr | ~$600–1,500 |
| 5 | Analysis | ~2 min | $0 |
| 6 | Figures & tables | ~2 min | $0 |
| 7 | Dataset release | ~1 min | $0 |
| **Total** | | **~14–20 hr** | **~$850–1,750** |

---

## Running Specific Stages

```bash
# Run only stages 1 and 2
python run_pipeline.py --stages 1 2

# Start from stage 4 (after manually verifying pairs)
python run_pipeline.py --from-stage 4

# Resume after interruption (skips stages with existing outputs)
python run_pipeline.py --resume

# Check pipeline status
python run_pipeline.py --status
```

---

## Stage Details

### Stage 1: Seed Collection
Loads problems from GPQA Diamond, SciBench, and MMLU-STEM if present in `data/raw/`,
then fills gaps with LLM-generated problems.

**To use real benchmarks**, download and place in `data/raw/`:
- `gpqa_diamond.csv` — from [idavidrein/gpqa](https://huggingface.co/datasets/idavidrein/gpqa)
- `scibench.json` — from [mandyyyyii/scibench](https://huggingface.co/datasets/mandyyyyii/scibench)
- `mmlu_stem.json` — from [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (STEM subjects)

Without these files, stage 1 falls back to fully synthetic generation.

### Stage 2: Pair Generation
For each (source, target) domain mapping, selects seeds and generates
`CANDIDATES_PER_SEED=3` isomorphic partners using the LLM. Saves per-mapping
JSONs for easy resumption.

### Stage 3: Verification
Three LLM judges score each candidate pair on 4 criteria (1–5 each):
- Logical equivalence
- Domain independence
- Difficulty parity
- Self-containment

Pairs scoring ≥3.5 on all criteria are accepted. Target: 150 pairs.

**Human annotation mode** (recommended for publication):
```bash
# Generate annotation CSV
python src/stage3_verification.py --mode both --pilot 30

# → Distribute data/verified/annotation_sheet.csv to PhD annotators
# → Annotators fill in scores and ACCEPT/REJECT columns

# Load completed annotations
python src/stage3_verification.py --mode load-human --human-csv completed_annotations.csv
```

### Stage 4: Model Evaluation
Evaluates all 12 models. Resume-safe — saves partial results every 10 items.

```bash
# Evaluate only specific model
python src/stage4_evaluation.py --model deepseek/deepseek-r1

# Evaluate only on IsoSci (skip external benchmarks if files unavailable)
python src/stage4_evaluation.py --benchmark isosci

# Quick test with 5 items per benchmark
python src/stage4_evaluation.py --limit 5
```

### Stage 5: Analysis
Pure Python/NumPy — no API calls. Computes:
- Domain-asymmetric gains (∆acc per domain) + Spearman ρ
- IsoSci-150 decoupling (% of gains that are knowledge-dependent)
- Full accuracy tables + bootstrap CIs
- McNemar's tests

### Stage 6: Figures
Generates publication-quality PDFs + PNGs and LaTeX table source.
Outputs go to `outputs/`.

### Stage 7: Dataset Release
Packages the verified pairs for HuggingFace Hub with a dataset card,
train/test split (80/70), and a standalone evaluation script.

```bash
# Package only
python src/stage7_dataset_release.py

# Package and push to HuggingFace
export HF_TOKEN=your_hf_token
python src/stage7_dataset_release.py --push-to-hub --repo-id your-org/isosci-150
```

---

## Configuration

All parameters are in `config.py`. Key settings:

```python
TARGET_PAIRS = 150          # total pairs in dataset
PAIRS_PER_MAPPING = 25      # pairs per domain mapping (6 mappings)
CANDIDATES_PER_SEED = 3     # LLM-generated candidates per seed
AUTO_ACCEPT_THRESHOLD = 3.5 # min avg judge score for auto-acceptance
EVAL_TEMPERATURE = 0.0      # deterministic evaluation
NUMERIC_TOLERANCE = 0.02    # ±2% for numerical answer matching
```

To modify the model list, edit `MODEL_PAIRS` in `config.py`.

---

## Cost Management

The pipeline tracks API costs automatically. Check at any point:

```python
from src.api_client import cost_tracker
print(cost_tracker.summary())
```


---

## Outputs for Paper

After running all stages, the following are ready for inclusion in the paper:

| File | Used for |
|------|----------|
| `outputs/figure_1_domain_asymmetry.pdf` | Figure 1 |
| `outputs/figure_2_isosci_decoupling.pdf` | Figure 2 |
| `outputs/figure_3_heatmap.pdf` | Figure 3 |
| `outputs/table_1_domain_asymmetry.tex` | Table 1 |
| `outputs/table_2_isosci_decoupling.tex` | Table 2 |
| `analysis/summary_for_paper.json` | All headline numbers |
| `outputs/isosci150/` | HuggingFace release |

---


```
