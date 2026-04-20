"""
IsoSci-150 Pipeline Configuration
===================================
Central config for all pipeline stages. Set your OPENROUTER_API_KEY
in the environment or in a .env file before running.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PAIRS_DIR       = DATA_DIR / "pairs"
VERIFIED_DIR    = DATA_DIR / "verified"
RESULTS_DIR     = DATA_DIR / "results"
ANALYSIS_DIR    = ROOT / "analysis"
OUTPUTS_DIR     = ROOT / "outputs"
LOGS_DIR        = ROOT / "logs"

for d in [RAW_DIR, PAIRS_DIR, VERIFIED_DIR, RESULTS_DIR, ANALYSIS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS  = {
    "HTTP-Referer": "https://isosci-benchmark.github.io",
    "X-Title":      "IsoSci-150 Benchmark",
}

# Generation model (cheap + capable for pair construction)
GENERATION_MODEL = "anthropic/claude-sonnet-4-5"

# ── Model pairs to evaluate ────────────────────────────────────────────────
# Each tuple: (reasoning_model, standard_model, label)
MODEL_PAIRS = [
    ("openai/o3-mini",                            "openai/gpt-4o-mini-2024-07-18",         "o3-mini / GPT-4o-mini"),
    # ("anthropic/claude-opus-4-5",                 "anthropic/claude-sonnet-4-5",            "opus/ sonnet"),
    # ("nvidia/llama-3.1-nemotron-70b-instruct",    "meta-llama/llama-3.1-70b-instruct",     "NVIDIA-Reasoning-Llama-70B / Llama-70B"),
#deepseek/deepseek-r1-distill-llama-70b
    ("microsoft/phi-4-reasoning-plus",                              "microsoft/phi-4",            "phi4-reasoning/ phi4"),
    # ("deepseek/deepseek-r1-0528",                 "deepseek/deepseek-v3",                  "R1 / DeepSeek-V3"),
    # ("qwen/qwq-32b",                              "qwen/qwen-2.5-72b-instruct",            "QwQ-32B / Qwen2.5-72B"),

    # ("deepseek/deepseek-r1-distill-qwen-7b",      "qwen/qwen-2.5-7b-instruct",             "R1-Qwen-7B / Qwen2.5-7B"),
    # google/gemini-2.5-progoogle/gemini-2.0-flash-001
]

# Flat list of all models
ALL_MODELS = [m for pair in MODEL_PAIRS for m in pair[:2]]

# ── Domains and cross-domain mappings ─────────────────────────────────────
DOMAINS = ["physics", "chemistry", "biology", "earth_science"]

DOMAIN_PROFILES = {
    "physics":      {"K": 0.2, "L": 0.3, "C": 0.5, "label": "Physics",      "computation_heavy": True},
    "chemistry":    {"K": 0.4, "L": 0.3, "C": 0.3, "label": "Chemistry",    "computation_heavy": False},
    "biology":      {"K": 0.6, "L": 0.3, "C": 0.1, "label": "Biology",      "computation_heavy": False},
    "earth_science":{"K": 0.6, "L": 0.2, "C": 0.2, "label": "Earth Science","computation_heavy": False},
}

# 6 bidirectional cross-domain mappings, ~25 pairs each
DOMAIN_MAPPINGS = [
    ("physics",   "chemistry"),
    ("physics",   "biology"),
    ("physics",   "earth_science"),
    ("chemistry", "biology"),
    ("chemistry", "earth_science"),
    ("biology",   "earth_science"),
]
PAIRS_PER_MAPPING = 25   # 6 × 25 = 150 total

# ── Problem structure types (for filtering seeds) ─────────────────────────
STRUCTURE_TYPES = [
    "formula_recall_and_substitute",   # recall law/formula, plug in values, compute
    "unit_conversion_chain",           # multi-step unit reasoning
    "conservation_law_application",    # apply a conservation principle
    "proportional_reasoning",          # ratio / scaling problems
    "two_step_causal_chain",           # A causes B causes C (qualitative)
]

# ── Evaluation settings ───────────────────────────────────────────────────
EVAL_TEMPERATURE     = 0.0
EVAL_MAX_TOKENS      = 2048
EVAL_RETRIES         = 3
EVAL_RETRY_DELAY     = 5      # seconds
EVAL_CONCURRENT      = 4      # parallel API calls per model
NUMERIC_TOLERANCE    = 0.02   # ±2% for free-response numerical matching

ZERO_SHOT_COT_PROMPT = (
    "Think step by step. Show your reasoning clearly. "
    "Provide your final answer at the end in the format: "
    "**Final Answer:** <your answer>"
)

# ── Dataset sizes ─────────────────────────────────────────────────────────
TARGET_PAIRS         = 150
PILOT_PAIRS          = 30     # subset for inter-annotator study
CANDIDATES_PER_SEED  = 3      # LLM-generated candidates before expert filter
EXPECTED_REJECTION   = 0.30   # ~30% rejection rate → need ~215 seeds

# ── Existing benchmark files (place CSVs here after download) ─────────────
GPQA_FILE      = RAW_DIR / "gpqa_diamond.csv"
SCIBENCH_FILE  = RAW_DIR / "scibench.json"
MMLU_STEM_FILE = RAW_DIR / "mmlu_stem.json"

# ── Statistical settings ──────────────────────────────────────────────────
CONFIDENCE_LEVEL      = 0.95
MIN_PAIRS_PER_MAPPING = 20    # flag a warning if below this
BOOTSTRAP_SAMPLES     = 1000

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
