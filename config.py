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

# Generation model (used in stages 1-3, NOT evaluated in stage 4)
GENERATION_MODEL = "anthropic/claude-sonnet-4-5"

# ── Model pairs ────────────────────────────────────────────────────────────
# Format: (reasoning_model, standard_model, label, reasoning_extra, standard_extra)
#
# reasoning_extra / standard_extra: dict of extra API payload fields.
# For toggle-mode models (same model ID, reasoning on vs off):
#   reasoning_extra = {"reasoning": {"enabled": True}}
#   standard_extra  = {"reasoning": {"enabled": False}}
# For traditional separate-model pairs: both extras = {}

MODEL_PAIRS = [
    # ── Traditional pairs (different models) ──────────────────────────────
    (
        "openai/o3-mini",
        "openai/gpt-4o-mini-2024-07-18",
        "o3-mini / GPT-4o-mini",
        {},
        {},
    ),
    # (
    #     "qwen/qwq-32b",
    #     "qwen/qwen-2.5-72b-instruct",
    #     "QwQ-32B / Qwen2.5-72B",
    #     {},
    #     {},
    # ),
    # ── Same-model toggle pairs (cleanest A/B — only reasoning changes) ───
    (
        "qwen/qwen3-32b:nitro",
        "qwen/qwen3-32b:nitro",
        "Qwen3-32B thinking-on vs off",
        {"reasoning": {"enabled": True}},
        {"reasoning": {"enabled": False}},
    ),
    # (
    #     "qwen/qwen3-4b",
    #     "qwen/qwen3-4b",
    #     "Qwen3-4B thinking-on vs off",
    #     {"reasoning": {"enabled": True}},
    #     {"reasoning": {"enabled": False}},
    # ),
    (
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-001",
        "Gemini-2.0-Flash-001 thinking-on vs off",
        {"reasoning": {"enabled": True}},
        {"reasoning": {"enabled": False}},
    ),
]

# ── Derived model lists ────────────────────────────────────────────────────
# ALL_MODELS_WITH_FLAGS: flat list of (model_id, reasoning_flag) for evaluation.
# Toggle pairs appear TWICE — once with flag=True, once with flag=False.
# Traditional pairs appear once each with flag=None.

def _build_model_flags():
    seen = set()
    result = []
    for pair in MODEL_PAIRS:
        r_model = pair[0]
        s_model = pair[1]
        r_extra = pair[3]
        s_extra = pair[4]
        # None for traditional pairs (empty dict), True/False for toggle pairs
        r_flag = r_extra.get("reasoning", {}).get("enabled", None)
        s_flag = s_extra.get("reasoning", {}).get("enabled", None)
        for model_id, flag in [(r_model, r_flag), (s_model, s_flag)]:
            key = (model_id, flag)
            if key not in seen:
                seen.add(key)
                result.append((model_id, flag))
    return result

ALL_MODELS_WITH_FLAGS = _build_model_flags()

# Flat list of unique model IDs (for cost estimation etc.)
ALL_MODELS = list(dict.fromkeys(m for m, _ in ALL_MODELS_WITH_FLAGS))

# ── Domains ────────────────────────────────────────────────────────────────
DOMAINS = ["physics", "chemistry", "biology", "earth_science"]

DOMAIN_PROFILES = {
    "physics":       {"K": 0.2, "L": 0.3, "C": 0.5, "label": "Physics",       "computation_heavy": True},
    "chemistry":     {"K": 0.4, "L": 0.3, "C": 0.3, "label": "Chemistry",     "computation_heavy": False},
    "biology":       {"K": 0.6, "L": 0.3, "C": 0.1, "label": "Biology",       "computation_heavy": False},
    "earth_science": {"K": 0.6, "L": 0.2, "C": 0.2, "label": "Earth Science", "computation_heavy": False},
}

DOMAIN_MAPPINGS = [
    ("physics",   "chemistry"),
    ("physics",   "biology"),
    ("physics",   "earth_science"),
    ("chemistry", "biology"),
    ("chemistry", "earth_science"),
    ("biology",   "earth_science"),
]
PAIRS_PER_MAPPING = 42

# ── Problem structure types ────────────────────────────────────────────────
STRUCTURE_TYPES = [
    "formula_recall_and_substitute",
    "unit_conversion_chain",
    "conservation_law_application",
    "proportional_reasoning",
    "two_step_causal_chain",
]

# ── Evaluation settings ───────────────────────────────────────────────────
EVAL_TEMPERATURE     = 0.0
EVAL_MAX_TOKENS      = 8192
EVAL_RETRIES         = 3
EVAL_RETRY_DELAY     = 5
EVAL_CONCURRENT      = 4
NUMERIC_TOLERANCE    = 0.02

ZERO_SHOT_COT_PROMPT = (
    "Think step by step. Show your reasoning clearly. "
    "Provide your final answer at the end in the format: "
    "**Final Answer:** <your answer>"
)

# ── Dataset sizes ─────────────────────────────────────────────────────────
TARGET_PAIRS         = 150
PILOT_PAIRS          = 30
CANDIDATES_PER_SEED  = 3
EXPECTED_REJECTION   = 0.30
SEEDS_PER_DOMAIN     = 108

# ── Benchmark file paths ──────────────────────────────────────────────────
GPQA_FILE      = RAW_DIR / "gpqa_diamond.csv"
SCIBENCH_FILE  = RAW_DIR / "scibench.json"
MMLU_STEM_FILE = RAW_DIR / "mmlu_stem.json"

# ── Statistical settings ──────────────────────────────────────────────────
CONFIDENCE_LEVEL  = 0.95
BOOTSTRAP_SAMPLES = 1000

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
