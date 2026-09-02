"""
Central configuration for the copula benchmark analysis pipeline.

All paths, dataset definitions, benchmark names, and analysis constants
are defined here to avoid hardcoding across scripts.
"""

from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)

# ── OLLM v2 benchmarks (6 benchmarks, n=4,497 models) ────────────────────────
OLLM_BENCHMARKS = [
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
]

# Short names used in file columns and figures
OLLM_SHORT_NAMES = {
    "IFEval": "IFEval",
    "BBH": "BBH",
    "MATH Lvl 5": "MATH",
    "GPQA": "GPQA",
    "MUSR": "MUSR",
    "MMLU-PRO": "MMLU-PRO",
}

# ── Copula families ───────────────────────────────────────────────────────────
COPULA_FAMILIES_8 = [
    "Gaussian", "Student-t", "Clayton", "Gumbel",
    "Frank", "Joe", "BB1", "BB7",
]

COPULA_FAMILIES_EXPANDED = COPULA_FAMILIES_8 + ["BB6", "BB8", "Tawn"]

# ── Analysis constants ────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_BOOTSTRAP = 1000
ALPHA = 0.05
CONDITIONAL_PERCENTILE = 0.80      # default percentile for ρ_up
DECEPTIVE_THRESHOLD = 0.5          # ρ_up < threshold * ρ_S → deceptive
MIN_SAMPLE_SIZE = 35               # recommended minimum n for copula fits

# ── HuggingFace dataset identifiers ──────────────────────────────────────────
HF_BENCHPRESS = "yzeng58/benchpress-score-matrix"
HF_OLLM = "open-llm-leaderboard/results"
HF_WILD = "kensho/WILD"

# ── Figure style ──────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = ["png", "pdf"]

FAMILY_COLORS = {
    "Frank": "#1f77b4",
    "Gaussian": "#aec7e8",
    "Student-t": "#ff7f0e",
    "Gumbel": "#2ca02c",
    "Joe": "#d62728",
    "Clayton": "#9467bd",
    "BB1": "#8c564b",
    "BB7": "#e377c2",
    "BB6": "#7f7f7f",
    "BB8": "#bcbd22",
    "Tawn": "#17becf",
}
