"""
Generate publication-quality figures for EMNLP 2026 paper:
"Correlation Collapse in the Tails: A Copula Analysis of LLM Benchmarks"
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BENCHMARKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
SHORT_NAMES = {
    "IFEval": "IFEval",
    "BBH": "BBH",
    "MATH Lvl 5": "MATH",
    "GPQA": "GPQA",
    "MUSR": "MUSR",
    "MMLU-PRO": "MMLU-PRO",
}

FAMILY_COLORS = {
    "Frank": "#377eb8",
    "Gumbel": "#ff7f00",
    "Student-t": "#4daf4a",
    "BB1": "#e41a1c",
}


def load_score_matrix():
    df = pd.read_csv(DATA / "ollm_score_matrix.csv", index_col=0)
    df.index = [re.sub(r"<[^>]+>", "", s).strip().split("📑")[0].strip()
                for s in df.index]
    return df


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: The Punchline – Correlation Collapse Scatter
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1(scores, tails):
    pairs = [
        ("BBH", "MUSR"),
        ("IFEval", "BBH"),
        ("MATH Lvl 5", "MMLU-PRO"),
        ("BBH", "MMLU-PRO"),
    ]

    tail_lookup = {(r["benchmark_1"], r["benchmark_2"]): r for r in tails}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5))
    axes = axes.ravel()

    for idx, (b1, b2) in enumerate(pairs):
        ax = axes[idx]
        key = (b1, b2) if (b1, b2) in tail_lookup else (b2, b1)
        info = tail_lookup[key]
        rho_full = info["spearman_full"]
        rho_upper = info["spearman_upper_20pct"]

        x = scores[b1].values
        y = scores[b2].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        thresh_x = np.percentile(x, 80)
        thresh_y = np.percentile(y, 80)
        top = (x >= thresh_x) | (y >= thresh_y)

        ax.scatter(x[~top], y[~top], s=4, alpha=0.15, color="#bdbdbd",
                   edgecolors="none", rasterized=True)
        ax.scatter(x[top], y[top], s=8, alpha=0.55, color="#d62728",
                   edgecolors="none", rasterized=True)

        sign = "−" if rho_upper < 0 else ""
        txt = (f"$\\rho_{{\\mathrm{{all}}}}$ = {rho_full:.3f}\n"
               f"$\\rho_{{\\mathrm{{top20\\%}}}}$ = {rho_upper:.3f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85,
                          edgecolor="#cccccc"))

        ax.set_xlabel(SHORT_NAMES[b1])
        ax.set_ylabel(SHORT_NAMES[b2])
        ax.grid(False)
        sns.despine(ax=ax)

    fig.suptitle("Correlation Collapses Among Top-Performing Models",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(FIGURES / "fig1_collapse_scatter.pdf")
    fig.savefig(FIGURES / "fig1_collapse_scatter.png")
    plt.close(fig)
    print("✓ Figure 1 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Conditional Spearman Heatmap Comparison
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2(tails):
    n = len(BENCHMARKS)
    mat_full = np.eye(n)
    mat_upper = np.eye(n)

    idx_map = {b: i for i, b in enumerate(BENCHMARKS)}
    for rec in tails:
        i = idx_map[rec["benchmark_1"]]
        j = idx_map[rec["benchmark_2"]]
        mat_full[i, j] = mat_full[j, i] = rec["spearman_full"]
        mat_upper[i, j] = mat_upper[j, i] = rec["spearman_upper_20pct"]

    short = [SHORT_NAMES[b] for b in BENCHMARKS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    norm_full = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=1.0)
    norm_upper = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=1.0)

    sns.heatmap(mat_full, ax=ax1, annot=True, fmt=".2f", cmap="RdBu_r",
                norm=norm_full, xticklabels=short, yticklabels=short,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax1.set_title("Bulk Spearman Correlation", fontsize=11, fontweight="bold")

    sns.heatmap(mat_upper, ax=ax2, annot=True, fmt=".2f", cmap="RdBu_r",
                norm=norm_upper, xticklabels=short, yticklabels=short,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax2.set_title("Correlation Among Top 20%", fontsize=11, fontweight="bold")

    for ax in (ax1, ax2):
        ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_heatmap_comparison.pdf")
    fig.savefig(FIGURES / "fig2_heatmap_comparison.png")
    plt.close(fig)
    print("✓ Figure 2 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Copula Family + ΔBIC Evidence
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3(fits):
    df = pd.DataFrame(fits)
    df["pair"] = df.apply(
        lambda r: f"{SHORT_NAMES[r['benchmark_1']]}–{SHORT_NAMES[r['benchmark_2']]}",
        axis=1)
    df = df.sort_values("delta_bic", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    colors = [FAMILY_COLORS[f] for f in df["family"]]
    bars = ax.barh(range(len(df)), df["delta_bic"], color=colors, edgecolor="none",
                   height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["pair"], fontsize=9)
    ax.set_xlabel("ΔBIC (vs. Gaussian copula)", fontsize=10)
    ax.axvline(x=6, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(6 + 5, len(df) - 0.5, "Strong evidence\n(ΔBIC > 6)",
            fontsize=8, va="top", color="#555555")

    handles = [mpatches.Patch(color=c, label=f)
               for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, title="Best Copula Family",
              loc="lower right", framealpha=0.9, fontsize=8)

    ax.set_title("Evidence Against Gaussian Dependence (All 15 Pairs)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_copula_summary.pdf")
    fig.savefig(FIGURES / "fig3_copula_summary.png")
    plt.close(fig)
    print("✓ Figure 3 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: BenchPress Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4(sensitivity):
    df = pd.DataFrame(sensitivity)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    x = df["min_n"]
    observed = df["observed_nongaussian_rate"] * 100
    fpr = df["expected_fpr_from_sim"] * 100

    ax.plot(x, observed, "o-", color="#d62728", linewidth=2, markersize=6,
            label="Observed non-Gaussian rate", zorder=3)
    ax.plot(x, fpr, "s--", color="#377eb8", linewidth=1.5, markersize=5,
            label="Simulation FPR (Gaussian null)", zorder=3)
    ax.fill_between(x, fpr, observed, alpha=0.15, color="#d62728",
                    label="Excess signal")

    ax.set_xlabel("Minimum model count threshold ($n_{\\min}$)")
    ax.set_ylabel("% pairs with ΔBIC > 2")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Sensitivity to Sample Size Threshold",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_sensitivity.pdf")
    fig.savefig(FIGURES / "fig4_sensitivity.png")
    plt.close(fig)
    print("✓ Figure 4 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    scores = load_score_matrix()
    tails = load_json(RESULTS / "ollm_nonparametric_tails.json")
    fits = load_json(RESULTS / "ollm_bivariate_fits.json")
    sensitivity = load_json(RESULTS / "sensitivity_by_n.json")

    print(f"Score matrix: {scores.shape[0]} models × {scores.shape[1]} benchmarks")

    make_fig1(scores, tails)
    make_fig2(tails)
    make_fig3(fits)
    make_fig4(sensitivity)

    print("\nAll figures saved to:", FIGURES)
