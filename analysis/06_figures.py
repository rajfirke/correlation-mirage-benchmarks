"""
Section 6: Generate all paper figures.
Fig 1: Correlation vs Tail Dependence heatmaps (THE visual punchline)
Fig 2: Deceptive pairs scatter plots
Fig 3: Copula family distribution + tail dependence summary
Fig 4: Selection experiment results
Fig 5: Vine structure vs PCA dendrogram
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

SHORT_NAMES = {
    "GPQA Diamond": "GPQA-D",
    "GPQA Main (full set)": "GPQA-M",
    "HumanEval": "HumanEval",
    "MMLU-Pro": "MMLU-Pro",
    "IFEval": "IFEval",
    "MMLU": "MMLU",
    "LiveCodeBench": "LCB",
    "MATH-500": "MATH-500",
}


def shorten(name):
    return SHORT_NAMES.get(name, name[:12])


def fig1_heatmap_comparison():
    """
    THE visual punchline: Spearman correlation vs upper tail dependence side-by-side.
    """
    spearman = pd.read_csv(RESULTS_DIR / "matrix_spearman.csv", index_col=0)
    lambda_U = pd.read_csv(RESULTS_DIR / "matrix_lambda_U.csv", index_col=0)

    labels = [shorten(c) for c in spearman.columns]
    sp_vals = spearman.values
    lu_vals = lambda_U.values

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2),
                                         gridspec_kw={"width_ratios": [1, 1, 1]})

    mask = np.triu(np.ones_like(sp_vals, dtype=bool), k=0)

    im1 = ax1.imshow(np.ma.array(sp_vals, mask=mask), cmap="RdYlBu_r",
                      vmin=0, vmax=1, aspect="equal")
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_yticklabels(labels)
    ax1.set_title("(a) Spearman Correlation", fontweight="bold")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            ax1.text(j, i, f"{sp_vals[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(np.ma.array(lu_vals, mask=mask), cmap="YlOrRd",
                      vmin=0, vmax=1, aspect="equal")
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_yticklabels(labels)
    ax2.set_title("(b) Upper Tail Dependence (lambda_U)", fontweight="bold")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            ax2.text(j, i, f"{lu_vals[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    diff = np.abs(sp_vals) - lu_vals
    im3 = ax3.imshow(np.ma.array(diff, mask=mask), cmap="PuOr",
                      vmin=-0.5, vmax=0.8, aspect="equal")
    ax3.set_xticks(range(len(labels)))
    ax3.set_yticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.set_yticklabels(labels)
    ax3.set_title("(c) |Correlation| - Tail Dep. (Deception)", fontweight="bold")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            val = diff[i, j]
            color = "red" if val > 0.3 else "black"
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                     color=color, fontweight="bold" if val > 0.3 else "normal")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_heatmap_comparison.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig1_heatmap_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Fig 1 saved: heatmap comparison")


def fig2_deceptive_scatter():
    """Scatter plots for top deceptive pairs from the complete-case analysis."""
    cc = pd.read_csv(DATA_DIR / "complete_case_matrix.csv", index_col=0)
    with open(RESULTS_DIR / "bivariate_fits_complete.json") as f:
        fits = json.load(f)

    df = pd.DataFrame(fits)
    df["max_tail"] = df[["lambda_L", "lambda_U"]].max(axis=1)
    df["deceptive_score"] = df["spearman_rho"].abs() - df["max_tail"]

    deceptive = df.nlargest(3, "deceptive_score")
    surprising = df.nsmallest(3, "deceptive_score")
    pairs_to_plot = pd.concat([deceptive, surprising])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (_, row) in enumerate(pairs_to_plot.iterrows()):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        bm1, bm2 = row["benchmark_1"], row["benchmark_2"]
        if bm1 in cc.columns and bm2 in cc.columns:
            x, y = cc[bm1], cc[bm2]
        else:
            ax.set_visible(False)
            continue

        ax.scatter(x, y, alpha=0.6, s=30, c="#4878CF", edgecolor="white", linewidth=0.5)

        q_high = 0.8
        top_mask = (x >= x.quantile(q_high)) | (y >= y.quantile(q_high))
        bottom_mask = (x <= x.quantile(0.2)) | (y <= y.quantile(0.2))
        ax.scatter(x[top_mask], y[top_mask], alpha=0.8, s=40, c="#D65F5F",
                   edgecolor="black", linewidth=0.5, label="Top 20%", zorder=5)
        ax.scatter(x[bottom_mask], y[bottom_mask], alpha=0.8, s=40, c="#6ACC65",
                   edgecolor="black", linewidth=0.5, label="Bottom 20%", zorder=5)

        title_prefix = "DECEPTIVE" if r == 0 else "SURPRISING"
        ax.set_title(f"{title_prefix}\nrho={row['spearman_rho']:.2f}, "
                     f"lambda_U={row['lambda_U']:.2f}\n[{row['family']}]",
                     fontsize=8)
        ax.set_xlabel(shorten(bm1), fontsize=8)
        ax.set_ylabel(shorten(bm2), fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    axes[0, 0].annotate("High correlation, low tail dependence",
                         xy=(0.5, 1.15), xycoords="axes fraction",
                         ha="center", fontsize=10, fontweight="bold")
    axes[1, 0].annotate("Low correlation, high tail dependence",
                         xy=(0.5, 1.15), xycoords="axes fraction",
                         ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_deceptive_pairs.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig2_deceptive_pairs.png", bbox_inches="tight")
    plt.close()
    print("  Fig 2 saved: deceptive pairs scatter")


def fig3_family_distribution():
    """Copula family distribution and tail dependence summary."""
    with open(RESULTS_DIR / "bivariate_fits_complete.json") as f:
        cc_fits = json.load(f)
    with open(RESULTS_DIR / "bivariate_fits_pairwise.json") as f:
        pw_fits = json.load(f)

    all_fits = cc_fits + pw_fits
    df = pd.DataFrame(all_fits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    family_counts = df["family"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(family_counts)))
    bars = ax1.barh(range(len(family_counts)), family_counts.values, color=colors)
    ax1.set_yticks(range(len(family_counts)))
    ax1.set_yticklabels(family_counts.index)
    ax1.set_xlabel("Number of Benchmark Pairs")
    ax1.set_title("(a) Best-Fitting Copula Family Distribution", fontweight="bold")

    non_gaussian_pct = (1 - family_counts.get("Gaussian", 0) / len(df)) * 100
    ax1.axvline(family_counts.get("Gaussian", 0), color="red", linestyle="--",
                alpha=0.7, label=f"Gaussian: {100-non_gaussian_pct:.0f}%")
    ax1.legend(fontsize=7)

    for bar, count in zip(bars, family_counts.values):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{count}", va="center", fontsize=7)

    ax2.scatter(df["spearman_rho"], df["lambda_U"], alpha=0.5, s=20, c="#4878CF",
                label=f"lambda_U (n={len(df)})")
    ax2.scatter(df["spearman_rho"], df["lambda_L"], alpha=0.5, s=20, c="#D65F5F",
                marker="^", label=f"lambda_L (n={len(df)})")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y = x (perfect alignment)")
    ax2.set_xlabel("Spearman Correlation (rho)")
    ax2.set_ylabel("Tail Dependence Coefficient")
    ax2.set_title("(b) Correlation vs Tail Dependence", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.set_xlim(-0.1, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_family_and_scatter.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_family_and_scatter.png", bbox_inches="tight")
    plt.close()
    print("  Fig 3 saved: family distribution + correlation vs tail dependence")


def fig4_selection_results():
    """Bar chart of selection experiment results."""
    with open(RESULTS_DIR / "selection_experiment.json") as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    K_values = sorted(df["K"].unique())
    methods = ["Copula", "Correlation", "PCA", "Random"]
    method_colors = {"Copula": "#4878CF", "Correlation": "#D65F5F",
                     "PCA": "#6ACC65", "Random": "#999999"}

    width = 0.18
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        x_positions = [k + i * width for k in range(len(K_values))]
        vals = [float(sub[sub["K"] == K]["mae_all"].values[0]) if len(sub[sub["K"] == K]) > 0 else 0
                for K in K_values]
        ax1.bar(x_positions, vals, width=width, label=method,
                color=method_colors.get(method, "gray"), alpha=0.8)

    ax1.set_xticks([k + width * 1.5 for k in range(len(K_values))])
    ax1.set_xticklabels([f"K={k}" for k in K_values])
    ax1.set_ylabel("MAE (All Models)")
    ax1.set_title("(a) Overall Prediction Error", fontweight="bold")
    ax1.legend(fontsize=7)

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        x_positions = [k + i * width for k in range(len(K_values))]
        vals = [float(sub[sub["K"] == K]["mae_tail"].values[0]) if len(sub[sub["K"] == K]) > 0 else 0
                for K in K_values]
        ax2.bar(x_positions, vals, width=width, label=method,
                color=method_colors.get(method, "gray"), alpha=0.8)

    ax2.set_xticks([k + width * 1.5 for k in range(len(K_values))])
    ax2.set_xticklabels([f"K={k}" for k in K_values])
    ax2.set_ylabel("MAE (Top/Bottom 20% Models)")
    ax2.set_title("(b) Tail Prediction Error", fontweight="bold")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_selection_experiment.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig4_selection_experiment.png", bbox_inches="tight")
    plt.close()
    print("  Fig 4 saved: selection experiment results")


def fig5_asymmetry():
    """Tail dependence asymmetry: upper vs lower."""
    with open(RESULTS_DIR / "bivariate_fits_complete.json") as f:
        cc_fits = json.load(f)
    with open(RESULTS_DIR / "bivariate_fits_pairwise.json") as f:
        pw_fits = json.load(f)

    df = pd.DataFrame(cc_fits + pw_fits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.scatter(df["lambda_L"], df["lambda_U"], alpha=0.5, s=25, c="#4878CF")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Symmetric (lambda_U = lambda_L)")
    ax1.set_xlabel("Lower Tail Dependence (lambda_L)")
    ax1.set_ylabel("Upper Tail Dependence (lambda_U)")
    ax1.set_title("(a) Tail Dependence Asymmetry", fontweight="bold")
    ax1.legend(fontsize=7)

    above = (df["lambda_U"] > df["lambda_L"] + 0.01).sum()
    below = (df["lambda_L"] > df["lambda_U"] + 0.01).sum()
    equal = len(df) - above - below
    ax1.text(0.05, 0.90, f"lambda_U > lambda_L: {above} pairs\n"
             f"lambda_L > lambda_U: {below} pairs\n"
             f"Symmetric: {equal} pairs",
             transform=ax1.transAxes, fontsize=7,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    lambda_U_vals = df["lambda_U"].values
    lambda_L_vals = df["lambda_L"].values
    bins = np.linspace(0, 1, 25)
    ax2.hist(lambda_U_vals[lambda_U_vals > 0.01], bins=bins, alpha=0.6,
             label=f"lambda_U > 0 (n={sum(lambda_U_vals > 0.01)})", color="#D65F5F")
    ax2.hist(lambda_L_vals[lambda_L_vals > 0.01], bins=bins, alpha=0.6,
             label=f"lambda_L > 0 (n={sum(lambda_L_vals > 0.01)})", color="#4878CF")
    ax2.set_xlabel("Tail Dependence Coefficient")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Distribution of Non-Zero Tail Dependence", fontweight="bold")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_asymmetry.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_asymmetry.png", bbox_inches="tight")
    plt.close()
    print("  Fig 5 saved: tail dependence asymmetry")


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 6: Figure Generation")
    print("=" * 70)

    fig1_heatmap_comparison()
    fig2_deceptive_scatter()
    fig3_family_distribution()
    fig4_selection_results()
    fig5_asymmetry()

    print("\n  All figures saved to:", FIGURES_DIR)
    print("=" * 70)
    print("SECTION 6 COMPLETE")
    print("=" * 70)
