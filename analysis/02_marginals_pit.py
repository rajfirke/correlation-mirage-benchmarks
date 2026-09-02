"""
Section 2: Marginal analysis and Probability Integral Transform (PIT).
Converts raw benchmark scores to pseudo-uniform [0,1] via empirical CDF.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def rank_transform_pit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Probability Integral Transform using scaled ranks.
    u_i = rank(x_i) / (n + 1) to avoid exact 0 and 1.
    """
    n = len(df)
    pit = df.rank(method="average") / (n + 1)
    return pit


def assess_marginals(df: pd.DataFrame, pit: pd.DataFrame):
    """Check each benchmark's marginal distribution + PIT uniformity."""
    results = {}
    for col in df.columns:
        raw = df[col].dropna()
        u = pit[col].dropna()

        ks_stat, ks_p = stats.kstest(u, "uniform")
        skewness = float(stats.skew(raw))
        kurtosis = float(stats.kurtosis(raw))
        floor_pct = float((raw <= raw.quantile(0.05)).mean())
        ceil_pct = float((raw >= raw.quantile(0.95)).mean())

        results[col] = {
            "n": int(len(raw)),
            "mean": float(raw.mean()),
            "std": float(raw.std()),
            "min": float(raw.min()),
            "max": float(raw.max()),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "pit_uniform_ok": bool(ks_p > 0.05),
            "floor_pct": floor_pct,
            "ceil_pct": ceil_pct,
        }

    return results


def plot_marginals(df: pd.DataFrame, pit: pd.DataFrame, out_dir: Path):
    """Generate histograms of raw scores and QQ plots of PIT values."""
    n_cols = len(df.columns)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(df.columns):
        raw = df[col].dropna().values
        u = pit[col].dropna().values

        axes[0, i].hist(raw, bins=15, edgecolor="black", alpha=0.7, color="#4878CF")
        axes[0, i].set_title(col, fontsize=8, fontweight="bold")
        axes[0, i].set_xlabel("Score", fontsize=7)
        if i == 0:
            axes[0, i].set_ylabel("Count", fontsize=7)
        axes[0, i].tick_params(labelsize=6)

        axes[1, i].hist(u, bins=15, edgecolor="black", alpha=0.7, color="#6ACC65",
                        density=True)
        axes[1, i].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Uniform")
        axes[1, i].set_xlabel("PIT value", fontsize=7)
        if i == 0:
            axes[1, i].set_ylabel("Density", fontsize=7)
        axes[1, i].tick_params(labelsize=6)
        axes[1, i].set_xlim(0, 1)

    axes[0, 0].set_ylabel("Raw scores\nCount", fontsize=7)
    axes[1, 0].set_ylabel("PIT (should be uniform)\nDensity", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "marginals_and_pit.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(out_dir / "marginals_and_pit.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved marginal plots -> {out_dir / 'marginals_and_pit.pdf'}")


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 2: Marginal Analysis & PIT Transformation")
    print("=" * 70)

    cc = pd.read_csv(DATA_DIR / "complete_case_matrix.csv", index_col=0)
    print(f"  Loaded complete-case matrix: {cc.shape}")

    print("\n--- 2.1: PIT Transformation ---")
    pit = rank_transform_pit(cc)
    pit.to_csv(DATA_DIR / "pit_transformed.csv")
    print(f"  PIT-transformed matrix saved: {pit.shape}")
    print(f"  Value range: [{pit.min().min():.4f}, {pit.max().max():.4f}]")

    print("\n--- 2.2: Marginal Adequacy Assessment ---")
    marginal_stats = assess_marginals(cc, pit)

    for col, s in marginal_stats.items():
        uniform_flag = "OK" if s["pit_uniform_ok"] else "WARN"
        print(f"  {col}: n={s['n']}, range=[{s['min']:.1f}, {s['max']:.1f}], "
              f"skew={s['skewness']:.2f}, KS p={s['ks_pvalue']:.3f} [{uniform_flag}]")

    with open(RESULTS_DIR / "marginal_stats.json", "w") as f:
        json.dump(marginal_stats, f, indent=2)

    pathological = [c for c, s in marginal_stats.items() if not s["pit_uniform_ok"]]
    if pathological:
        print(f"\n  WARNING: {len(pathological)} benchmarks failed KS uniformity test: {pathological}")
        print("  (This is expected — rank-PIT is always uniform by construction for complete data.")
        print("   Non-uniformity only arises with ties or discrete scores.)")
    else:
        print("\n  All benchmarks pass PIT uniformity test (as expected for rank-based PIT).")

    print("\n--- Generating Plots ---")
    plot_marginals(cc, pit, FIGURES_DIR)

    print("\n" + "=" * 70)
    print("SECTION 2 COMPLETE")
    print(f"  PIT matrix: {pit.shape}")
    print(f"  Marginal stats saved to: results/marginal_stats.json")
    print("=" * 70)
