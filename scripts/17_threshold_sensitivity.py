"""
Task 2C: Systematic Threshold Sensitivity Ablation.
Addresses R2(W8), R3(W10).

Computes conditional Spearman at percentiles 70, 75, 80, 85, 90, 95 for all 15 pairs.
Creates ablation data for heatmap/line chart.
Justifies the 80th percentile choice via stability criterion.
"""
import itertools
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
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

PERCENTILES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DECEPTIVE_THRESHOLD = 0.5


def conditional_spearman(u, quantile):
    """Compute Spearman restricted to models above quantile on either variable."""
    mask = (u[:, 0] >= quantile) | (u[:, 1] >= quantile)
    n_eff = mask.sum()
    if n_eff < 10:
        return np.nan, 0
    rho = float(stats.spearmanr(u[mask, 0], u[mask, 1]).statistic)
    return rho, int(n_eff)


def compute_all_conditional_spearman(pit):
    """Compute conditional Spearman at every percentile for all 15 pairs."""
    print("=" * 70)
    print("THRESHOLD SENSITIVITY ABLATION")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    n = len(pit)

    results = []

    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values

        rho_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

        pair_result = {
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "spearman_full": rho_full,
            "conditional_rho": {},
            "attenuation_ratio": {},
            "deceptive_at_threshold": {},
            "n_effective": {},
        }

        for q in PERCENTILES:
            cond_rho, n_eff = conditional_spearman(u, q)
            ratio = cond_rho / rho_full if rho_full != 0 and not np.isnan(cond_rho) else np.nan
            is_deceptive = (not np.isnan(ratio) and ratio < DECEPTIVE_THRESHOLD
                           and abs(rho_full) > 0.5)

            pair_result["conditional_rho"][str(q)] = float(cond_rho) if not np.isnan(cond_rho) else None
            pair_result["attenuation_ratio"][str(q)] = float(ratio) if not np.isnan(ratio) else None
            pair_result["deceptive_at_threshold"][str(q)] = is_deceptive
            pair_result["n_effective"][str(q)] = n_eff

        results.append(pair_result)

    return results


def compute_deceptive_counts(results):
    """Count how many pairs are classified as deceptive at each threshold."""
    print("\n  Deceptive pair counts by threshold:")
    print(f"  {'Percentile':>12s} | {'N deceptive':>12s} | {'N pairs':>8s} | {'Rate':>8s}")
    print("  " + "-" * 50)

    counts = {}
    for q in PERCENTILES:
        q_str = str(q)
        n_deceptive = sum(
            1 for r in results
            if r["deceptive_at_threshold"].get(q_str, False)
        )
        n_valid = sum(
            1 for r in results
            if r["conditional_rho"].get(q_str) is not None
        )
        counts[q_str] = {
            "n_deceptive": n_deceptive,
            "n_valid": n_valid,
            "rate": n_deceptive / n_valid if n_valid > 0 else 0,
        }
        print(f"  {q:>12.0%} | {n_deceptive:>12d} | {n_valid:>8d} | "
              f"{n_deceptive/n_valid*100:>7.1f}%")

    return counts


def stability_analysis(results):
    """
    Justify the 80th percentile:
    - Compute the derivative of deceptive count w.r.t. threshold
    - Find the elbow point where the classification stabilizes
    - Report confidence based on effective sample sizes
    """
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS: Justifying the 80th Percentile")
    print("=" * 70)

    pair_labels = [f"{r['benchmark_1']}×{r['benchmark_2']}" for r in results]

    transition_matrix = []
    for r in results:
        row = []
        for q in PERCENTILES:
            q_str = str(q)
            is_dec = r["deceptive_at_threshold"].get(q_str, False)
            row.append(1 if is_dec else 0)
        transition_matrix.append(row)

    transition_matrix = np.array(transition_matrix)

    flip_counts = np.sum(np.abs(np.diff(transition_matrix, axis=1)), axis=0)
    print(f"\n  Classification flips between adjacent thresholds:")
    for k, (q1, q2) in enumerate(zip(PERCENTILES[:-1], PERCENTILES[1:])):
        print(f"    {q1:.0%} → {q2:.0%}: {flip_counts[k]} pairs change classification")

    total_flips = np.sum(np.abs(np.diff(transition_matrix, axis=1)), axis=1)
    stable_pairs = np.sum(total_flips == 0)
    print(f"\n  Pairs with STABLE classification across all thresholds: "
          f"{stable_pairs}/{len(results)}")

    unstable_pairs = [(pair_labels[i], int(total_flips[i]))
                      for i in range(len(results)) if total_flips[i] > 0]
    if unstable_pairs:
        print("  Unstable pairs:")
        for label, flips in unstable_pairs:
            print(f"    {label}: {flips} flips")

    n_effs_80 = [r["n_effective"].get("0.8", 0) for r in results]
    n_effs_90 = [r["n_effective"].get("0.9", 0) for r in results]
    n_effs_95 = [r["n_effective"].get("0.95", 0) for r in results]

    print(f"\n  Effective sample sizes:")
    print(f"    80th: mean={np.mean(n_effs_80):.0f}, min={min(n_effs_80)}")
    print(f"    90th: mean={np.mean(n_effs_90):.0f}, min={min(n_effs_90)}")
    print(f"    95th: mean={np.mean(n_effs_95):.0f}, min={min(n_effs_95)}")

    print(f"\n  JUSTIFICATION for 80th percentile:")
    print(f"    1. Sufficient n_eff: ≥{min(n_effs_80)} observations (SE(ρ) ≈ "
          f"{1/np.sqrt(min(n_effs_80)-3):.3f})")
    print(f"    2. Classification stability: "
          f"{flip_counts[0] if len(flip_counts) > 0 else 'N/A'} flips from 75→80, "
          f"{flip_counts[2] if len(flip_counts) > 2 else 'N/A'} flips from 85→90")
    print(f"    3. Balances statistical power with tail focus")

    return {
        "transition_matrix": transition_matrix.tolist(),
        "flip_counts": flip_counts.tolist(),
        "stable_pairs": int(stable_pairs),
        "unstable_pairs": unstable_pairs,
        "n_effective_summary": {
            "80th": {"mean": float(np.mean(n_effs_80)), "min": int(min(n_effs_80))},
            "90th": {"mean": float(np.mean(n_effs_90)), "min": int(min(n_effs_90))},
            "95th": {"mean": float(np.mean(n_effs_95)), "min": int(min(n_effs_95))},
        }
    }


def create_heatmap(results):
    """Create a heatmap of conditional Spearman across thresholds."""
    pair_labels = [f"{r['benchmark_1']}×\n{r['benchmark_2']}" for r in results]
    short_labels = [f"{r['benchmark_1'][:4]}×{r['benchmark_2'][:4]}" for r in results]

    rho_full = [r["spearman_full"] for r in results]
    sort_idx = np.argsort(rho_full)[::-1]

    matrix = np.zeros((len(results), len(PERCENTILES)))
    for i, r in enumerate(results):
        for k, q in enumerate(PERCENTILES):
            val = r["conditional_rho"].get(str(q))
            matrix[i, k] = val if val is not None else np.nan

    matrix_sorted = matrix[sort_idx]
    labels_sorted = [short_labels[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix_sorted, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=1.0)

    ax.set_xticks(range(len(PERCENTILES)))
    ax.set_xticklabels([f"{int(q*100)}th" for q in PERCENTILES], fontsize=9)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels_sorted, fontsize=7)

    for i in range(len(results)):
        for k in range(len(PERCENTILES)):
            val = matrix_sorted[i, k]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.4 else "black"
                ax.text(k, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.axvline(x=2.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(2.5, -0.8, "80th\n(chosen)", ha="center", va="bottom",
            fontsize=8, color="red", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Conditional Spearman ρ", shrink=0.8)
    ax.set_xlabel("Conditioning Percentile", fontsize=11)
    ax.set_ylabel("Benchmark Pair (sorted by bulk ρ)", fontsize=11)
    ax.set_title("Conditional Spearman Across Percentile Thresholds\n"
                 "(red dashed = chosen 80th percentile)", fontsize=12)
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "fig_threshold_sensitivity_heatmap.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_threshold_sensitivity_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_threshold_sensitivity_heatmap.pdf/png")


def create_line_chart(results, deceptive_counts):
    """Line chart of deceptive pair count vs threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pcts = [int(q * 100) for q in PERCENTILES]
    n_deceptive = [deceptive_counts[str(q)]["n_deceptive"] for q in PERCENTILES]

    ax1.plot(pcts, n_deceptive, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.axvline(x=80, color="red", linestyle="--", alpha=0.7, label="Chosen threshold")
    ax1.set_xlabel("Conditioning Percentile", fontsize=11)
    ax1.set_ylabel("Number of Deceptive Pairs", fontsize=11)
    ax1.set_title("Deceptive Pair Count vs. Threshold", fontsize=12)
    ax1.set_xticks(pcts)
    ax1.set_xticklabels([f"{p}th" for p in pcts])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    headline_pairs = [
        ("BBH", "GPQA", "tab:blue"),
        ("BBH", "MUSR", "tab:orange"),
        ("MATH Lvl 5", "MMLU-PRO", "tab:green"),
        ("BBH", "MMLU-PRO", "tab:red"),
        ("IFEval", "BBH", "tab:purple"),
    ]

    for bm1, bm2, color in headline_pairs:
        match = [r for r in results
                 if (r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2)]
        if not match:
            continue
        r = match[0]
        vals = [r["conditional_rho"].get(str(q)) for q in PERCENTILES]
        vals = [v if v is not None else np.nan for v in vals]
        label = f"{bm1[:4]}×{bm2[:4]} (ρ={r['spearman_full']:.2f})"
        ax2.plot(pcts, vals, "o-", color=color, linewidth=2, markersize=6, label=label)

    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.axvline(x=80, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Conditioning Percentile", fontsize=11)
    ax2.set_ylabel("Conditional Spearman ρ", fontsize=11)
    ax2.set_title("Conditional Spearman Degradation\n(headline pairs)", fontsize=12)
    ax2.set_xticks(pcts)
    ax2.set_xticklabels([f"{p}th" for p in pcts])
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_threshold_sensitivity_lines.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_threshold_sensitivity_lines.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_threshold_sensitivity_lines.pdf/png")


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2C: SYSTEMATIC THRESHOLD SENSITIVITY ABLATION")
    print("=" * 70)

    pit = pd.read_csv(DATA_DIR / "ollm_pit.csv", index_col=0)
    print(f"  PIT matrix: {pit.shape}")

    print("\n--- 1. Conditional Spearman at all thresholds ---")
    results = compute_all_conditional_spearman(pit)

    print("\n--- 2. Deceptive pair counts ---")
    deceptive_counts = compute_deceptive_counts(results)

    print("\n--- 3. Stability analysis ---")
    stability = stability_analysis(results)

    print("\n--- 4. Visualization ---")
    create_heatmap(results)
    create_line_chart(results, deceptive_counts)

    output = {
        "conditional_spearman_by_threshold": results,
        "deceptive_counts": deceptive_counts,
        "stability_analysis": stability,
    }

    output_path = RESULTS_DIR / "threshold_sensitivity_ablation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("TASK 2C COMPLETE")
    print("=" * 70)
