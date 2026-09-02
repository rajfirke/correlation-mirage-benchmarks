"""
Rebuttal experiments for Reviewer SU96.

Addresses:
  1. Collider/Berkson bias: intersection vs union conditioning comparison
  2. Code mismatch: verify both conditioning schemes produce collapse
  3. Discrete scores / ties: quantify tie fraction per benchmark
  4. Multiple thresholds under both conditioning schemes
  5. Berkson simulation under INTERSECTION conditioning (eliminates collider concern)
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_release"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BENCHMARKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]


def load_data():
    scores = pd.read_csv(DATA_DIR / "ollm_score_matrix.csv", index_col=0)
    pit = pd.read_csv(DATA_DIR / "ollm_pit.csv", index_col=0)
    return scores, pit


def conditional_spearman_union(u, quantile):
    mask = (u[:, 0] >= quantile) | (u[:, 1] >= quantile)
    n_eff = mask.sum()
    if n_eff < 10:
        return np.nan, 0, np.nan
    result = stats.spearmanr(u[mask, 0], u[mask, 1])
    return float(result.statistic), int(n_eff), float(result.pvalue)


def conditional_spearman_intersection(u, quantile):
    mask = (u[:, 0] >= quantile) & (u[:, 1] >= quantile)
    n_eff = mask.sum()
    if n_eff < 10:
        return np.nan, 0, np.nan
    result = stats.spearmanr(u[mask, 0], u[mask, 1])
    return float(result.statistic), int(n_eff), float(result.pvalue)


def experiment_1_intersection_vs_union(pit):
    """Compare conditional Spearman under union vs intersection conditioning."""
    print("=" * 70)
    print("EXPERIMENT 1: Intersection vs Union Conditioning")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    quantiles = [0.80, 0.85, 0.90, 0.95]

    results = []
    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values

        rho_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

        pair_result = {
            "pair": f"{bm_i} x {bm_j}",
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "spearman_full": rho_full,
            "n_total": len(u),
            "quantiles": {}
        }

        for q in quantiles:
            rho_union, n_union, p_union = conditional_spearman_union(u, q)
            rho_inter, n_inter, p_inter = conditional_spearman_intersection(u, q)

            pair_result["quantiles"][str(q)] = {
                "union": {
                    "rho": rho_union,
                    "n_eff": n_union,
                    "p_value": p_union,
                    "drop_pct": round((1 - rho_union / rho_full) * 100, 1) if rho_full != 0 and not np.isnan(rho_union) else None
                },
                "intersection": {
                    "rho": rho_inter,
                    "n_eff": n_inter,
                    "p_value": p_inter,
                    "drop_pct": round((1 - rho_inter / rho_full) * 100, 1) if rho_full != 0 and not np.isnan(rho_inter) else None
                }
            }

        results.append(pair_result)

    # Print summary table
    print(f"\n{'Pair':<25} {'rho_S':>6} | {'Union 80%':>10} {'n':>5} | {'Inter 80%':>10} {'n':>5} | {'Union 95%':>10} {'n':>5} | {'Inter 95%':>10} {'n':>5}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: -x["spearman_full"]):
        q80 = r["quantiles"]["0.8"]
        q95 = r["quantiles"]["0.95"]
        u80 = q80["union"]["rho"]
        i80 = q80["intersection"]["rho"]
        u95 = q95["union"]["rho"]
        i95 = q95["intersection"]["rho"]
        print(f"{r['pair']:<25} {r['spearman_full']:>6.3f} | "
              f"{u80:>10.3f} {q80['union']['n_eff']:>5} | "
              f"{i80:>10.3f} {q80['intersection']['n_eff']:>5} | "
              f"{u95 if not np.isnan(u95) else 'N/A':>10} {q95['union']['n_eff']:>5} | "
              f"{i95 if not np.isnan(i95) else 'N/A':>10} {q95['intersection']['n_eff']:>5}")

    return results


def experiment_2_berkson_simulation_intersection(pit, n_sims=500, seed=42):
    """
    Simulate Berkson calibration under INTERSECTION conditioning.
    This eliminates the collider concern entirely since intersection
    does not condition on a common effect.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Berkson Calibration Under INTERSECTION Conditioning")
    print("=" * 70)

    headline_pairs = [
        ("BBH", "GPQA"),
        ("BBH", "MUSR"),
        ("MATH Lvl 5", "MMLU-PRO"),
    ]

    benchmarks = pit.columns.tolist()
    results = []

    for bm_i, bm_j in headline_pairs:
        u = pit[[bm_i, bm_j]].values
        rho_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)
        n = len(u)

        # Observed under intersection
        observed = {}
        for q in [0.8, 0.9, 0.95]:
            rho, n_eff, p = conditional_spearman_intersection(u, q)
            observed[str(q)] = {"rho": rho, "n_eff": n_eff}

        # Simulate Gaussian null under intersection conditioning
        gauss_cop = pv.Bicop(family=pv.BicopFamily.gaussian,
                             parameters=np.array([[rho_full]]))
        null_rhos = {str(q): [] for q in [0.8, 0.9, 0.95]}
        for s in range(n_sims):
            u_sim = gauss_cop.simulate(n, seeds=[seed + s])
            u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)
            for q in [0.8, 0.9, 0.95]:
                mask = (u_sim[:, 0] >= q) & (u_sim[:, 1] >= q)
                if mask.sum() >= 10:
                    rho_sim = float(stats.spearmanr(u_sim[mask, 0], u_sim[mask, 1]).statistic)
                    null_rhos[str(q)].append(rho_sim)

        # Simulate Student-t(df=5) null under intersection conditioning
        try:
            t_cop = pv.Bicop(family=pv.BicopFamily.student,
                             parameters=np.array([[rho_full, 5.0]]))
            t5_rhos = {str(q): [] for q in [0.8, 0.9, 0.95]}
            for s in range(n_sims):
                u_sim = t_cop.simulate(n, seeds=[seed + s])
                u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)
                for q in [0.8, 0.9, 0.95]:
                    mask = (u_sim[:, 0] >= q) & (u_sim[:, 1] >= q)
                    if mask.sum() >= 10:
                        rho_sim = float(stats.spearmanr(u_sim[mask, 0], u_sim[mask, 1]).statistic)
                        t5_rhos[str(q)].append(rho_sim)
        except Exception:
            t5_rhos = {str(q): [] for q in [0.8, 0.9, 0.95]}

        pair_result = {
            "pair": f"{bm_i} x {bm_j}",
            "spearman_full": rho_full,
            "n": n,
            "observed_intersection": observed,
            "null_predictions_intersection": {}
        }

        for q_str in ["0.8", "0.9", "0.95"]:
            gauss_vals = np.array(null_rhos[q_str])
            t5_vals = np.array(t5_rhos[q_str])
            obs = observed[q_str]["rho"]

            gauss_result = {}
            if len(gauss_vals) > 0:
                gauss_result = {
                    "mean": float(np.mean(gauss_vals)),
                    "ci": [float(np.percentile(gauss_vals, 2.5)),
                           float(np.percentile(gauss_vals, 97.5))],
                    "excess": float(obs - np.mean(gauss_vals)) if not np.isnan(obs) else None,
                    "p_value": float(np.mean(gauss_vals <= obs)) if not np.isnan(obs) else None,
                }

            t5_result = {}
            if len(t5_vals) > 0:
                t5_result = {
                    "mean": float(np.mean(t5_vals)),
                    "ci": [float(np.percentile(t5_vals, 2.5)),
                           float(np.percentile(t5_vals, 97.5))],
                    "excess": float(obs - np.mean(t5_vals)) if not np.isnan(obs) else None,
                    "p_value": float(np.mean(t5_vals <= obs)) if not np.isnan(obs) else None,
                }

            pair_result["null_predictions_intersection"][q_str] = {
                "gaussian": gauss_result,
                "student_t_df5": t5_result
            }

        results.append(pair_result)

        # Print
        print(f"\n{bm_i} x {bm_j} (rho_S = {rho_full:.3f}):")
        for q_str in ["0.8", "0.9", "0.95"]:
            obs = observed[q_str]["rho"]
            n_eff = observed[q_str]["n_eff"]
            gauss = pair_result["null_predictions_intersection"][q_str].get("gaussian", {})
            t5 = pair_result["null_predictions_intersection"][q_str].get("student_t_df5", {})
            gauss_mean = gauss.get("mean", "N/A")
            t5_mean = t5.get("mean", "N/A")
            gauss_excess = gauss.get("excess", "N/A")
            t5_excess = t5.get("excess", "N/A")
            gauss_p = gauss.get("p_value", "N/A")
            t5_p = t5.get("p_value", "N/A")
            obs_str = f"{obs:.3f}" if not np.isnan(obs) else "N/A"
            print(f"  q={q_str}: observed={obs_str} (n={n_eff}) | "
                  f"Gaussian null={gauss_mean if isinstance(gauss_mean, str) else f'{gauss_mean:.3f}'}, "
                  f"excess={gauss_excess if isinstance(gauss_excess, str) else f'{gauss_excess:.3f}'}, "
                  f"p={gauss_p if isinstance(gauss_p, str) else f'{gauss_p:.3f}'} | "
                  f"Student-t(5) null={t5_mean if isinstance(t5_mean, str) else f'{t5_mean:.3f}'}, "
                  f"excess={t5_excess if isinstance(t5_excess, str) else f'{t5_excess:.3f}'}, "
                  f"p={t5_p if isinstance(t5_p, str) else f'{t5_p:.3f}'}")

    return results


def experiment_3_tie_analysis(scores):
    """Quantify ties and score discreteness per benchmark."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Score Discreteness and Tie Analysis")
    print("=" * 70)

    results = {}
    for col in scores.columns:
        vals = scores[col].dropna()
        n = len(vals)
        n_unique = vals.nunique()
        n_ties = n - n_unique
        tie_fraction = n_ties / n

        # Check top-20% score spread
        q80 = vals.quantile(0.80)
        top_vals = vals[vals >= q80]
        full_range = vals.max() - vals.min()
        top_range = top_vals.max() - top_vals.min()
        spread_ratio = top_range / full_range if full_range > 0 else 0

        results[col] = {
            "n": int(n),
            "n_unique": int(n_unique),
            "tie_fraction": round(float(tie_fraction), 4),
            "full_range": round(float(full_range), 2),
            "top_20pct_range": round(float(top_range), 2),
            "spread_ratio": round(float(spread_ratio), 4),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "top_20pct_min": round(float(top_vals.min()), 2),
        }

        print(f"  {col:<15}: {n_unique:>5} unique / {n:>5} total "
              f"(tie frac = {tie_fraction:.3f}), "
              f"spread ratio = {spread_ratio:.3f} "
              f"(range {vals.min():.1f}–{vals.max():.1f}, "
              f"top-20% range {top_vals.min():.1f}–{top_vals.max():.1f})")

    return results


def experiment_4_intersection_correlation_with_union(pit):
    """
    Compute correlation between intersection-based and union-based
    conditional Spearman across all 15 pairs. High correlation proves
    conditioning scheme doesn't drive the results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Correlation Between Conditioning Schemes")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))

    union_vals = []
    inter_vals = []

    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values

        rho_u, _, _ = conditional_spearman_union(u, 0.8)
        rho_i, _, _ = conditional_spearman_intersection(u, 0.8)

        if not np.isnan(rho_u) and not np.isnan(rho_i):
            union_vals.append(rho_u)
            inter_vals.append(rho_i)

    r_spearman = stats.spearmanr(union_vals, inter_vals)
    r_pearson = stats.pearsonr(union_vals, inter_vals)

    print(f"  Spearman correlation between union and intersection rho_up: "
          f"r = {r_spearman.statistic:.3f} (p = {r_spearman.pvalue:.4f})")
    print(f"  Pearson correlation: r = {r_pearson.statistic:.3f} "
          f"(p = {r_pearson.pvalue:.4f})")
    print(f"  Number of pairs: {len(union_vals)}")

    # Count how many pairs are "deceptive" under each scheme
    benchmarks_list = pit.columns.tolist()
    pairs_list = list(itertools.combinations(range(len(benchmarks_list)), 2))
    deceptive_union = 0
    deceptive_inter = 0
    agree = 0

    for idx, (i, j) in enumerate(pairs_list):
        bm_i, bm_j = benchmarks_list[i], benchmarks_list[j]
        u = pit[[bm_i, bm_j]].values
        rho_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

        rho_u, _, _ = conditional_spearman_union(u, 0.8)
        rho_i, _, _ = conditional_spearman_intersection(u, 0.8)

        is_deceptive_u = (not np.isnan(rho_u)) and (rho_u < 0.5 * rho_full)
        is_deceptive_i = (not np.isnan(rho_i)) and (rho_i < 0.5 * rho_full)

        if is_deceptive_u:
            deceptive_union += 1
        if is_deceptive_i:
            deceptive_inter += 1
        if is_deceptive_u == is_deceptive_i:
            agree += 1

    print(f"\n  Deceptive pairs (rho_up < 0.5 * rho_S):")
    print(f"    Union conditioning:       {deceptive_union}/15")
    print(f"    Intersection conditioning: {deceptive_inter}/15")
    print(f"    Agreement:                 {agree}/15")

    return {
        "spearman_between_schemes": float(r_spearman.statistic),
        "spearman_p": float(r_spearman.pvalue),
        "pearson_between_schemes": float(r_pearson.statistic),
        "pearson_p": float(r_pearson.pvalue),
        "n_pairs": len(union_vals),
        "deceptive_union": deceptive_union,
        "deceptive_intersection": deceptive_inter,
        "classification_agreement": agree,
        "union_values": union_vals,
        "intersection_values": inter_vals,
    }


if __name__ == "__main__":
    scores, pit = load_data()
    print(f"Loaded: {scores.shape[0]} models, {scores.shape[1]} benchmarks\n")

    r1 = experiment_1_intersection_vs_union(pit)
    r2 = experiment_2_berkson_simulation_intersection(pit)
    r3 = experiment_3_tie_analysis(scores)
    r4 = experiment_4_intersection_correlation_with_union(pit)

    all_results = {
        "experiment_1_intersection_vs_union": r1,
        "experiment_2_berkson_intersection": r2,
        "experiment_3_tie_analysis": r3,
        "experiment_4_scheme_correlation": {
            k: v for k, v in r4.items()
            if k not in ("union_values", "intersection_values")
        },
    }

    out_path = RESULTS_DIR / "rebuttal_su96_experiments.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
