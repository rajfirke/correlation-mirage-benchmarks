"""
Fix 1: Nonparametric tail dependence estimation.
Addresses the core reviewer criticism that lambda_U=0 from Frank copula is definitional.
Implements:
  1. Empirical tail concentration function: lambda_hat(u) = P(U1>u, U2>u)/(1-u)
  2. Direct empirical estimator of upper tail dependence: lambda_hat(k) = #{i: U1_i in top-k AND U2_i in top-k} / k
  3. Bootstrap CIs for both estimators
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def empirical_tail_concentration(u1, u2, thresholds):
    """
    Compute the empirical upper tail concentration function:
        chi(u) = P(U1 > u, U2 > u) / (1 - u)

    For comparison, also compute the independence baseline: (1-u).
    And the "perfect dependence" baseline: 1.

    If chi(u) -> 0 as u -> 1, the pair has zero tail dependence.
    If chi(u) -> lambda_U > 0, the pair has positive tail dependence.
    """
    n = len(u1)
    results = []
    for u in thresholds:
        joint_exceed = np.sum((u1 > u) & (u2 > u))
        marginal_exceed = np.sum(u1 > u)
        p_joint = joint_exceed / n
        p_marginal = (1 - u)
        chi_u = p_joint / p_marginal if p_marginal > 0 else np.nan
        independence_baseline = p_marginal
        n_in_tail = int(np.sum(u1 > u))

        results.append({
            "threshold": float(u),
            "chi_u": float(chi_u),
            "independence_baseline": float(independence_baseline),
            "ratio_to_independence": float(chi_u / independence_baseline) if independence_baseline > 0 else np.nan,
            "joint_exceedances": int(joint_exceed),
            "marginal_exceedances": n_in_tail,
            "n_total": n,
        })
    return results


def empirical_lambda_U(u1, u2, k_range=None):
    """
    Direct empirical estimator of upper tail dependence:
        lambda_hat(k) = #{i: U1_i in top-k AND U2_i in top-k} / k

    This counts how many models are jointly in the top-k on BOTH benchmarks,
    normalized by k. If lambda_U > 0, this should stabilize at a positive
    value as k varies. If lambda_U = 0, it should decrease toward 0.
    """
    n = len(u1)
    if k_range is None:
        k_range = list(range(10, min(n // 4, 1000), max(1, n // 100)))

    r1 = stats.rankdata(u1)
    r2 = stats.rankdata(u2)

    estimates = []
    for k in k_range:
        threshold = n - k
        joint_top_k = np.sum((r1 > threshold) & (r2 > threshold))
        lambda_hat = joint_top_k / k
        estimates.append({
            "k": int(k),
            "pct": float(k / n * 100),
            "lambda_hat": float(lambda_hat),
            "joint_count": int(joint_top_k),
        })

    if not estimates:
        return {"emp_lambda_U": np.nan, "emp_estimates": []}

    lambdas = np.array([e["lambda_hat"] for e in estimates])
    small_k = [e for e in estimates if e["k"] <= n * 0.05]
    large_k = [e for e in estimates if e["k"] >= n * 0.10]
    emp_at_5pct = small_k[-1]["lambda_hat"] if small_k else lambdas[0]
    emp_at_10pct = large_k[0]["lambda_hat"] if large_k else lambdas[-1]

    slope = emp_at_5pct - emp_at_10pct
    trend = "decreasing" if slope < -0.02 else "increasing" if slope > 0.02 else "stable"

    return {
        "emp_lambda_U_5pct": float(emp_at_5pct),
        "emp_lambda_U_10pct": float(emp_at_10pct),
        "emp_lambda_U_20pct": float([e for e in estimates if e["pct"] >= 20][0]["lambda_hat"]) if any(e["pct"] >= 20 for e in estimates) else np.nan,
        "trend": trend,
        "slope_5_to_10": float(slope),
        "estimates": estimates[:30],
    }


def bootstrap_tail_ci(u1, u2, threshold=0.9, n_boot=500, seed=42):
    """Bootstrap CI for the empirical tail concentration at a given threshold."""
    rng = np.random.RandomState(seed)
    n = len(u1)
    chi_boots = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        u1b, u2b = u1[idx], u2[idx]
        joint = np.sum((u1b > threshold) & (u2b > threshold)) / n
        marginal = 1 - threshold
        chi = joint / marginal if marginal > 0 else np.nan
        chi_boots.append(chi)

    chi_boots = np.array(chi_boots)
    return {
        "chi_mean": float(np.nanmean(chi_boots)),
        "chi_ci_low": float(np.nanpercentile(chi_boots, 2.5)),
        "chi_ci_high": float(np.nanpercentile(chi_boots, 97.5)),
    }


def bootstrap_emp_lambda_ci(u1, u2, k_pct=5, n_boot=300, seed=42):
    """Bootstrap CI for the empirical lambda_U at the k-th percentile."""
    rng = np.random.RandomState(seed)
    n = len(u1)
    k = max(10, int(n * k_pct / 100))
    boots = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r1 = stats.rankdata(u1[idx])
        r2 = stats.rankdata(u2[idx])
        threshold = n - k
        joint = np.sum((r1 > threshold) & (r2 > threshold))
        boots.append(joint / k)

    boots = np.array(boots)
    return {
        "emp_lambda_ci_low": float(np.percentile(boots, 2.5)),
        "emp_lambda_ci_high": float(np.percentile(boots, 97.5)),
        "emp_lambda_mean": float(np.mean(boots)),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("FIX 1: Nonparametric Tail Dependence Estimation")
    print("=" * 70)

    pit = pd.read_csv(DATA_DIR / "ollm_pit.csv", index_col=0)
    benchmarks = pit.columns.tolist()
    print(f"  Data: {pit.shape[0]} models x {pit.shape[1]} benchmarks")

    with open(RESULTS_DIR / "ollm_bivariate_fits.json") as f:
        copula_fits = json.load(f)
    copula_families = {(r["benchmark_1"], r["benchmark_2"]): r["family"] for r in copula_fits}

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))

    all_results = []
    for idx, (i, j) in enumerate(pairs):
        bm1, bm2 = benchmarks[i], benchmarks[j]
        u1, u2 = pit.iloc[:, i].values, pit.iloc[:, j].values

        family = copula_families.get((bm1, bm2), copula_families.get((bm2, bm1), "Unknown"))

        tail_conc = empirical_tail_concentration(u1, u2, thresholds)
        emp_lambda = empirical_lambda_U(u1, u2)
        boot_chi90 = bootstrap_tail_ci(u1, u2, threshold=0.90, n_boot=500)
        boot_chi95 = bootstrap_tail_ci(u1, u2, threshold=0.95, n_boot=500)
        boot_emp5 = bootstrap_emp_lambda_ci(u1, u2, k_pct=5, n_boot=300)
        boot_emp10 = bootstrap_emp_lambda_ci(u1, u2, k_pct=10, n_boot=300)

        is_frank = "Frank" in str(family)
        spearman = float(stats.spearmanr(u1, u2).statistic)

        cond_u1_top20 = u1 >= np.percentile(u1, 80)
        cond_u2_top20 = u2 >= np.percentile(u2, 80)
        either_top20 = cond_u1_top20 | cond_u2_top20
        if either_top20.sum() > 10:
            cond_spearman_20 = float(stats.spearmanr(u1[either_top20], u2[either_top20]).statistic)
        else:
            cond_spearman_20 = np.nan

        result = {
            "benchmark_1": bm1,
            "benchmark_2": bm2,
            "copula_family": family,
            "is_frank": is_frank,
            "spearman_full": spearman,
            "cond_spearman_20pct": cond_spearman_20,
            "tail_concentration": tail_conc,
            "emp_lambda_5pct": emp_lambda["emp_lambda_U_5pct"],
            "emp_lambda_10pct": emp_lambda["emp_lambda_U_10pct"],
            "emp_lambda_20pct": emp_lambda.get("emp_lambda_U_20pct", np.nan),
            "emp_trend": emp_lambda["trend"],
            "emp_slope": emp_lambda["slope_5_to_10"],
            "emp_5pct_ci": [boot_emp5["emp_lambda_ci_low"], boot_emp5["emp_lambda_ci_high"]],
            "emp_10pct_ci": [boot_emp10["emp_lambda_ci_low"], boot_emp10["emp_lambda_ci_high"]],
            "chi_90_mean": boot_chi90["chi_mean"],
            "chi_90_ci": [boot_chi90["chi_ci_low"], boot_chi90["chi_ci_high"]],
            "chi_95_mean": boot_chi95["chi_mean"],
            "chi_95_ci": [boot_chi95["chi_ci_low"], boot_chi95["chi_ci_high"]],
        }
        all_results.append(result)

        print(f"  [{idx+1}/{len(pairs)}] {bm1} x {bm2} [{family}]: "
              f"emp_lambda(5%)={emp_lambda['emp_lambda_U_5pct']:.3f} [{boot_emp5['emp_lambda_ci_low']:.3f},{boot_emp5['emp_lambda_ci_high']:.3f}], "
              f"trend={emp_lambda['trend']}, cond_rho20={cond_spearman_20:.3f}")

    with open(RESULTS_DIR / "nonparametric_tail_dependence.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY: Frank vs Non-Frank Pairs")
    print("=" * 70)

    frank_pairs = [r for r in all_results if r["is_frank"]]
    nonfrank_pairs = [r for r in all_results if not r["is_frank"]]

    print(f"\n  FRANK pairs (n={len(frank_pairs)}):")
    for r in frank_pairs:
        print(f"    {r['benchmark_1']:20s} x {r['benchmark_2']:12s}: "
              f"emp_lam5={r['emp_lambda_5pct']:.3f}, "
              f"chi95={r['chi_95_mean']:.3f}, cond_rho20={r['cond_spearman_20pct']:.3f}")

    print(f"\n  NON-FRANK pairs (n={len(nonfrank_pairs)}):")
    for r in nonfrank_pairs:
        print(f"    {r['benchmark_1']:20s} x {r['benchmark_2']:12s}: "
              f"emp_lam5={r['emp_lambda_5pct']:.3f}, "
              f"chi95={r['chi_95_mean']:.3f}, cond_rho20={r['cond_spearman_20pct']:.3f}")

    frank_emp5 = [r["emp_lambda_5pct"] for r in frank_pairs if not np.isnan(r["emp_lambda_5pct"])]
    nonfrank_emp5 = [r["emp_lambda_5pct"] for r in nonfrank_pairs if not np.isnan(r["emp_lambda_5pct"])]
    frank_cond = [r["cond_spearman_20pct"] for r in frank_pairs if not np.isnan(r["cond_spearman_20pct"])]
    nonfrank_cond = [r["cond_spearman_20pct"] for r in nonfrank_pairs if not np.isnan(r["cond_spearman_20pct"])]

    print(f"\n  === NONPARAMETRIC COMPARISON ===")
    print(f"  Empirical lambda_U at 5%:")
    print(f"    Frank mean:     {np.mean(frank_emp5):.4f}")
    print(f"    Non-Frank mean: {np.mean(nonfrank_emp5):.4f}")
    if frank_emp5 and nonfrank_emp5:
        mw_stat, mw_p = stats.mannwhitneyu(frank_emp5, nonfrank_emp5, alternative="less")
        print(f"    Mann-Whitney (Frank < Non-Frank): p = {mw_p:.4f}")

    print(f"\n  Conditional Spearman (top 20%):")
    print(f"    Frank mean:     {np.mean(frank_cond):.4f}")
    print(f"    Non-Frank mean: {np.mean(nonfrank_cond):.4f}")
    if frank_cond and nonfrank_cond:
        mw_stat, mw_p = stats.mannwhitneyu(frank_cond, nonfrank_cond, alternative="less")
        print(f"    Mann-Whitney (Frank < Non-Frank): p = {mw_p:.4f}")
    
    print(f"\n  Trend analysis (5% -> 10% slope):")
    frank_trends = [r["emp_trend"] for r in frank_pairs]
    nonfrank_trends = [r["emp_trend"] for r in nonfrank_pairs]
    print(f"    Frank trends: {dict(pd.Series(frank_trends).value_counts())}")
    print(f"    Non-Frank trends: {dict(pd.Series(nonfrank_trends).value_counts())}")

    print("\n" + "=" * 70)
    print("FIX 1 COMPLETE")
    print("=" * 70)
