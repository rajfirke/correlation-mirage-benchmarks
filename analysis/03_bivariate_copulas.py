"""
Section 3: Bivariate copula fitting, tail dependence, and deceptive pair detection.
This is the CORE CONTRIBUTION of the paper.
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

COPULA_FAMILIES = [
    pv.BicopFamily.gaussian,
    pv.BicopFamily.student,
    pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.frank,
    pv.BicopFamily.joe,
    pv.BicopFamily.bb1,
    pv.BicopFamily.bb7,
]

FAMILY_NAMES = {
    pv.BicopFamily.gaussian: "Gaussian",
    pv.BicopFamily.student: "Student-t",
    pv.BicopFamily.clayton: "Clayton",
    pv.BicopFamily.gumbel: "Gumbel",
    pv.BicopFamily.frank: "Frank",
    pv.BicopFamily.joe: "Joe",
    pv.BicopFamily.bb1: "BB1",
    pv.BicopFamily.bb7: "BB7",
}


def compute_tail_dependence(cop: pv.Bicop) -> dict:
    """
    Compute upper and lower tail dependence coefficients from a fitted copula.
    Uses the closed-form expressions for each copula family.
    For families without closed-form, use empirical approximation.
    """
    fam = cop.family
    par = cop.parameters

    lambda_L = 0.0
    lambda_U = 0.0

    if fam == pv.BicopFamily.gaussian:
        lambda_L = 0.0
        lambda_U = 0.0
    elif fam == pv.BicopFamily.student:
        rho = par[0, 0]
        nu = par[0, 1] if par.shape[1] > 1 else par[1, 0]
        from scipy.stats import t as tdist
        val = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_L = 2 * tdist.cdf(-val, df=nu + 1)
        lambda_U = lambda_L
    elif fam == pv.BicopFamily.clayton:
        theta = par[0, 0]
        if theta > 0:
            lambda_L = 2 ** (-1.0 / theta)
        lambda_U = 0.0
    elif fam == pv.BicopFamily.gumbel:
        theta = par[0, 0]
        lambda_U = 2 - 2 ** (1.0 / theta)
        lambda_L = 0.0
    elif fam == pv.BicopFamily.frank:
        lambda_L = 0.0
        lambda_U = 0.0
    elif fam == pv.BicopFamily.joe:
        theta = par[0, 0]
        lambda_U = 2 - 2 ** (1.0 / theta)
        lambda_L = 0.0
    elif fam == pv.BicopFamily.bb1:
        theta = par[0, 0]
        delta = par[0, 1] if par.shape[1] > 1 else par[1, 0]
        lambda_U = 2 - 2 ** (1.0 / delta)
        lambda_L = 2 ** (-1.0 / (theta * delta)) if theta * delta > 0 else 0.0
    elif fam == pv.BicopFamily.bb7:
        theta = par[0, 0]
        delta = par[0, 1] if par.shape[1] > 1 else par[1, 0]
        lambda_U = 2 - 2 ** (1.0 / theta)
        lambda_L = 2 ** (-1.0 / delta) if delta > 0 else 0.0
    else:
        lambda_L = _empirical_tail_dep(cop, "lower")
        lambda_U = _empirical_tail_dep(cop, "upper")

    return {"lambda_L": float(lambda_L), "lambda_U": float(lambda_U)}


def _empirical_tail_dep(cop: pv.Bicop, tail: str, n_sim: int = 50000, q: float = 0.05):
    """Empirical tail dependence via simulation."""
    u = cop.simulate(n_sim, seeds=[42])
    if tail == "lower":
        mask = u[:, 0] <= q
        if mask.sum() == 0:
            return 0.0
        return float((u[mask, 1] <= q).mean())
    else:
        mask = u[:, 0] >= (1 - q)
        if mask.sum() == 0:
            return 0.0
        return float((u[mask, 1] >= (1 - q)).mean())


def fit_bivariate_copulas(pit: pd.DataFrame):
    """
    Section 3.1: Fit parametric copula families to ALL benchmark pairs.
    Returns a list of dicts with fit results.
    """
    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    print(f"  Fitting {len(pairs)} benchmark pairs...")

    results = []
    for idx, (i, j) in enumerate(pairs):
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u_data = pit[[bm_i, bm_j]].dropna().values

        if len(u_data) < 10:
            continue

        u_data = np.clip(u_data, 1e-6, 1 - 1e-6)

        cop = pv.Bicop()
        controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
        cop.select(u_data, controls)

        td = compute_tail_dependence(cop)
        ktau = float(stats.kendalltau(u_data[:, 0], u_data[:, 1]).statistic)
        spearman = float(stats.spearmanr(u_data[:, 0], u_data[:, 1]).statistic)

        results.append({
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "n_obs": int(len(u_data)),
            "family": FAMILY_NAMES.get(cop.family, str(cop.family)),
            "family_id": str(cop.family),
            "parameters": cop.parameters.tolist(),
            "loglik": float(cop.loglik(u_data)),
            "aic": float(cop.aic(u_data)),
            "bic": float(cop.bic(u_data)),
            "kendall_tau": ktau,
            "spearman_rho": spearman,
            "lambda_L": td["lambda_L"],
            "lambda_U": td["lambda_U"],
            "tail_asymmetry": abs(td["lambda_U"] - td["lambda_L"]),
        })

        if (idx + 1) % 5 == 0 or idx == len(pairs) - 1:
            print(f"    [{idx+1}/{len(pairs)}] {bm_i} x {bm_j}: "
                  f"{FAMILY_NAMES.get(cop.family, '?')} | "
                  f"rho={spearman:.3f} | lambda_U={td['lambda_U']:.3f} | lambda_L={td['lambda_L']:.3f}")

    return results


def fit_pairwise_from_wide(wide: pd.DataFrame):
    """
    Fit bivariate copulas to ALL pairs in the FULL wide matrix (not just complete-case).
    Uses pairwise-complete observations.
    """
    benchmarks = wide.columns.tolist()
    min_obs = 25

    bench_fill = wide.notna().sum(axis=0).sort_values(ascending=False)
    top_benchmarks = bench_fill[bench_fill >= min_obs].index.tolist()
    print(f"  Benchmarks with >= {min_obs} models: {len(top_benchmarks)}")

    pairs = list(itertools.combinations(range(len(top_benchmarks)), 2))
    print(f"  Pairwise copula fits to attempt: {len(pairs)}")

    results = []
    for idx, (i, j) in enumerate(pairs):
        bm_i, bm_j = top_benchmarks[i], top_benchmarks[j]
        pair_data = wide[[bm_i, bm_j]].dropna()

        if len(pair_data) < min_obs:
            continue

        n = len(pair_data)
        u = pair_data.rank(method="average").values / (n + 1)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        try:
            cop = pv.Bicop()
            controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
            cop.select(u, controls)
        except Exception:
            continue

        td = compute_tail_dependence(cop)
        spearman = float(stats.spearmanr(pair_data.iloc[:, 0], pair_data.iloc[:, 1]).statistic)

        results.append({
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "n_obs": int(n),
            "family": FAMILY_NAMES.get(cop.family, str(cop.family)),
            "spearman_rho": spearman,
            "lambda_L": td["lambda_L"],
            "lambda_U": td["lambda_U"],
            "tail_asymmetry": abs(td["lambda_U"] - td["lambda_L"]),
        })

    print(f"  Successfully fitted: {len(results)} pairs")
    return results


def identify_deceptive_pairs(results: list):
    """
    Section 3.3: Find benchmark pairs where correlation and tail dependence disagree.
    'Deceptive' = high correlation but low tail dependence (or vice versa).
    """
    df = pd.DataFrame(results)
    df["max_tail"] = df[["lambda_L", "lambda_U"]].max(axis=1)

    df["deceptive_score"] = df["spearman_rho"].abs() - df["max_tail"]

    high_corr_low_tail = df.nlargest(10, "deceptive_score")
    low_corr_high_tail = df.nsmallest(10, "deceptive_score")

    print("\n  === DECEPTIVE PAIRS: High Correlation but LOW Tail Dependence ===")
    print("  (These benchmarks LOOK redundant but are INDEPENDENT at the extremes)")
    for _, row in high_corr_low_tail.head(5).iterrows():
        print(f"    {row['benchmark_1']} x {row['benchmark_2']}: "
              f"rho={row['spearman_rho']:.3f}, lambda_max={row['max_tail']:.3f}, "
              f"deceptive={row['deceptive_score']:.3f} [{row['family']}]")

    print("\n  === SURPRISING PAIRS: Low Correlation but HIGH Tail Dependence ===")
    print("  (These benchmarks look independent but cluster at the extremes)")
    for _, row in low_corr_high_tail.head(5).iterrows():
        print(f"    {row['benchmark_1']} x {row['benchmark_2']}: "
              f"rho={row['spearman_rho']:.3f}, lambda_max={row['max_tail']:.3f}, "
              f"deceptive={row['deceptive_score']:.3f} [{row['family']}]")

    return {
        "high_corr_low_tail": high_corr_low_tail.to_dict("records"),
        "low_corr_high_tail": low_corr_high_tail.to_dict("records"),
    }


def build_tail_dependence_matrix(results: list, benchmarks: list):
    """Build matrices of lambda_U, lambda_L, and Spearman for heatmap comparison."""
    n = len(benchmarks)
    lambda_U_mat = np.zeros((n, n))
    lambda_L_mat = np.zeros((n, n))
    spearman_mat = np.eye(n)

    bm_idx = {bm: i for i, bm in enumerate(benchmarks)}

    for r in results:
        b1, b2 = r["benchmark_1"], r["benchmark_2"]
        if b1 in bm_idx and b2 in bm_idx:
            i, j = bm_idx[b1], bm_idx[b2]
            lambda_U_mat[i, j] = lambda_U_mat[j, i] = r["lambda_U"]
            lambda_L_mat[i, j] = lambda_L_mat[j, i] = r["lambda_L"]
            spearman_mat[i, j] = spearman_mat[j, i] = r["spearman_rho"]

    for name, mat in [("lambda_U", lambda_U_mat), ("lambda_L", lambda_L_mat), ("spearman", spearman_mat)]:
        df = pd.DataFrame(mat, index=benchmarks, columns=benchmarks)
        df.to_csv(RESULTS_DIR / f"matrix_{name}.csv")

    return lambda_U_mat, lambda_L_mat, spearman_mat


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 3: Bivariate Copula Fitting & Tail Dependence")
    print("=" * 70)

    pit = pd.read_csv(DATA_DIR / "pit_transformed.csv", index_col=0)
    wide = pd.read_csv(DATA_DIR / "benchpress_wide_all.csv", index_col=0)
    print(f"  PIT matrix: {pit.shape}")
    print(f"  Full wide matrix: {wide.shape}")

    print("\n--- 3.1: Bivariate Copula Fitting (Complete-Case) ---")
    cc_results = fit_bivariate_copulas(pit)

    with open(RESULTS_DIR / "bivariate_fits_complete.json", "w") as f:
        json.dump(cc_results, f, indent=2)
    print(f"  Saved {len(cc_results)} pair fits -> bivariate_fits_complete.json")

    print("\n--- 3.1b: Pairwise Copula Fitting (Full Matrix) ---")
    pw_results = fit_pairwise_from_wide(wide)

    with open(RESULTS_DIR / "bivariate_fits_pairwise.json", "w") as f:
        json.dump(pw_results, f, indent=2)
    print(f"  Saved {len(pw_results)} pair fits -> bivariate_fits_pairwise.json")

    print("\n--- 3.2: Tail Dependence Matrices ---")
    benchmarks = pit.columns.tolist()
    lU, lL, sp = build_tail_dependence_matrix(cc_results, benchmarks)
    print(f"  lambda_U range: [{lU[np.triu_indices_from(lU, k=1)].min():.3f}, "
          f"{lU[np.triu_indices_from(lU, k=1)].max():.3f}]")
    print(f"  lambda_L range: [{lL[np.triu_indices_from(lL, k=1)].min():.3f}, "
          f"{lL[np.triu_indices_from(lL, k=1)].max():.3f}]")
    print(f"  Spearman range: [{sp[np.triu_indices_from(sp, k=1)].min():.3f}, "
          f"{sp[np.triu_indices_from(sp, k=1)].max():.3f}]")

    print("\n--- 3.3: Deceptive Pair Detection ---")
    all_results = cc_results + pw_results
    deceptive = identify_deceptive_pairs(all_results)

    with open(RESULTS_DIR / "deceptive_pairs.json", "w") as f:
        json.dump(deceptive, f, indent=2, default=str)

    df_all = pd.DataFrame(all_results)
    family_counts = df_all["family"].value_counts()
    print(f"\n  Copula Family Distribution:")
    for fam, n in family_counts.items():
        print(f"    {fam}: {n} pairs ({n/len(df_all)*100:.1f}%)")

    gaussian_pct = family_counts.get("Gaussian", 0) / len(df_all) * 100
    print(f"\n  KEY INSIGHT: {100-gaussian_pct:.1f}% of pairs are best modeled by "
          f"NON-Gaussian copulas (which have non-zero tail dependence).")

    print("\n  Tail Dependence Summary:")
    has_upper = (df_all["lambda_U"] > 0.01).sum()
    has_lower = (df_all["lambda_L"] > 0.01).sum()
    has_asymmetric = (df_all["tail_asymmetry"] > 0.05).sum()
    print(f"    Pairs with upper tail dependence (lambda_U > 0.01): {has_upper}/{len(df_all)}")
    print(f"    Pairs with lower tail dependence (lambda_L > 0.01): {has_lower}/{len(df_all)}")
    print(f"    Pairs with asymmetric tail dependence: {has_asymmetric}/{len(df_all)}")

    print("\n" + "=" * 70)
    print("SECTION 3 COMPLETE")
    print(f"  Complete-case pairs: {len(cc_results)}")
    print(f"  Pairwise pairs: {len(pw_results)}")
    print(f"  Total pairs analyzed: {len(all_results)}")
    print("=" * 70)
