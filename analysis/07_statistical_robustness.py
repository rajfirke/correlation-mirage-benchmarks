"""
Section 7: Statistical robustness analysis.
Addresses reviewer concerns: bootstrap CIs, GoF tests, simulation study, Delta-BIC.
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

COPULA_FAMILIES = [
    pv.BicopFamily.gaussian, pv.BicopFamily.student,
    pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
    pv.BicopFamily.frank, pv.BicopFamily.joe,
    pv.BicopFamily.bb1, pv.BicopFamily.bb7,
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
    """Compute closed-form tail dependence from fitted copula."""
    fam = cop.family
    par = cop.parameters
    lambda_L, lambda_U = 0.0, 0.0

    if fam == pv.BicopFamily.student:
        rho = par[0, 0]
        nu = par[0, 1] if par.shape[1] > 1 else par[1, 0]
        val = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_L = lambda_U = 2 * stats.t.cdf(-val, df=nu + 1)
    elif fam == pv.BicopFamily.clayton:
        theta = par[0, 0]
        lambda_L = 2 ** (-1.0 / theta) if theta > 0 else 0.0
    elif fam == pv.BicopFamily.gumbel:
        lambda_U = 2 - 2 ** (1.0 / par[0, 0])
    elif fam == pv.BicopFamily.joe:
        lambda_U = 2 - 2 ** (1.0 / par[0, 0])
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

    return {"lambda_L": float(lambda_L), "lambda_U": float(lambda_U)}


def bootstrap_tail_dependence(u_data: np.ndarray, n_bootstrap: int = 1000,
                              seed: int = 42) -> dict:
    """
    Bootstrap confidence intervals for tail dependence coefficients.
    Resamples rows, refits copula, extracts tail dependence.
    """
    rng = np.random.RandomState(seed)
    n = len(u_data)

    lambda_U_samples = []
    lambda_L_samples = []
    family_samples = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        u_boot = u_data[idx]
        u_boot = np.clip(u_boot, 1e-6, 1 - 1e-6)

        try:
            cop = pv.Bicop()
            controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
            cop.select(u_boot, controls)
            td = compute_tail_dependence(cop)
            lambda_U_samples.append(td["lambda_U"])
            lambda_L_samples.append(td["lambda_L"])
            family_samples.append(FAMILY_NAMES.get(cop.family, str(cop.family)))
        except Exception:
            continue

    if not lambda_U_samples:
        return {"error": "all bootstrap iterations failed"}

    lu = np.array(lambda_U_samples)
    ll = np.array(lambda_L_samples)

    return {
        "lambda_U_mean": float(np.mean(lu)),
        "lambda_U_median": float(np.median(lu)),
        "lambda_U_ci_lower": float(np.percentile(lu, 2.5)),
        "lambda_U_ci_upper": float(np.percentile(lu, 97.5)),
        "lambda_U_std": float(np.std(lu)),
        "lambda_L_mean": float(np.mean(ll)),
        "lambda_L_median": float(np.median(ll)),
        "lambda_L_ci_lower": float(np.percentile(ll, 2.5)),
        "lambda_L_ci_upper": float(np.percentile(ll, 97.5)),
        "lambda_L_std": float(np.std(ll)),
        "family_stability": {
            fam: family_samples.count(fam) / len(family_samples)
            for fam in set(family_samples)
        },
        "n_successful": len(lambda_U_samples),
        "prob_lambda_U_positive": float((lu > 0.01).mean()),
        "prob_lambda_L_positive": float((ll > 0.01).mean()),
    }


def simulation_study_gaussian_null(n_samples_list=[25, 30, 50, 75, 100],
                                   rho_values=[0.3, 0.5, 0.7, 0.9],
                                   n_sims=500, seed=42):
    """
    Simulation study: generate data from Gaussian copula, check how often
    BIC incorrectly selects a non-Gaussian family (false positive rate).
    """
    print("\n  === Simulation Study: Gaussian Null ===")
    rng = np.random.RandomState(seed)
    results = []

    for n in n_samples_list:
        for rho in rho_values:
            non_gaussian_count = 0
            has_tail_dep_count = 0

            for sim in range(n_sims):
                gauss_cop = pv.Bicop(family=pv.BicopFamily.gaussian,
                                     parameters=np.array([[rho]]))
                u = gauss_cop.simulate(n, seeds=[seed + sim])
                u = np.clip(u, 1e-6, 1 - 1e-6)

                try:
                    cop = pv.Bicop()
                    controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
                    cop.select(u, controls)

                    if cop.family != pv.BicopFamily.gaussian:
                        non_gaussian_count += 1
                        td = compute_tail_dependence(cop)
                        if max(td["lambda_U"], td["lambda_L"]) > 0.01:
                            has_tail_dep_count += 1
                except Exception:
                    continue

            fpr = non_gaussian_count / n_sims
            tail_fpr = has_tail_dep_count / n_sims

            results.append({
                "n": n, "rho": rho,
                "false_positive_rate": fpr,
                "tail_dep_false_positive": tail_fpr,
                "n_sims": n_sims,
            })
            print(f"    n={n:3d}, rho={rho:.1f}: "
                  f"FPR(non-Gaussian)={fpr:.3f}, "
                  f"FPR(tail_dep>0.01)={tail_fpr:.3f}")

    return results


def compute_delta_bic(u_data: np.ndarray) -> dict:
    """
    Compute BIC for Gaussian copula and best non-Gaussian family.
    Returns Delta-BIC = BIC(Gaussian) - BIC(best non-Gaussian).
    Positive Delta-BIC means non-Gaussian is preferred.
    """
    u_data = np.clip(u_data, 1e-6, 1 - 1e-6)

    gauss_cop = pv.Bicop()
    gauss_controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.gaussian])
    gauss_cop.select(u_data, gauss_controls)
    bic_gaussian = gauss_cop.bic(u_data)

    non_gauss_families = [f for f in COPULA_FAMILIES if f != pv.BicopFamily.gaussian]
    best_cop = pv.Bicop()
    best_controls = pv.FitControlsBicop(family_set=non_gauss_families)
    best_cop.select(u_data, best_controls)
    bic_best = best_cop.bic(u_data)

    all_cop = pv.Bicop()
    all_controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
    all_cop.select(u_data, all_controls)

    return {
        "bic_gaussian": float(bic_gaussian),
        "bic_best_nongaussian": float(bic_best),
        "bic_overall_best": float(all_cop.bic(u_data)),
        "delta_bic": float(bic_gaussian - bic_best),
        "overall_best_family": FAMILY_NAMES.get(all_cop.family, str(all_cop.family)),
        "best_nongaussian_family": FAMILY_NAMES.get(best_cop.family, str(best_cop.family)),
    }


def gof_test_cramer_von_mises(u_data: np.ndarray, cop: pv.Bicop,
                               n_bootstrap: int = 200, seed: int = 42) -> dict:
    """
    Goodness-of-fit test using Cramér-von Mises statistic on the empirical copula.
    Compares empirical copula to the fitted parametric copula via parametric bootstrap.
    """
    n = len(u_data)
    u_data = np.clip(u_data, 1e-6, 1 - 1e-6)

    def cvm_statistic(u, fitted_cop):
        n_obs = len(u)
        empirical = np.zeros(n_obs)
        for k in range(n_obs):
            empirical[k] = np.mean((u[:, 0] <= u[k, 0]) & (u[:, 1] <= u[k, 1]))

        theoretical = fitted_cop.cdf(u)
        return float(np.sum((empirical - theoretical) ** 2))

    obs_stat = cvm_statistic(u_data, cop)

    rng = np.random.RandomState(seed)
    boot_stats = []
    for b in range(n_bootstrap):
        u_sim = cop.simulate(n, seeds=[seed + b])
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)

        try:
            boot_cop = pv.Bicop()
            controls = pv.FitControlsBicop(family_set=[cop.family])
            boot_cop.select(u_sim, controls)
            boot_stat = cvm_statistic(u_sim, boot_cop)
            boot_stats.append(boot_stat)
        except Exception:
            continue

    if not boot_stats:
        return {"p_value": np.nan, "test_statistic": obs_stat}

    p_value = float(np.mean(np.array(boot_stats) >= obs_stat))

    return {
        "test_statistic": float(obs_stat),
        "p_value": p_value,
        "n_bootstrap": len(boot_stats),
        "reject_at_05": p_value < 0.05,
    }


def run_robustness_on_key_pairs(wide: pd.DataFrame, pit: pd.DataFrame):
    """Run bootstrap CIs and GoF tests on the deceptive/surprising pairs from the paper."""
    print("\n  === Bootstrap CIs for Key Pairs ===")

    key_pairs_from_paper = [
        ("MMMU", "MathVista"),
        ("GPQA Diamond", "Aider Polyglot"),
        ("GPQA Main", "Aider Polyglot"),
        ("MATH-500", "SafetyBench"),
        ("AlpacaEval 2.0 LC", "SafetyBench"),
    ]

    all_bm = set(wide.columns.tolist()) | set(pit.columns.tolist())
    results = []

    for bm1_target, bm2_target in key_pairs_from_paper:
        bm1 = next((b for b in all_bm if bm1_target.lower() in b.lower()), None)
        bm2 = next((b for b in all_bm if bm2_target.lower() in b.lower()), None)

        if bm1 is None or bm2 is None:
            print(f"    SKIP: {bm1_target} x {bm2_target} (not found)")
            continue

        if bm1 in pit.columns and bm2 in pit.columns:
            pair_data = pit[[bm1, bm2]].dropna().values
        elif bm1 in wide.columns and bm2 in wide.columns:
            raw = wide[[bm1, bm2]].dropna()
            n = len(raw)
            pair_data = raw.rank(method="average").values / (n + 1)
        else:
            print(f"    SKIP: {bm1} x {bm2} (not in same matrix)")
            continue

        if len(pair_data) < 15:
            print(f"    SKIP: {bm1} x {bm2} (only {len(pair_data)} obs)")
            continue

        pair_data = np.clip(pair_data, 1e-6, 1 - 1e-6)
        print(f"\n    {bm1} x {bm2} (n={len(pair_data)}):")

        boot = bootstrap_tail_dependence(pair_data, n_bootstrap=1000)
        print(f"      lambda_U: {boot['lambda_U_mean']:.3f} "
              f"[{boot['lambda_U_ci_lower']:.3f}, {boot['lambda_U_ci_upper']:.3f}]")
        print(f"      lambda_L: {boot['lambda_L_mean']:.3f} "
              f"[{boot['lambda_L_ci_lower']:.3f}, {boot['lambda_L_ci_upper']:.3f}]")
        print(f"      P(lambda_U > 0.01) = {boot['prob_lambda_U_positive']:.3f}")
        print(f"      Family stability: {boot['family_stability']}")

        cop = pv.Bicop()
        controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
        cop.select(pair_data, controls)

        gof = gof_test_cramer_von_mises(pair_data, cop, n_bootstrap=200)
        print(f"      GoF (CvM): stat={gof['test_statistic']:.4f}, p={gof['p_value']:.3f}")

        dbic = compute_delta_bic(pair_data)
        print(f"      Delta-BIC: {dbic['delta_bic']:.2f} "
              f"(best={dbic['overall_best_family']})")

        results.append({
            "benchmark_1": bm1,
            "benchmark_2": bm2,
            "n_obs": len(pair_data),
            "bootstrap": boot,
            "gof_test": gof,
            "delta_bic": dbic,
        })

    return results


def run_delta_bic_distribution(wide: pd.DataFrame, min_obs: int = 25):
    """Compute Delta-BIC for all pairwise fits to get distribution."""
    print("\n  === Delta-BIC Distribution ===")
    benchmarks = wide.columns.tolist()
    bench_fill = wide.notna().sum(axis=0)
    top_bm = bench_fill[bench_fill >= min_obs].index.tolist()

    pairs = list(itertools.combinations(range(len(top_bm)), 2))
    delta_bics = []

    for idx, (i, j) in enumerate(pairs):
        bm_i, bm_j = top_bm[i], top_bm[j]
        pair_data = wide[[bm_i, bm_j]].dropna()
        if len(pair_data) < min_obs:
            continue

        n = len(pair_data)
        u = pair_data.rank(method="average").values / (n + 1)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        try:
            dbic = compute_delta_bic(u)
            delta_bics.append({
                "benchmark_1": bm_i,
                "benchmark_2": bm_j,
                "n_obs": n,
                **dbic,
            })
        except Exception:
            continue

    df = pd.DataFrame(delta_bics)
    if len(df) == 0:
        print("    No pairs computed")
        return []

    strong = (df["delta_bic"] > 6).sum()
    moderate = ((df["delta_bic"] > 2) & (df["delta_bic"] <= 6)).sum()
    weak = ((df["delta_bic"] > 0) & (df["delta_bic"] <= 2)).sum()
    gaussian_wins = (df["delta_bic"] <= 0).sum()

    print(f"    Total pairs: {len(df)}")
    print(f"    Strong non-Gaussian (ΔBIC > 6): {strong} ({strong/len(df)*100:.1f}%)")
    print(f"    Moderate non-Gaussian (2 < ΔBIC ≤ 6): {moderate} ({moderate/len(df)*100:.1f}%)")
    print(f"    Weak non-Gaussian (0 < ΔBIC ≤ 2): {weak} ({weak/len(df)*100:.1f}%)")
    print(f"    Gaussian preferred (ΔBIC ≤ 0): {gaussian_wins} ({gaussian_wins/len(df)*100:.1f}%)")
    print(f"    Median ΔBIC: {df['delta_bic'].median():.2f}")
    print(f"    Mean ΔBIC: {df['delta_bic'].mean():.2f}")

    return delta_bics


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 7: Statistical Robustness Analysis")
    print("=" * 70)

    pit = pd.read_csv(DATA_DIR / "pit_transformed.csv", index_col=0)
    wide = pd.read_csv(DATA_DIR / "benchpress_wide_all.csv", index_col=0)
    print(f"  PIT matrix: {pit.shape}")
    print(f"  Wide matrix: {wide.shape}")

    print("\n--- 7.1: Simulation Study (Gaussian Null) ---")
    sim_results = simulation_study_gaussian_null(
        n_samples_list=[25, 30, 42, 50, 75, 100],
        rho_values=[0.3, 0.5, 0.7, 0.9],
        n_sims=500,
    )
    with open(RESULTS_DIR / "simulation_gaussian_null.json", "w") as f:
        json.dump(sim_results, f, indent=2)

    print("\n--- 7.2: Bootstrap CIs for Key Pairs ---")
    key_pair_results = run_robustness_on_key_pairs(wide, pit)
    with open(RESULTS_DIR / "bootstrap_key_pairs.json", "w") as f:
        json.dump(key_pair_results, f, indent=2)

    print("\n--- 7.3: Delta-BIC Distribution ---")
    dbic_results = run_delta_bic_distribution(wide)
    with open(RESULTS_DIR / "delta_bic_distribution.json", "w") as f:
        json.dump(dbic_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SECTION 7 COMPLETE")
    print("=" * 70)
