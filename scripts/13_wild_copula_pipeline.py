"""
WILD (kensho) Copula Pipeline — Replication study on independent evaluation framework.
65 models × 27 tasks, 100% fill. C(27,2) = 351 bivariate pairs.

Validates whether the "deceptive pairs" phenomenon and conditional Spearman
collapse found in OLLM v2 replicate on a completely different evaluation setup.

NOTE: n=65 is small, so results will be noisier than OLLM v2 (n=4,576).
We look for directional consistency, not identical effect sizes.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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
    pv.BicopFamily.indep: "Independence",
}


def compute_tail_dependence(cop):
    fam = cop.family
    par = cop.parameters
    lL, lU = 0.0, 0.0
    if fam == pv.BicopFamily.student:
        rho = par[0, 0]
        nu = par[0, 1] if par.shape[1] > 1 else par[1, 0]
        val = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lL = lU = 2 * stats.t.cdf(-val, df=nu + 1)
    elif fam == pv.BicopFamily.clayton:
        lL = 2 ** (-1.0 / par[0, 0]) if par[0, 0] > 0 else 0.0
    elif fam == pv.BicopFamily.gumbel:
        lU = 2 - 2 ** (1.0 / par[0, 0])
    elif fam == pv.BicopFamily.joe:
        lU = 2 - 2 ** (1.0 / par[0, 0])
    elif fam == pv.BicopFamily.bb1:
        theta, delta = par[0, 0], (par[0, 1] if par.shape[1] > 1 else par[1, 0])
        lU = 2 - 2 ** (1.0 / delta)
        lL = 2 ** (-1.0 / (theta * delta)) if theta * delta > 0 else 0.0
    elif fam == pv.BicopFamily.bb7:
        theta, delta = par[0, 0], (par[0, 1] if par.shape[1] > 1 else par[1, 0])
        lU = 2 - 2 ** (1.0 / theta)
        lL = 2 ** (-1.0 / delta) if delta > 0 else 0.0
    return {"lambda_L": float(lL), "lambda_U": float(lU)}


# ============================================================
# SECTION 1: DATA PREPARATION
# ============================================================
def prepare_wild_data():
    print("=" * 70)
    print("SECTION 1: WILD Data Preparation")
    print("=" * 70)

    score_matrix = pd.read_csv(DATA_DIR / "wild_score_matrix.csv")
    score_matrix = score_matrix.set_index("model")
    score_matrix = score_matrix.dropna(axis=0, how="any")

    print(f"  Score matrix: {score_matrix.shape[0]} models × {score_matrix.shape[1]} tasks")
    print(f"  Fill rate: 100%")
    print(f"  Tasks: {list(score_matrix.columns)}")
    print(f"  Score ranges:")
    for bm in score_matrix.columns:
        print(f"    {bm}: [{score_matrix[bm].min():.4f}, {score_matrix[bm].max():.4f}]")

    n = len(score_matrix)
    pit = score_matrix.rank(method="average") / (n + 1)

    print(f"\n  PIT transform: {pit.shape[0]} models × {pit.shape[1]} tasks")
    print(f"  PIT range check: [{pit.values.min():.4f}, {pit.values.max():.4f}]")

    return score_matrix, pit


# ============================================================
# SECTION 2: BIVARIATE COPULA FITTING (351 pairs)
# ============================================================
def fit_all_bivariate(pit):
    print("\n" + "=" * 70)
    print("SECTION 2: Bivariate Copula Fitting (n={}, {} tasks)".format(
        len(pit), len(pit.columns)))
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    n_pairs = len(pairs)
    print(f"  Pairs to fit: C({len(benchmarks)},2) = {n_pairs}")

    results = []
    for idx, (i, j) in enumerate(pairs):
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values
        u = np.clip(u, 1e-6, 1 - 1e-6)

        try:
            cop = pv.Bicop()
            controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
            cop.select(u, controls)

            td = compute_tail_dependence(cop)
            spearman = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

            gauss_cop = pv.Bicop()
            gauss_controls = pv.FitControlsBicop(
                family_set=[pv.BicopFamily.gaussian])
            gauss_cop.select(u, gauss_controls)
            bic_gauss = gauss_cop.bic(u)

            non_gauss_fams = [f for f in COPULA_FAMILIES
                              if f != pv.BicopFamily.gaussian]
            ng_cop = pv.Bicop()
            ng_controls = pv.FitControlsBicop(family_set=non_gauss_fams)
            ng_cop.select(u, ng_controls)
            bic_ng = ng_cop.bic(u)

            delta_bic = float(bic_gauss - bic_ng)

            result = {
                "benchmark_1": bm_i, "benchmark_2": bm_j,
                "n_obs": len(u),
                "family": FAMILY_NAMES.get(cop.family, str(cop.family)),
                "parameters": cop.parameters.tolist(),
                "spearman_rho": spearman,
                "lambda_U": td["lambda_U"], "lambda_L": td["lambda_L"],
                "lambda_max": max(td["lambda_U"], td["lambda_L"]),
                "tail_asymmetry": abs(td["lambda_U"] - td["lambda_L"]),
                "delta_bic": delta_bic,
                "bic_gaussian": float(bic_gauss),
                "bic_best": float(cop.bic(u)),
                "loglik": float(cop.loglik(u)),
            }
        except Exception as e:
            result = {
                "benchmark_1": bm_i, "benchmark_2": bm_j,
                "n_obs": len(u),
                "family": "FAILED",
                "parameters": [],
                "spearman_rho": float(stats.spearmanr(u[:, 0], u[:, 1]).statistic),
                "lambda_U": 0.0, "lambda_L": 0.0, "lambda_max": 0.0,
                "tail_asymmetry": 0.0, "delta_bic": 0.0,
                "bic_gaussian": np.nan, "bic_best": np.nan, "loglik": np.nan,
                "error": str(e),
            }

        results.append(result)

        if (idx + 1) % 50 == 0 or idx == 0 or idx == n_pairs - 1:
            fam = result["family"]
            rho = result["spearman_rho"]
            print(f"  [{idx+1:3d}/{n_pairs}] {bm_i:20s} x {bm_j:20s}: "
                  f"{fam:10s} | rho={rho:.3f} | "
                  f"lU={result['lambda_U']:.3f} | lL={result['lambda_L']:.3f} | "
                  f"ΔBIC={result['delta_bic']:.1f}")

    return results


# ============================================================
# SECTION 3: CONDITIONAL SPEARMAN (upper 20%)
# ============================================================
def conditional_spearman_analysis(pit, min_obs=8):
    """Conditional Spearman in the upper tail.

    With n=65, the upper 20% per task has ~13 models. The OR-mask
    (either task in the top 20%) gives roughly 20-25 observations,
    enough for a noisy but directionally informative Spearman estimate.
    We use min_obs=8 as a floor.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Conditional Spearman Analysis (n={})".format(len(pit)))
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))

    results = []
    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values

        spearman_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

        upper_mask = (u[:, 0] >= 0.8) | (u[:, 1] >= 0.8)
        if upper_mask.sum() >= min_obs:
            res = stats.spearmanr(u[upper_mask, 0], u[upper_mask, 1])
            spearman_upper = float(res.statistic)
            p_upper = float(res.pvalue)
        else:
            spearman_upper = np.nan
            p_upper = np.nan

        lower_mask = (u[:, 0] <= 0.2) | (u[:, 1] <= 0.2)
        if lower_mask.sum() >= min_obs:
            res = stats.spearmanr(u[lower_mask, 0], u[lower_mask, 1])
            spearman_lower = float(res.statistic)
        else:
            spearman_lower = np.nan

        spearman_drop = spearman_full - spearman_upper if not np.isnan(spearman_upper) else np.nan

        results.append({
            "benchmark_1": bm_i, "benchmark_2": bm_j,
            "spearman_full": spearman_full,
            "spearman_upper_20pct": spearman_upper,
            "spearman_upper_pvalue": p_upper,
            "spearman_lower_20pct": spearman_lower,
            "spearman_drop": spearman_drop,
            "n_upper": int(upper_mask.sum()),
            "n_lower": int(lower_mask.sum()),
        })

    return results


# ============================================================
# SECTION 4: DECEPTIVE PAIRS & SUMMARY
# ============================================================
def analyze_deceptive_pairs(bivariate_results, cond_results):
    print("\n" + "=" * 70)
    print("SECTION 4: Deceptive Pairs Analysis")
    print("=" * 70)

    valid = [r for r in bivariate_results if r["family"] != "FAILED"]
    total = len(valid)

    n_nongauss = sum(1 for r in valid if r["family"] != "Gaussian")
    n_strong_bic = sum(1 for r in valid if r["delta_bic"] > 6)
    n_moderate_bic = sum(1 for r in valid if r["delta_bic"] > 2)

    deceptive = [r for r in valid
                 if r["spearman_rho"] > 0.7 and r["lambda_max"] < 0.05]
    n_deceptive = len(deceptive)

    deceptive_sorted = sorted(deceptive,
                              key=lambda r: r["spearman_rho"], reverse=True)

    family_counts = {}
    for r in valid:
        fam = r["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1

    rhos = [r["spearman_rho"] for r in valid]
    dbics = [r["delta_bic"] for r in valid]
    lUs = [r["lambda_U"] for r in valid]
    lLs = [r["lambda_L"] for r in valid]

    print(f"\n  --- FAMILY DISTRIBUTION ---")
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {fam:12s}: {cnt:3d} / {total} ({cnt/total*100:.1f}%)")

    print(f"\n  --- NON-GAUSSIANITY ---")
    print(f"    Non-Gaussian pairs:    {n_nongauss} / {total} ({n_nongauss/total*100:.1f}%)")
    print(f"    ΔBIC > 2 (positive):   {n_moderate_bic} / {total}")
    print(f"    ΔBIC > 6 (strong):     {n_strong_bic} / {total}")
    print(f"    Mean ΔBIC:             {np.mean(dbics):.2f}")
    print(f"    Median ΔBIC:           {np.median(dbics):.2f}")

    print(f"\n  --- TAIL DEPENDENCE ---")
    print(f"    Mean λ_U:              {np.mean(lUs):.4f}")
    print(f"    Mean λ_L:              {np.mean(lLs):.4f}")
    n_any_tail = sum(1 for r in valid if r["lambda_max"] > 0.05)
    n_no_tail = sum(1 for r in valid if r["lambda_max"] < 0.05)
    print(f"    Pairs with λ_max > 0.05:  {n_any_tail} / {total}")
    print(f"    Pairs with λ_max < 0.05:  {n_no_tail} / {total}")

    print(f"\n  --- DECEPTIVE PAIRS (ρ > 0.7, λ_max < 0.05) ---")
    print(f"    Count: {n_deceptive} / {total}")
    if deceptive_sorted:
        print(f"\n    Top 5 most deceptive (sorted by ρ):")
        for k, r in enumerate(deceptive_sorted[:5]):
            print(f"    {k+1}. {r['benchmark_1']:20s} x {r['benchmark_2']:20s} | "
                  f"ρ={r['spearman_rho']:.3f} | family={r['family']:10s} | "
                  f"λ_U={r['lambda_U']:.4f} | λ_L={r['lambda_L']:.4f} | "
                  f"ΔBIC={r['delta_bic']:.1f}")

    print(f"\n  --- CONDITIONAL SPEARMAN COLLAPSE ---")
    drops = [c["spearman_drop"] for c in cond_results
             if not np.isnan(c.get("spearman_drop", np.nan))]
    if drops:
        mean_drop = np.mean(drops)
        median_drop = np.median(drops)
        n_collapse = sum(1 for d in drops if d > 0.15)
        n_severe = sum(1 for d in drops if d > 0.3)
        n_valid_cond = len(drops)

        print(f"    Valid pairs (enough tail obs): {n_valid_cond} / {len(cond_results)}")
        print(f"    Mean Spearman drop (full→upper 20%): {mean_drop:.3f}")
        print(f"    Median Spearman drop:                {median_drop:.3f}")
        print(f"    Pairs with drop > 0.15:              {n_collapse} / {n_valid_cond}")
        print(f"    Pairs with drop > 0.30 (severe):     {n_severe} / {n_valid_cond}")

        top_collapse = sorted(
            [c for c in cond_results
             if not np.isnan(c.get("spearman_drop", np.nan))],
            key=lambda c: c["spearman_drop"], reverse=True)
        print(f"\n    Top 5 largest Spearman collapses:")
        for k, c in enumerate(top_collapse[:5]):
            print(f"    {k+1}. {c['benchmark_1']:20s} x {c['benchmark_2']:20s} | "
                  f"ρ_full={c['spearman_full']:.3f} → ρ_upper={c['spearman_upper_20pct']:.3f} "
                  f"(drop={c['spearman_drop']:.3f}, n_upper={c['n_upper']})")

        collapse_replicates = mean_drop > 0.05
        print(f"\n    REPLICATION VERDICT: ", end="")
        if collapse_replicates:
            print(f"YES — conditional Spearman collapse replicates on WILD "
                  f"(mean drop = {mean_drop:.3f})")
        else:
            print(f"NO — no clear conditional Spearman collapse on WILD "
                  f"(mean drop = {mean_drop:.3f})")
    else:
        print("    Not enough data for conditional Spearman analysis.")

    return {
        "total_pairs": total,
        "n_nongaussian": n_nongauss,
        "n_delta_bic_gt_6": n_strong_bic,
        "n_delta_bic_gt_2": n_moderate_bic,
        "n_deceptive": n_deceptive,
        "family_counts": family_counts,
        "mean_delta_bic": float(np.mean(dbics)),
        "mean_spearman_drop": float(np.mean(drops)) if drops else None,
        "median_spearman_drop": float(np.median(drops)) if drops else None,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("WILD (kensho) COPULA PIPELINE — REPLICATION STUDY")
    print("65 models × 27 tasks × 100% fill rate")
    print("C(27,2) = 351 bivariate pairs")
    print("=" * 70)

    score_matrix, pit = prepare_wild_data()

    bivariate_results = fit_all_bivariate(pit)

    cond_results = conditional_spearman_analysis(pit)

    summary = analyze_deceptive_pairs(bivariate_results, cond_results)

    output = {
        "dataset": "kensho/WILD",
        "n_models": int(score_matrix.shape[0]),
        "n_tasks": int(score_matrix.shape[1]),
        "n_pairs": len(bivariate_results),
        "tasks": list(score_matrix.columns),
        "bivariate_fits": bivariate_results,
        "conditional_spearman": cond_results,
        "summary": summary,
    }
    out_path = RESULTS_DIR / "wild_bivariate_fits.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    print("\n\n" + "=" * 70)
    print("WILD PIPELINE COMPLETE")
    print("=" * 70)
