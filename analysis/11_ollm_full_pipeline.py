"""
OLLM v2 Full Copula Pipeline — Primary high-power analysis (n=4,576 × 6 benchmarks).
Addresses fatal reviewer concerns: W1 (wide CIs), W2 (small n), W3 (underpowered selection).
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
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

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
def prepare_ollm_data():
    """Build the OLLM v2 score matrix and PIT transform."""
    print("=" * 70)
    print("SECTION 1: OLLM v2 Data Preparation")
    print("=" * 70)

    raw = pd.read_csv(DATA_DIR / "ollm_v2_raw.csv")
    print(f"  Raw OLLM v2: {raw.shape[0]} models x {raw.shape[1]} columns")

    benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    score_matrix = raw[["Model"] + benchmarks].copy()
    score_matrix = score_matrix.set_index("Model")
    score_matrix = score_matrix.dropna()

    dupes = score_matrix.index.duplicated(keep="first")
    score_matrix = score_matrix[~dupes]

    print(f"  Score matrix: {score_matrix.shape[0]} models x {score_matrix.shape[1]} benchmarks")
    print(f"  Fill rate: 100%")
    print(f"  Benchmarks: {list(score_matrix.columns)}")
    print(f"  Score ranges:")
    for bm in score_matrix.columns:
        print(f"    {bm}: [{score_matrix[bm].min():.1f}, {score_matrix[bm].max():.1f}]")

    score_matrix.to_csv(DATA_DIR / "ollm_score_matrix.csv")

    n = len(score_matrix)
    pit = score_matrix.rank(method="average") / (n + 1)
    pit.to_csv(DATA_DIR / "ollm_pit.csv")
    print(f"  PIT transform saved: {pit.shape}")

    return score_matrix, pit


# ============================================================
# SECTION 2: BIVARIATE COPULA FITTING
# ============================================================
def fit_all_bivariate(pit):
    """Fit copulas to all 15 pairs with n=4,576."""
    print("\n" + "=" * 70)
    print("SECTION 2: Bivariate Copula Fitting (n={})".format(len(pit)))
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    print(f"  Pairs to fit: {len(pairs)}")

    results = []
    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values
        u = np.clip(u, 1e-6, 1 - 1e-6)

        cop = pv.Bicop()
        controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
        cop.select(u, controls)

        td = compute_tail_dependence(cop)
        spearman = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)
        kendall = float(stats.kendalltau(u[:, 0], u[:, 1]).statistic)

        gauss_cop = pv.Bicop()
        gauss_controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.gaussian])
        gauss_cop.select(u, gauss_controls)
        bic_gauss = gauss_cop.bic(u)

        non_gauss_fams = [f for f in COPULA_FAMILIES if f != pv.BicopFamily.gaussian]
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
            "spearman_rho": spearman, "kendall_tau": kendall,
            "lambda_U": td["lambda_U"], "lambda_L": td["lambda_L"],
            "tail_asymmetry": abs(td["lambda_U"] - td["lambda_L"]),
            "delta_bic": delta_bic,
            "bic_gaussian": float(bic_gauss),
            "bic_best": float(cop.bic(u)),
            "loglik": float(cop.loglik(u)),
        }
        results.append(result)

        print(f"  {bm_i:12s} x {bm_j:12s}: {result['family']:10s} | "
              f"rho={spearman:.3f} | lU={td['lambda_U']:.3f} | lL={td['lambda_L']:.3f} | "
              f"ΔBIC={delta_bic:.1f}")

    return results


# ============================================================
# SECTION 3: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
def bootstrap_all_pairs(pit, n_bootstrap=1000, seed=42):
    """Bootstrap CIs for all 15 pairs — the key statistical validation."""
    print("\n" + "=" * 70)
    print(f"SECTION 3: Bootstrap CIs ({n_bootstrap} iterations)")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    rng = np.random.RandomState(seed)
    n = len(pit)

    all_results = []
    for pi, (i, j) in enumerate(pairs):
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u_full = pit[[bm_i, bm_j]].values
        u_full = np.clip(u_full, 1e-6, 1 - 1e-6)

        lU_samples, lL_samples, fam_samples = [], [], []
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            u_boot = u_full[idx]
            try:
                cop = pv.Bicop()
                controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
                cop.select(u_boot, controls)
                td = compute_tail_dependence(cop)
                lU_samples.append(td["lambda_U"])
                lL_samples.append(td["lambda_L"])
                fam_samples.append(FAMILY_NAMES.get(cop.family, str(cop.family)))
            except Exception:
                continue

        lU = np.array(lU_samples)
        lL = np.array(lL_samples)

        result = {
            "benchmark_1": bm_i, "benchmark_2": bm_j,
            "lambda_U_mean": float(np.mean(lU)),
            "lambda_U_ci_lower": float(np.percentile(lU, 2.5)),
            "lambda_U_ci_upper": float(np.percentile(lU, 97.5)),
            "lambda_U_ci_width": float(np.percentile(lU, 97.5) - np.percentile(lU, 2.5)),
            "lambda_L_mean": float(np.mean(lL)),
            "lambda_L_ci_lower": float(np.percentile(lL, 2.5)),
            "lambda_L_ci_upper": float(np.percentile(lL, 97.5)),
            "lambda_L_ci_width": float(np.percentile(lL, 97.5) - np.percentile(lL, 2.5)),
            "family_mode": max(set(fam_samples), key=fam_samples.count),
            "family_stability": fam_samples.count(max(set(fam_samples), key=fam_samples.count)) / len(fam_samples),
            "n_successful": len(lU_samples),
        }
        all_results.append(result)

        print(f"  [{pi+1:2d}/15] {bm_i:12s} x {bm_j:12s}: "
              f"lU={result['lambda_U_mean']:.3f} [{result['lambda_U_ci_lower']:.3f}, {result['lambda_U_ci_upper']:.3f}] "
              f"| lL={result['lambda_L_mean']:.3f} [{result['lambda_L_ci_lower']:.3f}, {result['lambda_L_ci_upper']:.3f}] "
              f"| {result['family_mode']} ({result['family_stability']:.0%})")

    print(f"\n  Mean CI width (lU): {np.mean([r['lambda_U_ci_width'] for r in all_results]):.4f}")
    print(f"  Mean CI width (lL): {np.mean([r['lambda_L_ci_width'] for r in all_results]):.4f}")
    print(f"  Mean family stability: {np.mean([r['family_stability'] for r in all_results]):.1%}")

    return all_results


# ============================================================
# SECTION 4: NONPARAMETRIC TAIL ESTIMATORS
# ============================================================
def nonparametric_tail_analysis(pit):
    """Conditional Spearman above 90th percentile + exceedance correlation."""
    print("\n" + "=" * 70)
    print("SECTION 4: Nonparametric Tail Estimators")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))

    results = []
    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values

        spearman_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

        upper_mask = (u[:, 0] >= 0.8) | (u[:, 1] >= 0.8)
        upper_joint = (u[:, 0] >= 0.8) & (u[:, 1] >= 0.8)
        if upper_mask.sum() >= 20:
            spearman_upper = float(stats.spearmanr(
                u[upper_mask, 0], u[upper_mask, 1]
            ).statistic)
        else:
            spearman_upper = np.nan

        lower_mask = (u[:, 0] <= 0.2) | (u[:, 1] <= 0.2)
        if lower_mask.sum() >= 20:
            spearman_lower = float(stats.spearmanr(
                u[lower_mask, 0], u[lower_mask, 1]
            ).statistic)
        else:
            spearman_lower = np.nan

        n = len(u)
        for q in [0.9, 0.95]:
            n_exceed_1 = (u[:, 0] >= q).sum()
            n_exceed_both = ((u[:, 0] >= q) & (u[:, 1] >= q)).sum()
            exceedance_upper = n_exceed_both / n_exceed_1 if n_exceed_1 > 0 else 0.0

            n_below_1 = (u[:, 0] <= (1 - q)).sum()
            n_below_both = ((u[:, 0] <= (1 - q)) & (u[:, 1] <= (1 - q))).sum()
            exceedance_lower = n_below_both / n_below_1 if n_below_1 > 0 else 0.0

            if q == 0.9:
                exc_u_90, exc_l_90 = exceedance_upper, exceedance_lower
            else:
                exc_u_95, exc_l_95 = exceedance_upper, exceedance_lower

        results.append({
            "benchmark_1": bm_i, "benchmark_2": bm_j,
            "spearman_full": spearman_full,
            "spearman_upper_20pct": spearman_upper,
            "spearman_lower_20pct": spearman_lower,
            "exceedance_upper_90": exc_u_90,
            "exceedance_upper_95": exc_u_95,
            "exceedance_lower_90": exc_l_90,
            "exceedance_lower_95": exc_l_95,
            "tail_asymmetry_exceedance": abs(exc_u_90 - exc_l_90),
        })

        print(f"  {bm_i:12s} x {bm_j:12s}: "
              f"rho_full={spearman_full:.3f} | "
              f"rho_upper={spearman_upper:.3f} | rho_lower={spearman_lower:.3f} | "
              f"exc_U90={exc_u_90:.3f} | exc_L90={exc_l_90:.3f}")

    return results


# ============================================================
# SECTION 5: SIMULATION STUDY + MULTIPLICITY CORRECTION
# ============================================================
def simulation_and_correction(bivariate_results, n_obs=4576, n_sims=500, seed=42):
    """Simulation FPR at n=4,576 + BH correction on ΔBIC results."""
    print("\n" + "=" * 70)
    print("SECTION 5: Simulation Study + Multiplicity Correction")
    print("=" * 70)

    rho_values = [0.3, 0.5, 0.7, 0.9]
    rng = np.random.RandomState(seed)
    sim_results = []

    print("  Simulation study (Gaussian null, n={}, {} sims):".format(n_obs, n_sims))
    for rho in rho_values:
        non_gauss_count = 0
        for s in range(n_sims):
            gauss_cop = pv.Bicop(family=pv.BicopFamily.gaussian,
                                 parameters=np.array([[rho]]))
            u = gauss_cop.simulate(n_obs, seeds=[seed + s])
            u = np.clip(u, 1e-6, 1 - 1e-6)
            try:
                cop = pv.Bicop()
                controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
                cop.select(u, controls)
                if cop.family != pv.BicopFamily.gaussian:
                    non_gauss_count += 1
            except:
                continue

        fpr = non_gauss_count / n_sims
        sim_results.append({"n": n_obs, "rho": rho, "fpr": fpr})
        print(f"    rho={rho:.1f}: FPR = {fpr:.4f}")

    delta_bics = [r["delta_bic"] for r in bivariate_results]
    p_values = []
    for db in delta_bics:
        p = np.exp(-0.5 * max(0, db))
        p_values.append(min(1.0, p))

    rejected_bh, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

    print(f"\n  BH/FDR Correction (15 tests):")
    print(f"    Pairs with ΔBIC > 0 (uncorrected non-Gaussian): {sum(1 for d in delta_bics if d > 0)}/15")
    print(f"    Pairs surviving BH correction at α=0.05: {sum(rejected_bh)}/15")

    correction_results = []
    for idx, r in enumerate(bivariate_results):
        correction_results.append({
            "pair": f"{r['benchmark_1']} x {r['benchmark_2']}",
            "delta_bic": r["delta_bic"],
            "approx_p": p_values[idx],
            "bh_corrected_p": float(pvals_corrected[idx]),
            "significant_bh_05": bool(rejected_bh[idx]),
        })

    return sim_results, correction_results


# ============================================================
# SECTION 6: VINE COPULA
# ============================================================
def fit_vine(pit):
    """Fit vine copula on OLLM v2 (6D, n=4,576)."""
    print("\n" + "=" * 70)
    print("SECTION 6: Vine Copula (6D, n={})".format(len(pit)))
    print("=" * 70)

    u = pit.values
    u = np.clip(u, 1e-6, 1 - 1e-6)

    controls = pv.FitControlsVinecop(
        family_set=COPULA_FAMILIES,
        trunc_lvl=5,
    )
    vc = pv.Vinecop(d=pit.shape[1])
    vc.select(u, controls)

    print(f"  Vine fitted: dim={vc.dim}, trunc_lvl={vc.trunc_lvl}")
    print(f"  Log-likelihood: {vc.loglik(u):.2f}")
    print(f"  AIC: {vc.aic(u):.2f}")
    print(f"  BIC: {vc.bic(u):.2f}")

    benchmarks = pit.columns.tolist()
    matrix = vc.matrix

    tree1_info = []
    print(f"\n  Tree 1 edges:")
    for j in range(vc.dim - 1):
        pair_cop = vc.get_pair_copula(0, j)
        fam = FAMILY_NAMES.get(pair_cop.family, str(pair_cop.family))
        td = compute_tail_dependence(pair_cop)

        idx_1 = int(matrix[0, j]) - 1
        idx_2 = int(matrix[j, j]) - 1
        bm1 = benchmarks[idx_1] if 0 <= idx_1 < len(benchmarks) else f"Var{idx_1}"
        bm2 = benchmarks[idx_2] if 0 <= idx_2 < len(benchmarks) else f"Var{idx_2}"

        edge_info = {
            "node_1": bm1, "node_2": bm2,
            "family": fam,
            "lambda_U": td["lambda_U"], "lambda_L": td["lambda_L"],
        }
        tree1_info.append(edge_info)
        print(f"    {bm1:12s} --- {bm2:12s} [{fam}] lU={td['lambda_U']:.3f} lL={td['lambda_L']:.3f}")

    return {"tree1": tree1_info, "loglik": float(vc.loglik(u)),
            "aic": float(vc.aic(u)), "bic": float(vc.bic(u))}


# ============================================================
# SECTION 7: SELECTION EXPERIMENT
# ============================================================
def selection_experiment(score_matrix, K_values=[3, 4, 5]):
    """Full selection experiment on OLLM v2 with 4,576 models."""
    print("\n" + "=" * 70)
    print("SECTION 7: Selection Experiment (n={})".format(len(score_matrix)))
    print("=" * 70)

    data = score_matrix.copy()
    spearman_corr = data.corr(method="spearman")
    benchmarks = data.columns.tolist()
    results = []

    for K in K_values:
        if K >= len(benchmarks):
            continue
        print(f"\n  === K = {K} ===")

        abs_corr = spearman_corr.abs()
        mean_corr = abs_corr.mean(axis=1)
        first = mean_corr.idxmax()
        corr_selected = [first]
        for _ in range(K - 1):
            remaining = [b for b in benchmarks if b not in corr_selected]
            best, best_score = None, -1
            for b in remaining:
                score = mean_corr[b] - abs_corr.loc[b, corr_selected].max()
                if score > best_score:
                    best_score = score
                    best = b
            if best:
                corr_selected.append(best)

        methods = {"Correlation": corr_selected}

        rng = np.random.RandomState(42)
        random_sets = [list(rng.choice(benchmarks, K, replace=False)) for _ in range(200)]

        for name, selected in methods.items():
            metrics = _evaluate(data, selected)
            results.append({"K": K, "method": name, "selected": selected, **metrics})
            print(f"    {name:14s}: {selected}")
            print(f"      MAE_all={metrics['mae_all']:.2f}, MAE_tail={metrics['mae_tail']:.2f}, "
                  f"tau_top20={metrics['tau_top20']:.3f}, QR_tails={metrics['qr_mae_tails']:.2f}")

        random_metrics = [_evaluate(data, rs) for rs in random_sets]
        avg = {k: float(np.nanmean([m[k] for m in random_metrics]))
               for k in random_metrics[0].keys()}
        results.append({"K": K, "method": "Random", "selected": "200 samples", **avg})
        print(f"    {'Random':14s}: MAE_all={avg['mae_all']:.2f}, MAE_tail={avg['mae_tail']:.2f}, "
              f"tau_top20={avg['tau_top20']:.3f}, QR_tails={avg['qr_mae_tails']:.2f}")

    return results


def _evaluate(data, selected, n_splits=5, seed=42):
    other = [b for b in data.columns if b not in selected]
    if not other:
        return {"mae_all": 0, "mae_tail": 0, "tau_top20": 0, "qr_mae_tails": 0}

    X = data[selected].values
    errors_all, errors_tail, tau_tops, qr_errors = [], [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for target_bm in other:
        y = data[target_bm].values
        preds = np.zeros_like(y, dtype=float)
        preds_qr_upper = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in kf.split(X):
            reg = LinearRegression()
            reg.fit(X[train_idx], y[train_idx])
            preds[test_idx] = reg.predict(X[test_idx])
            try:
                qr = QuantileRegressor(quantile=0.9, alpha=0.01, solver="highs")
                qr.fit(X[train_idx], y[train_idx])
                preds_qr_upper[test_idx] = qr.predict(X[test_idx])
            except:
                preds_qr_upper[test_idx] = preds[test_idx]

        errors_all.extend(np.abs(preds - y).tolist())
        top_mask = y >= np.percentile(y, 80)
        bottom_mask = y <= np.percentile(y, 20)
        if top_mask.sum() > 0:
            errors_tail.extend(np.abs(preds[top_mask] - y[top_mask]).tolist())
            qr_errors.extend(np.abs(preds_qr_upper[top_mask] - y[top_mask]).tolist())
        if bottom_mask.sum() > 0:
            errors_tail.extend(np.abs(preds[bottom_mask] - y[bottom_mask]).tolist())

        true_ranks = stats.rankdata(-y)
        pred_ranks = stats.rankdata(-preds)
        top_20_mask = true_ranks <= len(y) * 0.2
        if top_20_mask.sum() >= 3:
            tau, _ = stats.kendalltau(true_ranks[top_20_mask], pred_ranks[top_20_mask])
            tau_tops.append(tau)

    return {
        "mae_all": float(np.mean(errors_all)),
        "mae_tail": float(np.mean(errors_tail)) if errors_tail else np.nan,
        "tau_top20": float(np.nanmean(tau_tops)) if tau_tops else np.nan,
        "qr_mae_tails": float(np.mean(qr_errors)) if qr_errors else np.nan,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OLLM v2 FULL COPULA PIPELINE")
    print("4,576 models × 6 benchmarks × 100% fill rate")
    print("=" * 70)

    score_matrix, pit = prepare_ollm_data()

    bivariate_results = fit_all_bivariate(pit)
    with open(RESULTS_DIR / "ollm_bivariate_fits.json", "w") as f:
        json.dump(bivariate_results, f, indent=2)

    bootstrap_results = bootstrap_all_pairs(pit, n_bootstrap=1000)
    with open(RESULTS_DIR / "ollm_bootstrap_cis.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    nonparam_results = nonparametric_tail_analysis(pit)
    with open(RESULTS_DIR / "ollm_nonparametric_tails.json", "w") as f:
        json.dump(nonparam_results, f, indent=2)

    sim_results, correction_results = simulation_and_correction(bivariate_results)
    with open(RESULTS_DIR / "ollm_simulation_and_correction.json", "w") as f:
        json.dump({"simulation": sim_results, "bh_correction": correction_results}, f, indent=2)

    vine_results = fit_vine(pit)
    with open(RESULTS_DIR / "ollm_vine_structure.json", "w") as f:
        json.dump(vine_results, f, indent=2)

    selection_results = selection_experiment(score_matrix)
    with open(RESULTS_DIR / "ollm_selection_experiment.json", "w") as f:
        json.dump(selection_results, f, indent=2)

    print("\n\n" + "=" * 70)
    print("ALL SECTIONS COMPLETE")
    print("=" * 70)

    print("\n  KEY FINDINGS SUMMARY:")
    n_nongauss = sum(1 for r in bivariate_results if r["family"] != "Gaussian")
    n_strong = sum(1 for r in bivariate_results if r["delta_bic"] > 6)
    mean_ci = np.mean([r["lambda_U_ci_width"] for r in bootstrap_results])
    print(f"  Non-Gaussian pairs: {n_nongauss}/15 ({n_nongauss/15*100:.0f}%)")
    print(f"  Strong evidence (ΔBIC>6): {n_strong}/15")
    print(f"  Mean bootstrap CI width (λ_U): {mean_ci:.4f}")
    print(f"  Max family stability: {max(r['family_stability'] for r in bootstrap_results):.1%}")
