"""
Task 2A: Expanded Copula Candidate Set + Power Analysis.
Addresses R1(W3), R3(W4), R5(Q3).

Adds: BB6, BB8, Tawn families + rotated Clayton/Gumbel (via allow_rotations).
Reports which pairs (if any) change family assignment.
Formal power analysis: minimum detectable lambda_U at n=779.
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

ORIGINAL_FAMILIES = [
    pv.BicopFamily.gaussian, pv.BicopFamily.student,
    pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
    pv.BicopFamily.frank, pv.BicopFamily.joe,
    pv.BicopFamily.bb1, pv.BicopFamily.bb7,
]

EXPANDED_FAMILIES = ORIGINAL_FAMILIES + [
    pv.BicopFamily.bb6,
    pv.BicopFamily.bb8,
    pv.BicopFamily.tawn,
]

FAMILY_NAMES = {
    pv.BicopFamily.gaussian: "Gaussian",
    pv.BicopFamily.student: "Student-t",
    pv.BicopFamily.clayton: "Clayton",
    pv.BicopFamily.gumbel: "Gumbel",
    pv.BicopFamily.frank: "Frank",
    pv.BicopFamily.joe: "Joe",
    pv.BicopFamily.bb1: "BB1",
    pv.BicopFamily.bb6: "BB6",
    pv.BicopFamily.bb7: "BB7",
    pv.BicopFamily.bb8: "BB8",
    pv.BicopFamily.tawn: "Tawn",
    pv.BicopFamily.indep: "Independence",
}


def compute_tail_dependence(cop):
    fam = cop.family
    par = cop.parameters
    rot = cop.rotation
    lL, lU = 0.0, 0.0

    if fam == pv.BicopFamily.student:
        rho = par[0, 0]
        nu = par[0, 1] if par.shape[1] > 1 else par[1, 0]
        val = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lL = lU = 2 * stats.t.cdf(-val, df=nu + 1)
    elif fam == pv.BicopFamily.clayton:
        theta = par[0, 0]
        if theta > 0:
            lL = 2 ** (-1.0 / theta)
    elif fam == pv.BicopFamily.gumbel:
        lU = 2 - 2 ** (1.0 / par[0, 0])
    elif fam == pv.BicopFamily.joe:
        lU = 2 - 2 ** (1.0 / par[0, 0])
    elif fam == pv.BicopFamily.frank:
        lL = lU = 0.0
    elif fam == pv.BicopFamily.bb1:
        theta, delta = par[0, 0], (par[0, 1] if par.shape[1] > 1 else par[1, 0])
        lU = 2 - 2 ** (1.0 / delta)
        lL = 2 ** (-1.0 / (theta * delta)) if theta * delta > 0 else 0.0
    elif fam == pv.BicopFamily.bb7:
        theta, delta = par[0, 0], (par[0, 1] if par.shape[1] > 1 else par[1, 0])
        lU = 2 - 2 ** (1.0 / theta)
        lL = 2 ** (-1.0 / delta) if delta > 0 else 0.0
    else:
        u_sim = cop.simulate(50000, seeds=[42])
        q = 0.05
        mask_u = u_sim[:, 0] >= (1 - q)
        lU = float((u_sim[mask_u, 1] >= (1 - q)).mean()) if mask_u.sum() > 0 else 0.0
        mask_l = u_sim[:, 0] <= q
        lL = float((u_sim[mask_l, 1] <= q).mean()) if mask_l.sum() > 0 else 0.0

    if rot == 180:
        lL, lU = lU, lL
    elif rot == 90:
        lL, lU = 0.0, 0.0
    elif rot == 270:
        lL, lU = 0.0, 0.0

    return {"lambda_L": float(lL), "lambda_U": float(lU)}


def fit_with_family_set(u, family_set, allow_rot=False):
    cop = pv.Bicop()
    controls = pv.FitControlsBicop(
        family_set=family_set,
        allow_rotations=allow_rot,
    )
    cop.select(u, controls)
    return cop


def compare_original_vs_expanded(pit):
    """Fit all 15 pairs with original and expanded family sets, compare."""
    print("=" * 70)
    print("COMPARISON: Original (8 families) vs Expanded (11 + rotations)")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))

    results = []
    changes = []

    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values
        u = np.clip(u, 1e-6, 1 - 1e-6)

        cop_orig = fit_with_family_set(u, ORIGINAL_FAMILIES, allow_rot=False)
        cop_expanded = fit_with_family_set(u, EXPANDED_FAMILIES, allow_rot=True)

        td_orig = compute_tail_dependence(cop_orig)
        td_exp = compute_tail_dependence(cop_expanded)

        fam_orig = FAMILY_NAMES.get(cop_orig.family, str(cop_orig.family))
        fam_exp = FAMILY_NAMES.get(cop_expanded.family, str(cop_expanded.family))
        rot_exp = cop_expanded.rotation

        fam_exp_full = fam_exp
        if rot_exp > 0:
            fam_exp_full = f"{fam_exp} (rot={rot_exp})"

        bic_orig = float(cop_orig.bic(u))
        bic_exp = float(cop_expanded.bic(u))
        bic_improvement = bic_orig - bic_exp

        changed = (fam_orig != fam_exp) or (rot_exp > 0 and cop_orig.rotation == 0)

        result = {
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "n_obs": len(u),
            "original_family": fam_orig,
            "original_rotation": int(cop_orig.rotation),
            "original_bic": bic_orig,
            "original_lambda_U": td_orig["lambda_U"],
            "original_lambda_L": td_orig["lambda_L"],
            "expanded_family": fam_exp,
            "expanded_rotation": rot_exp,
            "expanded_bic": bic_exp,
            "expanded_lambda_U": td_exp["lambda_U"],
            "expanded_lambda_L": td_exp["lambda_L"],
            "bic_improvement": bic_improvement,
            "family_changed": changed,
            "expanded_family_full": fam_exp_full,
            "spearman_rho": float(stats.spearmanr(u[:, 0], u[:, 1]).statistic),
        }
        results.append(result)

        status = "CHANGED" if changed else "same"
        print(f"  {bm_i:12s} x {bm_j:12s}: "
              f"{fam_orig:10s} -> {fam_exp_full:18s} | "
              f"ΔBIC={bic_improvement:+.1f} | {status}")

        if changed:
            changes.append(result)

    n_changed = len(changes)
    print(f"\n  SUMMARY: {n_changed}/{len(results)} pairs changed family assignment")
    if n_changed > 0:
        print("  Changed pairs:")
        for c in changes:
            print(f"    {c['benchmark_1']} x {c['benchmark_2']}: "
                  f"{c['original_family']} -> {c['expanded_family_full']} "
                  f"(ΔBIC improvement = {c['bic_improvement']:.1f})")
    else:
        print("  No pairs changed — original 8-family set is sufficient.")

    return results


def power_analysis(n_values=[779, 1000, 2000, 4497], rho_values=[0.5, 0.7, 0.9],
                   lambda_U_targets=[0.05, 0.10, 0.15, 0.20, 0.30],
                   n_sims=500, seed=42):
    """
    Power analysis: at each n, simulate from copulas with known lambda_U,
    then check how often BIC correctly selects non-Gaussian.
    Uses Gumbel copula (which has closed-form lambda_U = 2 - 2^{1/theta}).
    """
    print("\n" + "=" * 70)
    print("POWER ANALYSIS: Minimum Detectable lambda_U")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    results = []

    for n in n_values:
        for lam_target in lambda_U_targets:
            theta = 1.0 / np.log2(2.0 / (2.0 - lam_target))
            if theta <= 1.0:
                theta = 1.001

            gumbel_cop = pv.Bicop(
                family=pv.BicopFamily.gumbel,
                parameters=np.array([[theta]])
            )
            actual_lam = 2 - 2 ** (1.0 / theta)

            correct_nongauss = 0
            detected_tail = 0

            for s in range(n_sims):
                u = gumbel_cop.simulate(n, seeds=[seed + s])
                u = np.clip(u, 1e-6, 1 - 1e-6)
                try:
                    cop = pv.Bicop()
                    controls = pv.FitControlsBicop(family_set=ORIGINAL_FAMILIES)
                    cop.select(u, controls)
                    if cop.family != pv.BicopFamily.gaussian:
                        correct_nongauss += 1
                    td = compute_tail_dependence(cop)
                    if td["lambda_U"] > 0.01:
                        detected_tail += 1
                except Exception:
                    continue

            power = correct_nongauss / n_sims
            tail_power = detected_tail / n_sims

            results.append({
                "n": n,
                "target_lambda_U": float(lam_target),
                "actual_lambda_U": float(actual_lam),
                "gumbel_theta": float(theta),
                "power_nongaussian": float(power),
                "power_tail_detected": float(tail_power),
                "n_sims": n_sims,
            })

            print(f"  n={n:5d} | λ_U={lam_target:.2f} (θ={theta:.2f}) | "
                  f"Power(non-Gauss)={power:.3f} | Power(λ_U>0.01)={tail_power:.3f}")

    print("\n  --- Minimum detectable λ_U at 80% power ---")
    for n in n_values:
        subset = [r for r in results if r["n"] == n]
        min_detectable = None
        for r in subset:
            if r["power_tail_detected"] >= 0.80:
                min_detectable = r["target_lambda_U"]
                break
        if min_detectable is not None:
            print(f"    n={n:5d}: min detectable λ_U = {min_detectable:.2f} (at 80% power)")
        else:
            print(f"    n={n:5d}: cannot reliably detect any tested λ_U at 80% power")

    return results


def student_t_df_sensitivity(pit):
    """
    Check if Student-t copula with varying df is important:
    fit Student-t and report estimated df for each pair.
    """
    print("\n" + "=" * 70)
    print("STUDENT-T DF SENSITIVITY")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    results = []

    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values
        u = np.clip(u, 1e-6, 1 - 1e-6)

        cop = pv.Bicop()
        controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.student])
        cop.select(u, controls)

        par = cop.parameters
        rho = par[0, 0]
        nu = par[0, 1] if par.shape[1] > 1 else par[1, 0]

        td = compute_tail_dependence(cop)

        result = {
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "student_rho": float(rho),
            "student_df": float(nu),
            "student_lambda_U": td["lambda_U"],
            "student_lambda_L": td["lambda_L"],
            "student_bic": float(cop.bic(u)),
        }
        results.append(result)
        print(f"  {bm_i:12s} x {bm_j:12s}: df={nu:.1f}, ρ={rho:.3f}, "
              f"λ_U={td['lambda_U']:.4f}, λ_L={td['lambda_L']:.4f}")

    df_values = [r["student_df"] for r in results]
    print(f"\n  Estimated df range: [{min(df_values):.1f}, {max(df_values):.1f}]")
    print(f"  Median df: {np.median(df_values):.1f}")
    print(f"  Pairs with df < 10 (heavy tails): "
          f"{sum(1 for d in df_values if d < 10)}/{len(results)}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2A: EXPANDED COPULA CANDIDATE SET + POWER ANALYSIS")
    print("=" * 70)

    ollm = pd.read_csv(DATA_DIR / "ollm_score_matrix.csv", index_col=0)
    pit = pd.read_csv(DATA_DIR / "ollm_pit.csv", index_col=0)
    print(f"  Score matrix: {ollm.shape}")
    print(f"  PIT matrix: {pit.shape}")

    print("\n--- 1. Original vs Expanded Family Comparison ---")
    comparison_results = compare_original_vs_expanded(pit)

    print("\n--- 2. Student-t df Sensitivity ---")
    student_results = student_t_df_sensitivity(pit)

    print("\n--- 3. Power Analysis ---")
    power_results = power_analysis(
        n_values=[779, 1000, 2000, 4497],
        lambda_U_targets=[0.05, 0.10, 0.15, 0.20, 0.30],
        n_sims=500,
    )

    output = {
        "family_comparison": comparison_results,
        "student_t_sensitivity": student_results,
        "power_analysis": power_results,
    }

    output_path = RESULTS_DIR / "expanded_copula_candidates.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    n_changed = sum(1 for r in comparison_results if r["family_changed"])
    print(f"\n  FINAL SUMMARY:")
    print(f"  Family changes: {n_changed}/15 pairs")
    print(f"  Expanded set adds {len(EXPANDED_FAMILIES) - len(ORIGINAL_FAMILIES)} families + rotations")

    for r in power_results:
        if r["n"] == 779 and r["power_tail_detected"] >= 0.80:
            print(f"  Min detectable λ_U at n=779: {r['target_lambda_U']:.2f}")
            break

    print("\n" + "=" * 70)
    print("TASK 2A COMPLETE")
    print("=" * 70)
