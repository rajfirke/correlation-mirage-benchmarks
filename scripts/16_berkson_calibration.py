"""
Task 2B: Broadened Berkson's Paradox Calibration.
Addresses R1(W1), R3(W4), R5(W3).

Extends the Gaussian-null calibration with:
  1. Student-t copula nulls (df=5, 10, 30)
  2. Permutation-based null (R5's suggestion)
  3. Excess-beyond-null for each alternative
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


def conditional_spearman(u, quantile=0.95):
    """Compute Spearman correlation restricted to top quantile on either variable."""
    mask = (u[:, 0] >= quantile) | (u[:, 1] >= quantile)
    if mask.sum() < 10:
        return np.nan
    return float(stats.spearmanr(u[mask, 0], u[mask, 1]).statistic)


def simulate_gaussian_null(rho, n, quantiles, n_sims=500, seed=42):
    """Simulate conditional Spearman under Gaussian copula null."""
    rng = np.random.RandomState(seed)
    cond_rhos = {q: [] for q in quantiles}

    gauss_cop = pv.Bicop(family=pv.BicopFamily.gaussian,
                          parameters=np.array([[rho]]))

    for s in range(n_sims):
        u = gauss_cop.simulate(n, seeds=[seed + s])
        u = np.clip(u, 1e-6, 1 - 1e-6)
        for q in quantiles:
            cr = conditional_spearman(u, q)
            if not np.isnan(cr):
                cond_rhos[q].append(cr)

    return {q: np.array(vals) for q, vals in cond_rhos.items()}


def simulate_student_t_null(rho, df, n, quantiles, n_sims=500, seed=42):
    """Simulate conditional Spearman under Student-t copula null."""
    cond_rhos = {q: [] for q in quantiles}

    try:
        t_cop = pv.Bicop(family=pv.BicopFamily.student,
                          parameters=np.array([[rho, df]]))
    except Exception:
        try:
            t_cop = pv.Bicop(family=pv.BicopFamily.student,
                              parameters=np.array([[rho], [df]]))
        except Exception:
            return {q: np.array([]) for q in quantiles}

    for s in range(n_sims):
        u = t_cop.simulate(n, seeds=[seed + s])
        u = np.clip(u, 1e-6, 1 - 1e-6)
        for q in quantiles:
            cr = conditional_spearman(u, q)
            if not np.isnan(cr):
                cond_rhos[q].append(cr)

    return {q: np.array(vals) for q, vals in cond_rhos.items()}


def permutation_null(u, quantiles, n_perms=500, seed=42):
    """
    Permutation-based null: independently shuffle ranks within each benchmark
    among the 'top' models, preserving marginals but destroying dependence.
    """
    rng = np.random.RandomState(seed)
    n = len(u)
    cond_rhos = {q: [] for q in quantiles}

    for p in range(n_perms):
        u_perm = u.copy()
        u_perm[:, 1] = rng.permutation(u_perm[:, 1])
        for q in quantiles:
            cr = conditional_spearman(u_perm, q)
            if not np.isnan(cr):
                cond_rhos[q].append(cr)

    return {q: np.array(vals) for q, vals in cond_rhos.items()}


def run_berkson_calibration(pit):
    """Full Berkson calibration for all 15 pairs."""
    print("=" * 70)
    print("BERKSON'S PARADOX CALIBRATION")
    print("=" * 70)

    benchmarks = pit.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    n = len(pit)
    quantiles = [0.80, 0.90, 0.95]

    results = []

    for pi, (i, j) in enumerate(pairs):
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = pit[[bm_i, bm_j]].values
        u = np.clip(u, 1e-6, 1 - 1e-6)

        rho_full = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)
        observed_cond = {q: conditional_spearman(u, q) for q in quantiles}

        print(f"\n  [{pi+1}/15] {bm_i} x {bm_j} (ρ={rho_full:.3f})")
        print(f"    Observed conditional ρ: " +
              " | ".join(f"q={q}: {observed_cond[q]:.3f}" for q in quantiles))

        gauss_null = simulate_gaussian_null(rho_full, n, quantiles, n_sims=500)

        t_nulls = {}
        for df in [5, 10, 30]:
            t_nulls[df] = simulate_student_t_null(rho_full, df, n, quantiles, n_sims=500)

        perm_null = permutation_null(u, quantiles, n_perms=500)

        pair_result = {
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "spearman_full": rho_full,
            "observed_conditional": {str(q): observed_cond[q] for q in quantiles},
            "null_predictions": {},
            "excess_beyond_null": {},
        }

        for q in quantiles:
            obs = observed_cond[q]
            if np.isnan(obs):
                continue

            nulls = {}

            g_vals = gauss_null[q]
            if len(g_vals) > 0:
                g_mean = float(np.mean(g_vals))
                g_ci = [float(np.percentile(g_vals, 2.5)),
                        float(np.percentile(g_vals, 97.5))]
                excess_g = obs - g_mean
                p_val_g = float(np.mean(g_vals <= obs))
                nulls["gaussian"] = {
                    "mean": g_mean, "ci": g_ci,
                    "excess": float(excess_g), "p_value_one_sided": p_val_g,
                }
                print(f"    q={q}: Gaussian null={g_mean:.3f}, excess={excess_g:+.3f}, "
                      f"p={p_val_g:.3f}")

            for df in [5, 10, 30]:
                t_vals = t_nulls[df].get(q, np.array([]))
                if len(t_vals) > 0:
                    t_mean = float(np.mean(t_vals))
                    t_ci = [float(np.percentile(t_vals, 2.5)),
                            float(np.percentile(t_vals, 97.5))]
                    excess_t = obs - t_mean
                    p_val_t = float(np.mean(t_vals <= obs))
                    nulls[f"student_t_df{df}"] = {
                        "mean": t_mean, "ci": t_ci,
                        "excess": float(excess_t), "p_value_one_sided": p_val_t,
                    }
                    print(f"    q={q}: Student-t(df={df}) null={t_mean:.3f}, "
                          f"excess={excess_t:+.3f}, p={p_val_t:.3f}")

            p_vals = perm_null.get(q, np.array([]))
            if len(p_vals) > 0:
                p_mean = float(np.mean(p_vals))
                p_ci = [float(np.percentile(p_vals, 2.5)),
                        float(np.percentile(p_vals, 97.5))]
                excess_p = obs - p_mean
                nulls["permutation"] = {
                    "mean": p_mean, "ci": p_ci,
                    "excess": float(excess_p),
                }

            pair_result["null_predictions"][str(q)] = nulls
            pair_result["excess_beyond_null"][str(q)] = {
                name: vals.get("excess", np.nan) for name, vals in nulls.items()
            }

        results.append(pair_result)

    return results


def summarize_headline_pairs(results):
    """Print detailed summary for the headline deceptive pairs."""
    print("\n" + "=" * 70)
    print("HEADLINE PAIRS: Excess Beyond Each Null")
    print("=" * 70)

    headline = [
        ("BBH", "GPQA"),
        ("BBH", "MUSR"),
        ("MATH Lvl 5", "MMLU-PRO"),
        ("IFEval", "BBH"),
    ]

    summary = []
    for bm1, bm2 in headline:
        match = [r for r in results
                 if (r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2) or
                    (r["benchmark_1"] == bm2 and r["benchmark_2"] == bm1)]
        if not match:
            continue
        r = match[0]
        print(f"\n  {r['benchmark_1']} x {r['benchmark_2']} (ρ={r['spearman_full']:.3f}):")

        for q in ["0.8", "0.9", "0.95"]:
            nulls = r["null_predictions"].get(q, {})
            obs = r["observed_conditional"].get(q, np.nan)
            if np.isnan(obs):
                continue
            print(f"    q={q}: observed={obs:.3f}")
            for null_name, null_vals in nulls.items():
                excess = null_vals.get("excess", np.nan)
                p_val = null_vals.get("p_value_one_sided", np.nan)
                marker = " ***" if (not np.isnan(p_val) and p_val < 0.05) else ""
                print(f"      {null_name:20s}: predicted={null_vals['mean']:.3f}, "
                      f"excess={excess:+.3f}{marker}")

        pair_summary = {
            "pair": f"{r['benchmark_1']} x {r['benchmark_2']}",
            "rho_full": r["spearman_full"],
        }
        for q in ["0.95"]:
            nulls = r["null_predictions"].get(q, {})
            for null_name, null_vals in nulls.items():
                pair_summary[f"excess_{null_name}_q{q}"] = null_vals.get("excess", np.nan)
                pair_summary[f"significant_{null_name}_q{q}"] = (
                    null_vals.get("p_value_one_sided", 1.0) < 0.05
                )
        summary.append(pair_summary)

    print("\n  KEY FINDING:")
    for s in summary:
        excess_g = s.get("excess_gaussian_q0.95", np.nan)
        excess_t5 = s.get("excess_student_t_df5_q0.95", np.nan)
        sig_t5 = s.get("significant_student_t_df5_q0.95", False)
        print(f"    {s['pair']}: Gaussian excess={excess_g:+.3f}, "
              f"Student-t(df=5) excess={excess_t5:+.3f} "
              f"{'(significant)' if sig_t5 else '(not significant)'}")

    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2B: BROADENED BERKSON'S PARADOX CALIBRATION")
    print("=" * 70)

    pit = pd.read_csv(DATA_DIR / "ollm_pit.csv", index_col=0)
    print(f"  PIT matrix: {pit.shape}")

    results = run_berkson_calibration(pit)
    headline_summary = summarize_headline_pairs(results)

    output = {
        "full_results": results,
        "headline_summary": headline_summary,
    }

    output_path = RESULTS_DIR / "berkson_calibration_broadened.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    print(f"\n  Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("TASK 2B COMPLETE")
    print("=" * 70)
