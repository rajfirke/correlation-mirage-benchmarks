"""
Sensitivity analysis: How do copula findings change as a function of minimum sample size?

Investigates whether the non-Gaussian dependence findings are robust to stricter
sample-size requirements, or whether they are partially driven by small-n noise.
"""

import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(RESULTS / "bivariate_fits_pairwise.json") as f:
    fits = json.load(f)

with open(RESULTS / "delta_bic_distribution.json") as f:
    delta_bic_data = json.load(f)

with open(RESULTS / "simulation_gaussian_null.json") as f:
    sim_null = json.load(f)

# ---------------------------------------------------------------------------
# Build simulation FPR lookup: average across rho for each simulated n
# ---------------------------------------------------------------------------
sim_fpr_by_n = {}
for entry in sim_null:
    n = entry["n"]
    sim_fpr_by_n.setdefault(n, []).append(entry["false_positive_rate"])

sim_n_values = sorted(sim_fpr_by_n.keys())
sim_fpr_means = {n: np.mean(rates) for n, rates in sim_fpr_by_n.items()}


def interpolate_fpr(target_n):
    """Linearly interpolate the mean FPR for a given n from simulation grid."""
    if target_n <= sim_n_values[0]:
        return sim_fpr_means[sim_n_values[0]]
    if target_n >= sim_n_values[-1]:
        return sim_fpr_means[sim_n_values[-1]]
    for i in range(len(sim_n_values) - 1):
        n_lo, n_hi = sim_n_values[i], sim_n_values[i + 1]
        if n_lo <= target_n <= n_hi:
            frac = (target_n - n_lo) / (n_hi - n_lo)
            return sim_fpr_means[n_lo] + frac * (sim_fpr_means[n_hi] - sim_fpr_means[n_lo])
    return sim_fpr_means[sim_n_values[-1]]


# ---------------------------------------------------------------------------
# Sensitivity analysis across minimum-n thresholds
# ---------------------------------------------------------------------------
thresholds = [25, 30, 35, 40, 50, 60, 75]
results = []

print("=" * 90)
print("SENSITIVITY ANALYSIS: Copula findings as a function of minimum sample size")
print("=" * 90)
print()

header = (
    f"{'Min n':>6} | {'Pairs':>6} | {'%ΔBIC>2':>8} | {'%ΔBIC>6':>8} | "
    f"{'%λ_U>0.05':>10} | {'Mean ΔBIC':>10} | {'Sim FPR':>8} | {'Excess':>8}"
)
print(header)
print("-" * len(header))

for min_n in thresholds:
    # Filter delta_bic entries
    bic_subset = [d for d in delta_bic_data if d["n_obs"] >= min_n]
    # Filter bivariate fits
    fits_subset = [f for f in fits if f["n_obs"] >= min_n]

    n_pairs_bic = len(bic_subset)
    n_pairs_fits = len(fits_subset)

    if n_pairs_bic == 0:
        continue

    # BIC-based metrics
    delta_bics = [d["delta_bic"] for d in bic_subset]
    pct_dbic_gt2 = 100.0 * sum(1 for d in delta_bics if d > 2) / n_pairs_bic
    pct_dbic_gt6 = 100.0 * sum(1 for d in delta_bics if d > 6) / n_pairs_bic
    mean_dbic = np.mean(delta_bics)

    # Tail dependence metric
    if n_pairs_fits > 0:
        pct_lambda_u = 100.0 * sum(
            1 for f in fits_subset if f.get("lambda_U", 0) > 0.05
        ) / n_pairs_fits
    else:
        pct_lambda_u = float("nan")

    # Simulation-based FPR at the median n of this subset
    median_n = np.median([d["n_obs"] for d in bic_subset])
    expected_fpr = interpolate_fpr(median_n)
    observed_rate = pct_dbic_gt2 / 100.0
    excess_signal = observed_rate - expected_fpr

    row = {
        "min_n": min_n,
        "n_pairs": n_pairs_bic,
        "pct_delta_bic_gt_2": round(pct_dbic_gt2, 1),
        "pct_delta_bic_gt_6": round(pct_dbic_gt6, 1),
        "pct_lambda_U_gt_005": round(pct_lambda_u, 1),
        "mean_delta_bic": round(mean_dbic, 2),
        "median_n_in_subset": float(median_n),
        "expected_fpr_from_sim": round(expected_fpr, 3),
        "observed_nongaussian_rate": round(observed_rate, 3),
        "excess_signal": round(excess_signal, 3),
    }
    results.append(row)

    print(
        f"{min_n:>6} | {n_pairs_bic:>6} | {pct_dbic_gt2:>7.1f}% | {pct_dbic_gt6:>7.1f}% | "
        f"{pct_lambda_u:>9.1f}% | {mean_dbic:>10.2f} | {expected_fpr:>7.1%} | {excess_signal:>+7.1%}"
    )

print()
print("=" * 90)
print("INTERPRETATION")
print("=" * 90)
print()

# Trend analysis
rates_gt2 = [r["pct_delta_bic_gt_2"] for r in results]
rates_gt6 = [r["pct_delta_bic_gt_6"] for r in results]
excess_signals = [r["excess_signal"] for r in results]

if len(rates_gt2) >= 2:
    trend_gt2 = rates_gt2[-1] - rates_gt2[0]
    trend_gt6 = rates_gt6[-1] - rates_gt6[0]
    trend_excess = excess_signals[-1] - excess_signals[0]

    print(f"ΔBIC > 2 rate: {rates_gt2[0]:.1f}% (n≥{thresholds[0]}) → "
          f"{rates_gt2[-1]:.1f}% (n≥{thresholds[-1]})")
    print(f"  → Change: {trend_gt2:+.1f} percentage points")
    print()
    print(f"ΔBIC > 6 rate: {rates_gt6[0]:.1f}% (n≥{thresholds[0]}) → "
          f"{rates_gt6[-1]:.1f}% (n≥{thresholds[-1]})")
    print(f"  → Change: {trend_gt6:+.1f} percentage points")
    print()
    print(f"Excess signal (observed - FPR): {excess_signals[0]:+.3f} (n≥{thresholds[0]}) → "
          f"{excess_signals[-1]:+.3f} (n≥{thresholds[-1]})")
    print(f"  → Change: {trend_excess:+.3f}")
    print()

    if trend_gt2 > 2:
        verdict = "INCREASES with larger n → finding is REAL and STRONGER with more data"
    elif trend_gt2 < -5:
        verdict = "DECREASES substantially → partially driven by small-n noise"
    elif abs(trend_gt2) <= 5:
        verdict = "STABLE across n thresholds → finding is ROBUST"
    else:
        verdict = "MIXED trend → interpret with caution"

    print(f"Verdict: Non-Gaussian rate {verdict}")
    print()

    if all(e > 0 for e in excess_signals):
        print("✓ Excess signal is POSITIVE at all thresholds → genuine non-Gaussian structure")
        print("  exists above and beyond what BIC selection noise would produce.")
    elif excess_signals[-1] > 0:
        print("~ Excess signal is positive for larger-n subsets but not universally.")
    else:
        print("✗ Excess signal vanishes at strict thresholds → findings may be artifactual.")

print()
print("=" * 90)
print("SIMULATION FPR REFERENCE (averaged across ρ = 0.3, 0.5, 0.7, 0.9)")
print("=" * 90)
print()
for n in sim_n_values:
    print(f"  n = {n:>3}: mean FPR = {sim_fpr_means[n]:.1%}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output_path = RESULTS / "sensitivity_by_n.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")
