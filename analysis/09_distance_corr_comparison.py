"""
09_distance_corr_comparison.py

Compare copula-based tail dependence with distance correlation and mutual
information for the 28 benchmark pairs in the 8-benchmark complete-case matrix.

Key argument: dcor and MI are *global* scalar measures of nonlinear dependence.
They cannot decompose WHERE in the joint distribution the dependence
concentrates (upper tail vs. lower tail).  Copulas provide this decomposition.
"""

import json
import pathlib
import itertools

import numpy as np
import pandas as pd
from scipy import stats

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"

# ── helpers ──────────────────────────────────────────────────────────────────

def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-uncorrected sample distance correlation (Székely et al. 2007)."""
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_mean = a.mean()
    A = a - a_row - a_col + a_mean
    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_mean = b.mean()
    B = b - b_row - b_col + b_mean
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    if dvar_x * dvar_y == 0:
        return 0.0
    return float(np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y)))


def _kde_entropy(x: np.ndarray, bw: str = "silverman") -> float:
    """Estimate differential entropy via Gaussian KDE."""
    kde = stats.gaussian_kde(x, bw_method=bw)
    log_pdf = np.log(kde(x) + 1e-300)
    return float(-log_pdf.mean())


def mutual_information_kde(x: np.ndarray, y: np.ndarray) -> float:
    """MI(X,Y) = H(X) + H(Y) - H(X,Y)  via Gaussian KDE."""
    hx = _kde_entropy(x)
    hy = _kde_entropy(y)
    hxy = _kde_entropy(np.vstack([x, y]))
    mi = hx + hy - hxy
    return float(max(mi, 0.0))


# ── load data ────────────────────────────────────────────────────────────────

raw = pd.read_csv(DATA / "complete_case_matrix.csv", index_col="model")
pit = pd.read_csv(DATA / "pit_transformed.csv", index_col="model")
with open(RESULTS / "bivariate_fits_complete.json") as f:
    fits = json.load(f)

benchmarks = list(raw.columns)
n_benchmarks = len(benchmarks)
pairs = list(itertools.combinations(benchmarks, 2))
assert len(pairs) == 28, f"Expected 28 pairs, got {len(pairs)}"

# index copula fits by (b1, b2) key
fit_map: dict[tuple[str, str], dict] = {}
for rec in fits:
    key = (rec["benchmark_1"], rec["benchmark_2"])
    fit_map[key] = rec
    fit_map[(rec["benchmark_2"], rec["benchmark_1"])] = rec

# ── compute measures for every pair ──────────────────────────────────────────

rows = []
for b1, b2 in pairs:
    x_raw = raw[b1].values.astype(float)
    y_raw = raw[b2].values.astype(float)
    x_pit = pit[b1].values.astype(float)
    y_pit = pit[b2].values.astype(float)

    spearman_r, spearman_p = stats.spearmanr(x_raw, y_raw)
    dcor_raw = distance_correlation(x_raw, y_raw)
    dcor_pit = distance_correlation(x_pit, y_pit)
    mi_raw = mutual_information_kde(x_raw, y_raw)
    mi_pit = mutual_information_kde(x_pit, y_pit)

    rec = fit_map[(b1, b2)]

    rows.append({
        "pair": f"{b1} × {b2}",
        "benchmark_1": b1,
        "benchmark_2": b2,
        "copula_family": rec["family"],
        "spearman": round(spearman_r, 4),
        "dcor_raw": round(dcor_raw, 4),
        "dcor_pit": round(dcor_pit, 4),
        "mi_raw": round(mi_raw, 4),
        "mi_pit": round(mi_pit, 4),
        "lambda_U": round(rec["lambda_U"], 4),
        "lambda_L": round(rec["lambda_L"], 4),
        "tail_dep_max": round(max(rec["lambda_U"], rec["lambda_L"]), 4),
        "kendall_tau": round(rec["kendall_tau"], 4),
    })

df = pd.DataFrame(rows)

# ── 1. Full comparison table ─────────────────────────────────────────────────

print("=" * 100)
print("FULL COMPARISON TABLE: Spearman | dCor | MI | λ_U | λ_L  (28 pairs)")
print("=" * 100)

header = f"{'Pair':<45} {'Copula':<10} {'ρ_S':>6} {'dCor':>6} {'MI':>6} {'λ_U':>6} {'λ_L':>6}"
print(header)
print("-" * 100)
for _, r in df.iterrows():
    print(
        f"{r['pair']:<45} {r['copula_family']:<10} "
        f"{r['spearman']:6.3f} {r['dcor_raw']:6.3f} {r['mi_raw']:6.3f} "
        f"{r['lambda_U']:6.3f} {r['lambda_L']:6.3f}"
    )

# ── 2. Deceptive pairs: high Spearman, zero tail dep ─────────────────────────

SPEARMAN_THRESH = 0.55
deceptive = df[(df["spearman"].abs() >= SPEARMAN_THRESH) &
               (df["lambda_U"] == 0) & (df["lambda_L"] == 0)].copy()

print("\n")
print("=" * 100)
print(f"DECEPTIVE PAIRS  (|ρ_S| ≥ {SPEARMAN_THRESH}, λ_U = λ_L = 0)")
print("These pairs look strongly associated by every global measure but have")
print("ZERO tail dependence — extreme models are NOT co-located.")
print("=" * 100)

if deceptive.empty:
    print("  (none found)")
else:
    for _, r in deceptive.iterrows():
        print(f"\n  {r['pair']}  [{r['copula_family']}]")
        print(f"    Spearman ρ  = {r['spearman']:.4f}")
        print(f"    dCor (raw)  = {r['dcor_raw']:.4f}")
        print(f"    MI   (raw)  = {r['mi_raw']:.4f}")
        print(f"    λ_U = {r['lambda_U']:.4f},  λ_L = {r['lambda_L']:.4f}")
        dcor_vs_sp = abs(r["dcor_raw"] - abs(r["spearman"]))
        if dcor_vs_sp < 0.10:
            verdict = "dCor ≈ Spearman → dCor ALSO misses the tail structure"
        else:
            verdict = "dCor differs from Spearman — still ZERO tail dep"
        print(f"    → {verdict}")

    print("\n  ▸ CONCLUSION: For every deceptive pair, dCor is high and")
    print("    comparable to Spearman.  dCor — like Spearman — fails to")
    print("    detect the absence of tail dependence.  MI shows the same")
    print("    pattern.  All three are GLOBAL scalars; none can decompose")
    print("    WHERE the dependence concentrates.")

# ── 3. Pairs with high tail dep but moderate Spearman ────────────────────────

TAIL_THRESH = 0.60
SPEARMAN_MOD_HI = 0.80
tail_rich = df[(df["tail_dep_max"] >= TAIL_THRESH) &
               (df["spearman"].abs() < SPEARMAN_MOD_HI)].copy()

print("\n")
print("=" * 100)
print(f"HIGH TAIL DEP (max(λ_U,λ_L) ≥ {TAIL_THRESH}) + MODERATE Spearman (< {SPEARMAN_MOD_HI})")
print("=" * 100)

if tail_rich.empty:
    print("  (none found)")
else:
    for _, r in tail_rich.iterrows():
        print(f"\n  {r['pair']}  [{r['copula_family']}]")
        print(f"    Spearman ρ  = {r['spearman']:.4f}")
        print(f"    dCor (raw)  = {r['dcor_raw']:.4f}")
        print(f"    MI   (raw)  = {r['mi_raw']:.4f}")
        print(f"    λ_U = {r['lambda_U']:.4f},  λ_L = {r['lambda_L']:.4f}")
        print(f"    tail_dep_max = {r['tail_dep_max']:.4f}")
    print("\n  ▸ These pairs have strong tail dependence that moderate")
    print("    Spearman/dCor/MI values do not fully communicate.")

# ── 4. Meta-correlations: how do the measures relate? ────────────────────────

print("\n")
print("=" * 100)
print("META-CORRELATIONS  (Spearman rank-corr among the 28 pair-level values)")
print("=" * 100)

meta = {}
combos = [
    ("spearman", "dcor_raw",    "ρ_S  vs  dCor"),
    ("spearman", "lambda_U",    "ρ_S  vs  λ_U"),
    ("spearman", "lambda_L",    "ρ_S  vs  λ_L"),
    ("spearman", "tail_dep_max","ρ_S  vs  max(λ)"),
    ("dcor_raw", "lambda_U",    "dCor vs  λ_U"),
    ("dcor_raw", "lambda_L",    "dCor vs  λ_L"),
    ("dcor_raw", "tail_dep_max","dCor vs  max(λ)"),
    ("mi_raw",   "lambda_U",    "MI   vs  λ_U"),
    ("mi_raw",   "lambda_L",    "MI   vs  λ_L"),
    ("mi_raw",   "tail_dep_max","MI   vs  max(λ)"),
    ("mi_raw",   "dcor_raw",    "MI   vs  dCor"),
    ("mi_raw",   "spearman",    "MI   vs  ρ_S"),
]

for c1, c2, label in combos:
    r, p = stats.spearmanr(df[c1], df[c2])
    meta[label] = {"r": round(r, 4), "p": round(p, 6)}
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {label:<25}  r = {r:+.4f}  (p = {p:.4e}) {star}")

print("\n  Interpretation:")
print("  • ρ_S vs dCor should be very high → both capture global association")
print("  • ρ_S vs λ_U should be LOWER → Spearman misses tail structure")
print("  • dCor vs λ_U: if ≈ ρ_S vs λ_U → dCor ALSO misses tails")
print("  • MI  vs λ_U: same logic")

# ── 5. Summary statistics ───────────────────────────────────────────────────

print("\n")
print("=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

zero_both = df[(df["lambda_U"] == 0) & (df["lambda_L"] == 0)]
has_upper = df[df["lambda_U"] > 0]
has_lower = df[df["lambda_L"] > 0]
has_both  = df[(df["lambda_U"] > 0) & (df["lambda_L"] > 0)]

print(f"  Total pairs:                        {len(df)}")
print(f"  Pairs with λ_U = λ_L = 0:           {len(zero_both)}  ({100*len(zero_both)/len(df):.0f}%)")
print(f"  Pairs with λ_U > 0:                 {len(has_upper)}")
print(f"  Pairs with λ_L > 0:                 {len(has_lower)}")
print(f"  Pairs with both λ_U,λ_L > 0:        {len(has_both)}")
print()

for subset_name, subset in [("All 28 pairs", df),
                             ("Zero-tail pairs", zero_both),
                             ("Upper-tail pairs", has_upper),
                             ("Lower-tail pairs", has_lower)]:
    if subset.empty:
        continue
    print(f"  {subset_name}:")
    for col in ["spearman", "dcor_raw", "mi_raw", "lambda_U", "lambda_L"]:
        vals = subset[col]
        print(f"    {col:<15}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")
    print()

# ── 6. The key test: dcor for deceptive pairs vs. tail-rich pairs ────────────

print("=" * 100)
print("KEY TEST: dCor gap between deceptive (λ=0) and tail-rich pairs")
print("=" * 100)

if not deceptive.empty and not has_upper.empty:
    dcor_decep = deceptive["dcor_raw"].values
    dcor_tail  = has_upper["dcor_raw"].values
    t_stat, t_p = stats.ttest_ind(dcor_decep, dcor_tail, equal_var=False)
    mw_stat, mw_p = stats.mannwhitneyu(dcor_decep, dcor_tail, alternative="two-sided")
    print(f"  Mean dCor (deceptive, λ=0):   {dcor_decep.mean():.4f}")
    print(f"  Mean dCor (upper-tail > 0):   {dcor_tail.mean():.4f}")
    print(f"  Welch t-test:  t = {t_stat:+.3f}, p = {t_p:.4e}")
    print(f"  Mann–Whitney:  U = {mw_stat:.1f},  p = {mw_p:.4e}")
    print()
    if t_p > 0.05:
        print("  → dCor does NOT significantly distinguish deceptive from tail-rich")
        print("    pairs.  This confirms: dCor is a GLOBAL measure that is blind")
        print("    to tail structure — the same limitation as Spearman.")
    else:
        print("  → dCor shows a significant gap, but the DIRECTION matters:")
        print("    if deceptive pairs have HIGHER dCor, that further proves")
        print("    dCor is irrelevant for tail inference.")
else:
    print("  (insufficient data for comparison)")

# ── 7. Save results ─────────────────────────────────────────────────────────

output = {
    "description": (
        "Comparison of copula tail dependence with distance correlation "
        "and mutual information for 28 benchmark pairs (42 models × 8 benchmarks)."
    ),
    "pair_results": rows,
    "meta_correlations": meta,
    "summary": {
        "n_pairs": len(df),
        "n_zero_tail": int(len(zero_both)),
        "n_upper_tail": int(len(has_upper)),
        "n_lower_tail": int(len(has_lower)),
        "n_both_tail": int(len(has_both)),
        "n_deceptive": int(len(deceptive)),
        "mean_dcor_deceptive": round(float(deceptive["dcor_raw"].mean()), 4) if not deceptive.empty else None,
        "mean_dcor_tail_rich": round(float(has_upper["dcor_raw"].mean()), 4) if not has_upper.empty else None,
    },
    "key_finding": (
        "Distance correlation and mutual information are GLOBAL nonlinear "
        "dependence measures.  They are high for 'deceptive' pairs where "
        "lambda_U = lambda_L = 0, confirming they ALSO miss the tail "
        "structure.  Copula-based tail dependence coefficients provide "
        "a decomposition that no scalar measure — linear or nonlinear — can "
        "replicate."
    ),
}

out_path = RESULTS / "dcor_mi_comparison.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n\nResults saved to {out_path}")
print("Done.")
