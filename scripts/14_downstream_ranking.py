"""
Task 1B: Downstream Ranking Validation Experiment

Demonstrates that the copula-identified "deceptive pairs" have concrete
consequences for model selection: benchmarks that appear redundant by
bulk Spearman provide non-redundant ranking information among top models.

Four analyses:
  1. Drop-one: impact of removing each benchmark on top-model rankings
  2. Pairwise ranking agreement: within-top-N Kendall's tau between
     individual benchmark rankings, compared by copula family
  3. Model-selection disagreement: pairwise preference reversal rate
     among top-50 models for deceptive vs non-deceptive pairs
  4. Tail-aware subset selection: using conditional Spearman (ρ_upper)
     instead of bulk ρ as the diversity criterion
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
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    score_matrix = pd.read_csv(DATA_DIR / "ollm_score_matrix.csv", index_col=0)
    with open(RESULTS_DIR / "ollm_bivariate_fits.json") as f:
        bivariate_fits = json.load(f)
    with open(RESULTS_DIR / "ollm_nonparametric_tails.json") as f:
        nonparam_tails = json.load(f)
    return score_matrix, bivariate_fits, nonparam_tails


def composite_ranking(score_matrix, benchmarks=None):
    if benchmarks is None:
        benchmarks = score_matrix.columns.tolist()
    composite = score_matrix[benchmarks].mean(axis=1)
    return composite.rank(ascending=False, method="min")


def ranking_disruption(full_rank, reduced_rank, top_n_values=(10, 20, 50)):
    tau_all, _ = stats.kendalltau(full_rank, reduced_rank)
    results = {"tau_all": float(tau_all)}

    for n in top_n_values:
        top_models = full_rank.nsmallest(n).index
        tau, _ = stats.kendalltau(
            full_rank.loc[top_models], reduced_rank.loc[top_models]
        )
        results[f"tau_top{n}"] = float(tau) if not np.isnan(tau) else 0.0

        reduced_top = set(reduced_rank.nsmallest(n).index)
        full_top = set(top_models)
        results[f"overlap_top{n}"] = len(reduced_top & full_top) / n

        displacements = (
            reduced_rank.loc[top_models] - full_rank.loc[top_models]
        ).abs()
        results[f"max_disp_top{n}"] = int(displacements.max())
        results[f"mean_disp_top{n}"] = float(displacements.mean())

    results["top1_preserved"] = bool(full_rank.idxmin() == reduced_rank.idxmin())
    return results


# ============================================================
# ANALYSIS 1: DROP-ONE
# ============================================================
def drop_one_analysis(score_matrix):
    benchmarks = score_matrix.columns.tolist()
    full_rank = composite_ranking(score_matrix)

    results = []
    for bm in benchmarks:
        remaining = [b for b in benchmarks if b != bm]
        reduced_rank = composite_ranking(score_matrix, remaining)
        disruption = ranking_disruption(full_rank, reduced_rank)
        disruption["dropped"] = bm
        results.append(disruption)

    return results


# ============================================================
# ANALYSIS 2: PAIRWISE RANKING AGREEMENT WITHIN TOP-N
# ============================================================
def pairwise_ranking_agreement(score_matrix, bivariate_fits):
    """For each benchmark pair, compute Kendall's tau between their
    individual rankings WITHIN the top-N models (by composite).
    Deceptive pairs should show lower within-top agreement."""
    benchmarks = score_matrix.columns.tolist()
    composite = score_matrix.mean(axis=1)

    results = []
    for fit in bivariate_fits:
        bm1, bm2 = fit["benchmark_1"], fit["benchmark_2"]
        rho_bulk = fit["spearman_rho"]
        family = fit["family"]
        lambda_U = fit["lambda_U"]
        is_deceptive = family == "Frank" and rho_bulk > 0.7

        rank_a_full = score_matrix[bm1].rank(ascending=False)
        rank_b_full = score_matrix[bm2].rank(ascending=False)
        tau_all, _ = stats.kendalltau(rank_a_full, rank_b_full)

        row = {
            "pair": f"{bm1} x {bm2}",
            "rho_bulk": float(rho_bulk),
            "family": family,
            "lambda_U": float(lambda_U),
            "is_deceptive": is_deceptive,
            "tau_all_models": float(tau_all),
        }

        for top_n in [20, 50, 100, 200]:
            top_models = composite.nlargest(top_n).index
            rank_a_top = score_matrix.loc[top_models, bm1].rank(ascending=False)
            rank_b_top = score_matrix.loc[top_models, bm2].rank(ascending=False)
            tau_top, _ = stats.kendalltau(rank_a_top, rank_b_top)
            row[f"tau_top{top_n}"] = float(tau_top) if not np.isnan(tau_top) else 0.0

        row["tau_ratio_top50"] = (
            row["tau_top50"] / row["tau_all_models"]
            if row["tau_all_models"] > 0.01
            else float("nan")
        )

        results.append(row)

    return results


# ============================================================
# ANALYSIS 3: MODEL-SELECTION DISAGREEMENT RATE
# ============================================================
def model_selection_disagreement(score_matrix, bivariate_fits, top_n=50):
    """Among top-N models, for each model pair (m1, m2):
    does benchmark A prefer the same model as benchmark B?
    Disagreement rate measures practical redundancy loss."""
    composite = score_matrix.mean(axis=1)
    top_models = composite.nlargest(top_n).index
    model_pairs = list(itertools.combinations(top_models, 2))
    n_pairs = len(model_pairs)

    results = []
    for fit in bivariate_fits:
        bm1, bm2 = fit["benchmark_1"], fit["benchmark_2"]
        rho_bulk = fit["spearman_rho"]
        family = fit["family"]
        is_deceptive = family == "Frank" and rho_bulk > 0.7

        scores_a = score_matrix.loc[top_models, bm1]
        scores_b = score_matrix.loc[top_models, bm2]

        disagreements = 0
        for m1, m2 in model_pairs:
            a_prefers = scores_a[m1] > scores_a[m2]
            b_prefers = scores_b[m1] > scores_b[m2]
            if a_prefers != b_prefers:
                disagreements += 1

        disagreement_rate = disagreements / n_pairs

        results.append(
            {
                "pair": f"{bm1} x {bm2}",
                "rho_bulk": float(rho_bulk),
                "family": family,
                "lambda_U": float(fit["lambda_U"]),
                "is_deceptive": is_deceptive,
                "n_model_pairs": n_pairs,
                "disagreements": disagreements,
                "disagreement_rate": float(disagreement_rate),
            }
        )

    return results


# ============================================================
# ANALYSIS 4: TAIL-AWARE SUBSET SELECTION
# ============================================================
def tail_aware_subset_selection(
    score_matrix, bivariate_fits, nonparam_tails, K_values=(3, 4, 5)
):
    """Compare subset selection using bulk ρ vs conditional ρ_upper as
    the redundancy metric. The tail-aware criterion uses ρ_upper20 instead
    of ρ_bulk to measure benchmark diversity."""
    benchmarks = score_matrix.columns.tolist()
    full_rank = composite_ranking(score_matrix)

    rho_bulk = {}
    rho_upper = {}
    for fit in bivariate_fits:
        key = (fit["benchmark_1"], fit["benchmark_2"])
        rho_bulk[key] = abs(fit["spearman_rho"])
        rho_bulk[(key[1], key[0])] = abs(fit["spearman_rho"])
    for np_entry in nonparam_tails:
        key = (np_entry["benchmark_1"], np_entry["benchmark_2"])
        rho_upper[key] = abs(np_entry["spearman_upper_20pct"])
        rho_upper[(key[1], key[0])] = abs(np_entry["spearman_upper_20pct"])

    results = []
    for K in K_values:
        all_subsets = list(itertools.combinations(benchmarks, K))
        subset_evals = []

        for subset in all_subsets:
            subset = list(subset)
            bulk_redund = sum(
                rho_bulk.get((bi, bj), 0)
                for bi, bj in itertools.combinations(subset, 2)
            )
            tail_redund = sum(
                rho_upper.get((bi, bj), 0)
                for bi, bj in itertools.combinations(subset, 2)
            )

            reduced_rank = composite_ranking(score_matrix, subset)
            disruption = ranking_disruption(full_rank, reduced_rank)

            subset_evals.append(
                {
                    "subset": subset,
                    "bulk_redundancy": float(bulk_redund),
                    "tail_redundancy": float(tail_redund),
                    **disruption,
                }
            )

        bulk_best = min(subset_evals, key=lambda x: x["bulk_redundancy"])
        tail_best = min(subset_evals, key=lambda x: x["tail_redundancy"])
        actual_best_top20 = max(subset_evals, key=lambda x: x["tau_top20"])

        results.append(
            {
                "K": K,
                "n_subsets": len(all_subsets),
                "bulk_best": bulk_best,
                "tail_best": tail_best,
                "actual_best_top20": actual_best_top20,
                "all_subsets": subset_evals,
            }
        )

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DOWNSTREAM RANKING VALIDATION EXPERIMENT")
    print("=" * 70)

    score_matrix, bivariate_fits, nonparam_tails = load_data()
    print(f"Loaded: {score_matrix.shape[0]} models x {score_matrix.shape[1]} benchmarks")

    # --- Analysis 1: Drop-one ---
    print("\n" + "=" * 70)
    print("ANALYSIS 1: DROP-ONE IMPACT ON TOP-MODEL RANKINGS")
    print("=" * 70)
    drop_one = drop_one_analysis(score_matrix)
    for r in sorted(drop_one, key=lambda x: x["tau_top20"]):
        ratio = r["tau_top20"] / r["tau_all"] if r["tau_all"] > 0 else 0
        print(
            f"  Drop {r['dropped']:12s}: "
            f"tau_all={r['tau_all']:.4f}  tau_top20={r['tau_top20']:.3f}  "
            f"ratio={ratio:.3f}  "
            f"overlap_top10={r['overlap_top10']:.0%}  "
            f"top1={'OK' if r['top1_preserved'] else 'CHANGED'}"
        )

    # --- Analysis 2: Pairwise ranking agreement ---
    print("\n" + "=" * 70)
    print("ANALYSIS 2: PAIRWISE RANKING AGREEMENT WITHIN TOP-N")
    print("=" * 70)
    agreement = pairwise_ranking_agreement(score_matrix, bivariate_fits)

    deceptive_pairs = [r for r in agreement if r["is_deceptive"]]
    non_deceptive_high = [
        r for r in agreement if not r["is_deceptive"] and r["rho_bulk"] > 0.7
    ]
    all_pairs = agreement

    print("\n  HIGH-ρ DECEPTIVE PAIRS (Frank, ρ > 0.7):")
    for r in sorted(deceptive_pairs, key=lambda x: -x["rho_bulk"]):
        print(
            f"    {r['pair']:24s} ρ={r['rho_bulk']:.3f}  "
            f"τ_all={r['tau_all_models']:.3f}  τ_top50={r['tau_top50']:.3f}  "
            f"τ_top20={r['tau_top20']:.3f}  ratio={r['tau_ratio_top50']:.3f}"
        )

    print("\n  HIGH-ρ NON-DECEPTIVE PAIRS (λ_U > 0, ρ > 0.7):")
    for r in sorted(non_deceptive_high, key=lambda x: -x["rho_bulk"]):
        print(
            f"    {r['pair']:24s} ρ={r['rho_bulk']:.3f}  "
            f"τ_all={r['tau_all_models']:.3f}  τ_top50={r['tau_top50']:.3f}  "
            f"τ_top20={r['tau_top20']:.3f}  ratio={r['tau_ratio_top50']:.3f}"
        )

    if deceptive_pairs and non_deceptive_high:
        dec_ratios = [r["tau_ratio_top50"] for r in deceptive_pairs]
        nondec_ratios = [
            r["tau_ratio_top50"]
            for r in non_deceptive_high
            if not np.isnan(r["tau_ratio_top50"])
        ]
        u_stat, p_val = stats.mannwhitneyu(
            dec_ratios, nondec_ratios, alternative="less"
        )
        print(
            f"\n  Mann-Whitney (tau_ratio deceptive < non-deceptive): "
            f"U={u_stat:.1f}, p={p_val:.4f}"
        )
        print(
            f"  Deceptive mean ratio: {np.mean(dec_ratios):.3f}  "
            f"Non-deceptive mean ratio: {np.mean(nondec_ratios):.3f}"
        )

    # --- Analysis 3: Model-selection disagreement ---
    print("\n" + "=" * 70)
    print("ANALYSIS 3: MODEL-SELECTION DISAGREEMENT (top-50 models)")
    print("=" * 70)
    disagreement = model_selection_disagreement(score_matrix, bivariate_fits, top_n=50)
    for r in sorted(disagreement, key=lambda x: -x["disagreement_rate"]):
        tag = "DECEPT" if r["is_deceptive"] else "      "
        print(
            f"  {tag} {r['pair']:24s} ρ={r['rho_bulk']:.3f}  "
            f"disagree={r['disagreement_rate']:.1%}  "
            f"({r['disagreements']}/{r['n_model_pairs']} pairs)"
        )

    dec_dis = [r for r in disagreement if r["is_deceptive"]]
    nondec_dis_high = [
        r for r in disagreement if not r["is_deceptive"] and r["rho_bulk"] > 0.7
    ]
    if dec_dis and nondec_dis_high:
        dec_rates = [r["disagreement_rate"] for r in dec_dis]
        nondec_rates = [r["disagreement_rate"] for r in nondec_dis_high]
        u_stat, p_val = stats.mannwhitneyu(
            dec_rates, nondec_rates, alternative="greater"
        )
        print(
            f"\n  Mann-Whitney (disagreement deceptive > non-deceptive): "
            f"U={u_stat:.1f}, p={p_val:.4f}"
        )
        print(
            f"  Deceptive mean: {np.mean(dec_rates):.1%}  "
            f"Non-deceptive mean: {np.mean(nondec_rates):.1%}"
        )

    # --- Analysis 4: Tail-aware subset selection ---
    print("\n" + "=" * 70)
    print("ANALYSIS 4: TAIL-AWARE vs BULK SUBSET SELECTION")
    print("=" * 70)
    subset_results = tail_aware_subset_selection(
        score_matrix, bivariate_fits, nonparam_tails
    )
    for res in subset_results:
        K = res["K"]
        print(f"\n  === K = {K} ({res['n_subsets']} subsets) ===")

        bb = res["bulk_best"]
        print(f"  Bulk-ρ best (min Σ|ρ_bulk|):")
        print(f"    Subset: {bb['subset']}")
        print(
            f"    tau_top20={bb['tau_top20']:.3f}  tau_top50={bb['tau_top50']:.3f}  "
            f"overlap_top10={bb['overlap_top10']:.0%}  "
            f"overlap_top20={bb['overlap_top20']:.0%}"
        )

        tb = res["tail_best"]
        print(f"  Tail-ρ best (min Σ|ρ_upper20|):")
        print(f"    Subset: {tb['subset']}")
        print(
            f"    tau_top20={tb['tau_top20']:.3f}  tau_top50={tb['tau_top50']:.3f}  "
            f"overlap_top10={tb['overlap_top10']:.0%}  "
            f"overlap_top20={tb['overlap_top20']:.0%}"
        )

        ab = res["actual_best_top20"]
        print(f"  Actually-best (max tau_top20):")
        print(f"    Subset: {ab['subset']}")
        print(
            f"    tau_top20={ab['tau_top20']:.3f}  tau_top50={ab['tau_top50']:.3f}  "
            f"overlap_top10={ab['overlap_top10']:.0%}  "
            f"overlap_top20={ab['overlap_top20']:.0%}"
        )

        if bb["subset"] != tb["subset"]:
            delta = tb["tau_top20"] - bb["tau_top20"]
            print(f"  ** Tail-aware advantage (tau_top20): {delta:+.3f} **")

    # --- Save all results ---
    all_results = {
        "drop_one": drop_one,
        "pairwise_agreement": agreement,
        "model_selection_disagreement": disagreement,
        "tail_aware_subsets": [
            {
                "K": r["K"],
                "n_subsets": r["n_subsets"],
                "bulk_best": r["bulk_best"],
                "tail_best": r["tail_best"],
                "actual_best_top20": r["actual_best_top20"],
            }
            for r in subset_results
        ],
    }

    out_path = RESULTS_DIR / "downstream_ranking_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to {out_path}")
