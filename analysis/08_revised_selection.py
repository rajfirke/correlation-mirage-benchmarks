"""
Section 8: Revised benchmark selection experiment.
Addresses reviewer concern: linear regression evaluation is biased toward correlation-based selection.
Adds:
  1. Quantile regression for tail prediction
  2. Ranking fidelity evaluation (Kendall's tau on top/bottom 20%)
  3. Improved copula selection criterion
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

COPULA_FAMILIES = [
    pv.BicopFamily.gaussian, pv.BicopFamily.student,
    pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
    pv.BicopFamily.frank, pv.BicopFamily.joe,
]


def correlation_based_selection(corr_matrix: pd.DataFrame, K: int) -> list:
    """Greedy selection maximizing informativeness (high avg corr, low redundancy)."""
    benchmarks = corr_matrix.columns.tolist()
    abs_corr = corr_matrix.abs()
    mean_corr = abs_corr.mean(axis=1)
    first = mean_corr.idxmax()
    selected = [first]

    for _ in range(K - 1):
        remaining = [b for b in benchmarks if b not in selected]
        best, best_score = None, -1
        for b in remaining:
            max_corr_to_selected = abs_corr.loc[b, selected].max()
            info_score = mean_corr[b] - max_corr_to_selected
            if info_score > best_score:
                best_score = info_score
                best = b
        if best:
            selected.append(best)
    return selected


def pca_based_selection(data: pd.DataFrame, K: int) -> list:
    """Select benchmarks with highest loadings on top-K PCA components."""
    X = StandardScaler().fit_transform(data.values)
    pca = PCA(n_components=min(K, data.shape[1]))
    pca.fit(X)
    loadings = np.abs(pca.components_)
    selected = []
    used = set()
    for k in range(pca.n_components_):
        ranking = np.argsort(-loadings[k])
        for idx in ranking:
            bm = data.columns[idx]
            if bm not in used:
                selected.append(bm)
                used.add(bm)
                break
    while len(selected) < K:
        remaining = [b for b in data.columns if b not in selected]
        if not remaining:
            break
        total_loading = loadings[:, [data.columns.tolist().index(b) for b in remaining]].sum(axis=0)
        selected.append(remaining[np.argmax(total_loading)])
    return selected[:K]


def copula_tail_selection(data: pd.DataFrame, K: int) -> list:
    """
    Improved copula-based selection: maximize DIVERSE tail coverage.
    Key fix: avoid selecting benchmarks that are tail-redundant with each other.
    Uses nonparametric tail dependence estimation for robustness.
    """
    benchmarks = data.columns.tolist()
    n_bm = len(benchmarks)

    tail_upper = np.zeros((n_bm, n_bm))
    tail_lower = np.zeros((n_bm, n_bm))
    spearman = np.zeros((n_bm, n_bm))

    for i, j in itertools.combinations(range(n_bm), 2):
        pair = data.iloc[:, [i, j]].dropna()
        if len(pair) < 15:
            continue

        n = len(pair)
        u = pair.rank(method="average").values / (n + 1)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        q_upper = 0.8
        q_lower = 0.2
        mask_u = u[:, 0] >= q_upper
        mask_l = u[:, 0] <= q_lower

        if mask_u.sum() >= 3:
            tail_upper[i, j] = tail_upper[j, i] = float(
                (u[mask_u, 1] >= q_upper).mean()
            )
        if mask_l.sum() >= 3:
            tail_lower[i, j] = tail_lower[j, i] = float(
                (u[mask_l, 1] <= q_lower).mean()
            )

        spearman[i, j] = spearman[j, i] = abs(float(
            stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
        ))

    tail_max = np.maximum(tail_upper, tail_lower)
    diversity_score = spearman - tail_max

    mean_diversity = diversity_score.mean(axis=1)
    first_idx = int(np.argmax(mean_diversity))
    selected_idx = [first_idx]

    for _ in range(K - 1):
        remaining = [i for i in range(n_bm) if i not in selected_idx]
        best_idx, best_score = None, -np.inf
        for r in remaining:
            max_tail_to_selected = max(
                (tail_max[r, s] for s in selected_idx), default=0
            )
            coverage = diversity_score[r].mean()
            score = coverage - 0.5 * max_tail_to_selected
            if score > best_score:
                best_score = score
                best_idx = r
        if best_idx is not None:
            selected_idx.append(best_idx)

    return [benchmarks[i] for i in selected_idx[:K]]


def random_selection(benchmarks: list, K: int, n_repeats: int = 100,
                     seed: int = 42) -> list:
    rng = np.random.RandomState(seed)
    return [list(rng.choice(benchmarks, K, replace=False)) for _ in range(n_repeats)]


def evaluate_ranking_fidelity(data: pd.DataFrame, selected: list,
                              n_splits: int = 5, seed: int = 42) -> dict:
    """
    RANKING-BASED evaluation: how well does the selected subset preserve
    the ranking of models, especially at the top/bottom?

    Metrics:
    - Kendall's tau between true and predicted rankings (all models)
    - Kendall's tau for top-20% models only
    - Top-K overlap: what fraction of the true top-K are in predicted top-K
    """
    other = [b for b in data.columns if b not in selected]
    if not other:
        return {}

    X = data[selected].values
    tau_all_list = []
    tau_top_list = []
    tau_bottom_list = []
    top_k_overlap_list = []

    kf = KFold(n_splits=min(n_splits, len(data)), shuffle=True, random_state=seed)

    for target_bm in other:
        y = data[target_bm].values
        mask = ~np.isnan(y)
        X_valid, y_valid = X[mask], y[mask]
        if len(y_valid) < 10:
            continue

        preds = np.zeros_like(y_valid)
        for train_idx, test_idx in kf.split(X_valid):
            reg = LinearRegression()
            reg.fit(X_valid[train_idx], y_valid[train_idx])
            preds[test_idx] = reg.predict(X_valid[test_idx])

        true_ranks = stats.rankdata(-y_valid)
        pred_ranks = stats.rankdata(-preds)

        tau_all, _ = stats.kendalltau(true_ranks, pred_ranks)
        tau_all_list.append(tau_all)

        n_models = len(y_valid)
        top_20_mask = true_ranks <= n_models * 0.2
        bottom_20_mask = true_ranks >= n_models * 0.8

        if top_20_mask.sum() >= 3:
            tau_top, _ = stats.kendalltau(
                true_ranks[top_20_mask], pred_ranks[top_20_mask]
            )
            tau_top_list.append(tau_top)

            k = int(top_20_mask.sum())
            true_top_k = set(np.where(top_20_mask)[0])
            pred_top_k = set(np.argsort(pred_ranks)[:k])
            overlap = len(true_top_k & pred_top_k) / k
            top_k_overlap_list.append(overlap)

        if bottom_20_mask.sum() >= 3:
            tau_bot, _ = stats.kendalltau(
                true_ranks[bottom_20_mask], pred_ranks[bottom_20_mask]
            )
            tau_bottom_list.append(tau_bot)

    return {
        "tau_all": float(np.nanmean(tau_all_list)) if tau_all_list else np.nan,
        "tau_top20": float(np.nanmean(tau_top_list)) if tau_top_list else np.nan,
        "tau_bottom20": float(np.nanmean(tau_bottom_list)) if tau_bottom_list else np.nan,
        "top_k_overlap": float(np.nanmean(top_k_overlap_list)) if top_k_overlap_list else np.nan,
    }


def evaluate_quantile_prediction(data: pd.DataFrame, selected: list,
                                  n_splits: int = 5, seed: int = 42) -> dict:
    """
    Evaluate using quantile regression at tau=0.9 and tau=0.1.
    Better suited for tail prediction than OLS.
    """
    other = [b for b in data.columns if b not in selected]
    if not other:
        return {}

    X = data[selected].values
    errors_upper = []
    errors_lower = []

    kf = KFold(n_splits=min(n_splits, len(data)), shuffle=True, random_state=seed)

    for target_bm in other:
        y = data[target_bm].values
        mask = ~np.isnan(y)
        X_valid, y_valid = X[mask], y[mask]
        if len(y_valid) < 10:
            continue

        preds_upper = np.zeros_like(y_valid)
        preds_lower = np.zeros_like(y_valid)

        for train_idx, test_idx in kf.split(X_valid):
            try:
                qr_upper = QuantileRegressor(quantile=0.9, alpha=0.1,
                                             solver="highs")
                qr_upper.fit(X_valid[train_idx], y_valid[train_idx])
                preds_upper[test_idx] = qr_upper.predict(X_valid[test_idx])

                qr_lower = QuantileRegressor(quantile=0.1, alpha=0.1,
                                             solver="highs")
                qr_lower.fit(X_valid[train_idx], y_valid[train_idx])
                preds_lower[test_idx] = qr_lower.predict(X_valid[test_idx])
            except Exception:
                reg = LinearRegression()
                reg.fit(X_valid[train_idx], y_valid[train_idx])
                preds_upper[test_idx] = reg.predict(X_valid[test_idx])
                preds_lower[test_idx] = reg.predict(X_valid[test_idx])

        top_mask = y_valid >= np.percentile(y_valid, 80)
        bottom_mask = y_valid <= np.percentile(y_valid, 20)

        if top_mask.sum() > 0:
            errors_upper.extend(
                np.abs(preds_upper[top_mask] - y_valid[top_mask]).tolist()
            )
        if bottom_mask.sum() > 0:
            errors_lower.extend(
                np.abs(preds_lower[bottom_mask] - y_valid[bottom_mask]).tolist()
            )

    return {
        "qr_mae_top": float(np.mean(errors_upper)) if errors_upper else np.nan,
        "qr_mae_bottom": float(np.mean(errors_lower)) if errors_lower else np.nan,
        "qr_mae_tails": float(np.mean(errors_upper + errors_lower)) if (errors_upper or errors_lower) else np.nan,
    }


def evaluate_mae(data: pd.DataFrame, selected: list,
                 n_splits: int = 5, seed: int = 42) -> dict:
    """Standard MAE evaluation using linear regression (for comparability)."""
    other = [b for b in data.columns if b not in selected]
    if not other:
        return {}

    X = data[selected].values
    errors_all, errors_tail = [], []

    kf = KFold(n_splits=min(n_splits, len(data)), shuffle=True, random_state=seed)

    for target_bm in other:
        y = data[target_bm].values
        mask = ~np.isnan(y)
        X_valid, y_valid = X[mask], y[mask]
        if len(y_valid) < 10:
            continue

        preds = np.zeros_like(y_valid)
        for train_idx, test_idx in kf.split(X_valid):
            reg = LinearRegression()
            reg.fit(X_valid[train_idx], y_valid[train_idx])
            preds[test_idx] = reg.predict(X_valid[test_idx])

        errors_all.extend(np.abs(preds - y_valid).tolist())
        top_mask = y_valid >= np.percentile(y_valid, 80)
        bottom_mask = y_valid <= np.percentile(y_valid, 20)
        if top_mask.sum() > 0:
            errors_tail.extend(np.abs(preds[top_mask] - y_valid[top_mask]).tolist())
        if bottom_mask.sum() > 0:
            errors_tail.extend(np.abs(preds[bottom_mask] - y_valid[bottom_mask]).tolist())

    return {
        "mae_all": float(np.mean(errors_all)) if errors_all else np.nan,
        "mae_tail": float(np.mean(errors_tail)) if errors_tail else np.nan,
    }


def run_revised_experiment(data: pd.DataFrame, K_values: list = [3, 4, 5]):
    """Run the complete revised selection experiment with all evaluation methods."""
    spearman_corr = data.corr(method="spearman")
    results = []

    for K in K_values:
        if K >= data.shape[1]:
            continue
        print(f"\n  === K = {K} ===")

        methods = {
            "Correlation": correlation_based_selection(spearman_corr, K),
            "PCA": pca_based_selection(data, K),
            "Copula-Tail": copula_tail_selection(data, K),
        }

        for name, selected in methods.items():
            mae = evaluate_mae(data, selected)
            ranking = evaluate_ranking_fidelity(data, selected)
            qr = evaluate_quantile_prediction(data, selected)

            result = {
                "K": K, "method": name, "selected": selected,
                **mae, **ranking, **qr,
            }
            results.append(result)

            print(f"    {name:14s}: MAE_all={mae.get('mae_all', 0):.2f}, "
                  f"MAE_tail={mae.get('mae_tail', 0):.2f}, "
                  f"tau_top20={ranking.get('tau_top20', 0):.3f}, "
                  f"overlap={ranking.get('top_k_overlap', 0):.3f}, "
                  f"QR_tails={qr.get('qr_mae_tails', 0):.2f}")

        random_sets = random_selection(data.columns.tolist(), K)
        random_results_list = []
        for rs in random_sets:
            mae = evaluate_mae(data, rs)
            ranking = evaluate_ranking_fidelity(data, rs)
            qr = evaluate_quantile_prediction(data, rs)
            random_results_list.append({**mae, **ranking, **qr})

        avg_random = {
            key: float(np.nanmean([r.get(key, np.nan) for r in random_results_list]))
            for key in ["mae_all", "mae_tail", "tau_all", "tau_top20",
                       "tau_bottom20", "top_k_overlap", "qr_mae_top",
                       "qr_mae_bottom", "qr_mae_tails"]
        }
        results.append({"K": K, "method": "Random", "selected": "100 samples", **avg_random})
        print(f"    {'Random':14s}: MAE_all={avg_random['mae_all']:.2f}, "
              f"MAE_tail={avg_random['mae_tail']:.2f}, "
              f"tau_top20={avg_random['tau_top20']:.3f}, "
              f"overlap={avg_random['top_k_overlap']:.3f}, "
              f"QR_tails={avg_random['qr_mae_tails']:.2f}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 8: Revised Benchmark Selection Experiment")
    print("=" * 70)

    cc = pd.read_csv(DATA_DIR / "complete_case_matrix.csv", index_col=0)
    print(f"  Data: {cc.shape[0]} models x {cc.shape[1]} benchmarks")
    print(f"  Benchmarks: {list(cc.columns)}")

    results = run_revised_experiment(cc, K_values=[3, 4, 5])

    with open(RESULTS_DIR / "revised_selection_experiment.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n  === SUMMARY TABLE ===")
    df = pd.DataFrame(results)
    for K in sorted(df["K"].unique()):
        print(f"\n  K={K}:")
        sub = df[df["K"] == K]
        for _, row in sub.iterrows():
            print(f"    {row['method']:14s} | "
                  f"MAE_all={row.get('mae_all', 0):5.2f} | "
                  f"MAE_tail={row.get('mae_tail', 0):5.2f} | "
                  f"tau_top20={row.get('tau_top20', 0):+.3f} | "
                  f"top_overlap={row.get('top_k_overlap', 0):.3f} | "
                  f"QR_tails={row.get('qr_mae_tails', 0):5.2f}")

    print("\n" + "=" * 70)
    print("SECTION 8 COMPLETE")
    print("=" * 70)
