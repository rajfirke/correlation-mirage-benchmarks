"""
Section 5: Benchmark selection experiment.
Compares copula-based, correlation-based, PCA-based, and random selection.
KEY experiment for boosting P(Accept).
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    """
    Greedy selection minimizing max pairwise correlation.
    Standard approach from Redundancy Principles (ACL 2025).
    """
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
    """
    Select benchmarks with highest loadings on top-K PCA components.
    BenchScope-style approach.
    """
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
        best_idx = np.argmax(total_loading)
        selected.append(remaining[best_idx])
    return selected[:K]


def copula_based_selection(data: pd.DataFrame, K: int) -> list:
    """
    Copula-based selection: maximize tail dependence coverage.
    Greedy: pick benchmark with highest avg tail dependence to others,
    then iteratively add benchmarks that contribute NEW tail information.
    """
    benchmarks = data.columns.tolist()
    n = len(data)

    tail_dep = {}
    for i, j in itertools.combinations(range(len(benchmarks)), 2):
        pair_data = data.iloc[:, [i, j]].dropna()
        if len(pair_data) < 15:
            continue
        u = pair_data.rank(method="average").values / (len(pair_data) + 1)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        try:
            cop = pv.Bicop()
            controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
            cop.select(u, controls)

            fam = cop.family
            par = cop.parameters
            lam_U = 0.0
            if fam == pv.BicopFamily.gumbel:
                lam_U = 2 - 2 ** (1.0 / par[0, 0])
            elif fam == pv.BicopFamily.joe:
                lam_U = 2 - 2 ** (1.0 / par[0, 0])
            elif fam == pv.BicopFamily.student:
                rho = par[0, 0]
                nu = par[0, 1] if par.shape[1] > 1 else par[1, 0]
                from scipy.stats import t as tdist
                lam_U = 2 * tdist.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), df=nu+1)
            elif fam == pv.BicopFamily.bb1:
                delta = par[0, 1] if par.shape[1] > 1 else par[1, 0]
                lam_U = 2 - 2 ** (1.0 / delta)
            elif fam == pv.BicopFamily.bb7:
                theta = par[0, 0]
                lam_U = 2 - 2 ** (1.0 / theta)

            tail_dep[(benchmarks[i], benchmarks[j])] = lam_U
            tail_dep[(benchmarks[j], benchmarks[i])] = lam_U
        except Exception:
            pass

    avg_tail = {}
    for bm in benchmarks:
        deps = [v for (a, b), v in tail_dep.items() if a == bm]
        avg_tail[bm] = np.mean(deps) if deps else 0

    first = max(avg_tail, key=avg_tail.get)
    selected = [first]

    for _ in range(K - 1):
        remaining = [b for b in benchmarks if b not in selected]
        best, best_score = None, -1
        for b in remaining:
            new_tail_info = sum(
                tail_dep.get((b, s), 0) for s in selected
            ) / len(selected)
            coverage_bonus = avg_tail[b]
            redundancy_penalty = max(
                (tail_dep.get((b, s), 0) for s in selected), default=0
            )
            score = coverage_bonus + 0.5 * new_tail_info - 0.3 * redundancy_penalty
            if score > best_score:
                best_score = score
                best = b
        if best:
            selected.append(best)

    return selected[:K]


def random_selection(benchmarks: list, K: int, n_repeats: int = 100, seed: int = 42) -> list:
    """Random baseline: return all random selections for averaging."""
    rng = np.random.RandomState(seed)
    return [list(rng.choice(benchmarks, K, replace=False)) for _ in range(n_repeats)]


def evaluate_selection(data: pd.DataFrame, selected: list, n_splits: int = 5, seed: int = 42):
    """
    Evaluate a benchmark selection by predicting held-out benchmarks.
    Uses K-fold CV on models.
    """
    other = [b for b in data.columns if b not in selected]
    if not other:
        return {"rmse_all": 0, "mae_all": 0, "rmse_tail": 0, "mae_tail": 0}

    X = data[selected].values
    errors_all = []
    errors_top = []
    errors_bottom = []

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
            errors_top.extend(np.abs(preds[top_mask] - y_valid[top_mask]).tolist())
        if bottom_mask.sum() > 0:
            errors_bottom.extend(np.abs(preds[bottom_mask] - y_valid[bottom_mask]).tolist())

    return {
        "rmse_all": float(np.sqrt(np.mean(np.array(errors_all) ** 2))) if errors_all else np.nan,
        "mae_all": float(np.mean(errors_all)) if errors_all else np.nan,
        "rmse_tail": float(np.sqrt(np.mean(
            (np.array(errors_top + errors_bottom)) ** 2
        ))) if (errors_top or errors_bottom) else np.nan,
        "mae_tail": float(np.mean(errors_top + errors_bottom)) if (errors_top or errors_bottom) else np.nan,
        "mae_top": float(np.mean(errors_top)) if errors_top else np.nan,
        "mae_bottom": float(np.mean(errors_bottom)) if errors_bottom else np.nan,
        "n_predictions": len(errors_all),
    }


def run_experiment(data: pd.DataFrame, K_values: list = [3, 4, 5, 6]):
    """Run the full benchmark selection comparison experiment."""
    spearman_corr = data.corr(method="spearman")
    results = []

    for K in K_values:
        if K >= data.shape[1]:
            continue
        print(f"\n  === K = {K} benchmarks ===")

        methods = {}
        methods["Correlation"] = correlation_based_selection(spearman_corr, K)
        methods["PCA"] = pca_based_selection(data, K)
        methods["Copula"] = copula_based_selection(data, K)

        for name, selected in methods.items():
            metrics = evaluate_selection(data, selected)
            print(f"    {name:12s}: {selected}")
            print(f"      MAE_all={metrics['mae_all']:.2f}, MAE_tail={metrics['mae_tail']:.2f}, "
                  f"MAE_top={metrics['mae_top']:.2f}, MAE_bottom={metrics['mae_bottom']:.2f}")
            results.append({
                "K": K,
                "method": name,
                "selected": selected,
                **metrics,
            })

        random_sets = random_selection(data.columns.tolist(), K)
        random_metrics = []
        for rs in random_sets:
            m = evaluate_selection(data, rs)
            random_metrics.append(m)

        avg_random = {
            key: float(np.nanmean([m[key] for m in random_metrics]))
            for key in ["rmse_all", "mae_all", "rmse_tail", "mae_tail", "mae_top", "mae_bottom"]
        }
        avg_random["n_predictions"] = int(np.mean([m["n_predictions"] for m in random_metrics]))
        print(f"    {'Random':12s}: (averaged over {len(random_sets)} samples)")
        print(f"      MAE_all={avg_random['mae_all']:.2f}, MAE_tail={avg_random['mae_tail']:.2f}")
        results.append({
            "K": K,
            "method": "Random",
            "selected": "100 random samples",
            **avg_random,
        })

    return results


def statistical_tests(results: list):
    """Compare methods pairwise for significance."""
    df = pd.DataFrame(results)
    comparisons = []

    for K in df["K"].unique():
        sub = df[df["K"] == K]
        copula_row = sub[sub["method"] == "Copula"]
        if copula_row.empty:
            continue

        for other in ["Correlation", "PCA", "Random"]:
            other_row = sub[sub["method"] == other]
            if other_row.empty:
                continue

            diff_all = float(other_row["mae_all"].values[0] - copula_row["mae_all"].values[0])
            diff_tail = float(other_row["mae_tail"].values[0] - copula_row["mae_tail"].values[0])

            comparisons.append({
                "K": int(K),
                "comparison": f"Copula vs {other}",
                "mae_all_improvement": diff_all,
                "mae_tail_improvement": diff_tail,
                "copula_better_all": diff_all > 0,
                "copula_better_tail": diff_tail > 0,
            })

    return comparisons


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 5: Benchmark Selection Experiment")
    print("=" * 70)

    cc = pd.read_csv(DATA_DIR / "complete_case_matrix.csv", index_col=0)
    print(f"  Data: {cc.shape[0]} models x {cc.shape[1]} benchmarks")
    print(f"  Benchmarks: {list(cc.columns)}")

    print("\n--- 5.1-5.3: Running Selection Experiment ---")
    results = run_experiment(cc, K_values=[3, 4, 5, 6])

    with open(RESULTS_DIR / "selection_experiment.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- 5.4: Statistical Comparisons ---")
    comparisons = statistical_tests(results)
    for c in comparisons:
        direction_all = "BETTER" if c["copula_better_all"] else "worse"
        direction_tail = "BETTER" if c["copula_better_tail"] else "worse"
        print(f"  K={c['K']}, {c['comparison']}: "
              f"MAE_all {direction_all} by {abs(c['mae_all_improvement']):.2f}, "
              f"MAE_tail {direction_tail} by {abs(c['mae_tail_improvement']):.2f}")

    with open(RESULTS_DIR / "selection_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    print("\n--- Practical Recommendation ---")
    df = pd.DataFrame(results)
    for K in sorted(df["K"].unique()):
        sub = df[(df["K"] == K) & (df["method"] != "Random")]
        best = sub.loc[sub["mae_tail"].idxmin()]
        print(f"  K={K}: Best method = {best['method']} "
              f"(MAE_tail={best['mae_tail']:.2f}), selected: {best['selected']}")

    print("\n" + "=" * 70)
    print("SECTION 5 COMPLETE")
    print("=" * 70)
