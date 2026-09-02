"""
Section 1.2 + 1.3: Download benchmark score matrices and compute baseline statistics.
Sources: BenchPress (HuggingFace), Open LLM Leaderboard v2, anadim/llm-benchmark-matrix.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def download_benchpress():
    """Download BenchPress score matrix from HuggingFace."""
    from datasets import load_dataset

    print("[1/3] Downloading BenchPress score matrix...")
    configs = ["scores_all", "scores_paper", "models", "benchmarks"]
    all_dfs = {}
    for cfg in configs:
        try:
            ds = load_dataset("yzeng58/benchpress-score-matrix", cfg, split="train")
            df = ds.to_pandas()
            out = DATA_DIR / f"benchpress_{cfg}.csv"
            df.to_csv(out, index=False)
            print(f"  {cfg}: {df.shape[0]} rows x {df.shape[1]} cols -> {out.name}")
            all_dfs[cfg] = df
        except Exception as e:
            print(f"  {cfg}: FAILED ({e})")
    return all_dfs


def download_open_llm_leaderboard():
    """Download Open LLM Leaderboard v2 results."""
    from datasets import load_dataset

    print("[2/3] Downloading Open LLM Leaderboard v2...")
    try:
        ds = load_dataset("open-llm-leaderboard/results", split="train")
        df = ds.to_pandas()
        out = DATA_DIR / "ollm_v2_raw.csv"
        df.to_csv(out, index=False)
        print(f"  Saved: {df.shape[0]} rows x {df.shape[1]} cols -> {out.name}")
        return df
    except Exception as e:
        print(f"  Could not download OLLM v2: {e}")
        print("  Will proceed with BenchPress as primary dataset.")
        return None


def download_anadim_matrix():
    """Download anadim/llm-benchmark-matrix from GitHub."""
    import urllib.request

    print("[3/3] Downloading anadim/llm-benchmark-matrix...")
    url = "https://raw.githubusercontent.com/anadim/llm-benchmark-matrix/main/benchmark_matrix.csv"
    out = DATA_DIR / "anadim_benchmark_matrix.csv"
    try:
        urllib.request.urlretrieve(url, out)
        df = pd.read_csv(out)
        print(f"  Saved: {df.shape[0]} rows x {df.shape[1]} cols -> {out.name}")
        return df
    except Exception as e:
        print(f"  Could not download anadim matrix: {e}")
        return None


def build_wide_matrix(data_dir: Path):
    """
    Build wide-format score matrices from BenchPress long-format data.
    Uses scores_all (larger, 188 models x 316 benchmarks) as primary.
    Returns wide DataFrame.
    """
    print("\n=== Building Wide Score Matrix ===")

    for fname in ["benchpress_scores_all.csv", "benchpress_scores_paper.csv"]:
        path = data_dir / fname
        if path.exists():
            raw = pd.read_csv(path)
            print(f"  Source: {fname} ({raw.shape[0]} entries)")
            print(f"  Unique models: {raw['model_name'].nunique()}")
            print(f"  Unique benchmarks: {raw['benchmark_name'].nunique()}")

            wide = raw.pivot_table(
                index="model_name", columns="benchmark_name",
                values="score", aggfunc="first",
            )
            wide.index.name = "model"
            out = data_dir / fname.replace("scores_", "wide_").replace(".csv", ".csv")
            wide.to_csv(out)
            print(f"  Pivoted: {wide.shape[0]} x {wide.shape[1]}, saved -> {out.name}")

    wide_all = pd.read_csv(data_dir / "benchpress_wide_all.csv", index_col=0)
    fill_rate = wide_all.notna().sum().sum() / (wide_all.shape[0] * wide_all.shape[1])
    print(f"\n  Primary wide matrix: {wide_all.shape[0]} models x {wide_all.shape[1]} benchmarks")
    print(f"  Fill rate: {fill_rate:.1%}")

    bench_fill = wide_all.notna().sum(axis=0).sort_values(ascending=False)
    print(f"  Top-15 most populated benchmarks:")
    for bm, n in bench_fill.head(15).items():
        print(f"    {bm}: {n}/{wide_all.shape[0]} ({n/wide_all.shape[0]:.0%})")

    return wide_all


def compute_complete_case_matrix(wide: pd.DataFrame, min_models: int = 25):
    """
    Build TWO complete-case submatrices for copula analysis:
    1. A DENSE matrix (more models, fewer benchmarks) for vine copula
    2. A BROAD matrix (fewer models, more benchmarks) for benchmark selection
    Also builds pairwise-complete counts for bivariate analysis.
    """
    print("\n=== Building Complete-Case Submatrices ===")

    bench_fill = wide.notna().sum(axis=0).sort_values(ascending=False)

    configs = {}
    for n_bm in range(30, 4, -1):
        cols = bench_fill.head(n_bm).index.tolist()
        cc = wide[cols].dropna()
        if cc.shape[0] >= min_models:
            configs[n_bm] = cc.shape[0]

    print("  Feasible complete-case configurations (benchmarks -> models):")
    for nb, nm in sorted(configs.items(), reverse=True)[:10]:
        print(f"    {nb} benchmarks -> {nm} models")

    dense_nb = max((nb for nb, nm in configs.items() if nm >= 40), default=6)
    broad_nb = max(configs.keys())
    dense = wide[bench_fill.head(dense_nb).index.tolist()].dropna()
    broad = wide[bench_fill.head(broad_nb).index.tolist()].dropna()

    for label, df, fname in [
        ("Dense", dense, "complete_dense.csv"),
        ("Broad", broad, "complete_broad.csv"),
    ]:
        const_cols = df.columns[df.std() < 1e-10]
        if len(const_cols) > 0:
            df = df.drop(columns=const_cols)
        print(f"\n  {label} matrix: {df.shape[0]} models x {df.shape[1]} benchmarks")
        print(f"  Benchmarks: {list(df.columns)}")
        df.to_csv(DATA_DIR / fname)

    pw = wide.notna().astype(int)
    pairwise_n = pw.T.dot(pw)
    pairwise_n.to_csv(RESULTS_DIR / "pairwise_complete_counts.csv")
    n_pairs_30plus = (pairwise_n.where(np.triu(np.ones(pairwise_n.shape), k=1).astype(bool)) >= 30).sum().sum()
    print(f"\n  Pairwise co-observation matrix saved. Pairs with n>=30: {n_pairs_30plus}")

    primary = dense
    primary.to_csv(DATA_DIR / "complete_case_matrix.csv")
    return primary


def compute_baselines(cc: pd.DataFrame):
    """Section 1.3: Pearson/Spearman correlation, PCA, effective dimensionality."""
    print("\n=== Computing Baseline Statistics ===")

    pearson = cc.corr(method="pearson")
    spearman = cc.corr(method="spearman")

    pearson.to_csv(RESULTS_DIR / "correlation_pearson.csv")
    spearman.to_csv(RESULTS_DIR / "correlation_spearman.csv")
    print(f"  Pearson correlation: {pearson.shape}")
    print(f"  Spearman correlation: {spearman.shape}")

    mean_spearman = spearman.where(np.triu(np.ones(spearman.shape), k=1).astype(bool)).stack().mean()
    print(f"  Mean pairwise Spearman: {mean_spearman:.3f}")

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(cc.values)
    pca = PCA()
    pca.fit(X)
    explained = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

    ed_results = {
        "n_models": int(cc.shape[0]),
        "n_benchmarks": int(cc.shape[1]),
        "effective_dimensionality": float(participation_ratio),
        "explained_variance_top3": [float(v) for v in explained[:3]],
        "cumulative_variance_90pct_components": int(np.searchsorted(np.cumsum(explained), 0.90) + 1),
        "mean_pairwise_spearman": float(mean_spearman),
        "eigenvalues": [float(v) for v in eigenvalues[:10]],
    }

    with open(RESULTS_DIR / "baseline_statistics.json", "w") as f:
        json.dump(ed_results, f, indent=2)

    print(f"\n  Effective Dimensionality (participation ratio): {participation_ratio:.2f} / {cc.shape[1]}")
    print(f"  Top-3 PCA explained variance: {explained[:3]}")
    print(f"  Components for 90% variance: {ed_results['cumulative_variance_90pct_components']}")

    return ed_results


if __name__ == "__main__":
    print("=" * 70)
    print("IDEA 61: Data Acquisition & Baseline Statistics")
    print("=" * 70)

    download_benchpress()
    download_open_llm_leaderboard()
    download_anadim_matrix()

    wide = build_wide_matrix(DATA_DIR)
    cc = compute_complete_case_matrix(wide)
    stats = compute_baselines(cc)

    print("\n" + "=" * 70)
    print("SECTION 1 COMPLETE")
    print(f"  Wide matrix: {wide.shape}")
    print(f"  Complete-case matrix: {cc.shape}")
    print(f"  Effective dimensionality: {stats['effective_dimensionality']:.2f}")
    print("=" * 70)
