"""
Section 4: Vine copula structure fitting + comparison with PCA structure.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

COPULA_FAMILIES = [
    pv.BicopFamily.gaussian,
    pv.BicopFamily.student,
    pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.frank,
    pv.BicopFamily.joe,
    pv.BicopFamily.bb1,
    pv.BicopFamily.bb7,
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


def fit_vine_copula(pit: pd.DataFrame):
    """
    Section 4.1: Fit a vine copula to the full benchmark matrix.
    Returns the fitted Vinecop object and structure summary.
    """
    print("  Fitting vine copula...")
    u_data = pit.values
    u_data = np.clip(u_data, 1e-6, 1 - 1e-6)

    controls = pv.FitControlsVinecop(
        family_set=COPULA_FAMILIES,
        trunc_lvl=pit.shape[1] - 1,
    )

    vc = pv.Vinecop(d=pit.shape[1])
    vc.select(u_data, controls)

    print(f"  Vine structure type: R-vine")
    print(f"  Dimensions: {vc.dim}")
    print(f"  Truncation level: {vc.trunc_lvl}")
    print(f"  Log-likelihood: {vc.loglik(u_data):.2f}")
    print(f"  AIC: {vc.aic(u_data):.2f}")
    print(f"  BIC: {vc.bic(u_data):.2f}")

    return vc


def extract_vine_structure(vc: pv.Vinecop, benchmarks: list):
    """Extract the vine tree structure as a human-readable summary."""
    matrix = vc.matrix
    d = vc.dim
    print(f"\n  Vine matrix (R-vine structure):")

    structure_info = {
        "dim": d,
        "trunc_lvl": vc.trunc_lvl,
        "matrix": matrix.tolist(),
        "benchmarks": benchmarks,
        "trees": [],
    }

    print(f"\n  Tree 1 (strongest unconditional dependencies):")
    tree1_edges = []
    for j in range(d - 1):
        pair_cop = vc.get_pair_copula(0, j)
        fam = FAMILY_NAMES.get(pair_cop.family, str(pair_cop.family))

        idx_1 = int(matrix[0, j]) - 1
        idx_2 = int(matrix[j, j]) - 1
        bm1 = benchmarks[idx_1] if 0 <= idx_1 < len(benchmarks) else f"Var{j}_a"
        bm2 = benchmarks[idx_2] if 0 <= idx_2 < len(benchmarks) else f"Var{j}_b"

        tree1_edges.append({
            "node_1": bm1,
            "node_2": bm2,
            "family": fam,
            "parameters": pair_cop.parameters.tolist(),
        })
        print(f"    {bm1} --- {bm2} [{fam}]")

    structure_info["trees"].append({"level": 1, "edges": tree1_edges})

    if vc.trunc_lvl >= 2:
        print(f"\n  Tree 2 (conditional dependencies):")
        tree2_edges = []
        for j in range(d - 2):
            pair_cop = vc.get_pair_copula(1, j)
            fam = FAMILY_NAMES.get(pair_cop.family, str(pair_cop.family))

            idx_1 = int(matrix[1, j]) - 1
            idx_2 = int(matrix[j, j]) - 1
            cond = int(matrix[0, j]) - 1
            bm1 = benchmarks[idx_1] if 0 <= idx_1 < len(benchmarks) else f"Var{j}_a"
            bm2 = benchmarks[idx_2] if 0 <= idx_2 < len(benchmarks) else f"Var{j}_b"
            bm_cond = benchmarks[cond] if 0 <= cond < len(benchmarks) else f"Var{j}_c"

            tree2_edges.append({
                "node_1": bm1,
                "node_2": bm2,
                "given": bm_cond,
                "family": fam,
            })
            print(f"    {bm1} --- {bm2} | {bm_cond} [{fam}]")

        structure_info["trees"].append({"level": 2, "edges": tree2_edges})

    return structure_info


def compare_vine_with_pca(pit: pd.DataFrame, benchmarks: list, vine_info: dict):
    """
    Section 4.2: Compare vine dependency structure with PCA/correlation structure.
    """
    print("\n  === Vine vs PCA/Correlation Comparison ===")

    spearman = pit.corr(method="spearman")
    dist_matrix = 1 - spearman.abs().values
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.clip(dist_matrix, 0, None)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    dendrogram(Z, labels=benchmarks, ax=ax, leaf_rotation=45, leaf_font_size=8)
    ax.set_title("Correlation-Based Hierarchical Clustering", fontsize=11)
    ax.set_ylabel("Distance (1 - |Spearman rho|)", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_dendrogram.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "correlation_dendrogram.png", dpi=200, bbox_inches="tight")
    plt.close()

    tree1_edges = vine_info["trees"][0]["edges"]
    vine_central = {}
    for edge in tree1_edges:
        for node in [edge["node_1"], edge["node_2"]]:
            vine_central[node] = vine_central.get(node, 0) + 1

    vine_central_sorted = sorted(vine_central.items(), key=lambda x: -x[1])

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(pit.values)
    pca = PCA()
    pca.fit(X)
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=benchmarks,
        columns=["PC1", "PC2", "PC3"],
    )
    pca_central = loadings.abs().sum(axis=1).sort_values(ascending=False)

    comparison = {
        "vine_centrality": vine_central_sorted,
        "pca_centrality": [(bm, float(v)) for bm, v in pca_central.items()],
        "vine_most_central": vine_central_sorted[0][0] if vine_central_sorted else None,
        "pca_most_central": pca_central.index[0],
        "agreement": vine_central_sorted[0][0] == pca_central.index[0] if vine_central_sorted else None,
    }

    print(f"  Vine most central benchmark: {comparison['vine_most_central']}")
    print(f"  PCA most central benchmark: {comparison['pca_most_central']}")
    print(f"  Agreement: {comparison['agreement']}")

    print(f"\n  Vine centrality (degree in Tree 1):")
    for bm, deg in vine_central_sorted:
        print(f"    {bm}: degree {deg}")

    print(f"\n  PCA centrality (sum of |loadings| on PC1-3):")
    for bm, score in pca_central.items():
        print(f"    {bm}: {score:.3f}")

    independence_in_vine = sum(
        1 for tree in vine_info["trees"]
        for edge in tree["edges"]
        if edge["family"] == "Independence"
    )
    total_vine_edges = sum(len(tree["edges"]) for tree in vine_info["trees"])
    print(f"\n  Independence copulas in vine: {independence_in_vine}/{total_vine_edges}")
    print(f"  (These pairs are conditionally independent given their vine neighbors)")

    return comparison


if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 4: Vine Copula Structure")
    print("=" * 70)

    pit = pd.read_csv(DATA_DIR / "pit_transformed.csv", index_col=0)
    benchmarks = pit.columns.tolist()
    print(f"  Data: {pit.shape[0]} models x {pit.shape[1]} benchmarks")

    print("\n--- 4.1: Fit Vine Copula ---")
    vc = fit_vine_copula(pit)
    vine_info = extract_vine_structure(vc, benchmarks)

    with open(RESULTS_DIR / "vine_structure.json", "w") as f:
        json.dump(vine_info, f, indent=2)

    print("\n--- 4.2: Compare with PCA/Correlation ---")
    comparison = compare_vine_with_pca(pit, benchmarks, vine_info)

    with open(RESULTS_DIR / "vine_vs_pca_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("SECTION 4 COMPLETE")
    print("=" * 70)
