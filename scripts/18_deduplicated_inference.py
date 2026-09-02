"""
Task 2D: Report All Results at Deduplicated n=779.
Addresses R2(W3), R3(W5), R5(implicit).

Recomputes headline BIC values, p-values, CIs at n=779.
Implements cluster-robust bootstrap (resample base-model families, not individual models).
Reports dual-n results for Table 1.
"""
import itertools
import json
import re
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


def extract_base_family(name):
    name_lower = name.lower()
    patterns = [
        (r"(llama[-_ ]?3[\.\d]*)", "Llama-3"),
        (r"(llama[-_ ]?2)", "Llama-2"),
        (r"(llama)", "Llama-1"),
        (r"(qwen2\.5|qwen[-_ ]?2\.5)", "Qwen-2.5"),
        (r"(qwen[-_ ]?2(?![\.\d]))", "Qwen-2"),
        (r"(qwen)", "Qwen-1"),
        (r"(gemma[-_ ]?2)", "Gemma-2"),
        (r"(gemma)", "Gemma-1"),
        (r"(mistral[-_ ]?nemo)", "Mistral-Nemo"),
        (r"(mixtral)", "Mixtral"),
        (r"(mistral)", "Mistral"),
        (r"(phi[-_ ]?[234])", "Phi"),
        (r"(deepseek)", "DeepSeek"),
        (r"(yi)", "Yi"),
        (r"(internlm)", "InternLM"),
        (r"(falcon)", "Falcon"),
        (r"(olmo)", "OLMo"),
        (r"(command)", "Command"),
    ]
    for pattern, family in patterns:
        if re.search(pattern, name_lower):
            return family
    org = name.split("/")[0] if "/" in name else ""
    return f"Other-{org}" if org else "Unknown"


def extract_param_size(name):
    m = re.search(r"(\d+\.?\d*)[bB]", name)
    return float(m.group(1)) if m else 0.0


def build_cluster_labels(ollm):
    """Assign cluster labels to each model for cluster bootstrap."""
    clean_names = []
    for idx in ollm.index:
        m = re.search(r'href="[^"]*huggingface\.co/([^"]+)"', str(idx))
        name = m.group(1) if m else str(idx)[:80]
        clean_names.append(name)

    families = [extract_base_family(n) for n in clean_names]
    sizes = [extract_param_size(n) for n in clean_names]

    size_buckets = []
    for s in sizes:
        if s == 0:
            size_buckets.append("unknown")
        elif s <= 3:
            size_buckets.append("<=3B")
        elif s <= 8:
            size_buckets.append("4-8B")
        elif s <= 15:
            size_buckets.append("9-15B")
        elif s <= 40:
            size_buckets.append("16-40B")
        else:
            size_buckets.append(">40B")

    clusters = [f"{fam}_{sb}" for fam, sb in zip(families, size_buckets)]
    return np.array(clusters)


def fit_bivariate_full(data):
    """Full bivariate copula analysis on a dataset."""
    n = len(data)
    pit = data.rank(method="average") / (n + 1)

    benchmarks = data.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    results = []

    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        u = np.clip(pit[[bm_i, bm_j]].values, 1e-6, 1 - 1e-6)

        cop = pv.Bicop()
        controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
        cop.select(u, controls)

        td = compute_tail_dependence(cop)
        spearman = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)

        gauss_cop = pv.Bicop()
        gauss_cop.select(u, pv.FitControlsBicop(family_set=[pv.BicopFamily.gaussian]))
        bic_gauss = float(gauss_cop.bic(u))

        ng_fams = [f for f in COPULA_FAMILIES if f != pv.BicopFamily.gaussian]
        ng_cop = pv.Bicop()
        ng_cop.select(u, pv.FitControlsBicop(family_set=ng_fams))
        bic_ng = float(ng_cop.bic(u))

        top20_mask = (pit[bm_i].values >= 0.8) | (pit[bm_j].values >= 0.8)
        if top20_mask.sum() >= 10:
            cond_rho = float(stats.spearmanr(
                data.iloc[:, i].values[top20_mask],
                data.iloc[:, j].values[top20_mask]
            ).statistic)
        else:
            cond_rho = np.nan

        results.append({
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "n_obs": n,
            "family": FAMILY_NAMES.get(cop.family, str(cop.family)),
            "spearman_rho": spearman,
            "lambda_U": td["lambda_U"],
            "lambda_L": td["lambda_L"],
            "delta_bic": bic_gauss - bic_ng,
            "bic_gaussian": bic_gauss,
            "bic_best": float(cop.bic(u)),
            "cond_spearman_20pct": cond_rho,
        })

    return results


def cluster_bootstrap(data, cluster_labels, n_bootstrap=1000, seed=42):
    """
    Cluster bootstrap: resample entire base-model families (clusters).
    This accounts for non-independence within model families.
    """
    print("\n" + "=" * 70)
    print(f"CLUSTER BOOTSTRAP (n_clusters={len(np.unique(cluster_labels))}, "
          f"n_bootstrap={n_bootstrap})")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    benchmarks = data.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))

    all_boot_results = {
        f"{benchmarks[i]}x{benchmarks[j]}": {
            "lambda_U": [], "lambda_L": [],
            "delta_bic": [], "family": [],
            "cond_spearman": [],
        }
        for i, j in pairs
    }

    for b in range(n_bootstrap):
        boot_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        boot_indices = []
        for c in boot_clusters:
            cluster_mask = cluster_labels == c
            boot_indices.extend(np.where(cluster_mask)[0])

        boot_data = data.iloc[boot_indices].copy()
        boot_data = boot_data.reset_index(drop=True)

        n_boot = len(boot_data)
        pit_boot = boot_data.rank(method="average") / (n_boot + 1)

        for i, j in pairs:
            bm_i, bm_j = benchmarks[i], benchmarks[j]
            key = f"{bm_i}x{bm_j}"
            u = np.clip(pit_boot[[bm_i, bm_j]].values, 1e-6, 1 - 1e-6)

            try:
                cop = pv.Bicop()
                controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
                cop.select(u, controls)
                td = compute_tail_dependence(cop)
                fam = FAMILY_NAMES.get(cop.family, str(cop.family))

                gauss_cop = pv.Bicop()
                gauss_cop.select(u, pv.FitControlsBicop(
                    family_set=[pv.BicopFamily.gaussian]))
                ng_fams = [f for f in COPULA_FAMILIES if f != pv.BicopFamily.gaussian]
                ng_cop = pv.Bicop()
                ng_cop.select(u, pv.FitControlsBicop(family_set=ng_fams))
                dbic = float(gauss_cop.bic(u) - ng_cop.bic(u))

                top20_mask = (pit_boot[bm_i].values >= 0.8) | (pit_boot[bm_j].values >= 0.8)
                if top20_mask.sum() >= 10:
                    crho = float(stats.spearmanr(
                        boot_data.iloc[:, i].values[top20_mask],
                        boot_data.iloc[:, j].values[top20_mask]
                    ).statistic)
                else:
                    crho = np.nan

                all_boot_results[key]["lambda_U"].append(td["lambda_U"])
                all_boot_results[key]["lambda_L"].append(td["lambda_L"])
                all_boot_results[key]["delta_bic"].append(dbic)
                all_boot_results[key]["family"].append(fam)
                all_boot_results[key]["cond_spearman"].append(crho)
            except Exception:
                continue

        if (b + 1) % 200 == 0:
            print(f"  Bootstrap iteration {b+1}/{n_bootstrap}")

    summary = []
    for i, j in pairs:
        bm_i, bm_j = benchmarks[i], benchmarks[j]
        key = f"{bm_i}x{bm_j}"
        br = all_boot_results[key]

        if not br["lambda_U"]:
            continue

        lU = np.array(br["lambda_U"])
        lL = np.array(br["lambda_L"])
        dbics = np.array(br["delta_bic"])
        crhos = np.array([x for x in br["cond_spearman"] if not np.isnan(x)])
        fams = br["family"]

        fam_mode = max(set(fams), key=fams.count)
        fam_stability = fams.count(fam_mode) / len(fams)

        result = {
            "benchmark_1": bm_i,
            "benchmark_2": bm_j,
            "n_bootstrap": len(lU),
            "lambda_U_mean": float(np.mean(lU)),
            "lambda_U_ci_lower": float(np.percentile(lU, 2.5)),
            "lambda_U_ci_upper": float(np.percentile(lU, 97.5)),
            "lambda_L_mean": float(np.mean(lL)),
            "lambda_L_ci_lower": float(np.percentile(lL, 2.5)),
            "lambda_L_ci_upper": float(np.percentile(lL, 97.5)),
            "delta_bic_mean": float(np.mean(dbics)),
            "delta_bic_ci_lower": float(np.percentile(dbics, 2.5)),
            "delta_bic_ci_upper": float(np.percentile(dbics, 97.5)),
            "delta_bic_pct_positive": float((dbics > 0).mean()),
            "cond_spearman_mean": float(np.mean(crhos)) if len(crhos) > 0 else None,
            "cond_spearman_ci_lower": float(np.percentile(crhos, 2.5)) if len(crhos) > 0 else None,
            "cond_spearman_ci_upper": float(np.percentile(crhos, 97.5)) if len(crhos) > 0 else None,
            "family_mode": fam_mode,
            "family_stability": float(fam_stability),
        }
        summary.append(result)

        print(f"  {bm_i:12s} x {bm_j:12s}: "
              f"λU={result['lambda_U_mean']:.3f} [{result['lambda_U_ci_lower']:.3f}, "
              f"{result['lambda_U_ci_upper']:.3f}] | "
              f"ΔBIC={result['delta_bic_mean']:.1f} [{result['delta_bic_ci_lower']:.1f}, "
              f"{result['delta_bic_ci_upper']:.1f}] | "
              f"{fam_mode} ({fam_stability:.0%})")

    return summary


def build_dual_n_table(full_results, dedup_results, cluster_boot):
    """Build the dual-n comparison table for the paper."""
    print("\n" + "=" * 70)
    print("DUAL-N TABLE: Full (n=4,497) vs Deduplicated (n=779)")
    print("=" * 70)

    table = []
    for fr in full_results:
        bm1, bm2 = fr["benchmark_1"], fr["benchmark_2"]

        dr = [r for r in dedup_results
              if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        dr = dr[0] if dr else None

        cb = [r for r in cluster_boot
              if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        cb = cb[0] if cb else None

        row = {
            "pair": f"{bm1} x {bm2}",
            "full_n": fr["n_obs"],
            "full_family": fr["family"],
            "full_rho": fr["spearman_rho"],
            "full_lambda_U": fr["lambda_U"],
            "full_delta_bic": fr["delta_bic"],
            "full_cond_rho": fr["cond_spearman_20pct"],
        }

        if dr:
            row.update({
                "dedup_n": dr["n_obs"],
                "dedup_family": dr["family"],
                "dedup_rho": dr["spearman_rho"],
                "dedup_lambda_U": dr["lambda_U"],
                "dedup_delta_bic": dr["delta_bic"],
                "dedup_cond_rho": dr["cond_spearman_20pct"],
                "family_agrees": fr["family"] == dr["family"],
            })

        if cb:
            row.update({
                "cluster_lambda_U_ci": [cb["lambda_U_ci_lower"], cb["lambda_U_ci_upper"]],
                "cluster_delta_bic_ci": [cb["delta_bic_ci_lower"], cb["delta_bic_ci_upper"]],
                "cluster_family_stability": cb["family_stability"],
                "cluster_family_mode": cb["family_mode"],
            })

        table.append(row)

        print(f"  {bm1:12s} x {bm2:12s}:")
        print(f"    Full  (n={fr['n_obs']:5d}): {fr['family']:8s} ρ={fr['spearman_rho']:.3f} "
              f"λU={fr['lambda_U']:.3f} ΔBIC={fr['delta_bic']:.1f} "
              f"ρ_up={fr['cond_spearman_20pct']:.3f}")
        if dr:
            print(f"    Dedup (n={dr['n_obs']:5d}): {dr['family']:8s} ρ={dr['spearman_rho']:.3f} "
                  f"λU={dr['lambda_U']:.3f} ΔBIC={dr['delta_bic']:.1f} "
                  f"ρ_up={dr['cond_spearman_20pct']:.3f}"
                  f" {'✓' if fr['family'] == dr['family'] else '✗'}")
        if cb:
            print(f"    Cluster CI: λU=[{cb['lambda_U_ci_lower']:.3f}, {cb['lambda_U_ci_upper']:.3f}] "
                  f"ΔBIC=[{cb['delta_bic_ci_lower']:.1f}, {cb['delta_bic_ci_upper']:.1f}] "
                  f"{cb['family_mode']} ({cb['family_stability']:.0%})")

    n_agree = sum(1 for r in table if r.get("family_agrees", False))
    print(f"\n  Family agreement: {n_agree}/{len(table)}")

    return table


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2D: DEDUPLICATED n=779 INFERENCE WITH CLUSTER BOOTSTRAP")
    print("=" * 70)

    ollm = pd.read_csv(DATA_DIR / "ollm_score_matrix.csv", index_col=0)
    deduped = pd.read_csv(DATA_DIR / "ollm_deduped.csv", index_col=0)

    print(f"  Full data: {ollm.shape}")
    print(f"  Deduplicated data: {deduped.shape}")

    print("\n--- 1. Full bivariate analysis at n=4,497 ---")
    full_results = fit_bivariate_full(ollm)
    print(f"  {len(full_results)} pairs fitted")

    print("\n--- 2. Bivariate analysis at n=779 (deduplicated) ---")
    dedup_results = fit_bivariate_full(deduped)
    print(f"  {len(dedup_results)} pairs fitted")

    print("\n--- 3. Cluster bootstrap on deduplicated data ---")
    cluster_labels = build_cluster_labels(deduped)
    n_clusters = len(np.unique(cluster_labels))
    print(f"  Unique clusters: {n_clusters}")
    cluster_boot = cluster_bootstrap(deduped, cluster_labels, n_bootstrap=1000)

    print("\n--- 4. Dual-n comparison table ---")
    dual_n_table = build_dual_n_table(full_results, dedup_results, cluster_boot)

    output = {
        "full_results": full_results,
        "dedup_results": dedup_results,
        "cluster_bootstrap": cluster_boot,
        "dual_n_table": dual_n_table,
        "metadata": {
            "n_full": int(ollm.shape[0]),
            "n_dedup": int(deduped.shape[0]),
            "n_clusters": n_clusters,
            "n_bootstrap": 1000,
        }
    }

    output_path = RESULTS_DIR / "deduplicated_inference.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    print("\n  HEADLINE FINDINGS:")
    headline_pairs = [("BBH", "GPQA"), ("BBH", "MUSR"), ("MATH Lvl 5", "MMLU-PRO")]
    for bm1, bm2 in headline_pairs:
        fr = [r for r in full_results if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        dr = [r for r in dedup_results if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        cb = [r for r in cluster_boot if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        if fr and dr:
            fr, dr = fr[0], dr[0]
            survived = "SURVIVED" if fr["family"] == dr["family"] else "CHANGED"
            print(f"    {bm1} x {bm2}: {fr['family']} -> {dr['family']} [{survived}]")
            if cb:
                cb = cb[0]
                print(f"      Cluster CI: λU=[{cb['lambda_U_ci_lower']:.3f}, "
                      f"{cb['lambda_U_ci_upper']:.3f}], stability={cb['family_stability']:.0%}")

    print("\n" + "=" * 70)
    print("TASK 2D COMPLETE")
    print("=" * 70)
