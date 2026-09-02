"""
Fix 2: Base model deduplication.
Addresses reviewer concern that 4,497 models include fine-tune swarms,
inflating effective sample size.
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

COPULA_FAMILIES = [
    pv.BicopFamily.gaussian, pv.BicopFamily.student,
    pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
    pv.BicopFamily.frank, pv.BicopFamily.joe,
    pv.BicopFamily.bb1, pv.BicopFamily.bb7,
]

FAMILY_NAMES = {
    pv.BicopFamily.gaussian: "Gaussian", pv.BicopFamily.student: "Student-t",
    pv.BicopFamily.clayton: "Clayton", pv.BicopFamily.gumbel: "Gumbel",
    pv.BicopFamily.frank: "Frank", pv.BicopFamily.joe: "Joe",
    pv.BicopFamily.bb1: "BB1", pv.BicopFamily.bb7: "BB7",
}


def clean_model_name(html):
    m = re.search(r'href="[^"]*huggingface\.co/([^"]+)"', html)
    return m.group(1) if m else html[:80]


def extract_base_family(name):
    """
    Extract base model family from HuggingFace model path.
    Strategy: match known base model patterns, then extract param size.
    """
    name_lower = name.lower()

    patterns = [
        (r"(llama[-_ ]?3[\.\d]*)", "Llama-3"),
        (r"(llama[-_ ]?2)", "Llama-2"),
        (r"(llama[-_ ]?1|llama(?![-_ ]?\d))", "Llama-1"),
        (r"(qwen2\.5|qwen[-_ ]?2\.5)", "Qwen-2.5"),
        (r"(qwen[-_ ]?2(?![\.\d]))", "Qwen-2"),
        (r"(qwen(?![-_ ]?\d))", "Qwen-1"),
        (r"(gemma[-_ ]?2)", "Gemma-2"),
        (r"(gemma(?![-_ ]?\d))", "Gemma-1"),
        (r"(mistral[-_ ]?nemo)", "Mistral-Nemo"),
        (r"(mistral[-_ ]?large)", "Mistral-Large"),
        (r"(mistral[-_ ]?small)", "Mistral-Small"),
        (r"(mixtral)", "Mixtral"),
        (r"(mistral)", "Mistral"),
        (r"(phi[-_ ]?4)", "Phi-4"),
        (r"(phi[-_ ]?3\.5)", "Phi-3.5"),
        (r"(phi[-_ ]?3)", "Phi-3"),
        (r"(phi[-_ ]?2)", "Phi-2"),
        (r"(deepseek[-_ ]?v3)", "DeepSeek-V3"),
        (r"(deepseek[-_ ]?v2)", "DeepSeek-V2"),
        (r"(deepseek[-_ ]?coder)", "DeepSeek-Coder"),
        (r"(deepseek)", "DeepSeek"),
        (r"(yi[-_ ]?1\.5)", "Yi-1.5"),
        (r"(yi[-_ ]?(?:large|coder)?)", "Yi"),
        (r"(internlm2\.5|internlm[-_ ]?2\.5)", "InternLM-2.5"),
        (r"(internlm[-_ ]?2)", "InternLM-2"),
        (r"(internlm)", "InternLM"),
        (r"(falcon[-_ ]?(?:mamba)?)", "Falcon"),
        (r"(starcoder)", "StarCoder"),
        (r"(codellama|code[-_ ]?llama)", "CodeLlama"),
        (r"(olmo)", "OLMo"),
        (r"(mpt)", "MPT"),
        (r"(bloom)", "BLOOM"),
        (r"(pythia)", "Pythia"),
        (r"(command[-_ ]?r)", "Command-R"),
        (r"(solar)", "Solar"),
        (r"(openchat)", "OpenChat"),
        (r"(zephyr)", "Zephyr"),
        (r"(starling)", "Starling"),
        (r"(vicuna)", "Vicuna"),
        (r"(wizardlm|wizard[-_ ]?lm)", "WizardLM"),
        (r"(orca)", "Orca"),
        (r"(hermes)", "Hermes"),
        (r"(dolphin)", "Dolphin"),
        (r"(nous[-_ ]?hermes)", "Nous-Hermes"),
        (r"(opencoder)", "OpenCoder"),
        (r"(exaone)", "EXAONE"),
        (r"(aya[-_ ]?expanse|aya)", "Aya"),
        (r"(c4ai)", "C4AI"),
        (r"(granite)", "Granite"),
        (r"(jamba)", "Jamba"),
        (r"(nemotron)", "Nemotron"),
        (r"(smollm)", "SmolLM"),
        (r"(tulu)", "Tulu"),
        (r"(amber)", "Amber"),
        (r"(map[-_ ]?neo)", "MAP-Neo"),
    ]

    for pattern, family in patterns:
        if re.search(pattern, name_lower):
            return family

    org = name.split("/")[0] if "/" in name else ""
    return f"Other-{org}" if org else "Unknown"


def extract_param_size(name):
    """Extract parameter size (e.g., 7B, 13B, 70B) from model name."""
    m = re.search(r"(\d+\.?\d*)[bB]", name)
    if m:
        return float(m.group(1))
    return 0.0


def deduplicate(ollm, strategy="best_per_family_size"):
    """
    Deduplicate by keeping one representative per (base_family, param_bucket).
    Strategy: keep the model with the highest average score in each group.
    """
    clean_names = [clean_model_name(n) for n in ollm.index]
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

    meta = pd.DataFrame({
        "clean_name": clean_names,
        "family": families,
        "param_size": sizes,
        "size_bucket": size_buckets,
        "group": [f"{fam}_{sb}" for fam, sb in zip(families, size_buckets)],
        "avg_score": ollm.mean(axis=1).values,
    }, index=ollm.index)

    deduped_idx = meta.groupby("group")["avg_score"].idxmax()
    deduped = ollm.loc[deduped_idx]

    return deduped, meta


def run_copula_on_subset(data, label):
    """Run the bivariate copula pipeline on a data subset."""
    n = len(data)
    pit = data.rank(method="average") / (n + 1)
    pit_values = pit.values
    pit_values = np.clip(pit_values, 1e-6, 1 - 1e-6)

    benchmarks = data.columns.tolist()
    pairs = list(itertools.combinations(range(len(benchmarks)), 2))
    results = []

    for i, j in pairs:
        bm1, bm2 = benchmarks[i], benchmarks[j]
        u = pit_values[:, [i, j]]

        cop = pv.Bicop()
        controls = pv.FitControlsBicop(family_set=COPULA_FAMILIES)
        cop.select(u, controls)

        fam = FAMILY_NAMES.get(cop.family, str(cop.family))
        spearman = float(stats.spearmanr(data.iloc[:, i], data.iloc[:, j]).statistic)

        top20_mask = (pit_values[:, i] >= 0.8) | (pit_values[:, j] >= 0.8)
        if top20_mask.sum() > 10:
            cond_rho = float(stats.spearmanr(
                data.iloc[:, i].values[top20_mask],
                data.iloc[:, j].values[top20_mask]
            ).statistic)
        else:
            cond_rho = np.nan

        r1 = stats.rankdata(data.iloc[:, i].values)
        r2 = stats.rankdata(data.iloc[:, j].values)
        k = max(10, int(n * 0.05))
        threshold = n - k
        emp_lambda = int(np.sum((r1 > threshold) & (r2 > threshold))) / k

        results.append({
            "benchmark_1": bm1,
            "benchmark_2": bm2,
            "family": fam,
            "is_frank": "Frank" in fam,
            "spearman": spearman,
            "cond_spearman_20pct": cond_rho,
            "emp_lambda_5pct": float(emp_lambda),
            "bic": float(cop.bic(u)),
        })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FIX 2: Base Model Deduplication")
    print("=" * 70)

    ollm = pd.read_csv(DATA_DIR / "ollm_score_matrix.csv", index_col=0)
    print(f"  Original: {ollm.shape[0]} models x {ollm.shape[1]} benchmarks")

    deduped, meta = deduplicate(ollm)
    print(f"  Deduplicated: {deduped.shape[0]} models x {deduped.shape[1]} benchmarks")

    print(f"\n  Family distribution:")
    family_counts = meta["family"].value_counts()
    for fam, count in family_counts.head(20).items():
        n_groups = meta[meta["family"] == fam]["group"].nunique()
        print(f"    {fam:20s}: {count:5d} models -> {n_groups:3d} groups")

    total_groups = meta["group"].nunique()
    print(f"\n  Total unique (family, size) groups: {total_groups}")
    print(f"  Deduplication ratio: {ollm.shape[0]} -> {deduped.shape[0]} ({deduped.shape[0]/ollm.shape[0]*100:.1f}%)")

    deduped.to_csv(DATA_DIR / "ollm_deduped.csv")

    print("\n--- Running copula analysis on FULL data ---")
    full_results = run_copula_on_subset(ollm, "full")
    print(f"  Full: {len(full_results)} pairs")

    print("\n--- Running copula analysis on DEDUPLICATED data ---")
    dedup_results = run_copula_on_subset(deduped, "dedup")
    print(f"  Dedup: {len(dedup_results)} pairs")

    print("\n" + "=" * 70)
    print("COMPARISON: Full vs Deduplicated")
    print("=" * 70)

    print(f"\n  {'Pair':35s} | {'Full':25s} | {'Dedup':25s} | Match?")
    print("-" * 100)

    matches = 0
    for fr, dr in zip(full_results, dedup_results):
        pair = f"{fr['benchmark_1']} x {fr['benchmark_2']}"
        full_str = f"{fr['family']:8s} rho={fr['spearman']:.3f} crho={fr['cond_spearman_20pct']:.3f}"
        dedup_str = f"{dr['family']:8s} rho={dr['spearman']:.3f} crho={dr['cond_spearman_20pct']:.3f}"
        match = "YES" if fr["family"] == dr["family"] else "NO"
        if fr["family"] == dr["family"]:
            matches += 1
        print(f"  {pair:35s} | {full_str} | {dedup_str} | {match}")

    print(f"\n  Family agreement: {matches}/{len(full_results)} ({matches/len(full_results)*100:.0f}%)")

    full_frank = [r for r in full_results if r["is_frank"]]
    dedup_frank = [r for r in dedup_results if r["is_frank"]]
    print(f"  Frank pairs: Full={len(full_frank)}, Dedup={len(dedup_frank)}")

    full_cond = [r["cond_spearman_20pct"] for r in full_results if not np.isnan(r["cond_spearman_20pct"])]
    dedup_cond = [r["cond_spearman_20pct"] for r in dedup_results if not np.isnan(r["cond_spearman_20pct"])]
    corr = np.corrcoef(full_cond, dedup_cond)[0, 1]
    print(f"  Conditional Spearman correlation (full vs dedup): r = {corr:.4f}")

    headline_pairs = [("BBH", "GPQA"), ("BBH", "MUSR"), ("MATH Lvl 5", "MMLU-PRO")]
    print(f"\n  HEADLINE PAIRS survival check:")
    for bm1, bm2 in headline_pairs:
        fr = [r for r in full_results if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        dr = [r for r in dedup_results if r["benchmark_1"] == bm1 and r["benchmark_2"] == bm2]
        if fr and dr:
            fr, dr = fr[0], dr[0]
            survived = "SURVIVED" if dr["is_frank"] else "CHANGED"
            print(f"    {bm1} x {bm2}: Full={fr['family']}(crho={fr['cond_spearman_20pct']:.3f}) -> "
                  f"Dedup={dr['family']}(crho={dr['cond_spearman_20pct']:.3f}) [{survived}]")

    all_dedup_results = {
        "n_original": int(ollm.shape[0]),
        "n_deduped": int(deduped.shape[0]),
        "n_groups": int(total_groups),
        "family_agreement_pct": float(matches / len(full_results) * 100),
        "cond_spearman_correlation": float(corr),
        "full_results": full_results,
        "dedup_results": dedup_results,
    }

    with open(RESULTS_DIR / "deduplication_results.json", "w") as f:
        json.dump(all_dedup_results, f, indent=2)

    print("\n" + "=" * 70)
    print("FIX 2 COMPLETE")
    print("=" * 70)
