# The Correlation Mirage: Benchmark Dependence Collapses for Top-Performing LLMs

**EMNLP 2026 | Oral**
Raj Firke (Red Hat) · Rajeswari Kannan (Pimpri Chinchwad College of Engineering)

[📄 Paper](./The%20Correlation%20Mirage.pdf)

---

## Overview

This repository contains the code, data, and pre-computed results for our paper on copula-based dependence analysis of LLM benchmark suites.

**Key finding:** Benchmark pairs correlating at ρ<sub>S</sub> > 0.73 show **near-zero or reversed ranking agreement** among top-performing models, with conditional Spearman reductions of up to **97%**. 47% of benchmark pairs follow copula families with zero asymptotic tail dependence despite strong bulk association — meaning that dropping a "redundant" benchmark discards precisely the information that distinguishes frontier models.

### In plain terms

> A model ranked first on one benchmark but last on another is not unusual among frontier models — precisely the ones practitioners are deploying.

## Highlights

- **Conditional Spearman collapse:** Among the top 5% of 4,497 OLLM v2 models, ranking correlation drops 50–97% or reverses sign (BBH×MUSR: 0.73 → −0.52)
- **Copula-based explanation:** 7/15 benchmark pairs are best fit by Frank copulas with zero asymptotic tail dependence (ΔBIC up to 585)
- **Nonparametric validation:** Empirical tail concentration independently confirms the collapse without parametric assumptions
- **Robustness:** Findings hold after deduplication to 779 base-model families (≥96% bootstrap stability), and replicate across BenchPress (316 benchmarks) and kensho/WILD (27 benchmarks, 65 models)
- **No scalar measure detects this:** Spearman, distance correlation, and mutual information all fail to identify the tail decomposition

## Repository Structure

```
correlation-mirage-benchmarks/
├── analysis/                Core analysis pipeline (14 scripts)
│   ├── 01_download_and_clean.py     Data acquisition and preparation
│   ├── 02_marginals_pit.py          Marginal analysis and PIT transformation
│   ├── 03_bivariate_copulas.py      Bivariate copula fitting and tail dependence
│   ├── 04_vine_copula.py            R-vine copula structure analysis
│   ├── 05_benchmark_selection.py    Copula vs correlation vs PCA selection
│   ├── 06_figures.py                BenchPress figures
│   ├── 07_nonparametric_tail.py     Nonparametric tail dependence estimation
│   ├── 07_statistical_robustness.py Bootstrap CIs, GoF, simulation calibration
│   ├── 08_deduplication.py          Base-model family deduplication
│   ├── 08_revised_selection.py      Quantile regression and ranking fidelity
│   ├── 09_distance_corr_comparison.py  Distance correlation / MI comparison
│   ├── 10_sensitivity_analysis.py   Sample-size sensitivity analysis
│   ├── 11_ollm_full_pipeline.py     Complete OLLM v2 pipeline (n=4,497)
│   └── 12_final_figures.py          Publication-quality figures
├── scripts/                 Validation and replication (7 scripts)
│   ├── 13_wild_copula_pipeline.py   kensho/WILD replication (65 models × 27 tasks)
│   ├── 14_downstream_ranking.py     Downstream ranking disruption analysis
│   ├── 15_expanded_copula_candidates.py  Expanded families (BB6/BB8/Tawn) + power
│   ├── 16_berkson_calibration.py    Berkson calibration (Gaussian/Student-t/perm nulls)
│   ├── 17_threshold_sensitivity.py  Percentile threshold ablation (70th–95th)
│   ├── 18_deduplicated_inference.py Cluster-robust bootstrap (n=779)
│   └── 19_rebuttal_su96.py         Intersection vs union conditioning robustness
├── data/                    Processed score matrices (see data/README.md)
├── results/                 Pre-computed analysis outputs (32 JSON + 6 CSV)
├── figures/                 Publication figures (13 stems, PNG + PDF)
├── config.py                Central configuration (paths, benchmarks, constants)
├── run_analysis.py          Pipeline orchestrator
├── requirements.txt         Python dependencies
├── LICENSE                  MIT License (code)
└── DATA_LICENSE             CC BY 4.0 (data)
```

## Setup

```bash
git clone https://github.com/rajfirke/correlation-mirage-benchmarks
cd correlation-mirage-benchmarks
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, pyvinecopulib ≥ 0.7, pandas ≥ 2.2, scipy ≥ 1.12, numpy ≥ 1.26, scikit-learn ≥ 1.4, matplotlib ≥ 3.8, seaborn ≥ 0.13, statsmodels ≥ 0.14

## Reproducing Results

### Quick start: inspect pre-computed results

All JSON/CSV files in `results/` match the paper's tables and appendices — no re-running required.

### Full pipeline (all tables and figures)

```bash
python run_analysis.py
```

### Individual analyses

Scripts are numbered in recommended execution order. Each is standalone:

```bash
# Core copula pipeline on OLLM v2 (produces Tables 1-3, Figures 1-3)
python analysis/11_ollm_full_pipeline.py

# Publication figures
python analysis/12_final_figures.py

# BenchPress exploratory analysis (Section 4.5)
python analysis/01_download_and_clean.py    # downloads from HuggingFace
python analysis/02_marginals_pit.py
python analysis/03_bivariate_copulas.py
python analysis/04_vine_copula.py

# Robustness analyses (Section 6, Appendices D-I)
python analysis/07_nonparametric_tail.py
python analysis/07_statistical_robustness.py
python analysis/08_deduplication.py

# Validation and replication
python scripts/13_wild_copula_pipeline.py     # kensho/WILD (Appendix K)
python scripts/15_expanded_copula_candidates.py  # Expanded families (Appendix D)
python scripts/16_berkson_calibration.py       # Berkson calibration (Appendix G)
python scripts/17_threshold_sensitivity.py     # Threshold ablation (Appendix H)
python scripts/18_deduplicated_inference.py    # Cluster bootstrap (Appendix I)
```

### Script-to-paper mapping

| Paper Section | Script(s) | Key Outputs |
|---------------|-----------|-------------|
| §3 Method | `01`, `02`, `03` | Score matrices, PIT, copula fits |
| §4.1–4.2 Copula results | `11`, `03` | `ollm_bivariate_fits.json`, `bivariate_fits_pairwise.json` |
| §4.3 Nonparametric validation | `07_nonparametric_tail` | `nonparametric_tail_dependence.json` |
| §4.4 Conditional Spearman | `11`, `19` | Conditional Spearman tables, intersection vs union |
| §4.5 Multi-dataset validation | `10`, `13` | `sensitivity_by_n.json`, `wild_bivariate_fits.json` |
| §4.6 Vine copula | `04`, `11` | `vine_structure.json`, `ollm_vine_structure.json` |
| §6 Alternative explanations | `16`, `19` | `berkson_calibration_broadened.json` |
| Fig 1 (scatter collapse) | `12` | `fig1_collapse_scatter.{png,pdf}` |
| Fig 2 (heatmap comparison) | `12` | `fig2_heatmap_comparison.{png,pdf}` |
| Fig 3 (copula summary) | `12` | `fig3_copula_summary.{png,pdf}` |
| Fig 4 (sensitivity) | `12` | `fig4_sensitivity.{png,pdf}` |
| Appendix D (expanded families) | `15` | `expanded_copula_candidates.json` |
| Appendix G (Berkson calibration) | `16` | `berkson_calibration_broadened.json` |
| Appendix H (threshold sensitivity) | `17` | `threshold_sensitivity_ablation.json` |
| Appendix I (cluster bootstrap) | `18` | `deduplicated_inference.json` |
| Appendix K (WILD validation) | `13` | `wild_bivariate_fits.json` |

## Data

Seven processed score matrices are provided in `data/` (see [`data/README.md`](data/README.md) for full documentation):

| File | Description | Size |
|------|-------------|------|
| `ollm_score_matrix.csv` | Open LLM Leaderboard v2 (4,497 × 6) | 2.4 MB |
| `ollm_pit.csv` | PIT-transformed OLLM scores | 2.5 MB |
| `ollm_deduped.csv` | Deduplicated OLLM (779 models) | 423 KB |
| `wild_score_matrix.csv` | kensho/WILD (65 × 27) | 31 KB |
| `benchpress_wide_all.csv` | BenchPress (188 × 316, sparse) | 85 KB |
| `complete_case_matrix.csv` | BenchPress complete-case subset | 2.3 KB |
| `pit_transformed.csv` | PIT-transformed BenchPress | 7 KB |

Alternatively, run `python analysis/01_download_and_clean.py` to download raw data from HuggingFace.

**Data sources:**
- [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (Fourrier et al., 2024) — Apache 2.0
- [BenchPress](https://huggingface.co/datasets/yzeng58/benchpress-score-matrix) (Zeng, 2026) — CC-BY-4.0
- [kensho/WILD](https://huggingface.co/datasets/kensho/WILD) (Kensho Technologies, 2024)

## Pre-computed Results

All 32 JSON and 6 CSV files in `results/` can be inspected directly. Key files:

| File | Maps to |
|------|---------|
| `ollm_bivariate_fits.json` | Table 1 (all 15 OLLM v2 pairs) |
| `ollm_bootstrap_cis.json` | Bootstrap CIs for λ<sub>U</sub>, λ<sub>L</sub> |
| `ollm_nonparametric_tails.json` | Table 2 (empirical tail concentration) |
| `bivariate_fits_pairwise.json` | BenchPress pairwise-complete fits (398 pairs) |
| `wild_bivariate_fits.json` | Appendix K (WILD 351 pairs) |
| `berkson_calibration_broadened.json` | Appendix G (null model calibration) |
| `threshold_sensitivity_ablation.json` | Appendix H (threshold ablation) |
| `deduplicated_inference.json` | Appendix I (cluster bootstrap at n=779) |

## Practical Protocol

The paper proposes a three-step protocol for benchmark suite maintainers:

```python
import pandas as pd

# Step 1: Screen — identify candidate "redundant" pairs (ρS > 0.7)
rho = df.corr(method='spearman')

# Step 2: Check the tail — compute conditional Spearman in the upper 20%
mask = (df[['A', 'B']].rank(pct=True) > 0.8).any(axis=1)
rho_tail = df.loc[mask, ['A', 'B']].corr(method='spearman').iloc[0, 1]

# Step 3: Decide — if rho_tail < 0.5 * rho_bulk, the pair is potentially deceptive
if rho_tail < 0.5 * rho.loc['A', 'B']:
    print("⚠️ Deceptive pair — retain both benchmarks")
```

## Hardware

All analyses run on a standard laptop in **under 5 minutes** total. No GPU, no model inference, no API calls required.

## Citing

```bibtex
@inproceedings{firke2026mirage,
  title     = {The Correlation Mirage: Benchmark Dependence Collapses for Top-Performing {LLMs}},
  author    = {Firke, Raj and Kannan, Rajeswari},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
}
```

## Contact

If you use this codebase or build upon this work — whether reproducing our results, extending the analysis to new benchmarks or models, or exploring related research directions — we would love to hear from you. Please open an issue on this repository or contact the authors directly via the email addresses listed in the paper.

## License

Code: [MIT License](LICENSE). Data: [CC BY 4.0](DATA_LICENSE).
