# Data: The Correlation Mirage

## Files

### Primary Analysis Data (OLLM v2)

| File | Size | Description |
|------|------|-------------|
| `ollm_score_matrix.csv` | 2.4 MB | Open LLM Leaderboard v2 score matrix. **4,497 models × 6 benchmarks** (IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO). 100% fill rate. Derived from `open-llm-leaderboard/results` on HuggingFace. |
| `ollm_pit.csv` | 2.5 MB | Probability Integral Transform of the OLLM score matrix. Ranks scaled to (0, 1) via `rank / (n + 1)`. Ties handled via midranks. |
| `ollm_deduped.csv` | 423 KB | Deduplicated OLLM scores. One representative per (base_family, parameter_size) group, yielding **779 independent models** from **714 unique base-model families**. |

### Replication Study Data (WILD)

| File | Size | Description |
|------|------|-------------|
| `wild_score_matrix.csv` | 31 KB | kensho/WILD evaluation framework scores. **65 models × 27 tasks**, 100% fill rate. 65 independent-organization models. |

### Exploratory Analysis Data (BenchPress)

| File | Size | Description |
|------|------|-------------|
| `benchpress_wide_all.csv` | 85 KB | BenchPress wide-format score matrix. **188 models × 316 benchmarks** (sparse, 8.2% fill rate). From `yzeng58/benchpress-score-matrix` on HuggingFace. |
| `complete_case_matrix.csv` | 2.3 KB | Complete-case submatrix. Subset of BenchPress with no missing values, used for copula analysis requiring complete observations. |
| `pit_transformed.csv` | 7 KB | PIT-transformed complete-case matrix. Rank-based probability integral transform applied to complete_case_matrix. |

## Format

All files are comma-separated values (CSV). The first column is the model identifier (index). Values represent raw benchmark scores (percentage or normalized, depending on the benchmark). PIT files contain values in (0, 1) representing rank-transformed scores.

## Regenerating from Source

To download raw data from HuggingFace instead of using these processed files:

```bash
python analysis/01_download_and_clean.py
```

This downloads the BenchPress, OLLM v2, and WILD datasets and produces the intermediate files in `data/`.

## Data Sources and Licenses

| Dataset | HuggingFace URL | License |
|---------|----------------|---------|
| Open LLM Leaderboard v2 | [`open-llm-leaderboard/results`](https://huggingface.co/datasets/open-llm-leaderboard/results) | Apache 2.0 |
| BenchPress | [`yzeng58/benchpress-score-matrix`](https://huggingface.co/datasets/yzeng58/benchpress-score-matrix) | CC-BY-4.0 |
| kensho/WILD | [`kensho/WILD`](https://huggingface.co/datasets/kensho/WILD) | See dataset page |

## Benchmark Descriptions (OLLM v2)

| Benchmark | Reference | Measures |
|-----------|-----------|----------|
| IFEval | Zhou et al., 2023 | Instruction following |
| BBH | Suzgun et al., 2023 | Challenging reasoning (BIG-Bench Hard) |
| MATH Lvl 5 | Hendrycks et al., 2021 | Mathematical problem solving |
| GPQA | Rein et al., 2024 | Graduate-level Q&A |
| MUSR | Sprague et al., 2024 | Multi-step soft reasoning |
| MMLU-PRO | Wang et al., 2024 | Massive multitask language understanding |
