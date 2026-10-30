# Fig. 2 — Training-data exposure vs citations (`fig:training_olmo_merged`)

OLMo-2-1124-13B-Instruct (4-bit) recognition of 12,214 physicists, drawn from the random and most-cited samples that remained after name-disambiguation filtering. 1,262 of them (10.3%) are recognized.

```
python fig2_training_data/make_fig2.py      # run from the repo root; ~3 s
```
The script writes `output/fig2.png` (500 dpi) and `output/fig2.pdf` and prints the numbers quoted in the caption and §2.2. It draws all three panels from the shipped data and uses no static images.

## Data
`data/fig2_scientists.csv` has one row per scientist (12,214 rows), with rows shuffled.

| column | meaning |
|---|---|
| `id` | anonymous integer id (no link to names or Dimensions IDs) |
| `citations` | cumulative citations (Dimensions) |
| `mentions` | infini-gram count of the exact name string "First Last" in the OLMo-2 pretraining corpus, with middle names and initials dropped |
| `recognized` | 1 if OLMo-2-13B-Instruct answered YES for the full recorded name or for the reduced "First Last" name, else 0 |

The script clips zeros in `citations` and `mentions` to 0.5 so they can be placed on the log axes.

## Panels
- **a** Gaussian KDE on log10 citations, recognized vs not recognized. Dashed lines are the medians (11,954 vs 10,057). Columns: `citations`, `recognized`.
- **b** The same plot for `mentions` (medians 623 vs 58).
- **c** Recognition rate in 16 equal-width log bins with Wilson 95% CIs. Bins with n<10 are dropped, and so are the sparsest tail bins: the last 2 for mentions, and the first 2 and last 1 for citations. The line and band are a univariate logistic regression of `recognized` on log10(x), fitted to all 12,214 rows, with its 95% CI. The line stops at the last shown bin. Panel c's x-axis is on the same row as panel b's.

## Numbers checked against the text
These reproduce:
- N = 12,214, with 10.3% recognized.
- Medians of 11,954 vs 10,057 citations and 623 vs 58 mentions (about 10x).
- The fitted rate starts near 3% at the left end.
- Fitted rate at 10^4: 24.6% for mentions vs 11.7% for citations.
- Top bins: 32.7% vs 12.8%, a ratio of 2.55 (the text says "2.6-fold").
- Spearman rho(citations, mentions) = 0.3648 (0.36 in the text).

## Upstream (not included)
The shipped CSV is a column subset of `data/olmo_training/olmo_training_master_final.csv`: rows with `in_main_analysis == True`, columns `cum_citations`, `training_frequency` and `recognized_any`. These scripts produced that table:
- `code/olmo_training_10k_phy.py` collects the infini-gram counts (`collect`, `recount_cleaned`; index `v4_olmo-2-0325-32b-instruct_llama`) and runs the OLMo-2 recognition probe (`collect_recognition`, local MLX `mlx-community/OLMo-2-1124-13B-Instruct-4bit`).
- `code/refined_processing.py` builds the master table. It reduces names and re-counts authors with full middle names, applies the Dimensions dominant-holder name filter (`name_collision`, `citation_share`), fixes encoding errors and sets the `in_main_analysis` flag.
- `code/refine_recognition_cleaned.py` asks OLMo-2 again with the reduced name and adds `recognized_refined` and `recognized_any`.
- The original plots came from `test/make_dist_stacked.py` (a, b) and `test/make_compare_errorbar_trunc.py` (c). `code/align_fig2.py` composited them into `olmo_training_combined_v17.png`.

Raw sources:
- Dimensions researcher records, for cohort sampling, citations and name collisions.
- The OLMo-2 pretraining corpus (OLMo-mix / Dolma), queried through the infini-gram API.
- The OLMo-2-1124-13B-Instruct weights (AI2), in a 4-bit MLX conversion.
