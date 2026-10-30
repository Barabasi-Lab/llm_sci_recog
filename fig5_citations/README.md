# Fig. 5 — LLM-assisted papers cite LLM-recognized scientists more often

Main-text Fig. 5 (`\label{fig:llm_impact}`, published file `figure/llm_impact_v17.png`) and the
numbers of its caption and of §2.5.

```
python fig5_citations/make_fig5.py              # from the repository root; refits all models (~30 s - 3 min)
python fig5_citations/make_fig5.py --published  # plot data/published_margins.csv without refitting (~3 s)
```

Writes `fig5_citations/output/fig5.png` (3600 x 2946 px, 300 dpi), `fig5.pdf` (vector panels b, c) and
`fig5_margins.csv` (the refit margins). The script prints the panel-a counts, the margins and the largest
difference from the published margins (0.0000 pp at the time of release).
Dependencies: numpy, pandas, matplotlib, scipy, statsmodels (tested with Python 3.13, statsmodels 0.14.4,
matplotlib 3.10, pandas 2.2). Helvetica or Arial is used if installed, otherwise DejaVu Sans.

## Panels

| panel | content | drawn from | status |
|---|---|---|---|
| a | pipeline schematic: arXiv preprints -> α -> GPT-4o recognition of cited first authors | `assets/panel_a_schematic.png` (cut from the published figure); the six count lines under the boxes are written from `data/*.csv.gz` | schematic static, counts recomputed |
| b | adjusted P(cited first author recognized), human-written vs LLM-assisted, by the cited author's career stage (early ≤10 yr, mid 11–30, late >30) | `data/pairs.csv.gz` + `papers.csv.gz` + `authors.csv.gz`, refit | recomputed |
| c | the same by age of the cited work (0–4, 5–15, >15 yr) | same | recomputed |

**Model** (one logit per panel cell, plus a pooled one for the +0.84 pp headline):
`y ~ asst + log1p(cited_by_count) + log1p(works_count) + career_age + log1p(ref_age) + log1p(ref_cit) + C(ym)`,
y = GPT-4o says YES to the cited first author, asst = α > 0.1, SEs clustered by citing paper.
Counterfactual means set every pair to asst = 0 / 1 and average the predicted probabilities; 95% CIs and
the AME p-value by the delta method. Ages are period-aware: the citing year is 2024 for papers of the
2023–2024 window (`period = old`) and 2026 for the 2024-12 – 2026-07 window (`period = new`).
Sample: pairs with a recognition label and an α, and non-missing controls (620,954 pairs, 17,064 papers).

## Data (`data/`, 7.2 MB)

All identifiers are anonymous integers assigned by a random permutation (`prep/build_inputs.py`);
no names, OpenAlex IDs, arXiv IDs or text are included. Rows are not dropped or altered.

| file | rows | columns |
|---|---|---|
| `papers.csv.gz` (0.09 MB) | 18,211 citing arXiv physics preprints | `paper` (anon id), `ym` (YYMM of the arXiv ID, the fixed effect), `period` (old/new), `alpha_main` (per-paper α of the main text; empty for 1,147 papers without one) |
| `authors.csv.gz` (2.2 MB) | 276,133 cited first authors | `author` (anon id), `recog` (GPT-4o: YES/NO/UNKNOWN; empty for 6 unlabelled), `works_count`, `cited_by_count` (OpenAlex), `first_pub_year` (first OpenAlex publication year) |
| `pairs.csv.gz` (4.9 MB) | 661,992 citing paper x cited-work first author | `paper`, `author`, `ref_year` (publication year of the cited work), `ref_cit` (its OpenAlex citations) |
| `published_margins.csv` | 7 | the margins behind the published figure (pooled, panel b, panel c), for the comparison printed by the script |

Note: `works_count`, `cited_by_count` and `first_pub_year` are public OpenAlex bibliometrics and, in
combination, could in principle be matched back to an OpenAlex author profile.

## Numbers checked against the paper (all reproduce)

| quantity | paper | recomputed |
|---|---|---|
| preprints / LLM-assisted / human-written / without α | 18,211 / 2,236 / 14,828 / 1,147 | same |
| cited first authors; citations to recognized | 276,133; 21.2% | same |
| model sample | 620,954 pairs, 17,064 papers (SI) | same |
| pooled AME | +0.84 pp (95% CI 0.43–1.25), p < 0.001 | +0.84 (0.43–1.25), p = 6.6e-5 |
| early-career | 3.74% vs 3.32%, p = 0.011 | 3.74 vs 3.32, p = 0.011 |
| mid-career | 14.95% vs 14.49%, p = 0.06 | 14.95 vs 14.49, p = 0.060 |
| late-career | 35.81% vs 34.63%, p < 0.001 | 35.81 vs 34.63, p = 8.1e-4 |
| refs < 5 yr | 6.77% vs 7.09%, p = 0.07 | 6.77 vs 7.09, p = 0.065 |
| refs 5–15 yr | 18.22% vs 17.49%, p = 0.008 | 18.22 vs 17.49, p = 0.0081 |
| refs > 15 yr | 41.05% vs 38.96%, p < 0.001 | 41.05 vs 38.96, p = 1.1e-6 |
| pairs per ref-age bin (SI) | 172,611 / 258,731 / 189,612 | same |

## Upstream (not included; needs network access, API keys, BigQuery or raw arXiv sources)

Paths refer to the original project repository.

1. **arXiv corpus.** `code/get_arxiv.py`, `code/get_arxiv_physics.py` (also `code/arxiv_sources_physics.ipynb`):
   arXiv API listing of `physics.*` preprints and download of LaTeX source archives (`export.arxiv.org/e-print/<id>`),
   Jan 2023 – Jul 2026 (59,406 papers with source).
2. **References -> OpenAlex.** References parsed from each paper's LaTeX/`.bbl`/`.bib`, DOIs and arXiv IDs matched to
   OpenAlex works (live `openalex.works` table in BigQuery `ccnr-success`; tables `LLMSci.edge_old_bib`,
   `LLMSci.edge_new_bib`), first author of each cited work kept, papers with 20–60 distinct cited first authors kept.
   Author metadata from `openalex.authors` / `openalex.works`. Curation notes:
   `data/oa_recognition/frames_bib_0927/CURATION_README.md`; earlier OpenAlex-only version
   `code/01_curate_arxiv_openalex.ipynb`. The curation scripts themselves ran on the Northeastern Explorer cluster
   (`.../neural_detectors/curate/`).
3. **α (LLM-modified share).** `code/get_llm_detect_main.ipynb` (main text; `code/get_llm_detect.ipynb` for abstracts):
   distributional MLE of Liang et al. (2025, *Nat. Hum. Behav.*), using that work's public code and its physics
   reference distribution (`distribution/Phys.parquet`); per-paper α tables `LLMSci.alpha_old_pp`, `LLMSci.alpha_new_pp`.
4. **Recognition labels.** GPT-4o, temperature 0, max 40 tokens, recognition prompt of the paper's SI
   (`SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN`): 66,103 authors via `data/oa_recognition/label_oa_firstauthors.py`
   (OpenRouter, `openai/gpt-4o`), 210,024 via the OpenAI Batch API (`data/oa_recognition/batch_run/submit_batches.py`,
   `poll_batches.py`, model `gpt-4o`).
5. **Pair-level frame** `data/oa_recognition/frames_bib_0927/bib_0927_pairs.csv.gz` (= BigQuery `LLMSci.bib_0927`),
   turned into `data/` by `prep/build_inputs.py`.
6. **Original analysis/plot scripts** replaced by `make_fig5.py`: `code/llm_impact_regression.py` (pooled + panel b,
   SI coefficient figure), `code/llm_impact_layers.py` (panel c), `code/plot_panelC_MLE_merged.py` (style),
   `code/make_fig5_refage_panel.py` (panel c plot and assembly of v17 from the v16 panel a).

`prep/make_schematic_asset.py` documents how `assets/panel_a_schematic.png` was cut from the published figure.
Neither `prep/` script is needed to redraw the figure.
