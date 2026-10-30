# Fig. 1: LLM recognition of scientists is limited and uneven

Main-text Fig. 1 (`\label{fig:llm_margins}`, published as `figure/llm_margins_v18.png`).

```
python fig1_recognition_audit/make_fig1.py        # run from the repo root, ~10 s
```

Requires numpy, pandas, matplotlib, scipy and statsmodels (patsy comes with statsmodels).
The script recomputes every plotted value from `data/`, including the signal-detection
(SDT) correction, survey weighting, logit marginals and delta-method CIs. It writes:

| file | contents |
|---|---|
| `output/fig1.png`, `output/fig1.pdf` | the composite, 3850 px wide at 300 dpi (the PDF holds the same raster) |
| `output/panels/fig1{a..g}.pdf` | each panel as a vector PDF |
| `output/fig1_values.csv` | every plotted point with its 95% CI (in %) |

It also prints the caption and Section 2.1 numbers that refer to this figure.

## Panels

The three LLMs are Claude Sonnet 4.6 (`claude46`), Gemini 3.5 Flash (`gemini35`) and GPT-5.4
(`gpt54`). Recognition means the model answers YES. NO and UNKNOWN both count as 0, and an
unparsable answer is left missing.

Each LLM rate is projected onto a common false-alarm threshold, the false-positive rate (FPR)
of Claude Sonnet 4.6 on fabricated names:
`p_adj = Phi(z(p) - z(f_model) + z(f_ref))`.
Wikipedia, OpenAlex and Google are never corrected.

| panel | shows | inputs (file: columns) | estimator |
|---|---|---|---|
| a | physics recognition (20k) vs OpenAlex and Google baselines | `physicists_20k.csv`: `claude46, gemini35, gpt54, wikipedia, google_research_page, weight`. `fake_name_probe_counts.csv`. `openalex_coverage.csv` (physics row) | survey-weighted rate. Delta-method CI on z(p)-z(f) at the Kish effective n (17,497). Wikipedia: Wilson CI. Google: weighted share, with missing weights set to 1 as in the original |
| b | recognition by field vs bibliometric retrieval | physics row as in (a). `other_fields_counts.csv` (10k per field: `n_yes, n_no, n_unknown`). `openalex_coverage.csv` | as in (a), unweighted for the other fields |
| c | 14 model releases, 2024–2026 (2,000-physicist subset) | `timeline_counts.csv`: `n_yes, n_no, n_unknown` per release, `fake_yes, fake_n` from each model's 1,000-name timeline probe, `release` | the same SDT correction, with the reference threshold set to Sonnet 4.6's FPR on the timeline probe (19/1000) |
| d | recognition by citation ventile | `physicists_20k.csv`: `citations, weight`, model columns | bin edges are weighted 5% quantiles. Weighted rate per bin, Wilson CI at the Kish n, then shifted. x is the weighted median citation count in each bin. The zero-citation bin falls off the log axis |
| e | gender | `physicists_20k.csv`: `gender, career_stage, citations, publications, weight` + outcome | logit `y ~ log1p(pubs) + log1p(cites) + C(gender) + C(career_stage)` with robust (per-scientist cluster) SE. Counterfactual marginals averaged with survey weights. Delta-method CI. LLMs then shifted |
| f | career stage | as in (e) | as in (e) |
| g | region (6 regions; Oceania sampled but not shown) | as in (e), plus `region` | the same logit with `+ C(region)`, unweighted, restricted to the 6 regions |

No panel needs a static asset: all seven are redrawn from data, so `assets/` is empty and
not shipped. The thin rule above panel c and the position of the letter "c" copy the published
composite. In the published figure, that strip was cropped from an earlier composite.

## Data (`data/`, 1.1 MB)

| file | rows | contents |
|---|---|---|
| `physicists_20k.csv` | 20,000 | one row per sampled physicist. `id` is an anonymous integer and the row order is shuffled. `weight` is the survey weight (population share / sample share of the region; empty for the 398 with no recorded region). `gender` ∈ {male, female}. `career_stage` ∈ {early-, mid-, late-career}. `region` ∈ 7 world regions. `citations` and `publications` are cumulative counts. `claude46`, `gemini35`, `gpt54` are 1 for YES and 0 for NO/UNKNOWN, empty if unparsable (58 Gemini, 3 GPT). `wikipedia` is 1 if the scientist has a science-related Wikipedia article. `google_research_page` is 1 if a name-only Google search returns a research-related page bearing the name |
| `fake_name_probe_counts.csv` | 3 | YES count and number of parsed answers on 1,000 fabricated names, per model (Claude 26/1000, Gemini 263/991, GPT 18/1000) |
| `other_fields_counts.csv` | 9 | answer counts for the 10,000-scientist biology, chemistry and social-science samples |
| `timeline_counts.csv` | 14 | answer counts per model release, plus that model's fabricated-name YES count and n |
| `openalex_coverage.csv` | 4 | share of each field's sample that resolves to a distinct OpenAlex author (fixed inputs, see below) |

Names and Dimensions IDs are removed, as are raw model responses and record-level
Dimensions, OpenAlex and Google data. The per-scientist file still holds exact citation and
publication counts, because the logit in panels e–g uses them as continuous controls.

## Reproduction check

These numbers from the caption, Section 2.1 and SI S3 reproduce exactly from `data/`:
* physics 17.0 / 19.3 / 15.4% (Claude / Gemini / GPT), Wikipedia 3.98%, Google 79.5%
* by field: biology 14.3–16.3%, chemistry 13.2–16.4%, social sciences 16.9–18.0%, overall range 13.2–19.3%
* timeline: corrected range 7.8–25.8%; latest of each family 12.5–17.5% (GPT-5.4 12.5499%, Sonnet 4.6 17.45%); GPT-4.1 to GPT-5.4 11.7–13.2%
* citation ventiles: bottom 5% at 13.5 / 14.3 / 12.1%, top 5% at 40.9 / 48.5 / 36.4%, so 51–64% unrecognized
* gender: male/female +21–29% for the LLMs, +5% on Wikipedia; Wikipedia female−male difference z = −0.55, p = 0.58 (male coefficient z = 0.54, p = 0.59)
* career stage: late/early +85–112% for the LLMs, +171% on Wikipedia
* region (Claude): Western Europe 21.7%, Africa 12.6%, Latin America 13.6%
* Kish effective n 17,497 (design effect 1.12), with 398 scientists missing a region

Not recomputed:
* **OpenAlex baselines.** The values 95.2 / 96.2 / 97.3 / 96.7% are shipped as fixed inputs.
  They are not recomputed here.

## Upstream (original pipeline, not included)

Paths are relative to the project root.

* **Sample.** Physicists come from the Dimensions author panel (`author_panel_complete.csv`,
  1.09 M physicists, Dimensions on BigQuery). The original 10k sample and the 10k per other
  field were drawn in `code/query_recognition_dimensions_multifields.ipynb`. The region-balanced
  top-up and the weights come from `panel1_region20k/code/sample_region_topup.py`, which writes
  `physics_combined_20k.csv`.
* **LLM recognition, closed-book, via the OpenRouter API.**
  * The original 10k physics sample, the three other fields and the fabricated-name probe ran
    in `code/query_recognition_dimensions_multifields.ipynb`, with models
    `anthropic/claude-sonnet-4.6`, `google/gemini-3.5-flash` and `openai/gpt-5.4`.
  * The 10k region top-up ran in `panel1_region20k/code/query_recognition_topup_v2.py`, with the
    paper prompt.
  * The model timeline ran in `code/query_recognition_timeline_models.ipynb` (GPT-4o ×3, GPT-4.1,
    GPT-5.1, GPT-5.2, GPT-5.4), `code/query_recognition_timeline_claude.ipynb` (Claude 3 Haiku,
    3.5 Haiku, Sonnet 4, 4.5, 4.6) and `code/query_recognition_timeline_gemini.ipynb`
    (Gemini 2.5 Flash, 3.5 Flash). They were put on one threshold by `code/unify_timeline_ref.py`.
* **Wikipedia.** `panel1_region20k/code/run_wikidata_lookup_topup.py` queried the Wikidata and
  Wikipedia APIs and wrote `wikidata_lookup_20k.csv`.
* **Google search.** `panel1_region20k/code/google_benchmark.py` and `serper_ladder.py` ran
  name-only Google web search via Serper and wrote
  `data/google_benchmark/coverage_full20k_real.csv` (`S2_science_strict`).
* **OpenAlex retrievability.** The crosswalk of Dimensions authors to OpenAlex authors is on
  BigQuery (`ccnr-success.LLMSci.pall_oa_final`).
* **Original figure code.** Panels came from these scripts in `panel1_region20k/code/`:
  * `make_panel_ac_20k_coverage.py google` and `c`: panels a and b
  * `make_panel_g_quantile.py --ventiles`: panel d
  * `make_panel_defg_20k.py d e f`: panels e–g
  * `make_panel1_figures.py`, `make_panel_ac_20k.py`, `panel_style.py`: shared statistics and style

  `code/plot_timeline_nature.py` drew panel c, and `make_fig1_composite.py` assembled the
  composite. That run's logs are in `panel1_region20k/logs_v2prompt/`.
* **Colour difference in panel c.** In the published figure, panel c's colours are slightly
  washed out, because the panel was cropped from an earlier, re-rasterised composite. Here it
  uses the house palette shared by all panels.
