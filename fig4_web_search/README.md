# Fig. 4 — Web search reflects the model's parametric memory

Main-text Fig. 4 (`\label{fig:online_anchoring}`, published as `figure/online_search_v17.png`), plus the
numbers of Sec. 2.4, SI "web methods" and Table `tab:web_inclusion`.

```
python fig4_web_search/make_fig4.py      # run from the repo root; ~4 s
```

Writes `fig4_web_search/output/fig4.png` (300 dpi, 1980 x 2333 px, same canvas as the published file) and
`fig4.pdf`, and prints every count, retention rate, odds ratio and p-value it recomputes.
Dependencies: numpy, pandas, matplotlib, statsmodels.

## Data (`data/`)

One row per concept–scientist pair. Scientists are replaced by anonymous ids (`person_id`, `S0001`…,
randomly assigned and shared across the three files); concept names are kept. No names, no Dimensions
researcher ids, no raw model responses.

| file | rows | columns |
|---|---|---|
| `offline_pairs.csv` | 1,608 | `concept`, `person_id`, `position` (rank in the closed-book answer, after de-duplication) |
| `searched_pairs.csv` | 1,599 | `concept`, `person_id`, `stage` (`A` = first, pre-retrieval query batch; `B` = first named in a later batch), `in_final` (0/1, reaches the final online answer), `in_offline` (0/1, in the closed-book answer for that concept), `identity_source` (how the person was anchored to Dimensions: `seminal-doi`, `curated-fallback` name match, `dropped-web-recovered` = the 61 hand-identified searched-but-dropped names, `unresolved`; suffix `\|no-career-record`), `career_cites`, `career_pubs` (summed over the researcher's Dimensions publications; empty if unavailable), `relevance` (SPECTER2 topic relevance, mean cosine of the 3 closest of the person's 100 most-cited papers to "concept. definition") |
| `final_pairs.csv` | 1,623 | `concept`, `person_id`, `provenance` (`A`, `B`, or `untraced` = no logged query), `position` (1–10 in the final answer), `in_offline`, `untraced_reason` (for untraced pairs: `in_offline`, `surname_in_query`, `neither`) |

`assets/schematic_ab_v17.png` — the schematic of panels a–b, cropped from the top 1,495 px of
`online_search_v17.png`, with the four count bars blanked out (they are redrawn from the data).

## Panels

| panel | drawn from | content |
|---|---|---|
| a | asset + `offline_pairs.csv` (row count) | closed-book recommendation; count bar 1,608 |
| b | asset + `searched_pairs.csv` (`stage`) + `final_pairs.csv` (`provenance`) | Step A 983, Step B 616, Step C traced 793 + 283 = 1,076 (bar lengths on one shared scale) |
| c | `searched_pairs.csv`: `stage`, `in_final`, `career_cites`, `career_pubs`, `relevance` | P(searched name in final answer), Step B vs Step A. Raw = proportions with Wilson 95% CIs; adjusted = logistic regression of `in_final` on Step A indicator + log1p(cites) + log1p(pubs) (+ relevance), g-computed (mean prediction with every pair set to A / to B), 95% CIs from 300 pair bootstraps (seeds 0–299). Stars = Wald p of the Step A coefficient |
| d | `final_pairs.csv`: `provenance` in {A, B}, `position` | share of each group's traceable names at each position |

Static: the schematic drawing (boxes, dots, labels, "avg x / concept" texts, panel letters a, b).
Recomputed: all counts on the bars, panels c and d, and every number below.

## Numbers checked against the paper (1030 version)

All reproduce: 1,608 offline (8.0 ± 2.1 per concept); 983 Step A (4.9 ± 2.8, over 199 concepts);
616 Step B (3.1 ± 4.1, median 2); 1,623 final (8.1 ± 2.0); 1,076 traced (5.4 per concept) = 793 A + 283 B,
104 of the 283 also offline; 547 untraced = 28% offline / 10% surname in query / 63% neither;
identity 1,092 DOI-anchored / 317 name fallback / 61 hand-identified / 129 unresolved / 46 without career
record → 1,424 with career statistics (175 dropped); relevance 0.900 (web) vs 0.909 (parametric);
Table `tab:web_inclusion`: 45.9/80.7 (OR 4.91, p 4e-44, p_cl 1e-24), 52.5/84.6 (4.98, 6e-36, 2e-22),
53.5/84.3 (5.08, 8e-35, 2e-22); median position 3 vs 6; top-3 share 56% parametric (448/793 = 56.49%) vs 17% web-discovered.

Note: `final_pairs.in_offline` gives 103 (not 104) Step B final
names in the closed-book list, because the two upstream tables flag closed-book membership with
separately resolved name keys; the 104 in the SI is from `searched_pairs` (`stage==B & in_final & in_offline`), which the script uses.

## Upstream (not included here)

The original pipeline scripts (not included; paths relative to the project's `llm/code/` folder):

- **Recommendation runs** — `recommend_runner_v2.py {offline|online}`: GPT-5.4 via the **OpenAI API**
  (direct, not OpenRouter, because only the native `web_search` tool exposes the model's query batches);
  closed-book and with web search → `recommend_offline_v2.json`, `recommend_native_v2.json`.
- **Query-name extraction** — `extract_query_names_v2.py`: GPT-5.4 (`openai/gpt-5.4` via **OpenRouter**,
  temperature 0) lists person names literally present in each query batch → `query_names_v2.json`.
- **Name resolution / Step A–B units** — `query_resolver_v2.py`, `online_search_analysis.py`
  (→ `analysis/online_search_master.csv`, incl. the identity-based in-answer flag fix).
- **Recognition probes** (bare "as a scientist?" via OpenRouter; `extend_query_recog.py`, `recog_v2_allforms.py`)
  — not used by any Fig. 4 panel.
- **Dimensions matching and impact** — `control_relevance/resolve_by_seminal_doi.py`, `build_researcher_ids.py`,
  `build_curated_citations_rids.py`, `dropped_names_identify.py` (61 hand-identified names); Dimensions
  publications on Google BigQuery (`dimensions-ai.data_analytics.publications`)
  → `analysis/relevance/units_researcher_ids.csv`.
- **Topic relevance** — `control_relevance/gen_definitions.py` (GPT-5.4 concept definitions, OpenAI API),
  `control_relevance/embed_dim_relevance.py` (SPECTER2 `allenai/specter2_base` + proximity adapter)
  → `analysis/relevance/author_relevance_dim.csv`.
- **Untraced provenance / positions** — `analysis_untraced_provenance.py` → `analysis/final_pairs_provenance.csv`,
  `analysis/untraced_final_pairs.csv`.
- **Published panels** — `gen_panel_ab_v4_bars.py` (schematic + bars, SVG), `control_relevance/make_panel_e_bars.py --wide`
  (panel c), `make_rank_position_traced.py --wide` (panel d), `compose_online_search_v17.py` (composite).

The CSVs here were derived from `recommend_offline_v2.json` (resolved with `query_resolver_v2.build()` and
de-duplicated per concept), `units_researcher_ids.csv`, `author_relevance_dim.csv`, `online_search_master.csv`,
`final_pairs_provenance.csv` and `untraced_final_pairs.csv`, keeping only the columns above.
