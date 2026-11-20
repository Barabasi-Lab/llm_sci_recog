# 4. Web search (Section 2.4, Fig. 4)

```
python 4_web_search/make_fig4.py
```

Writes `output/fig4.{png,pdf}` and prints the counts, retention rates and odds ratios of Section 2.4. The schematic of panels a–b is static (`assets/`); its count bars are redrawn from the data.

## Data

One row per concept–scientist pair. Scientists are anonymous ids (`S0001`…, shared across files); concept names are kept.

| file | rows | columns |
|---|---|---|
| `offline_pairs.csv` | 1,608 | `concept`, `person_id`, `position` in the closed-book answer |
| `searched_pairs.csv` | 1,599 | `stage` (A = named in the first query batch, B = first named in a later batch), `in_final`, `in_offline`, `identity_source`, `career_cites`, `career_pubs`, `relevance` (SPECTER2 topic relevance) |
| `final_pairs.csv` | 1,623 | `provenance` (A, B, or untraced), `position` in the final answer, `in_offline`, `untraced_reason` |

## Estimation

- c: P(searched name reaches the final answer), Step B vs Step A. Raw rates with Wilson CIs. Adjusted rates come from a logit of `in_final` on stage + log1p(cites) + log1p(pubs) (+ relevance), g-computed, with 300 pair-bootstrap CIs.
- d: share of each group's traceable names at each position.

## Pipeline (`pipeline/`)

The GPT-5.4 runs and the query-log analysis behind `data/`. Intermediates go to `$FIG4_WORK` (default `pipeline/work/`). Raw model responses contain names and are not shipped. Linking names to Dimensions researcher records (SI) uses licensed data and is not included. Shipped inputs: `inputs/network_concepts_200.json` and `inputs/concept_definitions.json`.

| step | does | needs |
|---|---|---|
| `01_recommend.py offline\|online` | GPT-5.4 recommendations: closed-book, and with the native `web_search` tool (each search call logged as a query batch) | `OPENAI_API_KEY` |
| `02_extract_query_names.py` | Person names in the first vs later query batches | `OPENROUTER_API_KEY` |
| `query_resolver_v2.py` | Module: name → `surname\|initial` key | – |
| `03_search_units.py` | Step A/B units, final-answer and closed-book flags | – |
| `04_final_answer_provenance.py`, `05_untraced_reasons.py` | Trace each final-answer name to a query batch | – |
| `06_concept_definitions.py`, `07_relevance_specter2.py` | Topic relevance (SPECTER2 + proximity adapter) | `OPENAI_API_KEY`, torch |
