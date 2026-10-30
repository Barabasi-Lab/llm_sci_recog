# Fig. 3: causal tracing, training-data exposure and attribute probes

Main-text Fig. 3 (`\label{fig:causal_trace}`, published as `llm_process_causal_v18.png`).

```
python fig3_causal_tracing/make_fig3.py          # run from the repository root
```

This writes `fig3_causal_tracing/output/fig3.png` (5194 x 5299 px, 300 dpi, the published geometry) and `fig3.pdf`, and
prints every number quoted in the caption and in Sec. 2.3. It needs numpy, pandas, matplotlib and statsmodels and runs in
about 2 s. Arial is used for headings and panel letters when it is installed; otherwise DejaVu Sans.

## Panels

| Panel | Source | What is drawn |
|---|---|---|
| a–c | `assets/schematic_abc_from_v18.png` | Static schematic, cropped from the published figure. |
| d | `data/example_traces.npz`: `curie_ie` (8 tokens x 32 layers), `curie_tokens`, `curie_mentions` | Marie Curie, residual-stream restoration (window 1), IE in probability, colour scale 0–0.9. |
| e | `data/example_traces.npz` (`both_ie`, `both_tokens`, `both_mentions`) | Csaba Both, same setup, drawn from his raw residual trace. (`assets/panel_e_from_v18.png` is kept only as a fallback if `both_ie` is absent.) |
| f, g | `data/traces_wikidata9k.npz`: `residual_sem` (9,270 x 6 semantic positions x 32 layers); `data/mentions_wikidata9k.csv`: `olmo2_mentions` | Mean IE of the top (f) and bottom (g) quartiles of training-data mentions (2,318 scientists each). |
| h | `residual_sem[:, 2, 5:10]` (last subject token, layers 5–9), by quartile | Early-site AIE, mean ± 1.96 SE; tick labels give each quartile's median mention count. |
| i | `traces_wikidata9k.npz`: `attention_last_token[:, 15:20]` (attention, last token, layers 15–19) | Late-site AIE, same layout. |
| j, k | `data/attribute_probes_wikidata9k.csv`: `recognized`, `score_field` … `score_coauthor` (1 = correct) | Percent correct, recognized (1,994) vs not (7,276), Wilson 95% CIs, two-proportion z-tests (`***` p < 0.001; no mark when p ≥ 0.05). |

Quartiles: `pd.qcut(mentions.rank(method="first"), 4)`. Ties (786 scientists have 0 mentions) are broken by row order,
so keep the shipped row order when regrouping.

## Data

All scientists are replaced by anonymous ids (`P00001` …); the ids join the three cohort files. Only the two exemplars
shown in the figure (Marie Curie, Csaba Both) are named.

| File | Contents |
|---|---|
| `data/traces_wikidata9k.npz` (2.8 MB) | `id`; `residual_sem` float16 (9270, 6, 32): residual-stream IE, window 1, averaged into the 6 ROME positions (first / middle / last subject token, first subsequent token, further tokens, last token); `attention_last_token` float16 (9270, 32): attention-output IE, window 10, last-token row; `p_physicist` float32: clean-run p(" physicist"). Storing these as float16 leaves the reported site means unchanged. |
| `data/mentions_wikidata9k.csv` | `id`, `olmo2_mentions`: infini-gram count of the exact name string in the OLMo-2 training corpus. |
| `data/attribute_probes_wikidata9k.csv` | `id`, `recognized` (internal criterion: " physicist" is the top-1 next token, or in the top 5 with p > 0.1), and 0/1 scores for the field, country, career-stage, gender, most-cited-work, lead-author-work and frequent-coauthor probes. |
| `data/example_traces.npz` | Marie Curie's raw 8 x 32 residual trace and token labels, clean p(physicist) 0.805 and corrupted 1.3e-5, mention count 784,759; Csaba Both's raw 8 x 32 residual trace, token labels and mention count 19. |

The cohort is the 9,270 physicists with ≥ 10,000 citations and a Wikidata entry. Four hand-added exemplars that were
traced but have no mention count (Albert Einstein, Marie Curie, Mark Newman, Réka Albert) are not in the cohort files,
matching the published analysis.

## Numbers this reproduces

| Quantity | Paper | `make_fig3.py` |
|---|---|---|
| quartile size; median mentions Q1–Q4 | 2,318; 2, 47, 274, 1,705 | 2,318 (Q1, Q4); 2, 47, 274, 1,705 |
| early-site AIE Q4 vs Q1 | 0.026 vs 0.005, 5.7x | 0.0257 vs 0.0045, 5.68x |
| late-site AIE Q4 vs Q1 | 0.099 vs 0.029, 3.4x | 0.0988 vs 0.0288, 3.43x |
| Curie p(physicist), corrupted, restored | 0.805, 1.3e-5, 0.848 | 0.805, 1.3e-5, 0.848 (last subject token, layer 2) |
| field | 34.40% vs 6.75% (five-fold) | 34.40% vs 6.75% (5.10x) |
| career stage; gender | 51.91 vs 46.52; 75.58 vs 70.68 (p < 0.001) | same; p = 2.0e-5 and 1.7e-5 |
| country | (52–76% vs 47–71% range) | 56.17% vs 54.43%, p = 0.17 (no mark) |
| scholarly probes | 2.5–4.9% vs 0.8–1.1%, 2.3–4.9x | 2.51–4.91% vs 0.80–1.10%, 2.28–4.91x |

Caveats:
- Layers are counted from 0, as on the heatmap axes of panels d–g: the early and late sites average layers 5–9 and 15–19 (Python slices `5:10` and `15:20`). The static schematic in panels b,c numbers layers 1–32.
- The clean "film 2.7e-7", corrupted "film 0.385" and patched "film 4.1e-5" in panel b are part of the static schematic. They are not in the shipped data.
- Marie Curie's 784,759 is an infini-gram count query on the same index. It is stored as a constant in `example_traces.npz` because no project data file holds it. Csaba Both's 19 comes from the project's network-scientist count table.

## Upstream (not included; regenerating the traces needs a GPU)

- **Model:** OLMo-2-1124-7B-Instruct (AI2; Team OLMo, "2 OLMo 2 Furious", 2025), from Hugging Face `allenai/OLMo-2-1124-7B-Instruct`.
- **Causal tracing:** ROME-style restoration (Meng et al., "Locating and Editing Factual Associations in GPT", 2022) on the prompt `"[NAME] is by profession a"`, target token `" physicist"` (id 83323), corrupted run = subject replaced by `"Meryl Streep"`, metric = probability, restored component residual stream (window 1) and attention (window 10). The tracing code is Arnab Sen Sharma's (David Bau lab) implementation (`src/trace.py`, `trace_important_states`), patched to run under nnsight 0.5 via baukit `TraceDict`. In the project: `arnab _codes/aicr/run_batch_harmonized.py` (batch traces, `--cohort wikidata9k --variant profession`), `arnab _codes/aicr/make_figures_harmonized.py` (`extract_semantic_positions`, 6-position aggregation), `arnab _codes/aicr/plot_example_pair.py` (single-scientist panels d,e), and `export_semantic_traces.py` on the cluster (writes the per-name 6 x 32 matrices that `traces_wikidata9k.npz` is reduced from). Runs took about 10 s per name on one RTX Pro 6000.
- **Training-data mentions:** infini-gram (Liu et al., 2024) count API, index `v4_olmo-2-0325-32b-instruct_llama` (the OLMo-2 pre-/post-training corpus, built on Dolma), via `arnab _codes/aicr/fetch_infinigram.py` → `code/olmo_training_10k_phy.py`.
- **Attribute probes:** OLMo-2-7B asked for field, country, career stage, first-publication year, gender, most-cited work, lead-author work and frequent coauthor (JSON output), scored against each scientist's bibliometric and Wikidata record (SI, OLMo hallucination section). The scored per-name table is `data/llm_hallucination_olmo7b/scored_olmo7b_json_year_gender_enumlast.csv`. The script that generated and scored it was run on the cluster and is not in the local project tree. The figure scripts were `code/david_fig3_prevalence.py` and `code/make_fig3_v16.py`/`code/make_fig3_v18.py`. The published figure was assembled from the 1022 v15 composite, with f–i drawn by `make_fig3_v18.py`.
