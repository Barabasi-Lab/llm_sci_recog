# 5. LLM-assisted papers and citations (Section 2.5, Fig. 5)

```
python 5_llm_assisted_citations/make_fig5.py              # refits all models (~30 s)
python 5_llm_assisted_citations/make_fig5.py --published  # plot data/published_margins.csv without refitting
```

Writes `output/fig5.{png,pdf}` and `output/fig5_margins.csv`. It also prints the panel-a counts and the margins, and their largest difference from the published margins. The panel-a schematic is static (`assets/`); its counts are recomputed.

## Data

Identifiers are anonymous integers. No names, OpenAlex IDs, arXiv IDs or text are included.

| file | rows | columns |
|---|---|---|
| `papers.csv.gz` | 18,211 citing arXiv physics preprints | `paper`, `ym` (YYMM, fixed effect), `period` (old = 2023-01–2024-11, new = 2024-12–2026-07), `alpha_main` (main-text α; empty if none) |
| `authors.csv.gz` | 276,133 cited first authors | `author`, `recog` (GPT-4o: YES/NO/UNKNOWN), `works_count`, `cited_by_count`, `first_pub_year` (OpenAlex) |
| `pairs.csv.gz` | 661,992 citing paper × cited first author | `paper`, `author`, `ref_year`, `ref_cit` |
| `published_margins.csv` | 7 | margins behind the published figure, for comparison |

## Estimation

Logit `y ~ asst + log1p(cited_by_count) + log1p(works_count) + career_age + log1p(ref_age) + log1p(ref_cit) + C(ym)`, where y = GPT-4o recognizes the cited first author and asst = α > 0.1. SEs are clustered by citing paper. Fits: one pooled, and one per panel cell (b: career stage of the cited author; c: age of the cited work). Counterfactual means set every pair to asst = 0 or 1. CIs and p-values use the delta method. Ages are counted to 2024 (old window) or 2026 (new window).

## Pipeline (`pipeline/`)

α estimation and GPT-4o labelling behind `data/`. The arXiv corpus and the reference-to-OpenAlex linkage are described in the SI and not included.

| step | does | needs |
|---|---|---|
| `01_alpha_main.py`, `02_alpha_main_parallel.py` | Per-paper main-text α (Liang et al., 2025), old and new window | Liang et al. code and `distribution/Phys.parquet` (`--liang-repo`), spaCy `en_core_web_sm` |
| `03_label_openrouter.py` | GPT-4o recognition labels via OpenRouter (temperature 0, max 40 tokens) | `OPENROUTER_API_KEY` |
| `04_batch_build_jsonl.py`, `05_batch_submit.py`, `06_batch_poll.py` | Remaining GPT-4o labels via the OpenAI Batch API, same prompt and settings | `OPENAI_API_KEY` |

The recognition prompt is the standard one (SI). The α code is at <https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers> (not included).
