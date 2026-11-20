# The selective scientific memory of large language models

Code and data for

> Y. Liu, R. Dorantes-Gilardi, Á. Elekes, D. Bau, A.-L. Barabási. *The selective scientific memory of large language models.* (2026)

One folder per Results section of the main text. Each folder holds:

- `make_figN.py`: analysis and plot. It recomputes every statistic and plotted value of the figure from `data/` and prints the numbers quoted in that section.
- `data/`: the de-identified tables the analysis needs.
- `pipeline/`: the numbered scripts, in run order, that queried the models and computed the LLM-derived measures in `data/`.
- `README.md`: data dictionary, estimators and the pipeline steps.

| Folder | Section | Figure | What it shows |
|---|---|---|---|
| [`1_llm_sci_recognition/`](1_llm_sci_recognition/) | 2.1 Sparse and unequal scientific recognition of LLMs | Fig. 1 | LLM recognition of 20,000 physicists (and three other fields) against OpenAlex, Google and Wikipedia; by model release, citations, gender, career stage and region |
| [`2_olmo_training/`](2_olmo_training/) | 2.2 Training data frequency is a primary predictor of scientific recognition | Fig. 2 | OLMo-2 recognition versus citations and training-data mentions |
| [`3_causal_tracing/`](3_causal_tracing/) | 2.3 Recognition is a localized signal inside the model | Fig. 3 | Causal tracing in OLMo-2-7B by training-data quartile; attribute accuracy by internal recognition |
| [`4_web_search/`](4_web_search/) | 2.4 Parametric recognition shapes web search outcomes | Fig. 4 | GPT-5.4 recommendations for 200 network-science concepts, closed-book and with web search |
| [`5_llm_assisted_citations/`](5_llm_assisted_citations/) | 2.5 Papers with detectable LLM assistance preferentially cite recognized scientists | Fig. 5 | Citations to GPT-4o-recognized first authors in LLM-assisted versus human-written arXiv physics papers |

## Reproducing the figures

Python 3.10+ with the packages in `requirements.txt` (exact versions used for the shipped outputs):

```bash
pip install -r requirements.txt
python 1_llm_sci_recognition/make_fig1.py
python 2_olmo_training/make_fig2.py
python 3_causal_tracing/make_fig3.py
python 4_web_search/make_fig4.py
python 5_llm_assisted_citations/make_fig5.py
```

Installing the dependencies takes under a minute on a standard desktop computer. Each script writes `output/figN.png` and `output/figN.pdf` in its folder and can be run from any directory. All five run on a laptop in under 30 s in total. No API keys, GPU or network access are needed.

The scripts recompute every plotted value: signal-detection corrections, survey weights and logit marginals (Fig. 1), binned rates and logistic fits (Fig. 2), quartile means of the causal traces and probe accuracies (Fig. 3), retention rates and adjusted probabilities with bootstrap CIs (Fig. 4), and the citation regressions (Fig. 5). Schematic panels (Fig. 3a–c, Fig. 4a,b drawings, Fig. 5a) are static images.

## Pipeline

`pipeline/` holds the code that queried the models and produced the LLM-derived measures: recognition prompts and fabricated-name probes (Section 1), training-data counts and OLMo-2 probes (2), causal tracing and attribute probes (3), GPT-5.4 recommendations with and without web search and the query-log analysis (4), and α estimation and GPT-4o labelling (5). These scripts are not needed to redraw the figures. Running them needs API keys (OpenRouter, OpenAI; read from environment variables), the infini-gram API, a local MLX build of OLMo-2-13B, or a GPU for OLMo-2-7B. Extra packages are listed in `requirements-pipeline.txt`. Sample construction and record linkage use licensed Dimensions data; they are described in the SI and not included. Intermediates contain names and are written to `pipeline/work/` (git-ignored).

## Data

The shipped files are derived, de-identified tables. Scientists, authors and papers are identified only by random integer or string ids. Names are removed except for the two exemplars shown in Fig. 3 (Marie Curie and Csaba Both). No raw records, LLM responses or paper texts are included.

Upstream sources:

- **Dimensions** (Digital Science): scientist samples, publications, citations, affiliations. Available under licence; only derived variables are shared.
- **OpenAlex**: author identity crosswalk (Fig. 1) and cited works and authors (Fig. 5).
- **arXiv**: physics preprints, January 2023 – July 2026 (Fig. 5).
- **Wikidata / Wikipedia** and **Google search** (via Serper): reference benchmarks (Fig. 1).
- **OLMo-2** models and training corpus (Allen Institute for AI); name counts via the **infini-gram** API (Figs. 2, 3).
- **LLM queries**: Claude Sonnet 4.6, Gemini 3.5 Flash, GPT-5.4 and earlier releases (Fig. 1); GPT-5.4 with web search (Fig. 4); GPT-4o (Fig. 5).

Causal tracing follows ROME (Meng et al., 2022) and uses Arnab Sen Sharma's implementation; `3_causal_tracing/pipeline/baukit_patch.py` is our patch for nnsight 0.5. The α estimator is that of Liang et al. (2025), run with their public code and physics reference distribution.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0.

## Contact

Albert-László Barabási (barabasi@gmail.com).
