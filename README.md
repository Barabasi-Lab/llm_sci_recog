# The selective scientific memory of large language models

Code and data to reproduce the five main-text figures of

> Y. Liu, R. Dorantes-Gilardi, Á. Elekes, D. Bau, A.-L. Barabási. *The selective scientific memory of large language models.* (2026)

Each figure has its own folder with the processed data it needs, a single script that redraws it, and a README describing every panel, its inputs and the upstream pipeline that produced the data.

| Folder | Figure | What it shows |
|---|---|---|
| [`fig1_recognition_audit/`](fig1_recognition_audit/) | Fig. 1 | LLM recognition of 20,000 physicists (and three other fields) against OpenAlex, Google and Wikipedia; by model release, citations, gender, career stage and region |
| [`fig2_training_data/`](fig2_training_data/) | Fig. 2 | OLMo-2 recognition versus citations and training-data mentions |
| [`fig3_causal_tracing/`](fig3_causal_tracing/) | Fig. 3 | Causal tracing in OLMo-2-7B by training-data quartile; attribute accuracy by internal recognition |
| [`fig4_web_search/`](fig4_web_search/) | Fig. 4 | GPT-5.4 recommendations for 200 network-science concepts, closed-book and with web search |
| [`fig5_citations/`](fig5_citations/) | Fig. 5 | Citations to GPT-4o-recognized first authors in LLM-assisted versus human-written arXiv physics papers |

## Running

Python 3.10+ with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
python fig1_recognition_audit/make_fig1.py
python fig2_training_data/make_fig2.py
python fig3_causal_tracing/make_fig3.py
python fig4_web_search/make_fig4.py
python fig5_citations/make_fig5.py
```

Run from the repository root. Each script writes `output/figN.png` and `output/figN.pdf` in its folder and prints the numbers quoted in the paper for that figure. All five run on a laptop in under a minute in total (Fig. 5 refits its regressions in about 30 s). No API keys, GPU or network access are needed.

The scripts recompute every plotted value from the shipped data: signal-detection corrections, survey weights and logit marginals (Fig. 1), binned rates and logistic fits (Fig. 2), quartile means of the causal traces and probe accuracies (Fig. 3), retention rates and adjusted probabilities with bootstrap CIs (Fig. 4), and the citation regressions (Fig. 5). Schematic panels (Fig. 3a–c, Fig. 4a,b drawings, Fig. 5a) are included as static images.

## Data

The shipped files are derived, de-identified tables. Scientists, authors and papers are identified only by random integer or string ids; names are removed except for the two exemplars shown in Fig. 3 (Marie Curie and Csaba Both). No raw records, LLM responses or paper texts are included.

Upstream sources:

- **Dimensions** (Digital Science): scientist samples, publications, citations, affiliations. Available under licence; only derived variables are shared.
- **OpenAlex**: author identity crosswalk (Fig. 1) and cited works and authors (Fig. 5).
- **arXiv**: physics preprints, January 2023 – July 2026 (Fig. 5).
- **Wikidata / Wikipedia** and **Google search** (via Serper): reference benchmarks (Fig. 1).
- **OLMo-2** models and training corpus (Allen Institute for AI); name counts via the **infini-gram** API (Figs. 2, 3).
- **LLM queries**: Claude Sonnet 4.6, Gemini 3.5 Flash, GPT-5.4 and earlier releases (Fig. 1); GPT-5.4 with web search (Fig. 4); GPT-4o (Fig. 5).

The causal-tracing analyses build on the ROME implementation of causal tracing (Meng et al., 2022) and on code provided by Arnab Sen Sharma. Regenerating the traces requires a GPU and is not part of this repository; the per-scientist trace matrices are shipped instead.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0.

## Contact

Albert-László Barabási (barabasi@gmail.com).
