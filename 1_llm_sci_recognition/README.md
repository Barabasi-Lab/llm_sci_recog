# 1. Scientific recognition of LLMs (Section 2.1, Fig. 1)

```
python 1_llm_sci_recognition/make_fig1.py
```

Writes `output/fig1.{png,pdf}`, one vector PDF per panel (`output/panels/`) and `output/fig1_values.csv` (every plotted point with its 95% CI). It also prints the numbers quoted in Section 2.1.

## Data

| file | rows | contents |
|---|---|---|
| `physicists_20k.csv` | 20,000 | One row per sampled physicist; anonymous `id`, shuffled rows. `weight`: survey weight (empty for the 398 with no recorded region). `gender`, `career_stage` (early-/mid-/late-career), `region`, `citations`, `publications`. `claude46`, `gemini35`, `gpt54`: 1 = YES, 0 = NO/UNKNOWN, empty if unparsable. `wikipedia`: has a science-related Wikipedia article. `google_research_page`: a name-only Google search returns a research-related page bearing the name |
| `fake_name_probe_counts.csv` | 3 | YES count and parsed answers on 1,000 fabricated names, per model |
| `other_fields_counts.csv` | 9 | Answer counts for the 10,000-scientist biology, chemistry and social-science samples |
| `timeline_counts.csv` | 14 | Answer counts per model release, with that model's fabricated-name YES count |
| `openalex_coverage.csv` | 4 | Share of each field's sample that resolves to an OpenAlex author |

## Estimation

LLM rates are put on a common false-alarm threshold, the fabricated-name false-positive rate of Claude Sonnet 4.6: `p_adj = Φ(z(p) − z(f_model) + z(f_ref))`. Wikipedia, OpenAlex and Google are not corrected.

| panel | estimator |
|---|---|
| a, b | Survey-weighted rate (physics) or unweighted (other fields); delta-method CI at the Kish effective n; Wilson CI for Wikipedia |
| c | Same correction, reference = Sonnet 4.6's false-positive rate on the timeline probe |
| d | Weighted citation ventiles; weighted rate per bin, Wilson CI |
| e, f | Logit `y ~ log1p(pubs) + log1p(cites) + C(gender) + C(career_stage)`, robust SE, survey-weighted counterfactual marginals, delta-method CI |
| g | Same logit with `+ C(region)`, unweighted |

## Pipeline (`pipeline/`)

The model queries behind `data/`. Run from `pipeline/` in order. Intermediates go to `pipeline/work/` (git-ignored). The samples are drawn from the licensed Dimensions panel (SI) and are not shipped. The fabricated-name lists are not shipped either, because each name is a one-letter edit of a real surname; step 2 regenerates them.

| step | does | needs |
|---|---|---|
| `01_query_recognition.py` | Recognition prompt, 3 models × 4 fields (10,000 each) | `OPENROUTER_API_KEY` |
| `02_fake_names.py` | 1,000 fabricated names (one-edit surname perturbations, checked against every panel name) | Dimensions panel |
| `03_fake_name_probe.py` | Same prompt on the fabricated names | `OPENROUTER_API_KEY` |
| `04_query_recognition_topup.py` | Recognition prompt on the region-balanced physics top-up | `OPENROUTER_API_KEY` |
| `05_model_timeline.py --family {gpt,claude,gemini}` | 2,000-physicist subset × 14 releases, with fabricated-name probe | `OPENROUTER_API_KEY` |

Prompt (SI, "Standardized prompt"): system "You are a knowledgeable assistant. Answer concisely and honestly. Do not speculate."; user "Do you recognize {Last}, {First} as a scientist? If so, what is their field?" with the reply template; temperature 0. The first `SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN` in the reply is parsed.
