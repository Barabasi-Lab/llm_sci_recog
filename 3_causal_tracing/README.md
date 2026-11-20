# 3. Recognition inside the model (Section 2.3, Fig. 3)

```
python 3_causal_tracing/make_fig3.py
```

Writes `output/fig3.{png,pdf}` and prints the numbers quoted in Section 2.3. Panels a–c are a static schematic (`assets/`).

## Data

Scientists are anonymous ids (`P00001`…); only the two exemplars in the figure (Marie Curie, Csaba Both) are named. The cohort is the 9,270 physicists with ≥10,000 citations and a Wikidata entry.

| file | contents |
|---|---|
| `traces_wikidata9k.npz` | `residual_sem` (9,270 × 6 positions × 32 layers): residual-stream indirect effect, window 1, averaged into the six ROME positions (first/middle/last subject token, first subsequent token, further tokens, last token); `attention_last_token` (9,270 × 32): attention-output indirect effect, window 10, last token; `p_physicist`: clean-run p(" physicist") |
| `mentions_wikidata9k.csv` | `olmo2_mentions`: infini-gram count of the name in the OLMo-2 training corpus |
| `attribute_probes_wikidata9k.csv` | `recognized` (" physicist" is the top-1 next token, or in the top 5 with p > 0.1) and 0/1 scores for the field, country, career-stage, gender, most-cited-work, lead-author-work and coauthor probes |
| `example_traces.npz` | Raw 8 × 32 residual traces, token labels and mention counts for the two exemplars |

## Estimation

- d, e: single-scientist traces.
- f, g: mean trace of the top and bottom quartiles of training-data mentions.
- h: early site (last subject token, layers 5–9, residual), by quartile.
- i: late site (last token, layers 15–19, attention), by quartile. Mean ± 1.96 SE.
- j, k: percent correct, recognized vs not; Wilson 95% CIs and two-proportion z-tests.

Layers are counted from 0.

## Pipeline (`pipeline/`)

Causal tracing, training-data counts and attribute probes behind `data/`. Intermediates go to `pipeline/work/` (git-ignored; they hold names). The cohort and its ground truth come from licensed Dimensions records (SI) and are not shipped.

| step | does | needs |
|---|---|---|
| `01_trace_batch.py` | Causal tracing per name: residual (window 1) and attention (window 10) | GPU, `allenai/OLMo-2-1124-7B-Instruct`, tracing library |
| `baukit_patch.py` | Patch that runs the tracing library under nnsight 0.5 | `baukit` |
| `02_export_semantic_traces.py` | Reduce each trace to 6 positions × 32 layers | torch |
| `03_infinigram_counts.py` | Name counts in the OLMo-2 training corpus | infini-gram API |
| `04_attribute_probes.py` | p(" physicist") and the JSON attribute probe, greedy | GPU |
| `05_score_probes.py` | Recognition label and per-probe scores | `rapidfuzz`, `pycountry` |

Tracing: prompt `"[NAME] is by profession a"`, target `" physicist"`; the corrupted run replaces the subject with `"Meryl Streep"`; metric = probability. Causal tracing follows ROME (Meng et al., 2022), using Arnab Sen Sharma's implementation (David Bau lab).
