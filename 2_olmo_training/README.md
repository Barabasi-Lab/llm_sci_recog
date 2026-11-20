# 2. Training-data frequency and recognition (Section 2.2, Fig. 2)

```
python 2_olmo_training/make_fig2.py
```

Writes `output/fig2.{png,pdf}` and prints the numbers quoted in Section 2.2.

## Data

`data/fig2_scientists.csv`: 12,214 physicists (random and most-cited samples after name-disambiguation filtering), shuffled rows.

| column | meaning |
|---|---|
| `id` | anonymous integer |
| `citations` | cumulative citations (Dimensions) |
| `mentions` | infini-gram count of "First Last" in the OLMo-2 training corpus |
| `recognized` | 1 if OLMo-2-13B-Instruct answered YES for the recorded or the reduced "First Last" name |

## Estimation

- a, b: Gaussian KDE of log10 citations / mentions, recognized vs not; dashed lines are medians.
- c: recognition rate in log bins with Wilson 95% CIs. The line is a univariate logistic regression on log10(x), fitted to all rows. Zeros are clipped to 0.5 for the log axes.

## Pipeline (`pipeline/`)

Training-data counts and OLMo-2 recognition behind `data/fig2_scientists.csv`. The cohorts and name-disambiguation filters use licensed Dimensions records (SI) and are not shipped.

| step | does | needs |
|---|---|---|
| `01_infinigram_counts_and_olmo_recognition.py` | `--step collect / recount`: infini-gram name counts; `--step recognize`: OLMo-2 recognition probe | infini-gram API, MLX |
| `02_refine_recognition_cleaned.py` | Re-ask OLMo-2 with the reduced "First Last" name; `merge` adds `recognized_any` | MLX |

Settings: infini-gram index `v4_olmo-2-0325-32b-instruct_llama` (exact-string count); `mlx-community/OLMo-2-1124-13B-Instruct-4bit`, standard recognition prompt (SI), greedy decoding.
