#!/usr/bin/env python
"""Step 1 (GPU). Batch causal tracing of OLMo-2-7B-Instruct over the wikidata9k cohort (Fig. 3d-i).

From the project's run_batch_harmonized.py, as run for Fig. 3 (variant `profession`). Configuration:

    model         allenai/OLMo-2-1124-7B-Instruct, bf16
    prompt        "{name} is by profession a"
    answer token  " physicist" (id 83323), fixed for every scientist
    corruption    subject swap: the name is replaced by "Meryl Streep" (CLEAN_SUBJ below)
    metric        "prob", normalize=False: indirect_effects[token, layer] = p(" physicist") in the corrupted run
                  after restoring that (token, layer) state from the scientist's run; base_score = p in the
                  scientist's run, low_score = p in the corrupted run (~1e-5)
    restored      residual stream, --window 1 (Fig. 3d-h)   and   attention output, --window 10 (Fig. 3i)

The tracing itself is Arnab Sen Sharma's ROME implementation (`src` package, not included; see README), patched for
nnsight 0.5 by baukit_patch.py. `window_size` is passed to `src.trace.get_window`, which restores layers
max(0, l - w//2) .. min(L-1, l + w//2): window 1 = layer l only, window 10 = eleven layers centred on l.
Layers are 0-indexed (OLMo-2-7B: 32 layers).

Fig. 3 runs (16-way SLURM array; one GPU, ~10 s per name for both kinds):

    python 01_trace_batch.py --kinds residual --window 1  --shard $i --nshards 16
    python 01_trace_batch.py --kinds attention            --shard $i --nshards 16   # window 10
                                                          # (the run used --kinds mlp attention; MLP is SI only)

Exemplar outside the cohort (Csaba Both, Fig. 3e): a one-row CSV with column `final_name`,
    python 01_trace_batch.py --cohort netsci --cohort-csv names.csv --require-any --kinds residual --window 1 \
        --skip-singles

Out: <out-dir>/causal_traces_harmonized_<cohort>_profession/
        {residual_w1,attention}/<slug>__prob.pt   one CausalTracingResult per name (as a dict)
        classification_shard<k>.csv              top-5 next tokens of the profession prompt (recognition label)
Needs: the tracing library on PYTHONPATH (the directory that contains `src/`), torch, transformers, nnsight,
baukit, dataclasses_json; HF access to the model.
"""

import os
import re
import sys
import argparse
import logging
import traceback

import torch
import pandas as pd
from tqdm.auto import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
WORK = os.path.join(HERE, "work")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("batch_harmonized")

# ------------------------------------------------------------------ config
MODEL_KEY        = "allenai/OLMo-2-1124-7B-Instruct"

# "profession"  prompt = "{name} is by profession a", answer token = " physicist"
#               (fixed for everyone).  Measures: can restoring the subject
#               recover the *correct* factual claim?
# (The original script also had an exploratory "fieldtop1" variant, not used in the paper.)
VARIANTS = {
    "profession": dict(prompt="{} is by profession a",
                       answer="fixed", answer_token_id=83323),   # " physicist"
}
DEFAULT_VARIANT = "profession"

# set from --variant in main()
PROMPT_TEMPLATE  = VARIANTS[DEFAULT_VARIANT]["prompt"]
CLEAN_SUBJ       = "Meryl Streep"
WINDOW_SIZE      = 10
METRIC           = "prob"
KINDS            = ["mlp", "attention"]

# recognition-label prompt -- unchanged from the published classification
CLASSIFY_TEMPLATE = "{} is by profession a"
CLASSIFY_TOP_K    = 5
TARGET_TOKENS     = {"physicist", " physicist"}

# wikidata9k: every >=10k-citation Dimensions physicist that has a Wikidata person
# label (9,393 rows -> 9,270 unique names). The source table is not shipped.
DEFAULT_COHORT = "wikidata9k"
DEFAULT_COHORT_CSV = os.path.join(WORK, "physicists_dimensions_17k.csv")

# Scientists traced in addition to the cohort. The run traced Marie Curie (panel d)
# and three other well-known exemplars outside the cohort; none of them enters any
# aggregate (names without a mention count are dropped before export).
SINGLE_SUBJECTS = ["Marie Curie"]

OUT_ROOT = None


def out_root_for(cohort, variant, out_dir):
    return os.path.join(out_dir, f"causal_traces_harmonized_{cohort}_{variant}")


def sanitize(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_").lower()


def kind_dir(kind):
    # window 10 is the default; a different window is a different experiment,
    # so keep its traces in a separate directory (residual, window 1 -> residual_w1)
    return kind if WINDOW_SIZE == 10 else f"{kind}_w{WINDOW_SIZE}"


def out_path(kind, slug):
    return os.path.join(OUT_ROOT, kind_dir(kind), f"{slug}__{METRIC}.pt")


def already_done(slug, kinds):
    return all(os.path.isfile(out_path(k, slug)) for k in kinds)


def load_cohort(path, name_col="final_name", require_any=("wikidata_name", "wd_label")):
    """Return (df, list_of_names) for a cohort table."""
    df = pd.read_csv(path, low_memory=False)
    if require_any:
        keep = pd.concat([df[c].notna() for c in require_any], axis=1).any(axis=1)
        log.info("require_any%s: %d / %d rows kept", list(require_any),
                 int(keep.sum()), len(df))
        df = df[keep]
    nc = name_col
    # the working name column is always called `wikidata_name` downstream; if the
    # source file already has one (physicists_dimensions_17k.csv, where the
    # published label is only populated for 734 rows) get it out of the way first,
    # otherwise the rename below produces two columns with the same name.
    if "wikidata_name" in df.columns and nc != "wikidata_name":
        df = df.rename(columns={"wikidata_name": "wikidata_name_published"})
    df = df.rename(columns={nc: "wikidata_name"})
    df = df[df["wikidata_name"].notna() & (df["wikidata_name"].astype(str).str.len() > 1)]
    return df, df["wikidata_name"].drop_duplicates().tolist()


# ------------------------------------------------------------- recognition
def classify_names(mt, names, batch_size=16):
    """'physicist' in the top-5 continuations of '{name} is by profession a'.

    Copied from analyze_traces.py so the label definition is byte-identical.
    """
    from src.functional import predict_next_token

    results = []
    for i in tqdm(range(0, len(names), batch_size), desc="Classifying",
                  unit="batch"):
        batch_names = names[i:i + batch_size]
        inputs = [CLASSIFY_TEMPLATE.format(n) for n in batch_names]
        try:
            preds = predict_next_token(mt=mt, inputs=inputs, k=CLASSIFY_TOP_K)
        except Exception as e:
            log.warning("classify batch %d failed: %s", i, e)
            for name in batch_names:
                results.append({"wikidata_name": name,
                                "olmo_next_token": "ERROR",
                                "top_k_tokens": [],
                                "is_physicist": False})
            continue

        if len(batch_names) == 1 and not isinstance(preds[0], list):
            preds = [preds]

        for name, pred_list in zip(batch_names, preds):
            top_tokens = [getattr(p, "token", str(p))
                          for p in pred_list[:CLASSIFY_TOP_K]]
            stripped = [t.strip().lower() for t in top_tokens]
            is_phys = any(t.strip().lower() in stripped for t in TARGET_TOKENS)
            results.append({"wikidata_name": name,
                            "olmo_next_token": top_tokens[0] if top_tokens else "",
                            "top_k_tokens": top_tokens,
                            "is_physicist": is_phys})
    return pd.DataFrame(results)


def field_answer_tokens(mt, names, batch_size=16):
    """Model's own top-1 continuation of the tracing prompt, per scientist.

    Recorded in the classification file (field_top1_*); the AIE target is the
    fixed " physicist" token under the `profession` variant.
    """
    from src.functional import predict_next_token

    out = {}
    for i in tqdm(range(0, len(names), batch_size), desc="Answer tokens",
                  unit="batch"):
        batch_names = names[i:i + batch_size]
        inputs = [PROMPT_TEMPLATE.format(n) for n in batch_names]
        try:
            preds = predict_next_token(mt=mt, inputs=inputs, k=5)
        except Exception as e:
            log.warning("answer-token batch %d failed: %s", i, e)
            continue
        if len(batch_names) == 1 and not isinstance(preds[0], list):
            preds = [preds]
        for name, pred_list in zip(batch_names, preds):
            top = pred_list[0]
            out[name] = {"token": top.token,
                         "token_id": int(top.token_id),
                         "prob": float(getattr(top, "prob", float("nan")))}
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="trace only the first N cohort names (smoke test)")
    ap.add_argument("--kinds", nargs="+", default=KINDS,
                    help="residual / mlp / attention")
    ap.add_argument("--window", type=int, default=10,
                    help="sliding restore window: 1 for residual, 10 for mlp/attention")
    ap.add_argument("--skip-singles", action="store_true")
    ap.add_argument("--cohort", default=DEFAULT_COHORT,
                    help="cohort name, used only for the output directory")
    ap.add_argument("--cohort-csv", default=DEFAULT_COHORT_CSV)
    ap.add_argument("--name-col", default="final_name")
    ap.add_argument("--require-any", nargs="*", default=["wikidata_name", "wd_label"],
                    help="keep rows where any of these columns is non-null "
                         "(wikidata9k filter); pass no values to keep all rows")
    ap.add_argument("--out-dir", default=os.path.join(WORK, "results"))
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    ap.add_argument("--subshard", default=None, metavar="J/N",
                    help="further split this shard's names into N disjoint parts "
                         "and run part J -- turns one long job into N short ones "
                         "that the backfill scheduler can slot in easily")
    args = ap.parse_args()

    global OUT_ROOT, PROMPT_TEMPLATE, WINDOW_SIZE
    vspec = VARIANTS[args.variant]
    PROMPT_TEMPLATE = vspec["prompt"]
    WINDOW_SIZE = args.window
    OUT_ROOT = out_root_for(args.cohort, args.variant, args.out_dir)
    log.info("variant=%s  prompt=%r  answer=fixed id %s", args.variant,
             PROMPT_TEMPLATE, vspec["answer_token_id"])
    for k in args.kinds:
        os.makedirs(os.path.join(OUT_ROOT, kind_dir(k)), exist_ok=True)

    # ---- cohort ----
    df, cohort = load_cohort(args.cohort_csv, args.name_col, args.require_any)
    if args.limit:
        cohort = cohort[:args.limit]
    log.info("cohort %s: %d names from %s",
             args.cohort, len(cohort), os.path.basename(args.cohort_csv))

    names = list(cohort)
    if not args.skip_singles:
        names += [n for n in SINGLE_SUBJECTS if n not in cohort]

    shard_names = names[args.shard::args.nshards]
    if args.subshard:
        j, n = (int(x) for x in args.subshard.split("/"))
        shard_names = shard_names[j::n]
        log.info("shard %d/%d subshard %d/%d -> %d names",
                 args.shard, args.nshards, j, n, len(shard_names))
    else:
        log.info("shard %d/%d -> %d names", args.shard, args.nshards, len(shard_names))

    # ---- model + baukit patches ----
    from baukit_patch import install_baukit_patches
    from src.models import ModelandTokenizer
    from src.trace import trace_important_states

    install_baukit_patches()
    mt = ModelandTokenizer(model_key=MODEL_KEY, torch_dtype=torch.bfloat16,
                           device_map="auto")
    log.info("model loaded: %s", MODEL_KEY)

    # ---- labels + answer tokens for this shard ----
    class_df = classify_names(mt, shard_names)
    answers = field_answer_tokens(mt, shard_names)

    # secondary label: the recognition-survey answer, where the cohort has one
    survey_col = "olmo_answer"
    if survey_col in df.columns:
        survey = dict(zip(df["wikidata_name"], df[survey_col]))
        class_df["survey_label"] = class_df["wikidata_name"].map(survey)
    class_df["survey_label_col"] = survey_col if survey_col in df.columns else None
    class_df["in_cohort"] = class_df["wikidata_name"].isin(cohort)
    class_df["field_top1_token"] = class_df["wikidata_name"].map(
        lambda n: answers.get(n, {}).get("token"))
    class_df["field_top1_id"] = class_df["wikidata_name"].map(
        lambda n: answers.get(n, {}).get("token_id"))
    class_df["field_top1_prob"] = class_df["wikidata_name"].map(
        lambda n: answers.get(n, {}).get("prob"))

    # A --limit / --subshard run only classifies a slice of the shard, so it must
    # NOT overwrite the shard's full classification file.
    partial = bool(args.limit) or bool(args.subshard)
    cpath = os.path.join(
        OUT_ROOT,
        f"classification_shard{args.shard}"
        + (f"_partial{args.limit or ''}{(args.subshard or '').replace('/', 'of')}" if partial else "")
        + ".csv")
    class_df.to_csv(cpath, index=False)
    log.info("wrote %s  (%d recognized / %d not)", cpath,
             int(class_df["is_physicist"].sum()),
             int((~class_df["is_physicist"]).sum()))

    # ---- trace ----
    n_done = n_skip = n_fail = 0
    pbar = tqdm(shard_names, desc="Tracing", unit="name", dynamic_ncols=True)
    for name in pbar:
        slug = sanitize(name)

        if already_done(slug, args.kinds):
            n_skip += 1
            continue

        ans_tokens = [vspec["answer_token_id"]]
        for kind in args.kinds:
            save_path = out_path(kind, slug)
            if os.path.isfile(save_path):
                continue
            try:
                result = trace_important_states(
                    mt=mt,
                    prompt_template=PROMPT_TEMPLATE,
                    clean_subj=CLEAN_SUBJ,
                    patched_subj=name,
                    ans_tokens=ans_tokens,
                    kind=kind,
                    metric=METRIC,
                    window_size=WINDOW_SIZE,
                    normalize=False,
                )
                payload = {
                    "result": {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                               for k, v in result.__dict__.items()},
                    "wikidata_name": name,
                    "subject": name,
                    "kind": kind,
                    "metric": METRIC,
                    "prompt_template": PROMPT_TEMPLATE,
                    "clean_subj": CLEAN_SUBJ,
                    "window_size": WINDOW_SIZE,
                    "variant": args.variant,
                    "ans_tokens": ans_tokens,
                    "ans_token": answers.get(name),
                }
                torch.save(payload, save_path)
                n_done += 1
            except Exception as e:
                n_fail += 1
                log.error("FAILED %s [%s]: %s", name, kind, e)
                traceback.print_exc()
                with open(save_path.replace(".pt", ".error"), "w") as f:
                    f.write(f"{type(e).__name__}: {e}\n")

    log.info("shard %d done: %d traces written, %d names skipped, %d failures",
             args.shard, n_done, n_skip, n_fail)


if __name__ == "__main__":
    main()
