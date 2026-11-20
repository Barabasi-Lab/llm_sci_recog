#!/usr/bin/env python3
"""
Step 01 - infini-gram name counts + OLMo-2-13B recognition probe.

Steps (run once per cohort; see README for the exact commands):
  collect   : infini-gram count of the recorded name "First [middle] Last" in the
              OLMo-2 corpus -> `olmo_2_count`. Resumable, batched, with retries.
  recount   : for first-name fields that are `First` + middle INITIAL(s), re-count
              the reduced "First Last" -> `olmo_2_count_cleaned` (full middle names
              are re-counted with the same query).
  recognize : ask OLMo-2-1124-13B-Instruct (4-bit MLX, greedy decoding, 32 new
              tokens, batches of 16) the standard recognition prompt on the FULL
              recorded name -> `olmo2_recog_yn` (YES / NO / UNKNOWN).

Copied from the original script; its plotting step (superseded by make_fig2.py)
and unused columns are removed, and paths are relative to the working directory.

Needs: network access to the public infini-gram API (collect, recount);
       Apple-silicon Mac with mlx / mlx-lm and the OLMo-2 weights (recognize).

I/O (defaults = random cohort; pass --infile/--outfile/--recog-outfile for the
most-cited cohort):
  in   data/recognition/recognition_gpt54_physical_sciences.csv          (cohort file)
  out  data/olmo_training/recognition_gpt54_physical_sciences_infinigram.csv
  out  data/olmo_training/recognition_olmo2_13b_physical_sciences.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (relative to the working directory)
# ----------------------------------------------------------------------------
DATA_IN  = Path("data") / "recognition"
DATA_OUT = Path("data") / "olmo_training"

DEFAULT_INFILE  = str(DATA_IN  / "recognition_gpt54_physical_sciences.csv")
DEFAULT_OUTFILE = str(DATA_OUT / "recognition_gpt54_physical_sciences_infinigram.csv")

INFINI_INDEX = "v4_olmo-2-0325-32b-instruct_llama"
INFINI_URL   = "https://api.infini-gram.io/"

FREQ_COL  = "olmo_2_count"                         # collected by --step collect
CLEAN_COUNT_COL = "olmo_2_count_cleaned"           # re-query (--step recount) for
                                                    # middle-initial names ('John P
                                                    # Smith' -> 'John Smith')

# --- OLMo-2 recognition (local MLX), the WITHIN-MODEL outcome -----------------
# We ask OLMo-2 itself whether it recognizes each scientist, so the recognition
# OUTCOME comes from the same model whose training corpus we counted (FREQ_COL).
OLMO_RECOG_MODEL      = "mlx-community/OLMo-2-1124-13B-Instruct-4bit"
DEFAULT_RECOG_OUTFILE = str(DATA_OUT / "recognition_olmo2_13b_physical_sciences.csv")
OLMO_RECOG_COL   = "olmo2_recog_yn"      # YES / NO / UNKNOWN (OLMo-2-13B)
OLMO_RECOG_FIELD = "olmo2_recog_field"
OLMO_RECOG_CONF  = "olmo2_recog_conf"
OLMO_RECOG_RAW   = "olmo2_recog_raw"

# Exact standardized prompt/system used for the GPT-5.4 labels, so OLMo-2's
# labels are produced apples-to-apples.
SYSTEM_RECOGNITION = (
    "You are a knowledgeable assistant. Answer concisely and honestly. "
    "Do not speculate."
)


def build_name(df):
    """Full name from first + last (matches the strings sent to the LLM)."""
    return (df["author_first_name"].fillna("").astype(str).str.strip() + " " +
            df["author_last_name"].fillna("").astype(str).str.strip()).str.strip()


def _middle_initials_only(first):
    """True iff first-name field is `First` + one-or-more single-letter initials
    (e.g. 'John P', 'Mary A B') and NOT a long middle name (e.g. 'John Paul')."""
    toks = str(first).strip().split()
    if len(toks) < 2:
        return False
    return all(len(t.rstrip(".")) == 1 and t.rstrip(".").isalpha() for t in toks[1:])


def cleaned_first_last(first, last):
    """First token + last name, dropping middle initials ('John P Smith' -> 'John Smith')."""
    toks = str(first).strip().split()
    f = toks[0] if toks else ""
    return (f + " " + str(last).strip()).strip()



def build_prompt_recognition(first, last):
    """Standardized recognition prompt (Last, First name order)."""
    name = f"{last}, {first}"
    return (
        f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
        f"Reply in EXACTLY this format (nothing else):\n"
        f"SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\n"
        f"FIELD: [specific field, or \"Unknown\"]\n"
        f"CONFIDENCE: HIGH/MEDIUM/LOW"
    )


def parse_recognition_response(response_text):
    """Standardized parser -> dict(recognized, field, confidence, raw_response)."""
    import re
    result = {'recognized': None, 'field': None, 'confidence': None,
              'raw_response': response_text}
    if not response_text or str(response_text).startswith("ERROR"):
        return result
    text_upper = str(response_text).upper()
    m = re.search(r'SCIENTIFIC RECOGNITION:\s*(YES|NO|UNKNOWN)', text_upper)
    if m:
        result['recognized'] = m.group(1)
    m = re.search(r'FIELD:\s*(.+)', str(response_text), re.IGNORECASE)
    if m:
        field = m.group(1).strip().strip('"')
        result['field'] = field if field.upper() != 'UNKNOWN' else None
    m = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', text_upper)
    if m:
        result['confidence'] = m.group(1)
    return result


# ----------------------------------------------------------------------------
# Step 1 — resumable infini-gram collection
# ----------------------------------------------------------------------------
def query_infini_count(name, max_retries=4):
    """
    n-gram count of `name` in the OLMo-2 corpus.

    Returns an int count on success (0 is a valid answer = name not in corpus),
    or None on failure so the caller can tell a real failure apart from a 0 and
    retry it later. Retries transient errors (network, timeouts, and API error
    payloads such as rate limits) with exponential backoff + jitter.
    """
    import random
    import requests
    if not isinstance(name, str) or not name:
        return None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                INFINI_URL,
                json={"index": INFINI_INDEX, "query_type": "count", "query": name},
                timeout=30,
            ).json()
            if "count" in r:
                return r["count"]
            # otherwise the API returned an error payload (e.g. rate limit) -> retry
        except Exception:
            pass
        time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))  # backoff + jitter
    return None


def collect_counts(infile, outfile, workers=8, checkpoint_every=500, limit=None,
                   retry_rounds=2, batch_size=100, batch_sleep=120):
    """
    Resumable, failure-aware infini-gram collection. On (re-)run:
      - loads `outfile` if it exists (recovers already-collected counts),
        otherwise loads `infile`;
      - works only within `scope` = the first `limit` rows (or all rows), so
        `--limit 100` then `--limit 10000` is fully resumable;
      - a row is "pending" iff it has a name AND olmo_2_count is still missing.
        A real failure stays missing (None), so it is automatically picked up
        again — both on later retry rounds and on any future re-run;
      - processes pending rows in batches of `batch_size`, checkpointing after
        each batch and sleeping `batch_sleep` seconds between them so the
        rate-limiter resets (set batch_size<=0 to fire everything at once);
      - after the main pass it runs up to `retry_rounds` extra passes over only
        the rows that failed, backing off and using fewer workers each round;
      - never loses collected work on Ctrl-C (saves in a finally block);
      - prints a failure report (collected / failed / no-name) at the end.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    src = outfile
    try:
        df = pd.read_csv(outfile)
        print(f"Resuming from {outfile}")
    except FileNotFoundError:
        df = pd.read_csv(infile)
        src = infile
        print(f"Starting fresh from {infile}")

    df["name"] = build_name(df)
    if FREQ_COL not in df.columns:
        df[FREQ_COL] = pd.NA
    df[FREQ_COL] = pd.to_numeric(df[FREQ_COL], errors="coerce")

    scope = df.index if limit is None else df.index[:limit]   # rows we may touch
    no_name = int((df.loc[scope, "name"].str.len() == 0).sum())

    def pending():
        """Indices in scope with a name but no count yet (untried OR failed)."""
        sub = df.loc[scope]
        return sub.index[sub[FREQ_COL].isna() & sub["name"].str.len().gt(0)].tolist()

    scope_txt = f" (first {limit:,} rows)" if limit is not None else ""
    print(f"Loaded {len(df):,} rows from {src}{scope_txt}. "
          f"Have {int(df.loc[scope, FREQ_COL].notna().sum()):,}, "
          f"to query {len(pending()):,}, no-name (skipped) {no_name:,}.")

    if not pending():
        df.to_csv(outfile, index=False, encoding="utf-8")
        print("Nothing to do.")
        return df

    def run_pass(idx, desc, w):
        step = batch_size if (batch_size and batch_size > 0) else len(idx)
        step = max(1, step)
        n_batches = (len(idx) + step - 1) // step
        for b, start in enumerate(range(0, len(idx), step), 1):
            batch = idx[start:start + step]
            with ThreadPoolExecutor(max_workers=w) as ex:
                futures = {ex.submit(query_infini_count, df.at[i, "name"]): i
                           for i in batch}
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"{desc} {b}/{n_batches}", leave=False):
                    df.at[futures[fut], FREQ_COL] = fut.result()
            df.to_csv(outfile, index=False, encoding="utf-8")        # checkpoint
            ok = int(df.loc[scope, FREQ_COL].notna().sum())
            print(f"  [{desc}] batch {b}/{n_batches}: {ok:,}/{len(scope):,} collected")
            if batch_sleep and batch_sleep > 0 and (start + step) < len(idx):
                time.sleep(batch_sleep)

    try:
        for rnd in range(retry_rounds + 1):
            todo = pending()
            if not todo:
                break
            if rnd == 0:
                run_pass(todo, "infini-gram", workers)
            else:
                w = max(1, workers // 2)               # gentler on retries
                wait = min(60, 10 * rnd)
                print(f"\nRetry round {rnd}/{retry_rounds}: {len(todo):,} still "
                      f"failing — waiting {wait}s, then retrying with {w} workers...")
                time.sleep(wait)
                run_pass(todo, f"retry {rnd}", w)
            df.to_csv(outfile, index=False, encoding="utf-8")
    except KeyboardInterrupt:
        print("\nInterrupted — saving progress so far...")
    finally:
        df[FREQ_COL] = pd.to_numeric(df[FREQ_COL], errors="coerce")
        df.to_csv(outfile, index=False, encoding="utf-8")
        failed = pending()
        ok = int(df.loc[scope, FREQ_COL].notna().sum())
        print(f"\nSaved {outfile}")
        print(f"  collected : {ok:,}/{len(scope):,} rows in scope")
        print(f"  failed    : {len(failed):,} (still missing after {retry_rounds} retry rounds)")
        print(f"  no-name   : {no_name:,} (cannot be queried)")
        if failed:
            print(f"  -> re-run the SAME command to keep retrying the {len(failed):,} failures.")
    return df


# ----------------------------------------------------------------------------
# Step 1a' — re-query counts with CLEANED names, only for middle-initial authors
# ----------------------------------------------------------------------------
def recount_cleaned(countfile, workers=4, checkpoint_every=500,
                    batch_size=100, batch_sleep=120, retry_rounds=2):
    """
    For authors whose first-name is `First` + single middle INITIAL(s)
    (e.g. 'John P Smith'), re-query infini-gram with the cleaned name
    'First Last' ('John Smith') and store the result in a NEW column
    `olmo_2_count_cleaned` — `olmo_2_count` is left untouched. Authors with a
    long middle name or no middle token are skipped (their cleaned column stays
    NaN, so plotting falls back to the old count). Resumable + batched + cooldown,
    exactly like --step collect; operates in place on `countfile`.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    df = pd.read_csv(countfile)
    first = df["author_first_name"].fillna("").astype(str)
    last  = df["author_last_name"].fillna("").astype(str)
    redo = first.map(_middle_initials_only)
    df["_clean_name"] = [cleaned_first_last(a, b) for a, b in zip(first, last)]
    if CLEAN_COUNT_COL not in df.columns:
        df[CLEAN_COUNT_COL] = pd.NA
    df[CLEAN_COUNT_COL] = pd.to_numeric(df[CLEAN_COUNT_COL], errors="coerce")

    def pending():
        return df.index[redo & df[CLEAN_COUNT_COL].isna()
                        & df["_clean_name"].str.len().gt(0)].tolist()

    def save():
        df.drop(columns=["_clean_name"]).to_csv(countfile, index=False, encoding="utf-8")

    n_redo = int(redo.sum())
    print(f"{countfile}\n  middle-initial authors to redo: {n_redo:,}; "
          f"already cleaned {int((redo & df[CLEAN_COUNT_COL].notna()).sum()):,}; "
          f"to query {len(pending()):,}")
    if not pending():
        save(); print("  Nothing to do."); return df

    def run_pass(idx, desc, w):
        step = max(1, batch_size if (batch_size and batch_size > 0) else len(idx))
        nb = (len(idx) + step - 1) // step
        for b, s in enumerate(range(0, len(idx), step), 1):
            chunk = idx[s:s + step]
            with ThreadPoolExecutor(max_workers=w) as ex:
                fut = {ex.submit(query_infini_count, df.at[i, "_clean_name"]): i
                       for i in chunk}
                for f in tqdm(as_completed(fut), total=len(fut),
                              desc=f"{desc} {b}/{nb}", leave=False):
                    df.at[fut[f], CLEAN_COUNT_COL] = f.result()
            save()
            ok = int((redo & df[CLEAN_COUNT_COL].notna()).sum())
            print(f"  [{desc}] batch {b}/{nb}: {ok:,}/{n_redo:,} cleaned")
            if batch_sleep and batch_sleep > 0 and (s + step) < len(idx):
                time.sleep(batch_sleep)

    try:
        for rnd in range(retry_rounds + 1):
            todo = pending()
            if not todo:
                break
            if rnd == 0:
                run_pass(todo, "recount", workers)
            else:
                w = max(1, workers // 2); wait = min(60, 10 * rnd)
                print(f"\nRetry round {rnd}/{retry_rounds}: {len(todo):,} failed — "
                      f"waiting {wait}s, then {w} workers...")
                time.sleep(wait); run_pass(todo, f"retry {rnd}", w)
            save()
    except KeyboardInterrupt:
        print("\nInterrupted — saving progress so far...")
    finally:
        df[CLEAN_COUNT_COL] = pd.to_numeric(df[CLEAN_COUNT_COL], errors="coerce")
        save()
        miss = len(pending())
        print(f"\nSaved {countfile}")
        print(f"  cleaned : {int((redo & df[CLEAN_COUNT_COL].notna()).sum()):,}/{n_redo:,}")
        print(f"  missing : {miss:,} (re-run --step recount to retry)")
    return df


# ----------------------------------------------------------------------------
# Step 1b — resumable OLMo-2 recognition (local MLX, batched)
# ----------------------------------------------------------------------------
def collect_recognition(infile, outfile, model_path=OLMO_RECOG_MODEL,
                        batch_size=16, gen_max_tokens=32, limit=None):
    """
    Ask OLMo-2-13B (local, 4-bit MLX) whether it recognizes each scientist.

    Mirrors the count step: resumable, failure-aware, writes to its OWN file
    (so it never races the infini-gram collector). A row is pending iff it has
    a name AND olmo2_recog_yn is still missing; unparseable/failed generations
    stay missing and are retried on the next run. Checkpoints after every batch.
    """
    import mlx.core as mx
    from mlx_lm import load, batch_generate

    def free_metal():
        """Release MLX's Metal buffer cache so memory doesn't grow per batch."""
        for fn in (getattr(mx, "clear_cache", None),
                   getattr(getattr(mx, "metal", None), "clear_cache", None)):
            if callable(fn):
                try:
                    fn(); return
                except Exception:
                    pass

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    try:
        df = pd.read_csv(outfile)
        print(f"Resuming from {outfile}")
    except FileNotFoundError:
        df = pd.read_csv(infile)
        print(f"Starting fresh from {infile}")

    for col in (OLMO_RECOG_RAW, OLMO_RECOG_COL, OLMO_RECOG_FIELD, OLMO_RECOG_CONF):
        if col not in df.columns:
            df[col] = pd.NA

    scope = df.index if limit is None else df.index[:limit]
    first = df.loc[scope, "author_first_name"].fillna("").astype(str).str.strip()
    last  = df.loc[scope, "author_last_name"].fillna("").astype(str).str.strip()
    has_name = (first.str.len() + last.str.len()) > 0

    def pending():
        sub = df.loc[scope]
        return sub.index[sub[OLMO_RECOG_COL].isna() & has_name].tolist()

    todo = pending()
    scope_txt = f" (first {limit:,} rows)" if limit is not None else ""
    print(f"Loaded {len(df):,} rows{scope_txt}. "
          f"Have {int(df.loc[scope, OLMO_RECOG_COL].notna().sum()):,}, "
          f"to recognize {len(todo):,}.")
    if not todo:
        df.to_csv(outfile, index=False, encoding="utf-8")
        print("Nothing to do.")
        return df

    print(f"Loading {model_path} ...")
    model, tokenizer = load(model_path)

    n_batches = (len(todo) + batch_size - 1) // batch_size
    try:
        for b, start in enumerate(range(0, len(todo), batch_size), 1):
            idx = todo[start:start + batch_size]
            prompts = []
            for i in idx:
                f = df.at[i, "author_first_name"]; l = df.at[i, "author_last_name"]
                f = "" if pd.isna(f) else str(f); l = "" if pd.isna(l) else str(l)
                msgs = [{"role": "system", "content": SYSTEM_RECOGNITION},
                        {"role": "user", "content": build_prompt_recognition(f, l)}]
                prompts.append(tokenizer.apply_chat_template(msgs, add_generation_prompt=True))
            resp = batch_generate(model, tokenizer, prompts=prompts,
                                  max_tokens=gen_max_tokens, verbose=False)
            texts = resp.texts if hasattr(resp, "texts") else resp
            for i, t in zip(idx, texts):
                p = parse_recognition_response(t)
                df.at[i, OLMO_RECOG_RAW]   = p["raw_response"]
                df.at[i, OLMO_RECOG_COL]   = p["recognized"]
                df.at[i, OLMO_RECOG_FIELD] = p["field"]
                df.at[i, OLMO_RECOG_CONF]  = p["confidence"]
            df.to_csv(outfile, index=False, encoding="utf-8")        # checkpoint
            free_metal()                                             # avoid OOM creep
            ok = int(df.loc[scope, OLMO_RECOG_COL].notna().sum())
            print(f"  batch {b}/{n_batches}: {ok:,}/{len(scope):,} labelled")
    except KeyboardInterrupt:
        print("\nInterrupted — saving progress so far...")
    finally:
        df.to_csv(outfile, index=False, encoding="utf-8")
        vc = df.loc[scope, OLMO_RECOG_COL].value_counts(dropna=False).to_dict()
        miss = len(pending())
        print(f"\nSaved {outfile}")
        print(f"  label counts : {vc}")
        print(f"  still missing: {miss:,} (re-run --step recognize to retry)")
    return df

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["collect", "recount", "recognize"], required=True)
    ap.add_argument("--infile", default=DEFAULT_INFILE)
    ap.add_argument("--outfile", default=DEFAULT_OUTFILE)
    ap.add_argument("--recog-outfile", default=DEFAULT_RECOG_OUTFILE,
                    help="where OLMo-2 recognition labels are written")
    ap.add_argument("--model", default=OLMO_RECOG_MODEL,
                    help="MLX model for --step recognize")
    ap.add_argument("--recog-batch-size", type=int, default=16,
                    help="MLX batch_generate batch size for recognition "
                         "(lower if you hit Metal OOM)")
    ap.add_argument("--gen-max-tokens", type=int, default=32,
                    help="max new tokens per recognition answer")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel infini-gram requests (lower if rate-limited)")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=100,
                    help="query this many rows per batch, then sleep "
                         "(<=0 = fire everything at once)")
    ap.add_argument("--batch-sleep", type=float, default=120,
                    help="seconds to sleep between batches (lets the API "
                         "rate-limiter reset)")
    ap.add_argument("--retry-rounds", type=int, default=2,
                    help="extra passes over rows that failed, after the main pass")
    ap.add_argument("--limit", type=int, default=None,
                    help="only collect the first N rows (e.g. 100 for a sample). "
                         "Resumable across runs.")
    args = ap.parse_args()

    if args.step == "collect":
        collect_counts(args.infile, args.outfile, workers=args.workers,
                       checkpoint_every=args.checkpoint_every, limit=args.limit,
                       retry_rounds=args.retry_rounds,
                       batch_size=args.batch_size, batch_sleep=args.batch_sleep)
    if args.step == "recount":
        recount_cleaned(args.outfile, workers=args.workers,
                        checkpoint_every=args.checkpoint_every,
                        batch_size=args.batch_size, batch_sleep=args.batch_sleep,
                        retry_rounds=args.retry_rounds)
    if args.step == "recognize":
        collect_recognition(args.infile, args.recog_outfile, model_path=args.model,
                            batch_size=args.recog_batch_size,
                            gen_max_tokens=args.gen_max_tokens, limit=args.limit)


if __name__ == "__main__":
    main()
