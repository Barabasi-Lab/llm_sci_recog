#!/usr/bin/env python3
"""
Step 4: recognition queries on the 10k region top-up (3 models).

Source: panel1_region20k/code/query_recognition_topup_v2.py (verbatim; paths and
output name changed). This is the run used in the paper. An earlier run
(query_recognition_topup.py, not included) added 'If you don't know, say "No".'
to the system message; v2 restores the paper prompt and the notebook parser, so
both halves of the 20k panel use the same configuration as step 1:

    system      "You are a knowledgeable assistant. Answer concisely and
                 honestly. Do not speculate."
    user        Do you recognize {Last, First} as a scientist? If so, what is
                their field?  + the exact 3-line reply template
    temperature 0.0
    max_tokens  400
    reasoning   {'max_tokens': 128}   for gemini-3.5 only
    no web search / no tools          (parametric memory only)

Differences from the step-2 runner: up to 4 attempts per call; a failed call or a
reply without a parsable SCIENTIFIC RECOGNITION line is not written and is
re-queried on the next run (8 Gemini and 1 GPT replies never parsed).

The run wrote recognition_{model}_topup_v2.csv, then copied to the canonical
recognition_{model}_topup.csv read downstream; this copy writes the canonical name
directly.

Usage
    export OPENROUTER_API_KEY=...
    python 04_query_recognition_topup.py --smoke 30     # ~$0.08, validates setup
    python 04_query_recognition_topup.py                # full run, ~$28
    python 04_query_recognition_topup.py --models gpt54 # one model only

Input   REGION_DIR/physics_topup_10k.csv (region-balanced 10k top-up sample)
Output  REGION_DIR/recognition_{model}_topup.csv
        -> data/physicists_20k.csv model columns (top-up half)
"""
import argparse
import os
import re
import sys
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

from paths import REGION_DIR

OUT_DIR = REGION_DIR
TOPUP   = os.path.join(OUT_DIR, "physics_topup_10k.csv")

MODELS = [                      # (openrouter id, short name)
    ("anthropic/claude-sonnet-4.6", "claude46"),
    ("google/gemini-3.5-flash",     "gemini35"),
    ("openai/gpt-5.4",              "gpt54"),
]

# $/token, OpenRouter list prices — used only for the cost estimate print-out
PRICE = {"claude46": (3.00e-6, 15.00e-6),
         "gemini35": (1.50e-6,  9.00e-6),
         "gpt54":    (2.50e-6, 15.00e-6)}
TOK_IN, TOK_OUT = 91.1, {"claude46": 20.5, "gemini35": 18.8 + 128, "gpt54": 19.3}

N_WORKERS  = 10
CHECKPOINT = 500

SYSTEM_RECOGNITION = (          # verbatim from the original run (notebook cell 16)
    "You are a knowledgeable assistant. Answer concisely and honestly. "
    "Do not speculate."
)


def build_prompt(row):
    name = f"{row['author_last_name']}, {row['author_first_name']}"
    return (
        f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
        f"Reply in EXACTLY this format (nothing else):\n"
        f"SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\n"
        f'FIELD: [specific field, or "Unknown"]\n'
        f"CONFIDENCE: HIGH/MEDIUM/LOW"
    )


def parse_response(text):
    """The notebook's parse_recognition_response, verbatim. Returns None if the
    reply has no recognition decision, so the row is re-queried, never faked."""
    if not text:
        return None
    out = {"raw_response": text, "recognized": None, "field": None,
           "confidence": None}
    up = text.upper()
    m = re.search(r"SCIENTIFIC RECOGNITION:\s*(YES|NO|UNKNOWN)", up)
    if m:
        out["recognized"] = m.group(1)
    m = re.search(r"FIELD:\s*(.+)", text, re.IGNORECASE)
    if m:
        f = m.group(1).strip().strip('"')
        out["field"] = f if f.upper() != "UNKNOWN" else None
    m = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", up)
    if m:
        out["confidence"] = m.group(1)
    return out if out["recognized"] else None


def call_llm(client, prompt, model, max_retries=4):
    extra = {}
    if "gemini-3.5" in model:
        extra["reasoning"] = {"max_tokens": 128}     # identical to original run
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_RECOGNITION},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
                extra_body=extra,
            )
            return r.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    give up: {e}", flush=True)
                return None          # -> row not written, retried next run
            time.sleep(2 * (attempt + 1))
    return None


def run_model(client, df, model_id, short, out_path):
    cols = {c: f"{short}_recog_{c}" for c in ["raw", "yn", "field", "conf"]}
    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        have = set(done["author_id"].astype(str))
        print(f"[{short}] resuming: {len(done):,} done")
    else:
        done, have = pd.DataFrame(), set()
    todo = df[~df["author_id"].astype(str).isin(have)]
    if todo.empty:
        print(f"[{short}] already complete")
        return done
    print(f"[{short}] {len(todo):,} to query")

    rows = []

    def work(i):
        row = todo.loc[i]
        return i, parse_response(call_llm(client, build_prompt(row), model_id))

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(work, i): i for i in todo.index}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=short):
            i, parsed = fut.result()
            if parsed is None:
                continue                       # never fabricate a row
            src = todo.loc[i]
            rows.append({**src.to_dict(),
                         cols["raw"]: parsed["raw_response"],
                         cols["yn"]: parsed["recognized"],
                         cols["field"]: parsed["field"],
                         cols["conf"]: parsed["confidence"]})
            if len(rows) >= CHECKPOINT:
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(out_path, index=False)
                rows = []
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_csv(out_path, index=False)
    miss = len(df) - len(done)
    print(f"[{short}] saved {len(done):,}"
          + (f"  ({miss:,} still missing — re-run to retry)" if miss else ""))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="query only N authors (validation run)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of short names, e.g. --models gpt54")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set.\n"
                 "  export OPENROUTER_API_KEY=...   (do not hardcode it)")

    models = [m for m in MODELS if args.models is None or m[1] in args.models]
    df = pd.read_csv(TOPUP)
    if args.smoke:
        df = df.head(args.smoke)
    df = df.reset_index(drop=True)

    est = sum(len(df) * (TOK_IN * PRICE[s][0] + TOK_OUT[s] * PRICE[s][1])
              for _, s in models)
    print(f"{len(df):,} authors x {len(models)} models "
          f"= {len(df)*len(models):,} calls   estimated ${est:.2f}\n")

    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    os.makedirs(OUT_DIR, exist_ok=True)
    for model_id, short in models:
        suffix = "_smoke" if args.smoke else ""
        out = os.path.join(OUT_DIR, f"recognition_{short}_topup{suffix}.csv")
        print(f"\n>>> {model_id}")
        d = run_model(client, df, model_id, short, out)
        yn = d[f"{short}_recog_yn"].astype(str).str.upper()
        n = int((yn == "YES").sum())
        print(f"[{short}] YES {n:,}/{len(d):,} = {100*n/max(len(d),1):.2f}%")


if __name__ == "__main__":
    main()
