#!/usr/bin/env python3
"""Step 3: false-positive probe, 3 models x 1,000 fabricated names.

Source: query_recognition_dimensions_multifields.ipynb, cell 24, second half
("RUN PROBE"). Prompt, parameters and parser are those of step 1 (paper prompt, no
"say No" clause; temperature 0.0; max_tokens 400; gemini-3.5 reasoning capped at
128 tokens; 3 attempts). Unlike step 1 there is no resume/retry pass: a call that
fails three times stays missing (Gemini: 991 of 1,000 parsed).

The YES count over parsed answers is the model's false-positive rate f_m used by
the signal-detection correction in make_fig1.py.

Usage
    export OPENROUTER_API_KEY=...
    python 03_fake_name_probe.py              # query, then count
    python 03_fake_name_probe.py --counts     # count existing outputs only

Input   inputs/fake_names_physics.csv  (= step 2 output without _source_last)
Output  MULTI_DIR/fake_probe_{model}.csv  -> data/fake_name_probe_counts.csv
"""
import argparse
import os
import re
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from paths import INPUTS, MULTI_DIR

N_WORKERS = 10
PROBE_DIR = MULTI_DIR
FAKE_NAMES = os.path.join(INPUTS, "fake_names_physics.csv")

MODELS = [
    ('openai/gpt-5.4',              'gpt54'),
    ('anthropic/claude-sonnet-4.6', 'claude46'),
    ('google/gemini-3.5-flash',     'gemini35'),
]

client = None   # created in main()

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer concisely and honestly. "
    "Do not speculate."
)


def build_prompt(row):
    name = f"{row['author_last_name']}, {row['author_first_name']}"
    return (
        f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
        f"Reply in EXACTLY this format (nothing else):\n"
        f"SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\n"
        f"FIELD: [specific field, or \"Unknown\"]\n"
        f"CONFIDENCE: HIGH/MEDIUM/LOW"
    )


def call_llm(prompt, model, max_retries=3, sleep_time=1):
    extra_body = {}
    if 'gemini-3.5' in model:
        extra_body['reasoning'] = {'max_tokens': 128}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
                extra_body=extra_body,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[{model}] attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(sleep_time * (attempt + 1))
            else:
                return f"ERROR: {e}"


def parse_response(text):
    result = {'recognized': None, 'field': None, 'confidence': None,
              'raw_response': text}
    if not text or text.startswith('ERROR'):
        return result
    upper = text.upper()
    m = re.search(r'SCIENTIFIC RECOGNITION:\s*(YES|NO|UNKNOWN)', upper)
    if m:
        result['recognized'] = m.group(1)
    m = re.search(r'FIELD:\s*(.+)', text, re.IGNORECASE)
    if m:
        f = m.group(1).strip().strip('"')
        result['field'] = f if f.upper() != 'UNKNOWN' else None
    m = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', upper)
    if m:
        result['confidence'] = m.group(1)
    return result


def process_row(idx, row, model):
    prompt = build_prompt(row)
    resp = call_llm(prompt, model)
    parsed = parse_response(resp)
    return idx, parsed


def run_fake_probe(fake_df, model_id, short_name, n_workers=N_WORKERS):
    col_map = {
        'raw_response': f'{short_name}_recog_raw',
        'recognized':   f'{short_name}_recog_yn',
        'field':        f'{short_name}_recog_field',
        'confidence':   f'{short_name}_recog_conf',
    }
    df = fake_df.copy().reset_index(drop=True)
    for c in col_map.values():
        df[c] = None
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process_row, i, df.iloc[i], model_id): i
                   for i in df.index}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f'[{short_name}] fake-probe'):
            idx, parsed = fut.result()
            for k, c in col_map.items():
                df.at[idx, c] = parsed[k]
    return df


def get_fpr(probe_df, short):
    """(YES count, parsed count); same rule as the timeline notebooks."""
    yn = probe_df[f'{short}_recog_yn'].astype(str).str.upper()
    k_yes = int((yn == 'YES').sum())
    n_parsed = int(yn.isin(['YES', 'NO', 'UNKNOWN']).sum())
    return k_yes, n_parsed


def main():
    global client
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true", help="count existing outputs only")
    a = ap.parse_args()
    os.makedirs(PROBE_DIR, exist_ok=True)

    if not a.counts:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1")
        fake_df = pd.read_csv(FAKE_NAMES)
        for model_id, short in MODELS:
            print(f"\n>>> Probing {model_id}")
            res = run_fake_probe(fake_df, model_id, short)
            res.to_csv(os.path.join(PROBE_DIR, f'fake_probe_{short}.csv'), index=False)

    print(f"\n{'model':10s} {'fake_yes':>8s} {'fake_n':>7s}")
    for _, short in MODELS:
        k, n = get_fpr(pd.read_csv(os.path.join(PROBE_DIR, f'fake_probe_{short}.csv')), short)
        print(f"{short:10s} {k:8d} {n:7d}  FPR {100*k/n:.2f}%")


if __name__ == "__main__":
    main()
