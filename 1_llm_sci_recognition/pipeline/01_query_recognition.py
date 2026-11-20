#!/usr/bin/env python3
"""Step 1: closed-book recognition queries, 3 models x 4 field samples (10k each).

Source: query_recognition_dimensions_multifields.ipynb, cells 16 (prompt, API call,
parser, resumable runner), 18 (per-field wrapper), 19 (model loop) and 23 (answer
counts). Logic verbatim; the API key is read from the environment.

Prompt: the paper prompt (SI Fig. "Standardized prompt"), with NO "say No" clause:
    system  "You are a knowledgeable assistant. Answer concisely and honestly.
             Do not speculate."
    user    Do you recognize {Last}, {First} as a scientist? If so, what is their
            field? + the 3-line reply template
Parameters: OpenRouter chat completions, temperature 0.0, max_tokens 400, no tools /
web search; for gemini-3.5 reasoning={'max_tokens': 128}. Up to 3 attempts per
call. A call that still fails is stored as "ERROR: ..." and re-queried on the next
run; a reply without a parsable SCIENTIFIC RECOGNITION line is kept as missing
(not re-queried).

Usage
    export OPENROUTER_API_KEY=...
    python 01_query_recognition.py              # query (resumable), then summarise
    python 01_query_recognition.py --summary    # counts only, no API calls

Input   SAMPLE_DIR/sample_{field_slug}.csv                    (stratified 10k sample per field)
Output  MULTI_DIR/recognition_{model}_{field_slug}.csv         raw + parsed answers
        MULTI_DIR/recognition_summary_by_field.csv             YES/NO/UNKNOWN counts
            -> data/other_fields_counts.csv (non-physics rows)
            -> data/physicists_20k.csv model columns (physics, original 10k half)
"""
import argparse
import glob
import os
import re
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from paths import MULTI_DIR, SAMPLE_DIR

# === CONFIG ===
OUTPUT_DIR = MULTI_DIR
N_WORKERS = 10
CHECKPOINT_EVERY = 1000
MODELS = [                                    # (OpenRouter id, short name)
    ('anthropic/claude-sonnet-4.6', 'claude46'),
    ('google/gemini-3.5-flash',     'gemini35'),
    ('openai/gpt-5.4',              'gpt54'),
]

# === SYSTEM PROMPT ===
SYSTEM_RECOGNITION = (
    "You are a knowledgeable assistant. Answer concisely and honestly. "
    "Do not speculate."
)

client = None   # OpenAI client pointed at OpenRouter, created in main()


# === PROMPT BUILDER ===
def build_prompt_recognition(row):
    name = f"{row['author_last_name']}, {row['author_first_name']}"
    return (
        f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
        f"Reply in EXACTLY this format (nothing else):\n"
        f"SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\n"
        f"FIELD: [specific field, or \"Unknown\"]\n"
        f"CONFIDENCE: HIGH/MEDIUM/LOW"
    )


# === API CALLER ===
def call_llm(prompt, model, system=SYSTEM_RECOGNITION,
             max_retries=3, sleep_time=1):
    extra_body = {}
    if 'gemini-3.5' in model:
        # Reasoning must be enabled for this model; cap the budget.
        # 128 tokens is enough for a YES/NO/field decision.
        extra_body['reasoning'] = {'max_tokens': 128}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,            # 128 reasoning + ~50 answer + headroom
                extra_body=extra_body,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{model}] attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(sleep_time * (attempt + 1))
            else:
                return f"ERROR: {str(e)}"


# === RESPONSE PARSER ===
def parse_recognition_response(response_text: str) -> dict:
    result = {'recognized': None, 'field': None, 'confidence': None,
              'raw_response': response_text}
    if not response_text or response_text.startswith("ERROR"):
        return result

    text_upper = response_text.upper()

    match = re.search(r'SCIENTIFIC RECOGNITION:\s*(YES|NO|UNKNOWN)', text_upper)
    if match:
        result['recognized'] = match.group(1)

    match = re.search(r'FIELD:\s*(.+)', response_text, re.IGNORECASE)
    if match:
        field = match.group(1).strip().strip('"')
        result['field'] = field if field.upper() != 'UNKNOWN' else None

    match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', text_upper)
    if match:
        result['confidence'] = match.group(1)

    return result


# === SINGLE ROW WORKER ===
def process_row(idx, row, model):
    prompt = build_prompt_recognition(row)
    resp = call_llm(prompt, model=model)
    parsed = parse_recognition_response(resp)
    return idx, parsed


# === PARALLEL BATCH RUNNER (model-specific, robust resume) ===
def run_recognition_for_model(sample, model_id, short_name,
                              output_dir=OUTPUT_DIR,
                              n_workers=N_WORKERS,
                              checkpoint_every=CHECKPOINT_EVERY,
                              retry_errors=True,
                              force=False):
    """
    Run LLM recognition for a single model, resumable.

    - If output CSV exists, loads it and only processes rows where
      {short}_recog_raw is null OR starts with 'ERROR:' (if retry_errors).
    - If force=True, deletes the existing file and starts fresh.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'recognition_{short_name}.csv')

    col_map = {
        'raw_response': f'{short_name}_recog_raw',
        'recognized':   f'{short_name}_recog_yn',
        'field':        f'{short_name}_recog_field',
        'confidence':   f'{short_name}_recog_conf',
    }
    raw_col = col_map['raw_response']
    cols_needed = list(col_map.values())

    # Force re-run: wipe the file
    if force and os.path.exists(output_path):
        os.remove(output_path)
        print(f"[{short_name}] force=True — removed existing file")

    # Load or initialize
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        for col in cols_needed:
            if col not in df.columns:
                df[col] = None
    else:
        df = sample.copy().reset_index(drop=True)
        for col in cols_needed:
            df[col] = None

    # Identify rows still needing work
    raw = df[raw_col]
    if retry_errors:
        todo_mask = raw.isna() | raw.astype(str).str.startswith('ERROR')
    else:
        todo_mask = raw.isna()

    todo_indices = df.index[todo_mask].tolist()
    n_done = len(df) - len(todo_indices)

    if not todo_indices:
        print(f"[{short_name}] All {len(df):,} rows already processed. Skipping.")
        return df

    print(f"[{short_name}] {n_done:,}/{len(df):,} already done, "
          f"{len(todo_indices):,} to go. Using {model_id}.")

    # Process in checkpointed batches (over TODO rows, not all rows)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for batch_start in range(0, len(todo_indices), checkpoint_every):
            batch = todo_indices[batch_start:batch_start + checkpoint_every]
            futures = {
                executor.submit(process_row, i, df.iloc[i], model_id): i
                for i in batch
            }
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"[{short_name}] batch "
                                    f"{batch_start//checkpoint_every + 1}"):
                idx, parsed = future.result()
                for key, col in col_map.items():
                    df.at[idx, col] = parsed[key]

            df.to_csv(output_path, index=False)
            processed_so_far = batch_start + len(batch)
            print(f"[{short_name}] Checkpoint: {processed_so_far:,}/"
                  f"{len(todo_indices):,} TODO rows done")

    print(f"[{short_name}] Done. Saved to {output_path}")
    return df


# === FIELD-AWARE WRAPPER ===
def _field_slug(field_name):
    s = field_name.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s


def run_all_fields_for_model(samples_by_field, model_id, short_name,
                             output_dir=OUTPUT_DIR,
                             n_workers=N_WORKERS,
                             checkpoint_every=CHECKPOINT_EVERY,
                             retry_errors=True,
                             force=False):
    """
    Run one model across all field samples.
    Writes to recognition_{short_name}_{field_slug}.csv so files don't collide;
    columns become e.g. claude46_physical_sciences_recog_yn.
    """
    out = {}
    for field, sample in samples_by_field.items():
        if len(sample) == 0:
            print(f"[{short_name} / {field}] empty sample, skipping")
            continue
        slug = _field_slug(field)
        scoped_short = f'{short_name}_{slug}'
        print(f"\n>>> {short_name} on field='{field}' "
              f"(short={scoped_short}, n={len(sample):,})")
        out[field] = run_recognition_for_model(
            sample,
            model_id=model_id,
            short_name=scoped_short,
            output_dir=output_dir,
            n_workers=n_workers,
            checkpoint_every=checkpoint_every,
            retry_errors=retry_errors,
            force=force,
        )
    return out


# === ANSWER COUNTS PER (MODEL x FIELD)  (notebook cell 23) ===
def _summarize(df, yn_col, model, field):
    """Compute YES/NO/UNKNOWN/MISSING counts for one (model, field) slice."""
    vals = df[yn_col].fillna('MISSING').astype(str).str.upper()
    counts = vals.value_counts()

    n_yes = int(counts.get('YES', 0))
    n_no = int(counts.get('NO', 0))
    n_unknown = int(counts.get('UNKNOWN', 0))
    n_missing = len(df) - (n_yes + n_no + n_unknown)
    denom = n_yes + n_no + n_unknown

    return {
        'model':     model,
        'field':     field,
        'n_total':   len(df),
        'n_yes':     n_yes,
        'n_no':      n_no,
        'n_unknown': n_unknown,
        'n_missing': n_missing,
        'pct_yes':   100 * n_yes / denom if denom > 0 else float('nan'),
        'pct_no':    100 * n_no / denom if denom > 0 else float('nan'),
        'pct_unk':   100 * n_unknown / denom if denom > 0 else float('nan'),
    }


def _find_yn_col(df, short_name, slug):
    expected = f'{short_name}_{slug}_recog_yn'
    if expected in df.columns:
        return expected
    candidates = [c for c in df.columns if c.endswith('_recog_yn')]
    return candidates[0] if len(candidates) == 1 else None


def summarize_all():
    summary_rows = []
    for model_id, short_name in MODELS:
        pattern = os.path.join(OUTPUT_DIR, f'recognition_{short_name}_*.csv')
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[{short_name}] no per-field files found at {pattern}")
            continue

        per_field_dfs = []
        for path in files:
            slug = re.sub(rf'^recognition_{re.escape(short_name)}_(.+)\.csv$',
                          r'\1', os.path.basename(path))
            df = pd.read_csv(path)
            yn_col = _find_yn_col(df, short_name, slug)
            if yn_col is None:
                print(f"[{short_name} / {slug}] no *_recog_yn column found")
                continue
            if 'modal_field_name' in df.columns:
                mode_vals = df['modal_field_name'].dropna().mode()
                field_name = mode_vals.iloc[0] if len(mode_vals) else slug
            else:
                field_name = slug
            df = df.rename(columns={yn_col: '_recog_yn_unified'})
            summary_rows.append(_summarize(df, '_recog_yn_unified',
                                           short_name, field_name))
            per_field_dfs.append(df[['_recog_yn_unified']])

        if per_field_dfs:
            all_df = pd.concat(per_field_dfs, ignore_index=True)
            summary_rows.append(_summarize(all_df, '_recog_yn_unified',
                                           short_name, '__ALL__'))

    summary = pd.DataFrame(summary_rows)
    summary['_is_total'] = (summary['field'] == '__ALL__').astype(int)
    summary = (summary
               .sort_values(['model', '_is_total', 'field'])
               .drop(columns='_is_total')
               .reset_index(drop=True))
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'recognition_summary_by_field.csv'),
                   index=False)


def load_samples():
    out = {}
    for path in sorted(glob.glob(os.path.join(SAMPLE_DIR, 'sample_*.csv'))):
        s = pd.read_csv(path)
        out[s['modal_field_name'].dropna().iloc[0]] = s
    return out


def main():
    global client
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true",
                    help="only count answers in existing output files")
    a = ap.parse_args()
    if not a.summary:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1")
        samples_by_field = load_samples()
        for model_id, short_name in MODELS:
            run_all_fields_for_model(samples_by_field, model_id=model_id,
                                     short_name=short_name)
    summarize_all()


if __name__ == "__main__":
    main()
