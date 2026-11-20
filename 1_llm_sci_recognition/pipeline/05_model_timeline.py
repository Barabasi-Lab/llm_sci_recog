#!/usr/bin/env python3
"""Step 5: model timeline, 14 releases x 2,000 physicists + 1,000 fabricated names.

Source: query_recognition_timeline_models.ipynb (GPT), _claude.ipynb and
_gemini.ipynb, cells 3 (models), 5-6 (2k subset), 8 (reuse of 10k answers),
10 (API), 12 (queries), 14 (answer counts), 19 (fake names), 20 (fake probe) and
the counting half of 22. The three notebooks share this code; their differences
are kept as per-family settings in FAMILIES below. The per-family SDT cells (22-23)
and code/unify_timeline_ref.py are not included: make_fig1.py recomputes the
correction for all 14 releases on one threshold (Sonnet 4.6 on this name set).

Prompt: the paper prompt, no "say No" clause (as step 1). Temperature 0.0.
    gpt, claude   max_tokens 150, no reasoning parameter
    gemini        max_tokens 450, reasoning {'max_tokens': 128}
GPT-5.4, Claude Sonnet 4.6 and Gemini 3.5 Flash are not re-queried on the subset:
their answers are taken from the 10k run of step 1 (max_tokens 400). Their
fabricated-name probes are run here with the family settings above.

Subset: proportional stratified draw (Gender x career_stage, RANDOM_STATE 42) from
the original 10k physicists; identical for all three families.
Fake names: same perturbation engine as step 2 (RANDOM_STATE 2024), sourced from
the 2k subset, checked against the full panel; identical for all three families
(not shipped: each name is a one-letter edit of a real surname).

Usage
    export OPENROUTER_API_KEY=...
    python 05_model_timeline.py --family gpt              # all steps
    python 05_model_timeline.py --family claude --step summary
    steps: subset, extract, query, fake, probe, summary

Input   MULTI_DIR/recognition_{gpt54,claude46,gemini35}_physical_sciences.csv (step 1)
        PANEL_CSV (only for the fake-name step)
Output  TIMELINE_DIRS[family]/timeline_subset_ids.csv, recognition_{short}.csv,
        fake_names_timeline.csv, fake_probe_{short}.csv, timeline_summary.csv
        -> data/timeline_counts.csv: n_yes, n_no, n_unknown, n_error from
           timeline_summary.csv; fake_yes, fake_n from the probe counts printed
           by the summary step
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from paths import INPUTS, MULTI_DIR, PANEL_CSV, TIMELINE_DIRS

# ============================================================
# PER-FAMILY SETTINGS (notebook cell 3 and cell 10 of each notebook)
# (openrouter_id, short, display_label, release_date)
# ============================================================
_GEMINI_THINKING = {'gemini-2.5', 'gemini-3-flash', 'gemini-3.5'}

FAMILIES = {
    'gpt': dict(
        models=[
            ('openai/gpt-4o-2024-05-13', 'gpt4o',     'GPT-4o (05-13)', '2024-05-13'),
            ('openai/gpt-4o-2024-08-06', 'gpt4o0806', 'GPT-4o (08-06)', '2024-08-06'),
            ('openai/gpt-4o-2024-11-20', 'gpt4o1120', 'GPT-4o (11-20)', '2024-11-20'),
            ('openai/gpt-4.1',           'gpt41',     'GPT-4.1',        '2025-04-14'),
            ('openai/gpt-5.1',           'gpt51',     'GPT-5.1',        '2025-11-13'),
            ('openai/gpt-5.2',           'gpt52',     'GPT-5.2',        '2025-12-10'),
            ('openai/gpt-5.4',           'gpt54',     'GPT-5.4',        '2026-03-05'),
        ],
        # The notebook listed {'gpt4o', 'gpt54'}, but the saved GPT-4o (05-13) file
        # has the subset schema: it was queried on the subset with these settings.
        existing={'gpt54': 'gpt54'},
        max_tokens=150,
        thinking=set(),
    ),
    'claude': dict(
        models=[
            ('anthropic/claude-3-haiku',    'claude3h',  'Haiku 3',    '2024-03-14'),
            ('anthropic/claude-3.5-haiku',  'claude35h', 'Haiku 3.5',  '2024-10-22'),
            ('anthropic/claude-sonnet-4',   'claude4s',  'Sonnet 4',   '2025-05-22'),
            ('anthropic/claude-sonnet-4.5', 'claude45s', 'Sonnet 4.5', '2025-09-29'),
            ('anthropic/claude-sonnet-4.6', 'claude46s', 'Sonnet 4.6', '2026-02-17'),
        ],
        existing={'claude46s': 'claude46'},      # our short -> step-2 short
        max_tokens=150,
        thinking=set(),
    ),
    'gemini': dict(
        models=[
            # gemini-2.0-flash-lite and gemini-3-flash-preview were tried in the
            # notebook and are not part of the paper.
            ('google/gemini-2.5-flash', 'gemini25f', 'Flash 2.5', '2025-04-17'),
            ('google/gemini-3.5-flash', 'gemini35f', 'Flash 3.5', '2026-05-19'),
        ],
        existing={'gemini35f': 'gemini35'},
        max_tokens=450,
        thinking=_GEMINI_THINKING,
    ),
}

N_SAMPLE = 2000          # physicists per model
RANDOM_STATE = 42        # same seed in all three notebooks -> same 2k subset
N_WORKERS = 10
CHECKPOINT_EVERY = 500
N_FAKE = 1000
RANDOM_STATE_FAKE = 2024

FAM = None               # selected family settings
OUTPUT_DIR = None
client = None


# ============================================================
# 2k subset (cells 5-6)
# ============================================================
KEEP_COLS = [
    'author_id', 'author_first_name', 'author_last_name',
    'Gender', 'career_stage', 'modal_field_name',
    'cum_publication_count', 'cum_citations',
    'first_affiliation_countries', 'first_affiliation_regions',
    'first_affiliation_subregions',
]


def make_subset():
    _candidates = [
        os.path.join(MULTI_DIR, 'recognition_gpt54_physical_sciences.csv'),
        os.path.join(MULTI_DIR, 'recognition_gpt4o_physical_sciences.csv'),
    ]
    base_df = None
    for p in _candidates:
        if os.path.exists(p):
            base_df = pd.read_csv(p)
            print(f"Loaded base sample from {p}  ({len(base_df):,} rows)")
            break
    if base_df is None:
        raise FileNotFoundError(f"Tried: {_candidates}. Run step 1 first.")

    keep = [c for c in KEEP_COLS if c in base_df.columns]
    full_sample = base_df[keep].copy()

    # Stratified sub-sample of 2,000; strata Gender x career_stage
    strata_cols = ['Gender', 'career_stage']
    eligible = full_sample.dropna(subset=strata_cols).copy()

    rng = np.random.RandomState(RANDOM_STATE)

    groups = eligible.groupby(strata_cols)
    target = N_SAMPLE

    sampled_parts = []
    for name, grp in groups:
        frac = len(grp) / len(eligible)
        n_draw = max(1, int(round(frac * target)))
        n_draw = min(n_draw, len(grp))
        sampled_parts.append(grp.sample(n=n_draw, random_state=rng))

    subset = pd.concat(sampled_parts, ignore_index=True)

    # Trim or pad to exactly N_SAMPLE
    if len(subset) > N_SAMPLE:
        subset = subset.sample(n=N_SAMPLE, random_state=rng).reset_index(drop=True)
    elif len(subset) < N_SAMPLE:
        remaining = eligible[~eligible['author_id'].isin(subset['author_id'])]
        extra = remaining.sample(n=N_SAMPLE - len(subset), random_state=rng)
        subset = pd.concat([subset, extra], ignore_index=True)

    subset = subset.reset_index(drop=True)
    print(f"Timeline subset: {len(subset):,} physicists")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subset[['author_id']].to_csv(os.path.join(OUTPUT_DIR, 'timeline_subset_ids.csv'),
                                 index=False)
    return subset


# ============================================================
# Reuse 10k answers for the latest model of each family (cell 8)
# ============================================================
def extract_existing(subset):
    subset_author_ids = set(subset['author_id'].astype(str))
    for model_id, short, label, release in FAM['models']:
        if short not in FAM['existing']:
            continue
        out_path = os.path.join(OUTPUT_DIR, f'recognition_{short}.csv')
        if os.path.exists(out_path):
            print(f"[{short}] Already exists ({out_path}), skipping extraction.")
            continue
        src_short = FAM['existing'][short]
        src = os.path.join(MULTI_DIR, f'recognition_{src_short}_physical_sciences.csv')
        src_df = pd.read_csv(src)
        print(f"[{short}] Loaded source: {src} ({len(src_df):,} rows)")

        src_df['author_id_str'] = src_df['author_id'].astype(str)
        filtered = src_df[src_df['author_id_str'].isin(subset_author_ids)].copy()
        filtered = filtered.drop(columns=['author_id_str'])

        # {src_short}_physical_sciences_recog_* -> {short}_recog_*
        rename_map = {}
        for col in filtered.columns:
            if '_recog_' in col:
                suffix = col.split('_recog_')[-1]
                new_name = f'{short}_recog_{suffix}'
                if new_name != col:
                    rename_map[col] = new_name
        if rename_map:
            filtered = filtered.rename(columns=rename_map)

        filtered.to_csv(out_path, index=False)
        print(f"[{short}] Saved {len(filtered):,} rows to {out_path}")


# ============================================================
# API functions (cell 10; identical prompt to step 1)
# ============================================================
SYSTEM_RECOGNITION = (
    "You are a knowledgeable assistant. Answer concisely and honestly. "
    "Do not speculate."
)


def build_prompt_recognition(row):
    name = f"{row['author_last_name']}, {row['author_first_name']}"
    return (
        f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
        f"Reply in EXACTLY this format (nothing else):\n"
        f"SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\n"
        f"FIELD: [specific field, or \"Unknown\"]\n"
        f"CONFIDENCE: HIGH/MEDIUM/LOW"
    )


def call_llm(prompt, model, system=SYSTEM_RECOGNITION,
             max_retries=3, sleep_time=1):
    extra_body = {}
    if any(t in model for t in FAM['thinking']):      # gemini notebook only
        extra_body['reasoning'] = {'max_tokens': 128}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=FAM['max_tokens'],
                extra_body=extra_body if extra_body else {},
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{model}] attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(sleep_time * (attempt + 1))
            else:
                return f"ERROR: {str(e)}"


def parse_recognition_response(response_text: str) -> dict:
    result = {'recognized': None, 'field': None, 'confidence': None,
              'raw_response': response_text}
    if not response_text or response_text.startswith('ERROR'):
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


def process_row(idx, row, model):
    prompt = build_prompt_recognition(row)
    resp = call_llm(prompt, model=model)
    parsed = parse_recognition_response(resp)
    return idx, parsed


def run_recognition_for_model(sample, model_id, short_name,
                              n_workers=N_WORKERS,
                              checkpoint_every=CHECKPOINT_EVERY,
                              retry_errors=True,
                              force=False):
    """Resumable: re-queries rows whose raw answer is missing or 'ERROR: ...'."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'recognition_{short_name}.csv')

    col_map = {
        'raw_response': f'{short_name}_recog_raw',
        'recognized':   f'{short_name}_recog_yn',
        'field':        f'{short_name}_recog_field',
        'confidence':   f'{short_name}_recog_conf',
    }
    raw_col = col_map['raw_response']
    cols_needed = list(col_map.values())

    if force and os.path.exists(output_path):
        os.remove(output_path)

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        for col in cols_needed:
            if col not in df.columns:
                df[col] = None
    else:
        df = sample.copy().reset_index(drop=True)
        for col in cols_needed:
            df[col] = None

    raw = df[raw_col]
    if retry_errors:
        todo_mask = raw.isna() | raw.astype(str).str.startswith('ERROR')
    else:
        todo_mask = raw.isna()

    todo_indices = df.index[todo_mask].tolist()
    if not todo_indices:
        print(f"[{short_name}] All {len(df):,} rows already processed. Skipping.")
        return df

    print(f"[{short_name}] {len(df) - len(todo_indices):,}/{len(df):,} done, "
          f"{len(todo_indices):,} to go. Model: {model_id}")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for batch_start in range(0, len(todo_indices), checkpoint_every):
            batch = todo_indices[batch_start:batch_start + checkpoint_every]
            futures = {
                executor.submit(process_row, i, df.iloc[i], model_id): i
                for i in batch
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"[{short_name}]"):
                idx, parsed = future.result()
                for key, col in col_map.items():
                    df.at[idx, col] = parsed[key]
            df.to_csv(output_path, index=False)

    print(f"[{short_name}] Done. Saved to {output_path}")
    return df


def query_models(subset):
    for model_id, short, label, release in FAM['models']:
        if short in FAM['existing']:
            continue                     # taken from the 10k run (extract step)
        print(f"\n>>> {model_id}  (short={short})")
        run_recognition_for_model(subset, model_id=model_id, short_name=short)


# ============================================================
# Fake names from the 2k subset (cell 19; engine identical to step 2)
# ============================================================
VOWELS = set('aeiouAEIOU')


def _is_latin_name(name):
    s = str(name).strip()
    if len(s) < 3:
        return False
    return bool(re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ' -]+$", s))


def _perturb_part(s, rng):
    chars = list(s)
    if len(chars) < 3:
        return None
    op = rng.choice(['vowel_sub', 'vowel_sub', 'vowel_sub',
                     'double', 'double', 'swap', 'delete'])
    if op == 'vowel_sub':
        vpos = [i for i, c in enumerate(chars) if c in VOWELS and i > 0]
        if not vpos:
            return None
        i = rng.choice(vpos)
        alts = [v for v in 'aeiou' if v != chars[i].lower()]
        new = rng.choice(alts)
        chars[i] = new.upper() if chars[i].isupper() else new
    elif op == 'double':
        consonant_pos = [i for i in range(1, len(chars))
                         if chars[i] not in VOWELS and chars[i].isalpha()]
        if not consonant_pos:
            return None
        i = rng.choice(consonant_pos)
        if i + 1 < len(chars) and chars[i].lower() == chars[i + 1].lower():
            chars.pop(i + 1)
        else:
            chars.insert(i + 1, chars[i].lower())
    elif op == 'swap':
        positions = list(range(1, len(chars) - 1))
        if not positions:
            return None
        i = rng.choice(positions)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == 'delete':
        positions = list(range(1, len(chars) - 1))
        if not positions:
            return None
        i = rng.choice(positions)
        chars.pop(i)
    result = ''.join(chars)
    return result if result != s and len(result) >= 2 else None


def perturb_last_name(last, rng, max_tries=15):
    if not _is_latin_name(last):
        return None
    for _ in range(max_tries):
        if '-' in last:
            parts = last.split('-')
            idx = rng.randint(0, len(parts))
            new_part = _perturb_part(parts[idx], rng)
            if new_part is None:
                continue
            parts[idx] = new_part
            return '-'.join(parts)
        else:
            result = _perturb_part(last, rng)
            if result is not None:
                return result
    return None


def make_fake_names(subset):
    if os.path.exists(PANEL_CSV):
        _panel = pd.read_csv(PANEL_CSV, usecols=['author_first_name', 'author_last_name'])
        real_combos = set(zip(_panel['author_last_name'].astype(str),
                              _panel['author_first_name'].astype(str)))
        all_real_lasts = set(_panel['author_last_name'].astype(str).unique())
        del _panel
    else:
        # Fallback: use only the 2k subset names
        real_combos = set(zip(subset['author_last_name'].astype(str),
                              subset['author_first_name'].astype(str)))
        all_real_lasts = set(subset['author_last_name'].astype(str).unique())

    rng_fake = np.random.RandomState(RANDOM_STATE_FAKE)
    pool = subset.sample(frac=1, random_state=RANDOM_STATE_FAKE).reset_index(drop=True)

    fakes = []
    seen = set()
    pool_idx = 0
    while len(fakes) < N_FAKE and pool_idx < len(pool) * 5:
        row = pool.iloc[pool_idx % len(pool)]
        pool_idx += 1

        first = str(row['author_first_name']).strip()
        last = str(row['author_last_name']).strip()

        new_last = perturb_last_name(last, rng_fake)
        if new_last is None:
            continue

        combo = (new_last, first)
        if combo in real_combos or combo in seen:
            continue
        if new_last in all_real_lasts:
            continue

        seen.add(combo)
        fakes.append({
            'author_first_name': first,
            'author_last_name':  new_last,
            '_source_last':      last,
        })

    fake_df = pd.DataFrame(fakes)
    fake_df.to_csv(os.path.join(OUTPUT_DIR, 'fake_names_timeline.csv'), index=False)
    print(f"Generated {len(fake_df):,} fake names from {pool_idx:,} attempts")
    return fake_df


# ============================================================
# Fake-name probe (cell 20)
# ============================================================
def run_fake_probe(fake_df, model_id, short_name, n_workers=N_WORKERS):
    out_path = os.path.join(OUTPUT_DIR, f'fake_probe_{short_name}.csv')

    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        raw_col = f'{short_name}_recog_raw'
        if raw_col in df.columns and df[raw_col].notna().sum() >= len(df) * 0.95:
            print(f"[{short_name}] Fake probe already complete ({len(df):,} rows). Skipping.")
            return df

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

    df.to_csv(out_path, index=False)
    print(f"[{short_name}] Saved fake probe to {out_path}")
    return df


def load_fake_names():
    p = os.path.join(OUTPUT_DIR, 'fake_names_timeline.csv')
    if not os.path.exists(p):
        p = os.path.join(INPUTS, 'fake_names_timeline.csv')    # local copy, not shipped
    return pd.read_csv(p)


# ============================================================
# Answer counts (cell 14) and fake-name counts (cell 22)
# ============================================================
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, max(0, center - half), min(1, center + half)


def get_fpr(probe_df, short):
    yn = probe_df[f'{short}_recog_yn'].astype(str).str.upper()
    k_yes = int((yn == 'YES').sum())
    n_parsed = int(yn.isin(['YES', 'NO', 'UNKNOWN']).sum())
    return k_yes, n_parsed


def summarize():
    summary_rows = []
    for model_id, short, label, release in FAM['models']:
        path = os.path.join(OUTPUT_DIR, f'recognition_{short}.csv')
        if not os.path.exists(path):
            print(f"[{short}] File not found: {path}")
            continue
        df = pd.read_csv(path)
        yn_col = [c for c in df.columns if c.endswith('_recog_yn')][0]

        yn = df[yn_col].astype(str).str.upper()
        n_yes = int((yn == 'YES').sum())
        n_no = int((yn == 'NO').sum())
        n_unknown = int((yn == 'UNKNOWN').sum())
        n_parsed = n_yes + n_no + n_unknown
        n_error = len(df) - n_parsed

        p, ci_lo, ci_hi = wilson_ci(n_yes, n_parsed)
        summary_rows.append({
            'short': short, 'label': label, 'release': release,
            'n_total': len(df), 'n_parsed': n_parsed,
            'n_yes': n_yes, 'n_no': n_no, 'n_unknown': n_unknown, 'n_error': n_error,
            'pct_yes': 100 * p if not np.isnan(p) else np.nan,
            'ci_lo': 100 * ci_lo if not np.isnan(ci_lo) else np.nan,
            'ci_hi': 100 * ci_hi if not np.isnan(ci_hi) else np.nan,
        })
        print(f"[{short:10s}] YES={n_yes:5d}/{n_parsed:5d}  ({p*100:5.1f}%)  errors={n_error}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df['release'] = pd.to_datetime(summary_df['release'])
    summary_df = summary_df.sort_values('release').reset_index(drop=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'timeline_summary.csv'), index=False)

    print("\n===== FALSE-POSITIVE RATES (PERTURBED FAKE NAMES) =====")
    print(f"{'model':12s} {'k_yes':>6s} {'n':>6s} {'FPR':>8s}")
    for _, short, _, _ in FAM['models']:
        p = os.path.join(OUTPUT_DIR, f'fake_probe_{short}.csv')
        if not os.path.exists(p):
            continue
        ky, np_ = get_fpr(pd.read_csv(p), short)
        print(f"{short:12s} {ky:6d} {np_:6d} {100*ky/np_:7.2f}%")


def main():
    global FAM, OUTPUT_DIR, client
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--step", default="all",
                    choices=["all", "subset", "extract", "query", "fake", "probe", "summary"])
    a = ap.parse_args()
    FAM, OUTPUT_DIR = FAMILIES[a.family], TIMELINE_DIRS[a.family]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    steps = ["subset", "extract", "query", "fake", "probe", "summary"] if a.step == "all" else [a.step]

    if {"query", "probe"} & set(steps):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1")

    subset = make_subset() if {"subset", "extract", "query", "fake"} & set(steps) else None
    if "extract" in steps:
        extract_existing(subset)
    if "query" in steps:
        query_models(subset)
    if "fake" in steps:
        make_fake_names(subset)
    if "probe" in steps:
        fake_df = load_fake_names()
        for model_id, short, _, _ in FAM['models']:
            print(f"\n>>> Probing {model_id}")
            run_fake_probe(fake_df, model_id, short)
    if "summary" in steps:
        summarize()


if __name__ == "__main__":
    main()
