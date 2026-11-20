#!/usr/bin/env python
"""Step 3 (network, no GPU). Training-data mentions: infini-gram count of each cohort name in the OLMo-2 corpus.

Index `v4_olmo-2-0325-32b-instruct_llama` on the public infini-gram API (Liu et al., 2024), query_type "count".
The query string is the full Wikidata name label (`final_name`) exactly as traced, e.g. "Marie Curie"; middle names
are kept. 0 is a valid count (name not in the corpus); a failed call stays missing and is retried on re-run.

From the project's fetch_infinigram.py; the collector functions (`query_infini_count`, `collect_counts`,
`build_name`) are copied verbatim from code/olmo_training_10k_phy.py (Fig. 2 pipeline), which it imported.

Out: <out> with one row per unique name and column `olmo_2_count` (-> data/mentions_wikidata9k.csv).
"""

import os
import argparse
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
COHORT_CSV = os.path.join(WORK, "physicists_dimensions_17k.csv")
OUT_CSV = os.path.join(WORK, "physicists_infinigram_wikidata9k.csv")

INFINI_INDEX = "v4_olmo-2-0325-32b-instruct_llama"
INFINI_URL   = "https://api.infini-gram.io/"
FREQ_COL  = "olmo_2_count"


# ------------------------------------------- verbatim from olmo_training_10k_phy.py
def build_name(df):
    """Full name from first + last (matches the strings sent to the LLM)."""
    return (df["author_first_name"].fillna("").astype(str).str.strip() + " " +
            df["author_last_name"].fillna("").astype(str).str.strip()).str.strip()


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


# ------------------------------------------- fetch_infinigram.py
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-csv", default=COHORT_CSV)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--batch-sleep", type=int, default=120,
                    help="seconds between batches; the API's limiter needs this")
    ap.add_argument("--retry-rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"index: {INFINI_INDEX}\nurl:   {INFINI_URL}")

    m = pd.read_csv(args.cohort_csv, low_memory=False)
    sel = m[m.wikidata_name.notna() | m.wd_label.notna()].copy()     # wikidata9k filter
    sel = sel[sel.final_name.notna()].drop_duplicates("final_name")

    # the collector builds its query string from author_first_name/author_last_name,
    # but those are initials for most of this cohort. Feed it the resolved full
    # name instead, split at the last space (build_name re-joins it unchanged).
    parts = sel.final_name.astype(str).str.rsplit(" ", n=1)
    sel["author_first_name"] = parts.str[0]
    sel["author_last_name"] = parts.str[-1].where(parts.str.len() > 1, "")
    keep = ["author_id", "final_name", "author_first_name", "author_last_name",
            "total_citations", "total_publications", "first_year_of_publication",
            "gender_any", "country_any", "current_research_org_country_code",
            "wikidata_id", "wikidata_name", "wd_label"]
    sel = sel[[c for c in keep if c in sel.columns]]
    print(f"{len(sel)} unique names to query (wikidata9k cohort)")

    tmp_in = args.out.replace(".csv", "_input.csv")
    sel.to_csv(tmp_in, index=False)

    collect_counts(tmp_in, args.out,
                   workers=args.workers,
                   limit=args.limit,
                   batch_size=args.batch_size,
                   batch_sleep=args.batch_sleep,
                   retry_rounds=args.retry_rounds)

    d = pd.read_csv(args.out)
    col = FREQ_COL
    got = d[col].notna().sum()
    print(f"\ncollected {got}/{len(d)}  ({100 * got / len(d):.1f}%)")
    if got:
        q = d[col].dropna()
        print(f"  zero counts: {int((q == 0).sum())}")
        print("  quantiles:", {p: int(q.quantile(p / 100))
                               for p in (10, 25, 50, 75, 90, 99)})


if __name__ == "__main__":
    main()
