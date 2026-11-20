#!/usr/bin/env python3
"""Step 3: GPT-4o recognition labels, round A (OpenRouter, real-time, resumable).

Model `openai/gpt-4o` via OpenRouter, temperature 0, max_tokens 40, 32 threads, up to 4 attempts per
author. Name shown as "Last, First..." (last whitespace token of the OpenAlex display name first).
Prompt = recognition prompt of the SI (also in 04_batch_build_jsonl.py). Input: the queue of
label queue A (CSV files with author_id, display_name; 102,789 authors).
Output CSV author_id, gpt4o_recog (YES/NO/UNKNOWN), gpt4o_conf; logged 0 errors.
Needs OPENROUTER_API_KEY in the environment.

Run: python 03_label_openrouter.py --queue 'fa_*.csv' --out oa_firstauthor_gpt4o_labels.csv
"""
import pandas as pd, os, re, time, threading, glob, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--queue", required=True, help="glob of queue CSV files (author_id, display_name)")
ap.add_argument("--out", default="oa_firstauthor_gpt4o_labels.csv")
args = ap.parse_args()

OUT = args.out
MODEL = "openai/gpt-4o"
WORKERS = 32

client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1")
SYS = "You are a knowledgeable assistant. Answer concisely and honestly. Do not speculate."

def prompt(display_name):
    toks = str(display_name).split()
    name = f"{toks[-1]}, {' '.join(toks[:-1])}" if len(toks) >= 2 else str(display_name)
    return (f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
            f"Reply in EXACTLY this format (nothing else):\n"
            f"SCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\nFIELD: [specific field, or \"Unknown\"]\nCONFIDENCE: HIGH/MEDIUM/LOW")

def parse(t):
    U = str(t).upper()
    m = re.search(r'SCIENTIFIC RECOGNITION:\s*(YES|NO|UNKNOWN)', U)
    rec = m.group(1) if m else None
    mc = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', U)
    conf = mc.group(1) if mc else None
    return rec, conf

def call(aid, name):
    for _ in range(4):
        try:
            r = client.chat.completions.create(model=MODEL,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt(name)}],
                max_tokens=40, temperature=0)
            rec, conf = parse(r.choices[0].message.content)
            return aid, rec, conf
        except Exception:
            time.sleep(3)
    return aid, "ERROR", None

def main():
    df = pd.concat([pd.read_csv(f, dtype=str) for f in glob.glob(args.queue)], ignore_index=True).drop_duplicates("author_id")
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, dtype=str)
        done = set(prev.loc[prev["gpt4o_recog"] != "ERROR", "author_id"])
    todo = df[~df["author_id"].isin(done)].reset_index(drop=True)
    print(f"total {len(df):,}; done {len(done):,}; todo {len(todo):,}", flush=True)
    if len(todo) == 0:
        print("ALL DONE", flush=True); return
    lock = threading.Lock()
    header = not os.path.exists(OUT)
    n = 0; t0 = time.time()
    with open(OUT, "a") as fh:
        if header:
            fh.write("author_id,gpt4o_recog,gpt4o_conf\n")
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(call, r.author_id, r.display_name) for r in todo.itertuples()]
            for f in as_completed(futs):
                aid, rec, conf = f.result()
                with lock:
                    fh.write(f"{aid},{rec},{conf}\n"); n += 1
                    if n % 1000 == 0:
                        fh.flush(); rate = n / (time.time() - t0)
                        print(f"  {n:,}/{len(todo):,}  {rate:.1f}/s  eta {(len(todo)-n)/rate/60:.0f}min", flush=True)
    print(f"DONE: {n:,} in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
