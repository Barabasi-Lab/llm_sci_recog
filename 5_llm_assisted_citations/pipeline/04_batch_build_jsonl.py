#!/usr/bin/env python3
"""Step 4: OpenAI Batch API request files for recognition labels, round B.

One chat-completion request per author (custom_id = OpenAlex author ID): model `gpt-4o`,
temperature 0, max_tokens 40, same system prompt, name format and user prompt as round A
(03_label_openrouter.py). 40,000 requests per file -> batch_01.jsonl .. batch_06.jsonl (210,024).
Input: need_label_authors_2060.csv from label queue B. No network access.

Run (writes into --outdir): python 04_batch_build_jsonl.py --authors need_label_authors_2060.csv --outdir batch_run
"""
import argparse, csv, json, os, glob

ap = argparse.ArgumentParser()
ap.add_argument("--authors", default="need_label_authors_2060.csv")
ap.add_argument("--outdir", default="batch_run")
args = ap.parse_args()

AUTH = args.authors; OUTD = args.outdir
os.makedirs(OUTD, exist_ok=True)
SYS = "You are a knowledgeable assistant. Answer concisely and honestly. Do not speculate."
def full_prompt(dn):
    t = str(dn).split(); name = f"{t[-1]}, {' '.join(t[:-1])}" if len(t) >= 2 else str(dn)
    return (f"Do you recognize {name} as a scientist? If so, what is their field?\n\n"
            f"Reply in EXACTLY this format (nothing else):\nSCIENTIFIC RECOGNITION: YES/NO/UNKNOWN\nFIELD: [specific field, or \"Unknown\"]\nCONFIDENCE: HIGH/MEDIUM/LOW")
rows = [r for r in csv.DictReader(open(AUTH)) if r.get("display_name", "").strip()]
CHUNK = 40000; nf = 0; f = None; n = 0
for i, r in enumerate(rows):
    if i % CHUNK == 0:
        if f: f.close()
        nf += 1; f = open(f"{OUTD}/batch_{nf:02d}.jsonl", "w")
    body = {"model": "gpt-4o", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": full_prompt(r["display_name"])}], "max_tokens": 40, "temperature": 0}
    f.write(json.dumps({"custom_id": r["author_id"], "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n"); n += 1
if f: f.close()
print(f"authors={len(rows)}  wrote {nf} batch files, {n} requests total")
[print("  ", os.path.basename(x), f"{os.path.getsize(x)//1024//1024}MB") for x in sorted(glob.glob(OUTD + '/batch_*.jsonl'))]
