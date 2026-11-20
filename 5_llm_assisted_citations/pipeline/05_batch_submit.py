#!/usr/bin/env python3
"""Step 5: upload the batch_*.jsonl files and create OpenAI batches (24 h window), idempotent.

Batch IDs are kept in submitted.json; files already submitted are skipped.
Needs OPENAI_API_KEY in the environment. Run inside the directory holding batch_*.jsonl:
  cd batch_run && python ../05_batch_submit.py
"""
import os, json, glob, sys
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
SUB = "submitted.json"; sub = json.load(open(SUB)) if os.path.exists(SUB) else {}
for fp in sorted(glob.glob("batch_*.jsonl")):
    if fp in sub: print("skip (submitted):", fp, sub[fp]); continue
    up = client.files.create(file=open(fp, "rb"), purpose="batch")
    b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"src": fp})
    sub[fp] = b.id; json.dump(sub, open(SUB, "w"), indent=2); print("submitted", fp, "->", b.id)
print("all submitted:", len(sub), "batches")
