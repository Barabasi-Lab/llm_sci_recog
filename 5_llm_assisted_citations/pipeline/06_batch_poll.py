#!/usr/bin/env python3
"""Step 6: wait for the batches, download completed ones and parse the recognition answer, resumable.

Every 10 min: for each batch in submitted.json, append author_id, gpt4o_recog, gpt4o_conf to
results_2060.csv once it is completed (authors already written are skipped; unparsable answers are
written as NA), report failed/expired/cancelled batches, and stop when no batch is still running.
As run, 210,024 rows (20,428 YES / 2,008 NO / 187,588 UNKNOWN, no NA), as a long cluster job.
Needs OPENAI_API_KEY. Run inside the directory holding submitted.json (or pass --dir):
  python 06_batch_poll.py --dir batch_run
"""
import os, json, csv, re, time, argparse
from openai import OpenAI
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=".", help="directory with submitted.json; results_2060.csv is written there")
args = ap.parse_args()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
os.chdir(args.dir)
sub = json.load(open("submitted.json")); RES = "results_2060.csv"
def load_done():
    d = set()
    if os.path.exists(RES):
        for r in csv.DictReader(open(RES)): d.add(r["author_id"])
    return d
def parse(t):
    U = str(t).upper(); m = re.search(r'SCIENTIFIC RECOGNITION:\s*(YES|NO|UNKNOWN)', U); c = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', U)
    return (m.group(1) if m else "NA", c.group(1) if c else "NA")
while True:
    done = load_done(); newhdr = not os.path.exists(RES)
    out = open(RES, "a", newline=""); w = csv.writer(out)
    if newhdr: w.writerow(["author_id", "gpt4o_recog", "gpt4o_conf"])
    alldone = True
    for fp, bid in sub.items():
        b = client.batches.retrieve(bid)
        if b.status == "completed" and b.output_file_id:
            txt = client.files.content(b.output_file_id).text; n = 0
            for line in txt.splitlines():
                o = json.loads(line); aid = o["custom_id"]
                if aid in done: continue
                body = o.get("response", {}).get("body", {}); c = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                rec, conf = parse(c); w.writerow([aid, rec, conf]); done.add(aid); n += 1
            if n: print(f"{fp}: +{n} (total {len(done)})", flush=True)
        elif b.status in ("failed", "expired", "cancelled"):
            print(f"{fp}: {b.status}!", flush=True)
        else:
            alldone = False; print(f"{fp}: {b.status} {getattr(b.request_counts,'completed',0)}/{getattr(b.request_counts,'total',0)}", flush=True)
    out.close()
    if alldone: print("ALL DONE, total", len(done), flush=True); break
    time.sleep(600)
