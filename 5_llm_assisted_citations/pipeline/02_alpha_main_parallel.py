#!/usr/bin/env python3
"""Step 2: per-paper main-text alpha for the new window (2024-12 .. 2026-07), parallel.

Same estimator, main-text extraction, tokenizer and < 5-sentence rule as 05a (the extraction
functions are imported from 01_alpha_main.py; as run, they were exec'd from the same notebook cell).
Only the loop differs: tokenisation and MLE inference run in a multiprocessing pool. Resumable.
Run as for the paper (16 workers, CPU):
  python 02_alpha_main_parallel.py --paper-list data/all_list.csv --min-month 2024-12 \
      --sources data/arxiv_sources_physics_ext --out data/new_alpha_perpaper.csv \
      --liang-repo ~/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers --nproc 16
Output: arxiv_id, llm_alpha_main (rounded to 4 decimals; empty if no main text or < 5 sentences),
n_sent (-1 on an exception). 40,237 papers (as run the file got no header row, so loading it
with --skip_leading_rows=1 kept 40,236 in alpha_new_pp; the skipped paper is not in the Fig. 5 frame).
"""
import os, sys, re, csv, argparse, time, tempfile, importlib.util
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def _init(repo, sources):
    import spacy
    sys.path.insert(0, repo)
    from src.MLE import MLE
    spec = importlib.util.spec_from_file_location("alpha_main", os.path.join(HERE, "01_alpha_main.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    g = globals()
    g["nlp"] = spacy.load("en_core_web_sm")
    g["get_main_text_for_paper"] = m.get_main_text_for_paper
    g["SOURCES"] = sources
    g["MODEL"] = MLE(os.path.join(repo, "distribution", "Phys.parquet"))
    g["TMP"] = os.path.join(tempfile.gettempdir(), "ppw_%d.parquet" % os.getpid())

    def _tok(text):
        text = str(text).replace("\n", " "); out = []
        for s in nlp(text).sents:
            w = [x for x in re.findall(r"\b\w+\b", s.text.lower()) if not x.isdigit()]
            if w: out.append(np.array(w))
        return out
    g["_tok"] = _tok


def _work(aid):
    try:
        txt = get_main_text_for_paper(aid, SOURCES)
        if not txt: return (aid, "", 0)
        sents = _tok(txt)
        if len(sents) < 5: return (aid, "", len(sents))
        pd.DataFrame({"inference_sentence": sents}).to_parquet(TMP)
        al, ci = MODEL.inference(TMP, exploded_data=True)
        return (aid, round(float(al), 4), len(sents))
    except Exception:
        return (aid, "", -1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-list", required=True, help="step-2 all_list.csv")
    ap.add_argument("--min-month", default="2024-12")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--liang-repo", default=os.path.expanduser("~/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers"))
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    ids = [r["arxiv_id"] for r in csv.DictReader(open(a.paper_list)) if r["month"] >= a.min_month]
    OUT = a.out
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)): done.add(r["arxiv_id"])
    todo = [i for i in ids if i not in done]
    hdr = not os.path.exists(OUT); fh = open(OUT, "a", newline=""); w = csv.writer(fh)
    if hdr: w.writerow(["arxiv_id", "llm_alpha_main", "n_sent"])
    pool = Pool(a.nproc, initializer=_init, initargs=(a.liang_repo, a.sources)); n = len(done); t0 = time.time()
    for aid, al, ns in pool.imap_unordered(_work, todo, chunksize=16):
        w.writerow([aid, al, ns]); n += 1
        if n % 500 == 0: fh.flush(); print(f"{n} done {n/(time.time()-t0):.1f}/s", flush=True)
    fh.close(); print("DONE", n, flush=True)
