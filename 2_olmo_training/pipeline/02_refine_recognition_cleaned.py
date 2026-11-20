#!/usr/bin/env python3
"""
Step 02 - re-ask OLMo-2 with the reduced "First Last" name; add `recognized_any`.

The original recognition label (`olmo2_recog_yn`) was produced by asking OLMo-2
"Do you recognize {Last}, {First} as a scientist?" using the FULL first-name
field, i.e. middle initials/names included (e.g. "Smith, John A",
"Smith, John A B"). That is the same string problem that deflates infini-gram
frequency; we cleaned the frequency (-> training_frequency) but never re-ran
recognition on the cleaned name.

This script re-asks OLMo-2 the IDENTICAL question with the cleaned "First Last"
name (first token + intact last name, middle dropped) for every row that has a
middle initial/name, using the exact same system prompt, model, and generation
settings, so the two labels are apples-to-apples. Rows with no middle token are
unchanged by definition and are copied over.

Output checkpoint: data/olmo_training/recognition_refined_cleaned.csv
    author_id, clean_query, olmo2_recog_yn_refined, olmo2_recog_raw_refined
Merge into the master with merge_into_master().

Needs: Apple-silicon Mac with mlx / mlx-lm and the OLMo-2 weights (run step only).
Usage (from the working directory):
    python 02_refine_recognition_cleaned.py          # query OLMo-2 (resumable)
    python 02_refine_recognition_cleaned.py merge    # add recognized_refined / recognized_any to the master
"""
import os, sys
import pandas as pd

# prompt, parser and model id are imported from step 01 (file name starts with a digit)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "step04", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "01_infinigram_counts_and_olmo_recognition.py"))
_step04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step04)
SYSTEM_RECOGNITION = _step04.SYSTEM_RECOGNITION
build_prompt_recognition = _step04.build_prompt_recognition
parse_recognition_response = _step04.parse_recognition_response
OLMO_RECOG_MODEL = _step04.OLMO_RECOG_MODEL

MASTER = "data/olmo_training/olmo_training_master_final.csv"
CKPT   = "data/olmo_training/recognition_refined_cleaned.csv"
BATCH, MAXTOK = 16, 32


def clean_first(first):
    """First token only (drop middle initials/names): 'John A' -> 'John'."""
    toks = str(first).strip().split()
    return toks[0] if toks else ""


def run(limit=None):
    import mlx.core as mx
    from mlx_lm import load, batch_generate

    def free_metal():
        for fn in (getattr(mx, "clear_cache", None),
                   getattr(getattr(mx, "metal", None), "clear_cache", None)):
            if callable(fn):
                try:
                    fn(); return
                except Exception:
                    pass

    d = pd.read_csv(MASTER, dtype={"author_id": str})
    # the master-table build writes author_first_name / author_last_name; the master used in the
    # original run had them as first_name / last_name (no-op there)
    d = d.rename(columns={"author_first_name": "first_name", "author_last_name": "last_name"})
    # target = rows whose name actually changes under cleaning (a middle token exists)
    tgt = d[d["name_type"].isin(["middle_initial", "middle_fullname"])].copy()
    tgt["clean_first"] = tgt["first_name"].map(clean_first)
    tgt["clean_query"] = tgt["last_name"].astype(str).str.strip() + ", " + tgt["clean_first"]

    try:
        done = pd.read_csv(CKPT, dtype={"author_id": str})
        print(f"resuming from {CKPT} ({len(done):,} done)")
    except FileNotFoundError:
        done = pd.DataFrame(columns=["author_id", "clean_query",
                                     "olmo2_recog_yn_refined", "olmo2_recog_raw_refined"])
    todo = tgt[~tgt["author_id"].isin(set(done["author_id"]))]
    if limit:
        todo = todo.head(limit)
    print(f"middle-name rows: {len(tgt):,} | still to query: {len(todo):,}")
    if len(todo) == 0:
        print("nothing to do."); return

    print(f"loading {OLMO_RECOG_MODEL} ...")
    model, tok = load(OLMO_RECOG_MODEL)
    rows = todo.to_dict("records")
    nb = (len(rows) + BATCH - 1) // BATCH
    buf = []
    for b, s in enumerate(range(0, len(rows), BATCH), 1):
        chunk = rows[s:s + BATCH]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_RECOGNITION},
             {"role": "user", "content": build_prompt_recognition(r["clean_first"], r["last_name"])}],
            add_generation_prompt=True) for r in chunk]
        resp = batch_generate(model, tok, prompts=prompts, max_tokens=MAXTOK, verbose=False)
        texts = resp.texts if hasattr(resp, "texts") else resp
        for r, t in zip(chunk, texts):
            p = parse_recognition_response(t)
            buf.append({"author_id": r["author_id"], "clean_query": r["clean_query"],
                        "olmo2_recog_yn_refined": p["recognized"],
                        "olmo2_recog_raw_refined": p["raw_response"]})
        # checkpoint every batch
        pd.concat([done, pd.DataFrame(buf)], ignore_index=True).to_csv(CKPT, index=False, encoding="utf-8")
        free_metal()
        if b % 10 == 0 or b == nb:
            print(f"  batch {b}/{nb}: {len(done) + len(buf):,} labelled", flush=True)
    print(f"done. wrote {CKPT}")


def merge_into_master():
    """Add recognized_refined_yn / recognized_refined to the master.
    Middle-name rows take the cleaned-name label; all others keep the original."""
    d = pd.read_csv(MASTER, dtype={"author_id": str})
    ref = pd.read_csv(CKPT, dtype={"author_id": str})[["author_id", "olmo2_recog_yn_refined"]]
    d = d.merge(ref, on="author_id", how="left")
    # non-middle-name rows are unchanged by cleaning -> refined == original
    d["recognized_refined_yn"] = d["olmo2_recog_yn_refined"].fillna(d["olmo2_recog_yn"])
    d["recognized_refined"] = (d["recognized_refined_yn"] == "YES").astype(int)
    # recognized_any = recognized under EITHER name form (max/OR) -> the analysis outcome
    d["recognized_any"] = ((d["olmo2_recog_yn"] == "YES") | (d["recognized_refined_yn"] == "YES")).astype(int)
    d = d.drop(columns=["olmo2_recog_yn_refined"])
    d.to_csv(MASTER, index=False, encoding="utf-8")
    flips = int(((d["recognized_refined_yn"] != d["olmo2_recog_yn"]) &
                 d["name_type"].isin(["middle_initial", "middle_fullname"])).sum())
    gained = int(((d["olmo2_recog_yn"] != "YES") & (d["recognized_refined_yn"] == "YES")).sum())
    lost = int(((d["olmo2_recog_yn"] == "YES") & (d["recognized_refined_yn"] != "YES")).sum())
    print(f"merged into {MASTER}: {flips:,} labels changed under cleaning "
          f"(+{gained:,} gained YES, -{lost:,} lost YES)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_into_master()
    else:
        run()
