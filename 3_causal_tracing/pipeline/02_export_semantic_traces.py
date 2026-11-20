#!/usr/bin/env python
"""Step 2 (CPU). Reduce each per-name trace to the 6 ROME token positions and export one matrix per name.

Input : <root>/{residual_w1,attention}/*__prob.pt and <root>/classification_shard*.csv from 01_trace_batch.py
Output: <root>/classification.csv                       (shards merged; recognition label input)
        <out>  semantic_traces_wikidata9k.npz          {kind}_names, {kind}_sem [n, 6, 32], {kind}_base [n]
                                                        for kind in residual_w1 (window 1), attention (window 10)

The six positions (ROME naming, rows of {kind}_sem):
    0 first subject token        ie[subj_start]
    1 middle subject tokens      mean(ie[subj_start+1 : subj_end-1])
    2 last subject token         ie[subj_end-1]            <- early site (Fig. 3h: residual, layers 5-9)
    3 first subsequent token     ie[subj_end]
    4 further tokens             mean(ie[subj_end+1 : -1])
    5 last token                 ie[-1]                    <- late site (Fig. 3i: attention, layers 15-19)
{kind}_base is the clean-run p(" physicist") (`base_score`).

`load_all_results` and `extract_semantic_positions` are copied verbatim from the project's
make_figures_harmonized.py; the export loop is the cluster script export_semantic_traces.py. Unpickling the .pt
files needs the tracing library's `src` package importable (or stubbed).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")


# ------------------------------------------- verbatim from make_figures_harmonized.py
def load_all_results(results_dir, metric="prob"):
    files = sorted(glob.glob(os.path.join(results_dir, f"*__{metric}.pt")))
    records, skipped = [], 0
    for fpath in files:
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  Cannot load {os.path.basename(fpath)}: {e}")
            skipped += 1
            continue
        if "error" in data:
            skipped += 1
            continue
        result = data["result"]
        rec = {
            "indirect_effects": result["indirect_effects"],
            "base_score": result.get("base_score"),
            "low_score": result.get("low_score"),
            "subj_range": result.get("subj_range"),
            "trace_start_idx": result.get("trace_start_idx", 0),
            "patch_input_toks": result.get("patch_input_toks"),
            "corrupt_input_toks": result.get("corrupt_input_toks"),
            "normalized": result.get("normalized", False),
        }
        rec["wikidata_name"] = data["wikidata_name"]
        rec["metric"] = data.get("metric", metric)
        records.append(rec)
    print(f"Loaded {len(records)} records from {results_dir}" +
          (f" (skipped {skipped})" if skipped else ""))
    return records


def extract_semantic_positions(rec):
    """Map raw indirect_effects [n_tokens, n_layers] -> [6, n_layers]."""
    ie = rec["indirect_effects"].float()
    n_tok, n_layers = ie.shape
    trace_start = rec.get("trace_start_idx", 0)
    subj_range = rec.get("subj_range", (0, 1))

    subj_start = subj_range[0] - trace_start
    subj_end = subj_range[1] - trace_start
    subj_start = max(0, subj_start)
    subj_end = min(n_tok, subj_end)

    out = torch.full((6, n_layers), float("nan"))
    if subj_start < n_tok:
        out[0] = ie[subj_start]
    mid_start, mid_end = subj_start + 1, subj_end - 1
    if mid_end > mid_start:
        out[1] = ie[mid_start:mid_end].mean(dim=0)
    last_subj_idx = subj_end - 1
    if 0 <= last_subj_idx < n_tok:
        out[2] = ie[last_subj_idx]
    first_sub = subj_end
    if first_sub < n_tok:
        out[3] = ie[first_sub]
    further_start, further_end = subj_end + 1, n_tok - 1
    if further_end > further_start:
        out[4] = ie[further_start:further_end].mean(dim=0)
    out[5] = ie[n_tok - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        WORK, "results", "causal_traces_harmonized_wikidata9k_profession"))
    ap.add_argument("--out", default=os.path.join(WORK, "semantic_traces_wikidata9k.npz"))
    args = ap.parse_args()
    root = args.root

    # ---- merge the per-shard recognition labels (from make_figures_harmonized.py) ----
    cls = pd.concat([pd.read_csv(p) for p in
                     sorted(glob.glob(os.path.join(root, "classification_shard*.csv")))],
                    ignore_index=True).drop_duplicates("wikidata_name")
    cls.to_csv(os.path.join(root, "classification.csv"), index=False)

    # ---- export (export_semantic_traces.py) ----
    out = {}
    for kind in ["residual_w1", "attention"]:
        recs = load_all_results(os.path.join(root, kind), "prob")
        names = [r["wikidata_name"] for r in recs]
        sem = np.stack([extract_semantic_positions(r).numpy() for r in recs])
        base = np.array([np.nan if r["base_score"] is None else float(r["base_score"]) for r in recs])
        out[f"{kind}_names"] = np.array(names, dtype=object); out[f"{kind}_sem"] = sem; out[f"{kind}_base"] = base
        print(kind, sem.shape)
    np.savez_compressed(args.out, **out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
