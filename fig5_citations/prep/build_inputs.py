#!/usr/bin/env python3
"""Build the anonymized inputs in fig5_citations/data/ from the private pair-level frame.

NOT needed to redraw the figure; kept for provenance. It needs the non-public file
data/oa_recognition/frames_bib_0927/bib_0927_pairs.csv.gz of the original project
(one row per citing-paper x cited-work first author; 661,992 rows), passed as argv[1].

What it does
  * arXiv IDs -> random integer `paper` (only the year-month YYMM of the ID is kept, as `ym`);
  * OpenAlex author IDs -> random integer `author`;
  * splits the frame into three tables (paper-, author- and pair-level) without dropping rows.
No values are changed; every regression variable is rebuilt from these columns by make_fig5.py.

Run: python fig5_citations/prep/build_inputs.py /path/to/bib_0927_pairs.csv.gz
"""
import os
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
rng = np.random.default_rng(20261030)

d = pd.read_csv(SRC, dtype={"arxiv_id": str})
assert d.arxiv_id.str.match(r"^\d{4}\.\d{4,5}$").all()


def anon(s):
    u = s.unique()
    return s.map(dict(zip(u, rng.permutation(len(u)) + 1))).astype("int32")


d["paper"] = anon(d.arxiv_id)
d["author"] = anon(d.author_id)

papers = (d.groupby("paper").agg(ym=("arxiv_id", lambda s: int(s.iloc[0][:4])), period=("period", "first"),
                                 alpha_main=("alpha_main", "first")).reset_index())
assert d.groupby("paper").alpha_main.nunique(dropna=False).max() == 1
authors = (d.groupby("author").agg(recog=("recog", "first"), works_count=("works_count", "first"),
                                   cited_by_count=("cited_by_count", "first"),
                                   first_pub_year=("first_pub_year", "first")).reset_index())
assert d.groupby("author")[["recog", "works_count", "cited_by_count", "first_pub_year"]].nunique().max().max() <= 1
# career age / stage are deterministic in first_pub_year (checked): 2024|2026 - first_pub_year, <=10 / 11-30 / >30
ca = 2024 - d.first_pub_year
assert ((ca == d.career_age) | ca.isna()).all()
st = np.where(ca <= 10, "early", np.where(ca <= 30, "mid", "late"))
assert ((st == d.career_stage) | ca.isna()).all()

pairs = d[["paper", "author", "ref_year", "ref_cit"]].sample(frac=1, random_state=1).sort_values(["paper"],
                                                                                                    kind="stable")
papers.sort_values("paper").to_csv(os.path.join(OUT, "papers.csv.gz"), index=False)
authors.sort_values("author").to_csv(os.path.join(OUT, "authors.csv.gz"), index=False)
pairs.to_csv(os.path.join(OUT, "pairs.csv.gz"), index=False)
print(len(papers), "papers", len(authors), "authors", len(pairs), "pairs ->", OUT)
