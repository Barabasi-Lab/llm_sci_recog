"""Step 4. Query provenance and position of every final-answer (Step C) pair.
(original: analysis_untraced_provenance.py)

Each scientist in the online answer (resolved with query_resolver_v2, de-duplicated within concept,
rank = position in the answer) is labelled A or B if the master table (step 3, second run) has that
(concept, mkey) unit with y=1 (g=1 -> A, g=0 -> B), else `untraced`. Units whose key differs from the
answer key (identity-corrected) are matched by unique surname within the concept.
`recog`/`cites` are descriptive columns not used by Fig. 4.

Reads : $FIG4_WORK/recommend_{offline,native}_v2.json, query_names_v2.json, recog_recommended_as.csv,
        recog_online_additional_as.csv, recog_v2_bare.csv, analysis/{online_search_master.csv,
        author_citations_curated_v2.csv}, analysis/relevance/units_researcher_ids.csv
Writes: $FIG4_WORK/analysis/final_pairs_provenance.csv
"""
import sys, json, re, unicodedata
import numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from query_resolver_v2 import build
C = os.environ.get("FIG4_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")); OUT = f"{C}/analysis"
def nrm(n):
    s = unicodedata.normalize("NFKD", str(n)); s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", "", s)).strip()
def mk(n):
    t = nrm(n).split(); return f"{t[-1]}|{t[0][0] if t[0] else ''}" if t else None
OFF, ON, resolve = build()
nat = {r['concept']: r for r in json.load(open(f"{C}/recommend_native_v2.json"))}
off = {r['concept']: r for r in json.load(open(f"{C}/recommend_offline_v2.json"))}
qn = json.load(open(f"{C}/query_names_v2.json"))
bare = {}
for fn in ["recog_recommended_as.csv", "recog_online_additional_as.csv", "recog_v2_bare.csv"]:
    for _, r in pd.read_csv(f"{C}/{fn}").iterrows():
        k = mk(r['normalized_name'] if pd.notna(r.get('normalized_name')) else r['name'])
        if k and k not in bare: bare[k] = (str(r['yn']).upper() == 'YES')
u = pd.read_csv(f"{OUT}/relevance/units_researcher_ids.csv").dropna(subset=['career_cites']).drop_duplicates('mkey')
ccit = dict(zip(u.mkey, u.career_cites))
cur = pd.read_csv(f"{OUT}/author_citations_curated_v2.csv"); cit = dict(zip(cur.mkey, cur.cites))
M = pd.read_csv(f"{OUT}/online_search_master.csv")
inans = {(r.concept, r.mkey): r.g for r in M.itertuples() if r.y == 1}   # traced final-answer pairs -> g
rows = []
used = set()
for c, r in nat.items():
    offset = {resolve(s.get('normalized_name') or f"{s.get('first_name','')} {s.get('last_name','')}") for s in off.get(c, {}).get('scientists', [])} - {None}
    seen = set()
    for rank, s in enumerate(r.get('scientists', []), 1):
        k = resolve(s.get('normalized_name') or f"{s.get('first_name','')} {s.get('last_name','')}")
        if not k or k in seen: continue
        seen.add(k)
        g0 = inans.get((c, k))
        if g0 is None:   # identity-corrected units whose key differs from the answer key: match by surname within concept
            cand = [kk for (cc_, kk) in inans if cc_ == c and kk.split('|')[0] == k.split('|')[0] and (c, kk) not in used]
            if len(cand) == 1: g0 = inans[(c, cand[0])]; used.add((c, cand[0]))
        else: used.add((c, k))
        grp = {1: 'A', 0: 'B'}.get(g0, 'untraced')
        cc = ccit.get(k, cit.get(k, np.nan))
        rows.append(dict(concept=c, mkey=k, grp=grp, rank=rank, n_final=len(r.get('scientists', [])),
                         offline=int(k in offset), recog=bare.get(k, np.nan), cites=cc))
F = pd.DataFrame(rows)
F.to_csv(f"{OUT}/final_pairs_provenance.csv", index=False)
print(F.grp.value_counts().to_dict())
g = F.groupby('grp')
out = pd.DataFrame({
    'n': g.size(),
    'in_closed_book_%': 100 * g.offline.mean(),
    'GPT54_recognizes_%': 100 * g.recog.mean(),
    'recog_coverage_%': 100 * g.recog.apply(lambda x: x.notna().mean()),
    'median_rank': g['rank'].median(),
    'top3_%': 100 * g['rank'].apply(lambda x: (x <= 3).mean()),
    'median_cites': g.cites.median(),
}).round(1)
print(out.to_string())
