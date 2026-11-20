"""Step 3. Master table of searched (concept, scientist) units: Step A vs Step B, in-final-answer flag.
(original: online_search_analysis.py, reduced to the master table; its plots and statistics are
superseded by make_fig4.py)

Run twice: once to build the units that are then anchored to Dimensions researcher ids, and again
after that, when the identity-corrected in-answer flag `y` from units_researcher_ids.csv is merged in.

Unit = (concept, mkey) for every person name extracted from that concept's queries (step 2), resolved
with query_resolver_v2.resolve().
  g = 1 if the name is in the first (pre-retrieval) query batch (Step A), else 0 (Step B)
  y = 1 if the resolved person is in the final online answer
  offrec = 1 if in the closed-book recommendation for the same concept
  first_batch = index of the first batch whose text contains the surname (capped at 4)
  recog, log_cit, log_pub = bare-recognition flag and impact (not used by Fig. 4; Fig. 4 reads
  career_cites/career_pubs from units_researcher_ids.csv)

Reads : $FIG4_WORK/recommend_{offline,native}_v2.json, query_names_v2.json,
        analysis/author_citations_curated_v2.csv, recog_recommended_as.csv, recog_online_additional_as.csv,
        recog_v2_bare.csv (bare recognition probe; only the unused `recog` column),
        analysis/relevance/units_researcher_ids.csv (second run only)
Writes: $FIG4_WORK/analysis/online_search_master.csv
"""
import os, sys, json, re, unicodedata, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from query_resolver_v2 import build
warnings.filterwarnings('ignore')

C=os.environ.get("FIG4_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")); OUT=f"{C}/analysis"

# ------------------------------------------------------------------ helpers
def nrm(n):
    s=unicodedata.normalize("NFKD",str(n));s="".join(c for c in s if not unicodedata.combining(c)).lower();return re.sub(r"\s+"," ",re.sub(r"[^a-z\s]","",s)).strip()
def mk(n):
    t=nrm(n).split();return f"{t[-1]}|{t[0][0] if t[0] else ''}" if t else None

# ------------------------------------------------------------------ load ONCE
OFF, ON, resolve = build()                                   # distinct-people sets + resolver
nat = {r['concept']:r for r in json.load(open(f"{C}/recommend_native_v2.json"))}
off = {r['concept']:r for r in json.load(open(f"{C}/recommend_offline_v2.json"))}
qn  = json.load(open(f"{C}/query_names_v2.json"))
cur = pd.read_csv(f"{OUT}/author_citations_curated_v2.csv")
cit = dict(zip(cur.mkey,cur.cites)); pub = dict(zip(cur.mkey,cur.pubs))
# prefer DOI-anchored career citations (units_researcher_ids), fall back to old curated cites for coverage
try:
    _uc = pd.read_csv(f"{OUT}/relevance/units_researcher_ids.csv").dropna(subset=['career_cites']).drop_duplicates('mkey')
    ccit = dict(zip(_uc.mkey,_uc.career_cites)); cpub = dict(zip(_uc.mkey,_uc.career_pubs))
except FileNotFoundError:
    ccit = {}; cpub = {}
lc = lambda k: np.log1p(ccit[k]) if (k in ccit and pd.notna(ccit[k])) else (np.log1p(cit[k]) if (k in cit and pd.notna(cit[k])) else np.nan)
lp = lambda k: np.log1p(cpub[k]) if (k in cpub and pd.notna(cpub[k])) else (np.log1p(pub[k]) if (k in pub and pd.notna(pub[k])) else np.nan)
bare = {}
for fn in ["recog_recommended_as.csv","recog_online_additional_as.csv","recog_v2_bare.csv"]:
    for _,r in pd.read_csv(f"{C}/{fn}").iterrows():
        k=mk(r['normalized_name'] if pd.notna(r.get('normalized_name')) else r['name'])
        if k and k not in bare: bare[k]=(str(r['yn']).upper()=='YES')
# resolve() (not mk()) so answer/offline keys match the unit keys built from query names
def ppl(r): return {resolve(s.get('normalized_name') or f"{s.get('first_name','')} {s.get('last_name','')}") for s in r.get('scientists',[])}-{None}

# ------------------------------------------------------------------ MASTER TABLE
rows=[]
for c,r in nat.items():
    ans=ppl(r); offset=ppl(off.get(c,{}))
    first={resolve(n) for n in qn.get(c,{}).get('first',[])}-{None}
    allq =first | ({resolve(n) for n in qn.get(c,{}).get('later',[])}-{None})
    qb=r.get('query_batches') or ([r['first_batch']] if r.get('first_batch') else [])
    btok=[set((" "+nrm(" ".join(b))+" ").split()) for b in qb] if qb else []
    for k in allq:
        fb=next((i for i,t in enumerate(btok) if k.split('|')[0] in t),None)   # localize batch (finer split only)
        if fb is None: fb=0 if k in first else 1
        rows.append({'concept':c,'mkey':k,'g':int(k in first),'first_batch':min(fb+1,4),
                     'y':int(k in ans),'offrec':int(k in offset),
                     'recog':(1 if bare[k] else 0) if k in bare else np.nan,
                     'log_cit':lc(k),'log_pub':lp(k)})
D=pd.DataFrame(rows)
# identity-based y correction: a malformed recommended name can split one person into two keys
# (different first initial), leaving some in-answer names flagged y=0. Take the researcher-id-verified
# y from the relevance units (units_researcher_ids.csv) when available.
try:
    _u = pd.read_csv(f"{OUT}/relevance/units_researcher_ids.csv")[["concept","mkey","y"]].rename(columns={"y":"_yfix"})
    D = D.merge(_u, on=["concept","mkey"], how="left")
    _n = int(((D["_yfix"]==1) & (D["y"]==0)).sum())
    D["y"] = D["_yfix"].where(D["_yfix"].notna(), D["y"]).astype(int); D = D.drop(columns="_yfix")
    print(f"[y-fix] identity-corrected in-answer flags: +{_n} units")
except FileNotFoundError:
    print("[y-fix] units_researcher_ids.csv not found; skipping identity y-correction")
D.to_csv(f"{OUT}/online_search_master.csv",index=False)
print(f"[master] units={len(D)}  first(g=1)={int(D.g.sum())}  later_only={int((1-D.g).sum())}  concepts={D.concept.nunique()}")
