"""Shared resolver (imported by steps 03, 04, 05): map an extracted query name (often surname-only)
to a canonical recommended-person key `surname|first-initial` (mkey): same surname, compatible first
name (full==full, else initial; falls back to all same-surname keys if none is compatible), ties broken
by most-cited (cites from the Dimensions name match), then by key. A searched name whose surname matches nobody recommended
resolves to `surname|~` (searched but never recommended).

Reads: $FIG4_WORK/analysis/author_citations_curated_v2.csv, $FIG4_WORK/recommend_{offline,native}_v2.json
"""
import json,os,re,unicodedata,pandas as pd
from collections import defaultdict
C=os.environ.get("FIG4_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
def nrm(n):
    s=unicodedata.normalize("NFKD",str(n));s="".join(c for c in s if not unicodedata.combining(c)).lower();return re.sub(r"\s+"," ",re.sub(r"[^a-z\s]","",s)).strip()
def mk(n):
    t=nrm(n).split();return f"{t[-1]}|{t[0][0] if t[0] else ''}" if t else None
def fok(nf,af):
    if not nf or not af: return True
    return (nf==af) if (len(nf)>1 and len(af)>1) else (nf[0]==af[0])
def build():
    cur=pd.read_csv(f"{C}/analysis/author_citations_curated_v2.csv"); cites=dict(zip(cur.mkey,cur.cites.fillna(0)))
    ks=lambda r:[(mk(s.get('normalized_name') or f"{s.get('first_name','')} {s.get('last_name','')}"),
                  (nrm(s.get('first_name','')).split()[0] if nrm(s.get('first_name','')) else
                   (nrm(s.get('normalized_name','')).split()[0] if nrm(s.get('normalized_name','')) else '')))
                 for s in r.get('scientists',[])]
    OFF=set();ON=set();first_of={};sur2=defaultdict(set)
    for r in json.load(open(f"{C}/recommend_offline_v2.json")):
        for k,f in ks(r):
            if k: OFF.add(k); first_of.setdefault(k,f); sur2[k.split('|')[0]].add(k)
    for r in json.load(open(f"{C}/recommend_native_v2.json")):
        for k,f in ks(r):
            if k: ON.add(k); first_of.setdefault(k,f); sur2[k.split('|')[0]].add(k)
    def resolve(name):
        t=nrm(name).split()
        if not t: return None
        sur=t[-1]; first=(t[0] if len(t)>1 else '')   # surname-only -> first unknown
        cands=list(sur2.get(sur,[]))
        if not cands: return f"{sur}|~"                # novel: searched but not recommended
        if first: comp=[k for k in cands if fok(first,first_of.get(k,''))] or cands
        else: comp=cands
        return max(comp,key=lambda k:(cites.get(k,0) if pd.notna(cites.get(k,0)) else 0, k))  # deterministic tiebreak
    return OFF,ON,resolve
