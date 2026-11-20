"""Step 5. Why a final-answer pair has no recorded query: label each `untraced` pair of step 4.
(original: inline session code, verbatim apart from paths and removed print-outs)

why = 'surname in query (match miss)' if the surname occurs as a token anywhere in the concept's
normalized query text, else 'in closed-book list' if in the closed-book recommendation, else
'known offline' / 'unknown offline' (bare recognition; both become `neither` in the export).
The export checks closed-book membership first: in_offline -> surname_in_query -> neither (28/10/63%).

Reads : $FIG4_WORK/recommend_native_v2.json, analysis/final_pairs_provenance.csv
Writes: $FIG4_WORK/analysis/untraced_final_pairs.csv
"""
import json, os, re, unicodedata, pandas as pd, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from query_resolver_v2 import build
OFF,ON,resolve=build()
def nrm(n):
    s=unicodedata.normalize('NFKD',str(n));s=''.join(c for c in s if not unicodedata.combining(c)).lower();return re.sub(r'\s+',' ',re.sub(r'[^a-z\s]','',s)).strip()
C=os.environ.get("FIG4_WORK", os.path.join(HERE, "work"))
nat={r['concept']:r for r in json.load(open(f'{C}/recommend_native_v2.json'))}
full={}
for c,r in nat.items():
    for s in r.get('scientists',[]):
        k=resolve(s.get('normalized_name') or f"{s.get('first_name','')} {s.get('last_name','')}")
        if k and (c,k) not in full: full[(c,k)]=f"{s.get('first_name','')} {s.get('last_name','')}".strip()
raw={c:' '+nrm(' '.join(' '.join(b) if isinstance(b,list) else str(b) for b in (r.get('query_batches') or [])))+' ' for c,r in nat.items()}
F=pd.read_csv(f'{C}/analysis/final_pairs_provenance.csv'); U=F[F.grp=='untraced'].copy()
U['name']=U.apply(lambda r: full.get((r.concept,r.mkey),r.mkey),axis=1)
U['why']=U.apply(lambda r:'surname in query (match miss)' if (' '+r.mkey.split('|')[0]+' ') in raw[r.concept] else ('in closed-book list' if r.offline==1 else ('known offline' if r.recog==True else 'unknown offline')),axis=1)
U[['concept','name','rank','n_final','cites','offline','recog','why']].sort_values(['concept','rank']).to_csv(f'{C}/analysis/untraced_final_pairs.csv',index=False)
