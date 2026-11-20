"""Step 2. Person names literally present in the logged web-search queries, split into the FIRST
batch (pre-retrieval = Step A) and LATER batches (result-informed). (original: extract_query_names_v2.py)

GPT-5.4 via OpenRouter (`openai/gpt-5.4`), temperature 0, max_tokens 700, up to 4 attempts.
Strict: names must be written in the queries; eponyms count. Resumable.

Requires OPENROUTER_API_KEY.
Reads : $FIG4_WORK/recommend_native_v2.json
Writes: $FIG4_WORK/query_names_v2.json = {concept: {first:[names], later:[names]}}
"""
import os,re,json,time,unicodedata,pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
client=OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],base_url="https://openrouter.ai/api/v1")
MODEL="openai/gpt-5.4"; C=os.environ.get("FIG4_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
SYS="You extract person names literally present in text. Do not infer names that are not written."
nat={r['concept']:r for r in json.load(open(f"{C}/recommend_native_v2.json"))}
def batches(r):
    qb=r.get('query_batches') or []
    if qb: first=qb[0]; later=[q for b in qb[1:] for q in b]
    else: first=r.get('first_batch') or r.get('queries',[]); later=[]
    return first,later
path=f"{C}/query_names_v2.json"
out=json.load(open(path)) if os.path.exists(path) else {}
todo=[c for c in nat if c not in out]
print(f"concepts {len(nat)} | done {len(out)} | todo {len(todo)}",flush=True)
def extract(c):
    r=nat[c]; first,later=batches(r)
    if not first and not later: return c,{'first':[],'later':[]}
    p=(f'Web-search queries a system issued about the topic "{c}" (network science).\n'
       f'FIRST-BATCH queries (issued BEFORE seeing any results):\n{chr(10).join("- "+q for q in first) or "(none)"}\n\n'
       f'LATER queries (issued AFTER seeing results):\n{chr(10).join("- "+q for q in later) or "(none)"}\n\n'
       'List the PERSON names (scientists/authors) LITERALLY present in each group. Count eponyms '
       '(e.g. "Barabasi-Albert model" -> Barabasi, Albert; "Girvan-Newman" -> Girvan, Newman). '
       'Do NOT invent names that are not written. Return ONLY JSON: '
       '{"first": ["Full Name", ...], "later": ["Full Name", ...]}.')
    for a in range(4):
        try:
            rr=client.chat.completions.create(model=MODEL,temperature=0.0,max_tokens=700,
                messages=[{"role":"system","content":SYS},{"role":"user","content":p}])
            t=rr.choices[0].message.content or ''; m=re.search(r"\{.*\}",t,re.DOTALL)
            d=json.loads(m.group(0) if m else t)
            return c,{'first':[str(x) for x in (d.get('first') or [])],'later':[str(x) for x in (d.get('later') or [])]}
        except Exception: time.sleep(a+1)
    return c,{'first':[],'later':[],'error':1}
with ThreadPoolExecutor(max_workers=8) as ex:
    for i,fut in enumerate(as_completed([ex.submit(extract,c) for c in todo])):
        c,d=fut.result(); out[c]=d
        if (i+1)%40==0: json.dump(out,open(path,'w'),ensure_ascii=False,indent=1); print(f"  {i+1}/{len(todo)}",flush=True)
json.dump(out,open(path,'w'),ensure_ascii=False,indent=1)
nf=sum(len(v.get('first',[])) for v in out.values()); nl=sum(len(v.get('later',[])) for v in out.values())
print(f"saved query_names_v2.json | first-name mentions {nf} | later-name mentions {nl}",flush=True)
