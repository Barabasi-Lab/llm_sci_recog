"""Step 1. Concept recommendations from GPT-5.4, closed-book and with native web search.
(original: recommend_runner_v2.py)

Standardized output per scientist: {first_name, last_name, normalized_name, contribution,
works:[{title, year, doi}]}. Offline = chat completions, no tools. Online = Responses API with the
native `web_search` tool (falls back to `web_search_preview`); every web_search_call is logged as one
query batch (`query_batches`; `first_batch` = issued before any result returns = Step A).
OpenAI direct, not OpenRouter: only the native tool exposes the model's own queries.
Default sampling parameters (no temperature set). Resumable per concept.

Requires OPENAI_API_KEY.
Usage:  python 01_recommend.py offline | online      (test = one concept, printed only)
Reads : inputs/network_concepts_200.json
Writes: $FIG4_WORK/recommend_offline_v2.json, $FIG4_WORK/recommend_native_v2.json
"""
import os, re, json, sys, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL="gpt-5.4"; CDIR=os.environ.get("FIG4_WORK", os.path.join(HERE, "work"))
CONCEPTS=os.path.join(HERE, "inputs", "network_concepts_200.json")
SYSTEM=("You are a knowledgeable assistant on the history and methods of network science. "
        "Answer concisely and honestly. Do not speculate.")
def prompt(concept, web=False):
    verify = " Use web search to identify and verify them; give a DOI for each work when possible." if web else ""
    return (f'Who are the key scientists and what are the seminal works behind the concept '
            f'"{concept}" in network science?{verify}\n\n'
            'Return ONLY a JSON array of up to 10 elements, most important first, each:\n'
            '{"first_name": "", "last_name": "", "contribution": "<one short line>", '
            '"works": [{"title": "", "year": null, "doi": null}]}')
def norm(n):
    s=unicodedata.normalize("NFKD",str(n)); s="".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+"," ",re.sub(r"[^a-z\s]","",s)).strip()
def parse(text):
    if not text or str(text).startswith("ERROR"): return []
    m=re.search(r"\[.*\]",text,re.DOTALL)
    try: arr=json.loads(m.group(0) if m else text)
    except Exception: return []
    out=[]
    for e in arr:
        if not isinstance(e,dict): continue
        fn=(e.get("first_name") or "").strip(); ln=(e.get("last_name") or "").strip()
        full=f"{fn} {ln}".strip() or (e.get("name") or "").strip()
        works=[]
        for w in (e.get("works") or []):
            if isinstance(w,dict):
                works.append({"title":(w.get("title") or "").strip(),
                              "year":w.get("year"),"doi":(w.get("doi") or None)})
        out.append({"first_name":fn,"last_name":ln,"normalized_name":norm(full),
                    "contribution":(e.get("contribution") or "").strip(),"works":works})
    return out
def recommend_offline(concept):
    r=client.chat.completions.create(model=MODEL,messages=[{"role":"system","content":SYSTEM},
        {"role":"user","content":prompt(concept,web=False)}])
    t=r.choices[0].message.content
    return {"concept":concept,"mode":"offline","answer":t,"scientists":parse(t)}
def _queries(a):
    qs=getattr(a,"queries",None) or (a.get("queries") if isinstance(a,dict) else None)
    if qs is None:
        q=getattr(a,"query",None) or (a.get("query") if isinstance(a,dict) else None); qs=[q] if q else []
    return [q for q in qs if q]
def recommend_native(concept):
    for tool in ("web_search","web_search_preview"):
        try:
            resp=client.responses.create(model=MODEL,tools=[{"type":tool}],
                input=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt(concept,web=True)}]); break
        except Exception:
            if tool=="web_search_preview": raise
    batches,cites=[],[]
    for it in (resp.output or []):
        ty=getattr(it,"type",None)
        if ty in ("web_search_call","web_search_preview_call"):
            qs=_queries(getattr(it,"action",None)); batches.append(qs) if qs else None
        elif ty=="message":
            for blk in (getattr(it,"content",None) or []):
                for ann in (getattr(blk,"annotations",None) or []):
                    u=getattr(ann,"url",None) or (ann.get("url") if isinstance(ann,dict) else None)
                    if u: cites.append(u)
    q=[x for b in batches for x in b]; t=getattr(resp,"output_text","") or ""
    return {"concept":concept,"mode":"native","answer":t,"scientists":parse(t),
            "queries":q,"query_batches":batches,"first_batch":batches[0] if batches else [],
            "first_query":q[0] if q else None,"citations":cites}
def run(mode):
    concepts=json.load(open(CONCEPTS))
    fn=recommend_offline if mode=="offline" else recommend_native
    os.makedirs(CDIR, exist_ok=True)
    path=f"{CDIR}/recommend_{'offline' if mode=='offline' else 'native'}_v2.json"
    rec={r["concept"]:r for r in json.load(open(path))} if os.path.exists(path) else {}
    todo=[c for c in concepts if c not in rec]; print(f"[{mode}] {len(todo)} to run")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(fn,c):c for c in todo}
        for fut in as_completed(futs):
            c=futs[fut]
            try: rec[c]=fut.result()
            except Exception as e: rec[c]={"concept":c,"answer":f"ERROR:{e}","scientists":[]}
            json.dump(list(rec.values()),open(path,"w"),ensure_ascii=False,indent=1)
    print(f"saved -> {path}")
if __name__=="__main__":
    cmd=sys.argv[1] if len(sys.argv)>1 else "test"
    if cmd=="test":
        for m,fn in [("OFFLINE",recommend_offline),("ONLINE",recommend_native)]:
            r=fn("scale-free network"); print(f"\n===== {m} — standardized output (first 3) =====")
            if m=="ONLINE": print("first_batch queries:",r.get("first_batch"))
            print(json.dumps(r["scientists"][:3],ensure_ascii=False,indent=1))
    else: run(cmd)
