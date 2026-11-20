"""Step 7. Topic relevance of each unit's researcher to the concept (SPECTER2).
(original: control_relevance/embed_dim_relevance.py)

`allenai/specter2_base` + `allenai/specter2` proximity adapter, CLS token, L2-normalized, cosine.
Papers embedded as "title. abstract" (each researcher's top-100 cited papers with an abstract, max 512
tokens), once, cached to pubvec.npy. Relevance = mean cosine of the 3 papers CLOSEST to the anchor.
Two anchors: bare concept term (rel_top3_bare) and "concept. definition" (rel_top3_def; used by Fig. 4).
Runs on MPS if available, else CPU. Downloads the model weights from Hugging Face.

Reads : $FIG4_WORK/analysis/relevance/{dim_pubs.parquet, units_researcher_ids.csv}, concept_definitions.json
Writes: $FIG4_WORK/analysis/relevance/author_relevance_dim.csv (+ pubvec.npy cache)
"""
import pandas as pd, numpy as np, json, os, time
C = os.environ.get("FIG4_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")); OUT = f"{C}/analysis/relevance"
pubs = pd.read_parquet(f"{OUT}/dim_pubs.parquet").reset_index(drop=True)
units = pd.read_csv(f"{OUT}/units_researcher_ids.csv")
defs = json.load(open(f"{C}/concept_definitions.json")) if os.path.exists(f"{C}/concept_definitions.json") else {}
concepts = sorted(units.concept.unique())
print(f"{len(pubs)} pubs, {len(units)} units, {len(concepts)} concepts; definitions loaded: {len(defs)}")

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
model.to(dev).eval()
def embed(texts, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            inp = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            e = model(**inp).last_hidden_state[:, 0, :]
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
            if i % 3200 == 0: print(f"  embed {i}/{len(texts)}", flush=True)
    return np.vstack(out).astype("float32") if out else np.zeros((0, 768), "float32")

cache = f"{OUT}/pubvec.npy"
if os.path.exists(cache) and np.load(cache).shape[0] == len(pubs):
    pubvec = np.load(cache); print("loaded cached pubvec")
else:
    t = time.time(); pubvec = embed([f"{r.title}. {r.abstract}" for r in pubs.itertuples()])
    np.save(cache, pubvec); print("pub embed %.0fs -> cached" % (time.time()-t))

cbare = dict(zip(concepts, embed(concepts)))
cdef  = dict(zip(concepts, embed([f"{c}. {defs.get(c,'')}".strip() for c in concepts]))) if defs else cbare
idx_by_r = pubs.groupby("researcher_id").indices

def top3(vec_rows, anchor):
    s = vec_rows @ anchor; return float(np.mean(np.sort(s)[::-1][:3])), s
rows = []
for u in units.itertuples():
    r = {"concept": u.concept, "mkey": u.mkey, "name": u.name, "g": u.g, "y": u.y,
         "researcher_id": u.researcher_id, "n_pubs": 0, "rel_top3_bare": np.nan, "rel_top3_def": np.nan}
    for j in range(1, 4): r[f"top{j}_title"], r[f"top{j}_sim"], r[f"top{j}_year"] = "", np.nan, ""
    idx = idx_by_r.get(u.researcher_id) if pd.notna(u.researcher_id) else None
    if idx is not None and len(idx):
        V = pubvec[idx]; r["n_pubs"] = len(idx)
        r["rel_top3_bare"], _ = top3(V, cbare[u.concept])
        rd, sdef = top3(V, cdef[u.concept]); r["rel_top3_def"] = rd
        order = idx[np.argsort(-sdef)[:3]]
        for j, gi in enumerate(order, 1):
            r[f"top{j}_title"] = pubs.title.iloc[gi]; r[f"top{j}_sim"] = round(float(pubvec[gi] @ cdef[u.concept]), 3); r[f"top{j}_year"] = pubs.year.iloc[gi]
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/author_relevance_dim.csv", index=False)

RELCOL = "rel_top3_def" if defs else "rel_top3_bare"
d = df.dropna(subset=[RELCOL])
print("\nsaved author_relevance_dim.csv | scored", len(d), "/", len(df))
print("mean by group (0=web,1=parametric):  bare", df.groupby("g")["rel_top3_bare"].mean().round(3).to_dict(),
      " def", df.groupby("g")["rel_top3_def"].mean().round(3).to_dict())
if defs: print("corr(bare,def) = %.3f" % df[["rel_top3_bare","rel_top3_def"]].corr().iloc[0,1])
for col in (["rel_top3_bare","rel_top3_def"] if defs else ["rel_top3_bare"]):
    dd = df.dropna(subset=[col])
    print(f"  [{col}] dropped-web(g0,y0)={dd[(dd.g==0)&(dd.y==0)][col].mean():.3f} (n={len(dd[(dd.g==0)&(dd.y==0)])})  "
          f"first-in-answer(g1,y1)={dd[(dd.g==1)&(dd.y==1)][col].mean():.3f}  "
          f"web-in-answer(g0,y1)={dd[(dd.g==0)&(dd.y==1)][col].mean():.3f}")
