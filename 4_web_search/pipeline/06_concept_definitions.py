"""Step 6. A 2-3 sentence, person-free definition per concept: the relevance anchor is
"<concept>. <definition>". (original: control_relevance/gen_definitions.py)

GPT-5.4, OpenAI direct (as step 1), default sampling. Resumable. Requires OPENAI_API_KEY.
Reads : inputs/network_concepts_200.json
Writes: $FIG4_WORK/concept_definitions.json
"""
import json, os, time
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.environ.get("FIG4_WORK", os.path.join(HERE, "work"))
MODEL = "gpt-5.4"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
concepts = json.load(open(os.path.join(HERE, "inputs", "network_concepts_200.json")))
SYSTEM = "You are a network-science expert. Define concepts precisely, concisely, and neutrally."
PROMPT = ("Define the network-science concept \"{c}\" in 2-3 sentences: what it is and what it "
          "captures or measures, in neutral encyclopedic style. Do NOT mention any people, authors, "
          "institutions, or references.")

outp = f"{C}/concept_definitions.json"
defs = json.load(open(outp)) if os.path.exists(outp) else {}     # resume-safe
for i, c in enumerate(concepts):
    if defs.get(c): continue
    try:
        r = client.chat.completions.create(model=MODEL, messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(c=c)}])
        defs[c] = r.choices[0].message.content.strip().replace("\n", " ")
    except Exception as e:
        print("  err", c, repr(e)[:120]); time.sleep(3); continue
    if i % 10 == 0:
        json.dump(defs, open(outp, "w"), indent=1, ensure_ascii=False); print(f"  {i+1}/{len(concepts)}")
json.dump(defs, open(outp, "w"), indent=1, ensure_ascii=False)
print("saved", outp, "-", len(defs), "definitions")
