#!/usr/bin/env python
"""Step 4 (GPU). Attribute probes on OLMo-2-1124-7B-Instruct for the 9,270 wikidata9k physicists (Fig. 3j,k).

From the project's probe_olmo.py (hallucination_test), as run; exploratory prompt variants removed. Two runs:

    python 04_attribute_probes.py --mode pphys                    # p(" physicist") after "{name} is by profession a"
    python 04_attribute_probes.py --mode json_year_gender --enum-last --batch-size 48 --checkpoint-every 1000 --resume

pphys recomputes the probability arm of the `combined` recognition label (the traces' base_score, bf16).
json_year_gender asks, in one chat turn with greedy decoding, for a JSON object with field, country, career stage,
first-publication year, gender, most-cited work, a first/last-author work and a frequent coauthor; --enum-last puts
Physics last in the field options (with Physics first the model answers Physics for nearly every name, real or
fabricated). ~17 min on one RTX Pro 6000.

In : cohort_wikidata9k.csv (cohort table with ground truth)
Out: pphys_olmo7b.csv, probes_olmo7b_json_year_gender_enumlast.csv
"""
import os, re, sys, json, time, argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "work")

MODEL_KEY   = "allenai/OLMo-2-1124-7B-Instruct"
PHYS_TOKEN  = " physicist"
PROF_PROMPT = "{} is by profession a"

SYSTEM = "Answer with JSON only. Use \"Unknown\" if you do not know. Do not speculate."
JSON_USER = (
    'Scientist: {name}\n'
    'Reply with this JSON (use "Unknown" if unsure):\n'
    '{{"field":"Physics|Biology|Chemistry|Mathematics|Computer Science|Materials Science|Unknown",'
    '"country":"country or Unknown",'
    '"career_stage":"early-career (0-10 yrs since first publication) | mid-career (11-30) | late-career (31+) | Unknown",'
    '"decade":"decade like 1990s or Unknown",'
    '"top_work":"title of most cited paper or Unknown",'
    '"lead_work":"title where they are first/last author or Unknown",'
    '"coauthor":"frequent collaborator name or Unknown"}}'
)

PROBE_KEYS = ["field", "country", "career_stage", "decade",
              "top_work", "lead_work", "coauthor"]

# Variant of the JSON schema that asks for the exact first-publication YEAR
# instead of the decade. The decade probe is degenerate on this cohort -- OLMo
# answers "2010s" for ~85% of everyone -- so it measures the age distribution,
# not knowledge. Asking for a year gives a graded target (exact / +-5 y / MAE).
PROBE_KEYS_YEAR = ["field", "country", "career_stage", "first_pub_year",
                   "top_work", "lead_work", "coauthor"]

# The field enum lists Physics first, and this cohort is 100% Physical Sciences,
# so "Physics" is both the default answer and the right one -- the probe scores
# 96% for people the model calls a lawyer under the recognition prompt. Moving
# Physics to the end of the enum removes the position cue; anything left is the
# "Scientist:" header plus whatever the model actually knows.
ENUM_PHYSICS_FIRST = ("Physics|Biology|Chemistry|Mathematics|Computer Science|"
                      "Materials Science|Unknown")
ENUM_PHYSICS_LAST = ("Biology|Chemistry|Mathematics|Computer Science|"
                     "Materials Science|Physics|Unknown")

JSON_USER_YEAR = JSON_USER.replace(
    '"decade":"decade like 1990s or Unknown",',
    '"first_pub_year":"year of first publication like 1994 or Unknown",')

# Gender in the slot the date probes occupied. Both date framings (decade,
# absolute year, career age) turned out to measure the age distribution rather
# than knowledge, so the slot is better spent on an attribute the study already
# cares about. Ground truth is Wikidata `gender_any` (74.4% of the cohort).
# Caveat baked into the analysis: the female share is 4.3% among recognized and
# 19.2% among unrecognized, so raw accuracy rewards "always male" far more for
# the recognized group -- balanced accuracy is reported alongside.
#
# Keeping the date field is not cosmetic: work-probe refusal is 17.7% when the
# schema asks for a year, 48.0% for a decade and 95.3% when no date field is
# present at all. The whole JSON is one autoregressive sequence, so committing
# to a concrete number earlier keeps the model committing later. Gender is
# added rather than swapped in.
PROBE_KEYS_YEAR_GENDER = ["field", "country", "career_stage", "first_pub_year",
                          "gender", "top_work", "lead_work", "coauthor"]
JSON_USER_YEAR_GENDER = JSON_USER_YEAR.replace(
    '"top_work"', '"gender":"male or female or Unknown", "top_work"')

# ------------------------------------------------------------------ parsing
def parse_json_response(txt, keys=None):
    """Tolerant JSON extraction (same ladder as the GPT-5.4 runner)."""
    keys = keys or PROBE_KEYS
    out = {k: None for k in keys}
    if not isinstance(txt, str) or not txt.strip():
        return out
    s = txt.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    cand = []
    depth, start = 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                cand.append(s[start:i + 1])
                start = None
    for c in [s] + cand:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                for k in keys:
                    if k in d and d[k] is not None:
                        out[k] = str(d[k])
                if any(v is not None for v in out.values()):
                    return out
        except Exception:
            pass
    for k in keys:                             # last resort: line regex
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', s)
        if m:
            out[k] = m.group(1)
    return out


# --------------------------------------------------------------- generation
def build_chat(tok, system, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(chunk)}/{len(prompts)}", flush=True)
    return outs


@torch.no_grad()
def p_physicist(model, tok, names, batch_size=32):
    """p(' physicist') as the next token after '{name} is by profession a'."""
    tid = tok.encode(PHYS_TOKEN, add_special_tokens=False)
    assert len(tid) == 1, f"'{PHYS_TOKEN}' is not one token: {tid}"
    tid = tid[0]
    ps, top1 = [], []
    for i in range(0, len(names), batch_size):
        chunk = [PROF_PROMPT.format(n) for n in names[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        prob = torch.softmax(logits, dim=-1)
        ps.extend(prob[:, tid].tolist())
        top1.extend(tok.batch_decode(prob.argmax(-1, keepdim=True)))
    return ps, top1, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["json_year_gender", "pphys"],
                    default="json_year_gender")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--enum-last", action="store_true",
                    help="reorder the field enum so Physics comes last")
    ap.add_argument("--resume", action="store_true",
                    help="skip names already present in the output CSV")
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="rewrite the output CSV every N names")
    ap.add_argument("--data-dir", default=DATA)
    args = ap.parse_args()
    data = args.data_dir

    coh = pd.read_csv(os.path.join(data, "cohort_wikidata9k.csv"))
    names = coh["wikidata_name"].astype(str).tolist()
    if args.limit:
        # balanced smoke test: half top-1 physicist, half not
        rec = coh[coh["physicist_top1"]].head(args.limit // 2)["wikidata_name"].tolist()
        unr = coh[~coh["physicist_top1"]].head(args.limit - len(rec))["wikidata_name"].tolist()
        names = rec + unr
    if args.nshards > 1:
        names = names[args.shard::args.nshards]
    print(f"mode={args.mode}  names={len(names)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_KEY, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_KEY, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"loaded {MODEL_KEY} on {model.device}", flush=True)

    sfx = args.tag or ("" if args.nshards == 1 else f"_shard{args.shard}")
    t0 = time.time()

    if args.mode == "pphys":
        ps, top1, tid = p_physicist(model, tok, names, args.batch_size)
        out = pd.DataFrame({"wikidata_name": names, "p_physicist": ps,
                            "next_top1": top1})
        path = os.path.join(data, f"pphys_olmo7b{sfx}.csv")
        out.to_csv(path, index=False)
        print(f"token id for '{PHYS_TOKEN}' = {tid}")
        print(out["p_physicist"].describe())
    else:                                      # json_year_gender
        keys = PROBE_KEYS_YEAR_GENDER
        tmpl = JSON_USER_YEAR_GENDER
        if args.enum_last:
            assert ENUM_PHYSICS_FIRST in tmpl
            tmpl = tmpl.replace(ENUM_PHYSICS_FIRST, ENUM_PHYSICS_LAST)
        mx = args.max_new_tokens or 300
        path = os.path.join(data, f"probes_olmo7b_{args.mode}"
                                  f"{'_enumlast' if args.enum_last else ''}{sfx}.csv")
        rows, done = [], set()
        if args.resume and os.path.isfile(path):
            prev = pd.read_csv(path)
            rows = prev.to_dict("records")
            done = set(prev["wikidata_name"].astype(str))
            print(f"resuming: {len(done)} names already in {os.path.basename(path)}",
                  flush=True)
        todo = [n for n in names if n not in done]
        step = max(args.checkpoint_every, args.batch_size)
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            raw = generate(model, tok, [build_chat(tok, SYSTEM, tmpl.format(name=n))
                                        for n in chunk], mx, args.batch_size)
            for n, r in zip(chunk, raw):
                d = parse_json_response(r, keys)
                rows.append({"wikidata_name": n, "olmo_probe_raw": r,
                             **{f"olmo_probe_{k}": d[k] for k in keys}})
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  [checkpoint] {len(rows)}/{len(names)} -> "
                  f"{os.path.basename(path)}  ({time.time() - t0:.0f}s)", flush=True)
        out = pd.DataFrame(rows)
        got = {k: int(out[f"olmo_probe_{k}"].notna().sum()) for k in keys}
        print("parsed non-null:", got)

    print(f"wrote {path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
