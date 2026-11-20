#!/usr/bin/env python
"""Step 5 (CPU). Score the attribute probes against Dimensions/Wikidata ground truth (Fig. 3j,k).

From the project's score_probes.py (hallucination_test), as run:
    python 05_score_probes.py                    # = --mode json_year_gender_enumlast, combined label

Recognized (`combined`): physicist_top1 OR (physicist_top5 AND p_physicist > 0.1) -> 1,994 / 7,276.
Rules: a refusal ("Unknown", empty, ...) scores 0. Field: answer mapped to a Dimensions top-level field, must equal
the modal field (most_frequent_category). Country: current affiliation country only (names and synonyms via
pycountry). Career stage: early 0-10 / mid 11-30 / late 31+ years since first publication (as of 2026). Gender:
Wikidata gender_any (missing -> 0). First publication year: exact. Most-cited and first/last-author work:
rapidfuzz partial_ratio >= 60 against any of the top-3 titles. Coauthor: surname of any of the top-5 coauthors
appears in the answer.

In : cohort_wikidata9k.csv, gt_enrichment_wikidata9k.csv (Dimensions ground truth),
     pphys_olmo7b.csv and probes_olmo7b_json_year_gender_enumlast.csv (step 4)
Out: scored_olmo7b_json_year_gender_enumlast.csv   per-name 0/1 scores (-> data/attribute_probes_wikidata9k.csv)
     accuracy_by_recognition_*.csv                  per-probe accuracy + Wilson CI
     behavior_by_recognition_*.csv                  correct / wrong / refused counts (SI table)
"""
import os, re, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "work")
from rapidfuzz import fuzz                                    # noqa: E402
import pycountry                                              # noqa: E402

PROBE_KEYS = ["field", "country", "career_stage", "decade",
              "top_work", "lead_work", "coauthor"]
PROBE_KEYS_YEAR_GENDER = ["field", "country", "career_stage", "first_pub_year",
                          "gender", "top_work", "lead_work", "coauthor"]
YEAR_TOL = 0        # exact year required; --year-tol relaxes it as a check
CURRENT_YEAR = 2026
FUZZ_MIN = 60
REFUSAL_RE = re.compile(r"^\s*(unknown|i (do not|don't) know|n/?a|none|not sure|"
                        r"no (information|data)|unclear|cannot|can't)\b", re.I)

COUNTRY_GT_SETS = {
    "current": ("current_research_org_country_code",),
    "union":   ("current_research_org_country_code", "first_affiliation_country",
                "country_any"),
}
COUNTRY_GT_COLS = COUNTRY_GT_SETS["current"]

COUNTRY_SYNONYMS = {
    "US": ["united states", "usa", "u.s.", "america", "american"],
    "GB": ["united kingdom", "uk", "britain", "great britain", "england",
           "scotland", "wales", "british"],
    "CN": ["china", "prc", "people's republic of china", "chinese"],
    "KR": ["south korea", "korea", "republic of korea"],
    "RU": ["russia", "russian federation", "soviet union", "ussr"],
    "DE": ["germany", "german", "west germany"],
    "NL": ["netherlands", "holland", "the netherlands"],
    "CZ": ["czech republic", "czechia"],
    "IR": ["iran", "islamic republic of iran"],
    "TW": ["taiwan"],
    "CH": ["switzerland", "swiss"],
    "IT": ["italy", "italian"],
    "FR": ["france", "french"],
    "JP": ["japan", "japanese"],
    "IN": ["india", "indian"],
    "ES": ["spain", "spanish"],
    "CA": ["canada", "canadian"],
    "AU": ["australia", "australian"],
    "BR": ["brazil", "brazilian"],
    "IL": ["israel", "israeli"],
    "SE": ["sweden", "swedish"],
    "AT": ["austria", "austrian"],
    "BE": ["belgium", "belgian"],
    "PL": ["poland", "polish"],
    "DK": ["denmark", "danish"],
    "NO": ["norway", "norwegian"],
    "FI": ["finland", "finnish"],
    "SG": ["singapore"],
    "MX": ["mexico", "mexican"],
    "PT": ["portugal", "portuguese"],
    "GR": ["greece", "greek"],
    "HU": ["hungary", "hungarian"],
    "TR": ["turkey", "türkiye", "turkish"],
    "ZA": ["south africa"],
    "AR": ["argentina", "argentinian"],
    "CL": ["chile", "chilean"],
    "NZ": ["new zealand"],
    "IE": ["ireland", "irish"],
    "HK": ["hong kong"],
    "SA": ["saudi arabia", "saudi"],
    "AE": ["united arab emirates", "uae"],
}


def country_strings(code_or_name):
    """All accepted surface forms for one country code or name."""
    if not isinstance(code_or_name, str) or not code_or_name.strip():
        return set()
    v = code_or_name.strip()
    out = set()
    rec = None
    if len(v) == 2:
        rec = pycountry.countries.get(alpha_2=v.upper())
        out |= set(COUNTRY_SYNONYMS.get(v.upper(), []))
    else:
        try:
            rec = pycountry.countries.lookup(v)
        except LookupError:
            rec = None
        out.add(v.lower())
    if rec is not None:
        for attr in ("name", "official_name", "common_name"):
            nm = getattr(rec, attr, None)
            if nm:
                out.add(nm.lower())
        out |= set(COUNTRY_SYNONYMS.get(rec.alpha_2, []))
    return {s for s in out if len(s) > 2}


PROBE_TO_FIELD = {
    "physics": "Physical Sciences", "astronomy": "Physical Sciences",
    "astrophysics": "Physical Sciences", "geology": "Physical Sciences",
    "earth sciences": "Physical Sciences", "space science": "Physical Sciences",
    "biology": "Biological Sciences", "biomedicine": "Biological Sciences",
    "medicine": "Biological Sciences", "neuroscience": "Biological Sciences",
    "chemistry": "Chemical Sciences",
    "psychology": "Human Society", "economics": "Human Society",
    "political science": "Human Society", "sociology": "Human Society",
    "philosophy": "Human Society", "history": "Human Society",
}
_MISSING = {"", "unknown", "nan", "none"}


def probe_to_bucket(v):
    """Dimensions top-level field the answer maps to, or None."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in _MISSING:
        return None
    for part in s.split("|"):
        p = part.strip().lower()
        b = PROBE_TO_FIELD.get(p)
        if b:
            return b
        for key, bucket in PROBE_TO_FIELD.items():
            if key in p:
                return bucket
    return None


def is_refusal(ans):
    if not isinstance(ans, str) or not ans.strip():
        return True
    return bool(REFUSAL_RE.match(ans.strip())) or "unknown" == ans.strip().lower()


def titles_from(js, key):
    if not isinstance(js, str) or not js.strip() or js == "nan":
        return []
    try:
        arr = json.loads(js)
    except Exception:
        return []
    return [str(d.get("title") or "") for d in (arr or [])][:3]


def coauthors_from(js):
    if not isinstance(js, str) or not js.strip() or js == "nan":
        return []
    try:
        arr = json.loads(js)
    except Exception:
        return []
    return [(str(d.get("first_name") or ""), str(d.get("last_name") or ""))
            for d in (arr or [])][:5]


def score_row(r, probe):
    ans = r.get(f"olmo_probe_{probe}")
    if is_refusal(ans):
        return 0, "refused"
    a = str(ans).lower()

    if probe == "field":
        # Same bucket rule as the GPT-5.4 panel (plot_hallucination_results.ipynb
        # cell 19): map the answer to a Dimensions top-level field and require it
        # to equal the author's modal category. The cohort is Physical Sciences,
        # so astronomy/astrophysics count and "materials science" does not.
        bucket = probe_to_bucket(ans)
        truth = str(r.get("most_frequent_category") or "Physical Sciences").strip()
        ok = bucket is not None and bucket == truth
    elif probe == "country":
        # Default is the CURRENT affiliation country only. The GPT-5.4 rule (a
        # union of current org + first affiliation + Wikidata nationality) lets
        # a nationality guessed from the name count as correct, and famous
        # scientists have that column filled far more often (38.5% vs 26.2%) and
        # move country more, so the union hands the recognized group 18.3% of
        # its hits versus 13.2% for the unrecognized -- 5.1 pp of a 6.8 pp gap.
        cols = COUNTRY_GT_COLS
        gt = set()
        for col in cols:
            gt |= country_strings(r.get(col))
        ok = any(g in a for g in gt) if gt else False
    elif probe == "career_stage":
        gt = r.get("gt_career_stage")
        ok = isinstance(gt, str) and str(gt).lower() in a
    elif probe == "first_pub_year":
        y = pd.to_numeric(pd.Series([r.get("first_year_of_publication")]),
                          errors="coerce").iloc[0]
        m = re.search(r"(1[89]\d{2}|20\d{2})", a)
        if pd.isna(y) or m is None:
            ok = False
        else:
            ok = abs(int(m.group(1)) - int(y)) <= YEAR_TOL
    elif probe == "gender":
        g = str(r.get("gender_any") or "").strip().lower()
        if g not in ("male", "female"):
            ok = False                      # no Wikidata gender -> scores 0
        else:
            said_f = "female" in a or "woman" in a or a.strip() in ("f", "she")
            said_m = (not said_f) and ("male" in a or "man" in a
                                       or a.strip() in ("m", "he"))
            ok = (said_f and g == "female") or (said_m and g == "male")
    elif probe in ("top_work", "lead_work"):
        key = "top_works_any" if probe == "top_work" else "top_works_lead"
        truths = titles_from(r.get(key), key)
        if probe == "top_work" and not truths and isinstance(r.get("most_cited_title"), str):
            truths = [r["most_cited_title"]]
        ok = any(fuzz.partial_ratio(a, t.lower()) >= FUZZ_MIN for t in truths if t)
    elif probe == "coauthor":
        pairs = coauthors_from(r.get("top_coauthors"))
        lasts = [ln.lower() for _, ln in pairs if len(ln) > 2]
        if not lasts and isinstance(r.get("top_5_collaborators_names"), str):
            lasts = [n.strip().split()[-1].lower()
                     for n in r["top_5_collaborators_names"].split(",")
                     if len(n.strip().split()) > 1]
            lasts = [x for x in lasts if len(x) > 2]
        ok = any(l in a for l in lasts)
    else:
        raise ValueError(probe)
    return int(ok), ("correct" if ok else "wrong")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0, c - h), min(1, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["json_year_gender_enumlast"],
                    default="json_year_gender_enumlast")
    ap.add_argument("--tag", default="")
    ap.add_argument("--label", choices=["combined", "top1"], default="combined")
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--year-tol", type=int, default=0,
                    help="years of slack on the first-publication-year probe; "
                         "0 = the exact year must match")
    ap.add_argument("--country-gt", choices=["current", "union"], default="current",
                    help="ground truth for the country probe: the current "
                         "affiliation country only (default), or the GPT-5.4 "
                         "union that also accepts first affiliation and "
                         "Wikidata nationality")
    ap.add_argument("--permute", type=int, default=-1,
                    help="seed: score answers against SHUFFLED ground truth, i.e. the "
                         "accuracy a knowledge-free guesser gets from base rates alone")
    ap.add_argument("--data-dir", default=DATA)
    args = ap.parse_args()
    data = args.data_dir

    globals()["COUNTRY_GT_COLS"] = COUNTRY_GT_SETS[args.country_gt]
    globals()["YEAR_TOL"] = args.year_tol
    keys = PROBE_KEYS_YEAR_GENDER
    globals()["PROBE_KEYS"] = keys
    coh = pd.read_csv(os.path.join(data, "cohort_wikidata9k.csv"))
    pr  = pd.read_csv(os.path.join(data, f"probes_olmo7b_{args.mode}{args.tag}.csv"))
    df = coh.merge(pr, on="wikidata_name", how="inner")

    gt_path = os.path.join(data, "gt_enrichment_wikidata9k.csv")
    if os.path.isfile(gt_path):
        gt = pd.read_csv(gt_path)
        df = df.merge(gt, on="author_id", how="left")
        print(f"BQ ground truth merged: "
              f"{ {c: int(df[c].notna().sum()) for c in ['top_works_any','top_works_lead','top_coauthors'] if c in df} }")
    else:
        for c in ("top_works_any", "top_works_lead", "top_coauthors"):
            df[c] = None
        print("!! no BQ enrichment -- falling back to most_cited_title / "
              "top_5_collaborators_names; lead_work cannot be scored")

    pp_path = os.path.join(data, f"pphys_olmo7b{args.tag}.csv")
    if os.path.isfile(pp_path):
        df = df.merge(pd.read_csv(pp_path)[["wikidata_name", "p_physicist"]],
                      on="wikidata_name", how="left")
    else:
        df["p_physicist"] = np.nan

    # ---- recognition label ------------------------------------------------
    if args.label == "combined":
        df["recognized"] = df["physicist_top1"] | (
            df["physicist_top5"] & (df["p_physicist"].fillna(0) > args.p_min))
    elif args.label == "top1":
        df["recognized"] = df["physicist_top1"]
    print("recognized:", df["recognized"].value_counts(dropna=False).to_dict())

    # ---- chance baseline ---------------------------------------------------
    # Biographical probes are guessable in a single-field, US/EU-heavy cohort
    # ("Physics", "United States", "mid-career", "2010s" are right surprisingly
    # often). Scoring the same answers against a shuffled ground truth gives the
    # base-rate accuracy of a model that knows nothing about the individual.
    if args.permute >= 0:
        gt_cols = ["most_frequent_category", "current_research_org_country_code",
                   "first_affiliation_country", "country_any", "gt_career_stage",
                   "gt_decade", "top_works_any", "top_works_lead", "top_coauthors",
                   "most_cited_title", "top_5_collaborators_names"]
        gt_cols = [c for c in gt_cols if c in df.columns]
        perm = df[gt_cols].sample(frac=1.0, random_state=args.permute).reset_index(drop=True)
        df = df.reset_index(drop=True)
        df[gt_cols] = perm
        print(f"[permuted ground truth, seed={args.permute}] -> chance baseline")

    # ---- score ------------------------------------------------------------
    for k in PROBE_KEYS:
        sb = df.apply(lambda r: score_row(r, k), axis=1)
        df[f"score_{k}"]    = [s for s, _ in sb]
        df[f"behavior_{k}"] = [b for _, b in sb]
    df["score_total"] = df[[f"score_{k}" for k in PROBE_KEYS]].sum(axis=1)
    if "gender" in PROBE_KEYS:
        # Raw accuracy is dominated by the base rate: 4.3% of the recognized are
        # women versus 19.2% of the unrecognized, so "always male" alone scores
        # 95.7% vs 80.8%. Balanced accuracy (mean of the male and female rates)
        # is the number to read.
        g = df["gender_any"].astype(str).str.lower()
        lab = df[g.isin(["male", "female"])]
        for grp, l in [(True, "recognized"), (False, "unrecognized")]:
            sub = lab[lab["recognized"] == grp]
            gm = sub["gender_any"].astype(str).str.lower()
            am = sub.loc[gm == "male", "score_gender"].mean()
            af = sub.loc[gm == "female", "score_gender"].mean()
            allrows = df[df["recognized"] == grp]["score_gender"].mean()
            print(f"  gender {l:13s} all rows {100*allrows:5.2f}%   labelled only "
                  f"{100*sub['score_gender'].mean():5.2f}%   male {100*am:5.2f}% "
                  f"(n={int((gm=='male').sum())})   female {100*af:5.2f}% "
                  f"(n={int((gm=='female').sum())})   BALANCED {100*(am+af)/2:5.2f}%")
    if "first_pub_year" in PROBE_KEYS:
        pred = (df["olmo_probe_first_pub_year"].astype(str)
                  .str.extract(r"(1[89]\d{2}|20\d{2})")[0].astype(float))
        truth = pd.to_numeric(df["first_year_of_publication"], errors="coerce")
        df["year_pred"], df["year_err"] = pred, (pred - truth)
        df["score_first_pub_year_exact"] = (pred == truth).astype(int)
        df["score_first_pub_year_pm5"] = ((pred - truth).abs() <= 5).astype(int)
        for grp, lab in [(True, "recognized"), (False, "unrecognized")]:
            sub = df[df["recognized"] == grp]
            e = sub["year_err"].dropna()
            print(f"  year {lab:13s} exact {100*sub['score_first_pub_year_exact'].mean():5.2f}%"
                  f"  +-5y {100*sub['score_first_pub_year_pm5'].mean():5.2f}%"
                  f"  scored(tol={YEAR_TOL}) {100*sub['score_first_pub_year'].mean():5.2f}%"
                  f"  MAE {e.abs().mean():5.1f}  median bias {e.median():+.0f}"
                  f"  n_parsed {len(e)}")
    sfx = (args.tag + ("" if args.country_gt == "current" else "_ctryunion")
           + ("" if args.year_tol == 0 else f"_tol{args.year_tol}")
           + ("" if args.label == "combined" else f"_label{args.label}")
           + ("" if args.permute < 0 else f"_chance{args.permute}"))
    df.to_csv(os.path.join(data, f"scored_olmo7b_{args.mode}{sfx}.csv"), index=False)

    # ---- per-probe accuracy by recognition --------------------------------
    rows = []
    for k in PROBE_KEYS:
        for rec in (True, False):
            sub = df[df["recognized"] == rec]
            n, kk = len(sub), int(sub[f"score_{k}"].sum())
            lo, hi = wilson(kk, n)
            rows.append(dict(probe=k, recognized=rec, n=n, n_correct=kk,
                             accuracy=kk / n if n else np.nan,
                             ci_lo=lo, ci_hi=hi,
                             refusal_rate=(sub[f"behavior_{k}"] == "refused").mean()
                                          if n else np.nan))
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(data, f"accuracy_by_recognition_{args.mode}{sfx}.csv"),
               index=False)

    beh = (df.melt(id_vars=["recognized"],
                   value_vars=[f"behavior_{k}" for k in PROBE_KEYS],
                   var_name="probe", value_name="behavior")
             .assign(probe=lambda d: d["probe"].str.replace("behavior_", "", regex=False))
             .groupby(["probe", "recognized", "behavior"]).size()
             .rename("n").reset_index())
    beh.to_csv(os.path.join(data, f"behavior_by_recognition_{args.mode}{sfx}.csv"),
               index=False)

    piv = acc.pivot(index="probe", columns="recognized", values="accuracy") * 100
    piv["ratio"] = piv[True] / piv[False].replace(0, np.nan)
    print(piv.round(2).to_string())
    print("\nmean score_total  recognized: "
          f"{df.loc[df['recognized'] == True, 'score_total'].mean():.3f}  "
          f"unrecognized: {df.loc[df['recognized'] == False, 'score_total'].mean():.3f}")


if __name__ == "__main__":
    main()
