#!/usr/bin/env python3
"""Fig. 5 (fig:llm_impact): LLM-assisted arXiv papers cite LLM-recognized first authors more often.

  a  schematic (static asset cut from the published figure); its counts are recomputed here
  b  adjusted P(cited first author recognized by GPT-4o), human-written vs LLM-assisted, by career stage
  c  the same by age of the cited work (0-4, 5-15, >15 yr)

Model (as in the paper, SI "arxiv_regression"): logit of y = [cited first author recognized], on
asst = [alpha_main > 0.1] + log1p(author citations) + log1p(author works) + career age
+ log1p(reference age) + log1p(reference citations) + year-month fixed effects of the citing paper;
SEs clustered by citing paper. One model per career stage (panel b) and per reference-age bin (panel c).
Counterfactual means: every pair set to asst = 0 / 1, averaged; delta-method 95% CIs; the AME's p-value.
Ages and stages are period-aware: citing papers of the 2023-24 window use 2024, of the 2025-26 window 2026.

Run from the repository root:
    python fig5_citations/make_fig5.py                 # refits the 7 logits (~1-3 min)
    python fig5_citations/make_fig5.py --published     # plot the shipped margins table, no refit
Writes fig5_citations/output/fig5.{png,pdf} and fig5_citations/output/fig5_margins.csv
"""
import os
import sys
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)
HERE = "fig5_citations"
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
CONTROLS = "log_cit + log_works + cage + log_ref_age + log_ref_cit + C(ym)"
STAGES = [("early", "Early", "#9DB17C"), ("mid", "Mid", "#DDA15E"), ("late", "Late", "#C1666B")]
REFAGE = [("0-4 yr", "0–4 yr", "#86B3AA"), ("5-15 yr", "5–15 yr", "#7E9CC4"), (">15 yr", ">15 yr", "#9D86BA")]

# ---------------------------------------------------------------- data
papers = pd.read_csv(os.path.join(DATA, "papers.csv.gz"))
authors = pd.read_csv(os.path.join(DATA, "authors.csv.gz"))
pairs = pd.read_csv(os.path.join(DATA, "pairs.csv.gz"))
full = pairs.merge(papers, on="paper", how="left").merge(authors, on="author", how="left")

# panel a counts (all 18,211 papers / 661,992 pairs, before any model restriction)
pp = papers.set_index("paper")
COUNTS = dict(n_papers=len(papers), n_llm=int((pp.alpha_main > 0.1).sum()), n_human=int((pp.alpha_main <= 0.1).sum()),
              n_no_alpha=int(pp.alpha_main.isna().sum()), n_authors=full.author.nunique(),
              share_recog=(full.recog == "YES").mean(), n_pairs=len(full))

d = full[full.recog.notna() & full.alpha_main.notna()].copy()
d["y"] = (d.recog == "YES").astype(int)
d["asst"] = (d.alpha_main > 0.1).astype(int)
yr = np.where(d.period == "new", 2026, 2024)
d["cage"] = yr - d.first_pub_year
d["stage"] = np.select([d.cage <= 10, d.cage <= 30, d.cage > 30], ["early", "mid", "late"], None)
d["log_cit"] = np.log1p(d.cited_by_count)
d["log_works"] = np.log1p(d.works_count)
d["ref_age"] = yr - d.ref_year
d["log_ref_age"] = np.log1p(d.ref_age.clip(lower=0))
d["log_ref_cit"] = np.log1p(d.ref_cit)
d["ym"] = d.ym.astype(str)
d = d.dropna(subset=["stage", "cage", "log_cit", "log_works", "log_ref_age", "log_ref_cit"]).reset_index(drop=True)


def fit_margins(df):
    """logit + counterfactual mean P(y) at asst = 0 / 1 (delta-method CIs) and their difference (AME)."""
    f = smf.logit("y ~ asst + " + CONTROLS, data=df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(df.paper)[0]})
    X, b, V = f.model.exog.copy(), f.params.values, f.cov_params().values
    j = f.model.exog_names.index("asst")
    out = {}
    for a in (0, 1):
        X[:, j] = a
        p = 1 / (1 + np.exp(-X @ b))
        out[a] = (p.mean(), ((p * (1 - p))[:, None] * X).mean(axis=0))
    (m0, g0), (m1, g1) = out[0], out[1]
    se0, se1, sed = (np.sqrt(g @ V @ g) for g in (g0, g1, g1 - g0))
    ame = m1 - m0
    return dict(n_pairs=len(df), n_papers=df.paper.nunique(),
                p_human=m0, lo_human=m0 - 1.96 * se0, hi_human=m0 + 1.96 * se0,
                p_llm=m1, lo_llm=m1 - 1.96 * se1, hi_llm=m1 + 1.96 * se1,
                ame=ame, ame_lo=ame - 1.96 * sed, ame_hi=ame + 1.96 * sed, p=2 * norm.sf(abs(ame / sed)))


pub = pd.read_csv(os.path.join(DATA, "published_margins.csv"))
if "--published" in sys.argv:
    res = pub.copy()
else:
    rows = [dict(panel="pooled", group="pooled", **fit_margins(d))]
    for st, _, _ in STAGES:
        rows.append(dict(panel="b", group=st, **fit_margins(d[d.stage == st].reset_index(drop=True))))
    for key, m in [("0-4 yr", d.ref_age <= 4), ("5-15 yr", (d.ref_age >= 5) & (d.ref_age <= 15)),
                   (">15 yr", d.ref_age > 15)]:
        rows.append(dict(panel="c", group=key, **fit_margins(d[m].reset_index(drop=True))))
    res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "fig5_margins.csv"), index=False)

# ---------------------------------------------------------------- report vs. paper
print(f"papers {COUNTS['n_papers']:,} | LLM-assisted {COUNTS['n_llm']:,} | human {COUNTS['n_human']:,} | "
      f"no alpha {COUNTS['n_no_alpha']:,} | cited first authors {COUNTS['n_authors']:,} | "
      f"pairs {COUNTS['n_pairs']:,} | citations to recognized {100 * COUNTS['share_recog']:.1f}%")
show = res.copy()
for c in ["p_human", "lo_human", "hi_human", "p_llm", "lo_llm", "hi_llm", "ame", "ame_lo", "ame_hi"]:
    show[c] = (100 * show[c]).round(2)
show["p"] = show.p.map(lambda v: f"{v:.2g}")
print(show.drop(columns=["lo_human", "hi_human", "lo_llm", "hi_llm"]).to_string(index=False))
num = ["p_human", "lo_human", "hi_human", "p_llm", "lo_llm", "hi_llm", "ame", "ame_lo", "ame_hi"]
dev = (res.set_index(["panel", "group"])[num] - pub.set_index(["panel", "group"])[num]).abs().max().max()
print(f"max |difference| from the published margins: {100 * dev:.4f} pp")

# ---------------------------------------------------------------- figure
# Canvas and geometry of the published figure: 3600 x 2946 px at 300 dpi; panel a = rows 0-1545;
# panels b, c were drawn at 4.3 x 3.2 in and scaled to 1710 px wide each, 60 px gutters.
DPI, W, H = 300, 3600, 2946
PAD, HALF, YB = 60, (3600 - 3 * 60) // 2, 1545 + 60
FAM = ["Helvetica", "Arial", "DejaVu Sans"]
star = lambda p: "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "ytick.left": False, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.linestyle": "--", "grid.alpha": 0.3})


def draw(axes, groups, sub, k):
    """the three-axes point-range panel of the paper, every size multiplied by k."""
    for ax, (key, title, col) in zip(axes, groups):
        q = sub[sub.group == key].iloc[0]
        ph, pl = 100 * q.p_human, 100 * q.p_llm
        cih, cil = (100 * q.lo_human, 100 * q.hi_human), (100 * q.lo_llm, 100 * q.hi_llm)
        kw = dict(markersize=9 * k, capsize=3.5 * k, elinewidth=1.4 * k, capthick=1.4 * k)
        ax.errorbar([0], [ph], yerr=[[ph - cih[0]], [cih[1] - ph]], fmt="o", mfc="white", mec=col, mew=1.5 * k,
                    ecolor=col, **kw)
        ax.errorbar([1], [pl], yerr=[[pl - cil[0]], [cil[1] - pl]], fmt="o", color=col, **kw)
        lo, hi = min(cih[0], cil[0]), max(cih[1], cil[1])
        span = hi - lo if hi > lo else 1
        yb = hi + 0.12 * span
        ax.set_ylim(lo - 0.28 * span, yb + 0.22 * span)
        ax.plot([0, 0, 1, 1], [yb - 0.06 * span, yb, yb, yb - 0.06 * span], color="dimgray", lw=1 * k)
        ax.text(0.5, yb + 0.03 * span, star(q.p), ha="center", fontsize=11 * k, color="dimgray")
        ax.set_xticks([0, 1]); ax.set_xticklabels([])
        ax.tick_params(axis="x", length=4 * k, width=1 * k, color="black", pad=3.5 * k)
        ax.tick_params(axis="y", labelsize=9.5 * k, pad=3.5 * k)
        ax.grid(True, axis="y", lw=0.8 * k)
        ax.spines["bottom"].set_linewidth(0.8 * k)
        tr = ax.get_xaxis_transform()
        ax.text(-0.22, -0.075, "Human-\nwritten", transform=tr, ha="center", va="top", fontsize=8 * k, linespacing=1.0)
        ax.text(1.22, -0.075, "LLM-\nassisted", transform=tr, ha="center", va="top", fontsize=8 * k, linespacing=1.0)
        ax.set_xlim(-0.7, 1.7)
        ax.set_title(title, fontsize=13 * k, pad=6 * k)
    axes[0].set_ylabel("LLM recognition (%)", fontsize=11.5 * k, labelpad=4 * k)


def layout(groups, sub, single_line_first):
    """axes boxes of the 4.3 x 3.2 in panel after tight_layout, relative to its saved (tight + 0.1 in) bbox."""
    f, axes = plt.subplots(1, 3, figsize=(4.3, 3.2), dpi=DPI)
    draw(axes, groups, sub, 1.0)
    if single_line_first:           # panel b was laid out with one-line tick labels, then relabelled
        tmp = []
        for ax in axes:
            for t in ax.texts[1:]:
                t.set_visible(False)
            tr = ax.get_xaxis_transform()
            tmp += [ax.text(-0.12, -0.075, "Human", transform=tr, ha="center", va="top", fontsize=8.5),
                    ax.text(1.12, -0.075, "LLM", transform=tr, ha="center", va="top", fontsize=8.5)]
        f.tight_layout()
        for t in tmp:
            t.remove()
        for ax in axes:
            for t in ax.texts:
                t.set_visible(True)
    else:
        f.tight_layout()
    bb = f.get_tightbbox(f.canvas.get_renderer()).padded(0.1)
    boxes = [ax.get_window_extent().transformed(f.dpi_scale_trans.inverted()) for ax in axes]
    plt.close(f)
    return bb, boxes


fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
plt.rcParams["font.family"] = "sans-serif"
# ---- a: schematic + counts from the data
axa = fig.add_axes([0, 1 - 1545 / H, 1, 1545 / H])
axa.imshow(plt.imread(os.path.join(HERE, "assets", "panel_a_schematic.png")), extent=(0, W, 1545, 0),
           interpolation="none")
axa.set_xlim(0, W); axa.set_ylim(1545, 0); axa.axis("off")
ta = dict(ha="center", va="baseline", family=FAM)
c1, c2 = "#5D6471", "#707782"
axa.text(572, 1330, f"{COUNTS['n_papers']:,} preprints", fontsize=10.9, color=c1, **ta)
axa.text(572, 1381, "2023–2026", fontsize=10.9, color=c1, **ta)
axa.text(1800, 1330, f"{COUNTS['n_llm']:,} LLM-assisted", fontsize=10.9, color=c1, **ta)
axa.text(1800, 1381, f"{COUNTS['n_human']:,} Human-written", fontsize=10.9, color=c1, **ta)
axa.text(2985, 1327, f"{COUNTS['n_authors']:,} authors", fontsize=9.1, color=c2, **ta)
axa.text(2985, 1379, f"{100 * COUNTS['share_recog']:.1f}% of citations go to recognized authors", fontsize=9.1,
         color=c2, **ta)

# ---- b, c: refit margins, drawn as vector axes at the published size
for groups, panel, x0, letter, first in [(STAGES, "b", PAD, "b", True), (REFAGE, "c", 2 * PAD + HALF, "c", False)]:
    sub = res[res.panel == panel]
    bb, boxes = layout(groups, sub, first)
    s = HALF / (bb.width * DPI)                          # px of the big canvas per px of the small panel
    hpx = bb.height * DPI * s
    axes = []
    for bx in boxes:
        X = x0 + (bx.x0 - bb.x0) * DPI * s
        Y = (H - YB - hpx) + (bx.y0 - bb.y0) * DPI * s   # from the bottom
        axes.append(fig.add_axes([X / W, Y / H, bx.width * DPI * s / W, bx.height * DPI * s / H]))
    draw(axes, groups, sub, s)
    fig.text((0 if letter == "b" else PAD + HALF + 9) / W, 1 - (YB - 40 + 22) / H, letter, fontsize=26.4,
             fontweight="bold", family=["Arial", "Helvetica", "DejaVu Sans"], va="top", ha="left")

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig5.{e}"), dpi=DPI, facecolor="white")
plt.close(fig)
print("->", os.path.join(OUT, "fig5.png"), "and .pdf")
