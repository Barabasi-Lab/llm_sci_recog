"""Fig. 4 (fig:online_anchoring): web search reflects the model's parametric memory.

Run from the repo root:  python fig4_web_search/make_fig4.py
Writes fig4_web_search/output/fig4.{png,pdf} and prints every number quoted in the caption, Sec. 2.4,
SI "web methods" and Table "web_inclusion" as recomputed from fig4_web_search/data/.

Panels
  a, b  schematic (static asset cropped from online_search_v17.png) + count bars drawn here from the data
  c     retention: P(searched name reaches the final answer), Step A vs Step B, raw / impact / impact + relevance
  d     position of the traceable names in the final answer
"""
import os
import numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as mtransforms

HERE = os.path.join("fig4_web_search")
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

off = pd.read_csv(os.path.join(DATA, "offline_pairs.csv"))      # closed-book recommendation, 1 row / concept-scientist
sp = pd.read_csv(os.path.join(DATA, "searched_pairs.csv"))      # searched names (Step A / B), 1 row / pair
fp = pd.read_csv(os.path.join(DATA, "final_pairs.csv"))         # final answer (Step C), 1 row / pair
concepts = sorted(set(off.concept) | set(fp.concept))           # 200

# ------------------------------------------------------------------ numbers
def per_concept(d, over=concepts):
    s = d.groupby("concept").size().reindex(over, fill_value=0)
    return s.sum(), s.mean(), s.std(), s.median()

A, B = sp[sp.stage == "A"], sp[sp.stage == "B"]
traced = fp[fp.provenance.isin(["A", "B"])]
c_with_q = sorted(sp.concept.unique())                          # 199 concepts with names in their queries
n_off, m_off, s_off, _ = per_concept(off)
n_A, m_A, s_A, _ = per_concept(A, c_with_q)
n_B, m_B, s_B, med_B = per_concept(B, c_with_q)
n_C, m_C, s_C, _ = per_concept(fp)
n_tr, m_tr, _, _ = per_concept(traced)
nCA, nCB = (fp.provenance == "A").sum(), (fp.provenance == "B").sum()
unt = fp[fp.provenance == "untraced"]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / 2 / n) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

def logit(d, covs, cluster=False):
    X = sm.add_constant(d[["g"] + covs].astype(float), has_constant="add")
    m = sm.Logit(d.y, X)
    return m.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.cid}) if cluster else m.fit(disp=0)

def gcomp(f, d, covs):
    X = sm.add_constant(d[["g"] + covs].astype(float), has_constant="add")
    return [100 * f.predict(X.assign(g=float(v))).mean() for v in (0, 1)]

U = sp.assign(g=(sp.stage == "A").astype(int), y=sp.in_final, cid=pd.factorize(sp.concept)[0],
              log_cit=np.log1p(sp.career_cites), log_pub=np.log1p(sp.career_pubs), rel=sp.relevance)
conds = [("raw", []), ("impact-\nadjusted", ["log_cit", "log_pub"]),
         ("impact +\nrelevance", ["log_cit", "log_pub", "rel"])]
res = []
for lab, covs in conds:
    d = U.dropna(subset=["y"] + covs).reset_index(drop=True)
    f = logit(d, covs); fc = logit(d, covs, cluster=True)
    if not covs:
        web = wilson(d[d.g == 0].y.sum(), (d.g == 0).sum()); par = wilson(d[d.g == 1].y.sum(), (d.g == 1).sum())
    else:   # g-computation; 300 bootstrap resamples of pairs, same seeds as the original script
        a0, a1 = gcomp(f, d, covs)
        bs = np.array([gcomp(logit(s, covs), s, covs)
                       for s in (d.sample(len(d), replace=True, random_state=i) for i in range(300))])
        web = (a0, *np.percentile(bs[:, 0], [2.5, 97.5])); par = (a1, *np.percentile(bs[:, 1], [2.5, 97.5]))
    res.append(dict(lab=lab, n=len(d), web=web, par=par, OR=np.exp(f.params["g"]),
                    p=f.pvalues["g"], p_cl=fc.pvalues["g"]))

top3 = {g: 100 * (traced[traced.provenance == g].position <= 3).mean() for g in "AB"}
medpos = {g: traced[traced.provenance == g].position.median() for g in "AB"}
idsrc = sp.identity_source
withcar = sp.career_cites.notna()

print("== counts (Sec. 2.4, caption a/b, SI web methods) ==")
print(f"offline: {n_off} pairs, {m_off:.1f} +- {s_off:.1f} per concept")
print(f"Step A : {n_A} pairs, {m_A:.1f} +- {s_A:.1f} per concept over {len(c_with_q)} concepts "
      f"({n_A/len(concepts):.2f} over 200)")
print(f"Step B : {n_B} pairs, {m_B:.1f} +- {s_B:.1f}, median {med_B:.0f} over {len(c_with_q)} concepts")
print(f"Step C : {n_C} pairs, {m_C:.1f} +- {s_C:.1f} per concept; traced {n_tr} ({m_tr:.1f}/concept) = "
      f"{nCA} Step A + {nCB} Step B; Step B in final and offline: {((B.in_final==1)&(B.in_offline==1)).sum()}")
print(f"untraced: {len(unt)}; in offline {100*(unt.untraced_reason=='in_offline').mean():.0f}%, "
      f"surname in query {100*(unt.untraced_reason=='surname_in_query').mean():.0f}%, "
      f"neither {100*(unt.untraced_reason=='neither').mean():.0f}%")
print(f"identity: seminal-DOI {idsrc.str.startswith('seminal-doi').sum()}, name fallback "
      f"{idsrc.str.startswith('curated-fallback').sum()}, hand-identified {idsrc.str.startswith('dropped').sum()}, "
      f"unresolved {(idsrc=='unresolved').sum()}, resolved w/o career record {idsrc.str.endswith('no-career-record').sum()}, "
      f"with career stats {withcar.sum()} (drop {(~withcar).sum()})")
print(f"relevance defined {sp.relevance.notna().sum()}; mean web {B.relevance.mean():.3f} vs parametric {A.relevance.mean():.3f}")
print("== retention (panel c, Table web_inclusion) ==")
for r in res:
    print(f"{r['lab'].replace(chr(10), ' '):20s} n={r['n']}  web {r['web'][0]:.1f}  parametric {r['par'][0]:.1f}  "
          f"OR {r['OR']:.2f}  p={r['p']:.1g}  p_cl={r['p_cl']:.1g}")
print("== position (panel d) ==")
print(f"top-3: parametric {top3['A']:.1f}% vs web {top3['B']:.1f}%; median position {medpos['A']:.0f} vs {medpos['B']:.0f}")

# ------------------------------------------------------------------ figure
BLUE, GOLD = "#6f8fc0", "#d1a24f"                  # bar edges / lines
BFILL, GFILL = "#c4d4ec", "#ecd9a4"                # bar fills
W_IN = 6.6
TOP_PX, GAP_PX, BOT_IN = 1495, 70, 2.56            # geometry of the published composite (300 dpi)
TOP_IN, GAP_IN = TOP_PX / 300, GAP_PX / 300
H_IN = TOP_IN + GAP_IN + BOT_IN
fig = plt.figure(figsize=(W_IN, H_IN))

# --- a, b: static schematic + count bars, in the schematic's own SVG units (612 wide)
SVG_W = 612; U2PT = W_IN * 72 / SVG_W; SVG_H = TOP_PX * SVG_W / 1980
axs = fig.add_axes([0, 1 - TOP_IN / H_IN, 1, TOP_IN / H_IN])
axs.imshow(plt.imread(os.path.join(HERE, "assets", "schematic_ab_v17.png")),
           extent=(0, SVG_W, SVG_H, 0), aspect="auto", interpolation="lanczos")
axs.set_xlim(0, SVG_W); axs.set_ylim(SVG_H, 0); axs.axis("off")
X0, SC, BH, LFS = 420, 138 / n_off, 26, 10.5 * U2PT  # bar origin, units per name (offline bar = 138 units), height
BAR_FONT = dict(family=["Arial", "Helvetica", "DejaVu Sans"], fontsize=LFS)

def hbar(yc, segs, total=None):
    x = X0
    for n, edge in segs:
        axs.add_patch(Rectangle((x, yc - BH / 2), n * SC, BH, fc="white", ec=edge, lw=1.2 * U2PT, zorder=3))
        axs.text(x + 3, yc + LFS / U2PT * 0.36, f"{n:,}", color=edge, va="baseline", ha="left", zorder=4, **BAR_FONT)
        x += n * SC
    axs.plot([X0, X0], [yc - BH / 2 - 5, yc + BH / 2 + 5], color="#3a3f47", lw=1 * U2PT, zorder=4,
             solid_capstyle="butt")
    return x

hbar(76, [(n_off, BLUE)]); hbar(206, [(n_A, BLUE)]); hbar(302, [(n_B, GOLD)])
xe = hbar(405, [(nCA, BLUE), (nCB, GOLD)])
# Step C total in the blue -> gold gradient text of the original
lab = f"{nCA + nCB:,}"
fp_ = FontProperties(family=BAR_FONT["family"], size=LFS)
tp = TextPath((0, 0), lab, prop=fp_)
tw = tp.get_extents().width / U2PT                   # text width in svg units
x_t, y_t = xe + 5, 405 + LFS / U2PT * 0.36
tr = (mtransforms.Affine2D().scale(1 / U2PT, -1 / U2PT).translate(x_t, y_t) + axs.transData)
grad = np.linspace(0, 1, 256)[None, :]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mix", ["#3f6f9f", "#7f7a52", "#b8860b"])
gi = axs.imshow(grad, extent=(x_t, x_t + tw + 1, y_t + 3, y_t - LFS / U2PT), cmap=cmap, aspect="auto", zorder=4)
gi.set_clip_path(PathPatch(tp, transform=tr, fc="none", ec="none"))
axs.set_xlim(0, SVG_W); axs.set_ylim(SVG_H, 0)

# --- lower panels: 3.3 x 2.56 in canvases, inner axes 2.55 x 1.90 in at (0.62, 0.58)
LABFS, TICKFS, LEGFS = 10, 9, 8.5
def inner(x_off):
    l, b, w, h = 0.62, 0.58, 2.55, 1.90
    return fig.add_axes([(x_off + l) / W_IN, b / H_IN, w / W_IN, h / H_IN])
def style(ax):
    for s in ("top", "left", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.tick_params(left=False, length=0, labelsize=TICKFS); ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", which="major", ls="--", alpha=0.3); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

# c: retention bars
ax = inner(0); w = .38
for i, r in enumerate(res):
    for x, (m, lo, hi), fc, ec in ((i - w / 2, r["web"], GFILL, GOLD), (i + w / 2, r["par"], BFILL, BLUE)):
        ax.bar(x, m, w, yerr=[[m - lo], [hi - m]], color=fc, edgecolor=ec, lw=1.2, zorder=3,
               capsize=2.5, error_kw=dict(lw=1, ecolor=".4"))
        ax.text(x, hi + 2, f"{m:.0f}%", ha="center", va="bottom", fontsize=7.5, color=ec)
    yb = max(r["web"][2], r["par"][2]) + 14
    ax.plot([i - w / 2, i - w / 2, i + w / 2, i + w / 2], [yb - 2, yb, yb, yb - 2], lw=.9, color=".5")
    ax.text(i, yb + .5, stars(r["p"]), ha="center", va="bottom", fontsize=9, color=".3")
ax.set_xticks(range(len(res))); ax.set_xticklabels([r["lab"] for r in res])
ax.set_xlim(-0.55, len(res) - 0.45); ax.set_ylim(0, 124); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("% in the final answer", fontsize=LABFS); style(ax)
ax.legend(handles=[Patch(fc=GFILL, ec=GOLD, label="web-discovered"), Patch(fc=BFILL, ec=BLUE, label="parametric")],
          frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03),
          handlelength=1, columnspacing=0.8, handletextpad=0.4)

# d: position in the final answer (traceable names only)
ax = inner(3.3); ranks = np.arange(1, 11)
for g, col, lab_ in (("A", BLUE, "parametric"), ("B", GOLD, "web-discovered")):
    d = traced[traced.provenance == g]
    ax.plot(ranks, [100 * (d.position == k).mean() for k in ranks], "-o", color=col, lw=1.6, ms=5, mfc="white",
            mec=col, mew=1.4, zorder=3, label=f"{lab_} ({len(d)})")
ax.set_xticks(ranks); ax.set_xlim(0.5, 10.5); ax.set_ylim(0, 25); ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel("position in the final answer", fontsize=LABFS); ax.set_ylabel("% of the group", fontsize=LABFS)
style(ax)
ax.legend(frameon=False, fontsize=LEGFS, loc="upper right", bbox_to_anchor=(1.02, 1.03), handlelength=1.5)

# panel letters c, d (a, b are part of the schematic asset)
for lab_, x in (("c", 6), ("d", 990 + 6)):
    fig.text(x / 300 / W_IN, 1 - (TOP_PX + 18) / 300 / H_IN, lab_, fontsize=12, fontweight="bold",
             family=["Arial", "DejaVu Sans"], va="top", ha="left")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig4.{ext}"), dpi=300)
print("saved", os.path.join(OUT, "fig4.{png,pdf}"))
