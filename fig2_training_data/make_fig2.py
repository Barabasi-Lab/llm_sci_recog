#!/usr/bin/env python3
"""Fig. 2 (fig:training_olmo_merged): training-data exposure vs citations as predictors
of OLMo-2-13B-Instruct recognition, 12,214 physicists.

  a  KDE (log10 x) of citations, recognized vs not recognized, dashed medians
  b  same for mentions of the name in the OLMo-2 pretraining corpus (infini-gram)
  c  binned recognition rate (Wilson 95% CI) + univariate logistic fit (95% CI band)
     vs citations and vs mentions; c's x-axis sits on the same row as b's

Run from the repo root:  python fig2_training_data/make_fig2.py
Input:  fig2_training_data/data/fig2_scientists.csv  (id, citations, mentions, recognized)
Output: fig2_training_data/output/fig2.png, fig2.pdf; prints the numbers quoted in the text.
"""
import os, warnings
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr
import statsmodels.api as sm

HERE = "fig2_training_data"
DATA = f"{HERE}/data/fig2_scientists.csv"
OUT = f"{HERE}/output"

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linestyle": "--", "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "ytick.left": False})

REC, NOTC, REC_M, NOTC_M = "#A78FCE", "#A0C58C", "#7B60AC", "#628F49"   # a,b: recognized purple / not green
FREQ_C, CIT_C, FREQ_D, CIT_D = "#E8A598", "#B0C4DE", "#C57A6B", "#6E86A8"  # c: mentions coral / citations steel blue
fmt = lambda x: f"{int(Decimal(str(x)).quantize(Decimal(1), ROUND_HALF_UP)):,}"   # half-up: 10,056.5 -> 10,057

d = pd.read_csv(DATA)
cit = d.citations.clip(lower=0.5).values          # zeros -> 0.5 so they sit on the log axis
freq = d.mentions.clip(lower=0.5).values
r = d.recognized.values.astype(bool); y = r.astype(int)

# Layout (figure fractions) matches the published 4336x2108 composite.
W = 8.8; fig = plt.figure(figsize=(W, W * 972 / 2000))
axa = fig.add_axes([0.0815, 0.6276, 0.3775, 0.347])
axb = fig.add_axes([0.0815, 0.1358, 0.3775, 0.347])
axc = fig.add_axes([0.5925, 0.1358, 0.3925, 0.8076])      # square; bottom = b's x-axis

# ---------------------------------------------------------------- a, b
g = np.linspace(np.log10(0.4), np.log10(3e6), 400)
for ax, val, xlabel in [(axa, cit, "Citations"), (axb, freq, "Mentions in the training data")]:
    lv = np.log10(val); peak = 0
    for mask, c in [(~r, NOTC), (r, REC)]:
        dz = gaussian_kde(lv[mask])(g); peak = max(peak, dz.max())
        ax.fill_between(10**g, 0, dz, color=c, alpha=0.35, lw=0); ax.plot(10**g, dz, color=c, lw=1.6)
    mr, mn = np.median(val[r]), np.median(val[~r]); lh = peak * 1.06
    ax.plot([mn, mn], [0, lh], color=NOTC_M, ls="--", lw=0.8, zorder=5)
    ax.plot([mr, mr], [0, lh], color=REC_M, ls="--", lw=0.8, zorder=5)
    if max(mr, mn) / min(mr, mn) < 2.5:       # close medians: one bracket line, labels left/right
        c = np.sqrt(mn * mr); xa, xb = c * 10**-0.95, c * 10**0.95
        ax.plot([xa, c], [lh, lh], color=NOTC_M, ls="--", lw=0.8, zorder=5)
        ax.plot([c, xb], [lh, lh], color=REC_M, ls="--", lw=0.8, zorder=5)
        ax.text(c * 10**-0.05, lh, fmt(mn), color=NOTC_M, fontsize=10, ha="right", va="bottom")
        ax.text(c * 10**0.05, lh, fmt(mr), color=REC_M, fontsize=10, ha="left", va="bottom")
    else:
        ax.text(mr, lh, fmt(mr), color=REC_M, fontsize=10, ha="center", va="bottom")
        ax.text(mn, lh, fmt(mn), color=NOTC_M, fontsize=10, ha="center", va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.4, 1e6); ax.set_ylim(0, peak * 1.24)
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
axa.legend(handles=[Line2D([0], [0], color=REC, lw=3, label="recognized"),
                    Line2D([0], [0], color=NOTC, lw=3, label="not recognized")],
           frameon=False, fontsize=11, loc="upper left")

# ---------------------------------------------------------------- c
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return p, max(c - h, 0), c + h

P = lambda e: 1 / (1 + np.exp(-e)); stats = {}
for key, val, cf, cd, lab, sl in [("mentions", freq, FREQ_C, FREQ_D, "Mentions in training data", slice(None, -2)),
                                  ("citations", cit, CIT_C, CIT_D, "Citations", slice(2, -1))]:
    lx = np.log10(val)
    edges = np.logspace(lx.min(), lx.max(), 17)                      # 16 equal-width log bins
    dfb = pd.DataFrame({"v": val, "y": y}); dfb["b"] = pd.cut(dfb.v, edges, include_lowest=True)
    rows = []
    for _, gb in dfb.groupby("b", observed=True):
        if len(gb) < 10: continue                                     # drop bins with n<10
        p, lo, hi = wilson(int(gb.y.sum()), len(gb)); rows.append((gb.v.mean(), p, p - lo, hi - p, len(gb)))
    rows = rows[sl]                       # drop sparse tail bins: mentions last 2; citations first 2 + last 1
    xs, ps, los, his, ns = map(np.array, zip(*rows))
    axc.errorbar(xs, ps, yerr=[los, his], fmt="o", color=cd, ms=4.5, mfc=cf, mec=cd, mew=1,
                 capsize=2.5, elinewidth=1.1, lw=0, zorder=4)
    res = sm.Logit(y, sm.add_constant(lx)).fit(disp=0)               # logistic on raw 0/1, log10 x
    gx = np.linspace(lx.min(), np.log10(xs[-1]), 200); Xg = np.column_stack([np.ones_like(gx), gx])
    eta = Xg @ res.params; se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, res.cov_params(), Xg))
    axc.fill_between(10**gx, P(eta - 1.96*se), P(eta + 1.96*se), color=cf, alpha=0.25, lw=0, zorder=1)
    axc.plot(10**gx, P(eta), color=cf, lw=2.4, zorder=3, label=lab)
    stats[key] = dict(bins=[(round(float(a), 1), round(float(b), 4), int(c)) for a, b, c in zip(xs, ps, ns)], fit=lambda x, b=res.params: P(b[0] + b[1]*np.log10(x)))
axc.set_xscale("log"); axc.set_xlim(0.3, 1.5e5); axc.set_ylim(0, 0.45); axc.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axc.set_xlabel("Citations or Mentions"); axc.set_ylabel("Recognized probability")
axc.legend(frameon=False, fontsize=11, loc="upper left")

lab = dict(fontsize=16, fontweight="bold", va="top", ha="left",
           fontfamily=["Arial", "Helvetica", "DejaVu Sans"])
fig.text(0.0, 0.985, "a", **lab); fig.text(0.0, 0.508, "b", **lab); fig.text(0.508, 0.985, "c", **lab)

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/fig2.png", dpi=500, facecolor="white")
fig.savefig(f"{OUT}/fig2.pdf", facecolor="white")

# ---------------------------------------------------------------- numbers quoted in text/caption
print(f"N={len(d):,}  recognized={r.sum():,} ({r.mean():.1%})")
print(f"Spearman rho(citations, mentions) = {spearmanr(d.citations, d.mentions)[0]:.4f}")
print(f"median citations  recognized {fmt(np.median(cit[r]))}  vs not {fmt(np.median(cit[~r]))}")
print(f"median mentions   recognized {fmt(np.median(freq[r]))}  vs not {fmt(np.median(freq[~r]))}"
      f"  (ratio {np.median(freq[r]) / np.median(freq[~r]):.1f}x)")
for k in ("mentions", "citations"):
    s = stats[k]; f = s["fit"]
    print(f"{k:9s} fit: {f(0.5):.3f} at x=0.5, {f(1e4):.3f} at 1e4;  last bin x={s['bins'][-1][0]:,} rate={s['bins'][-1][1]:.3f}")
    print(f"          bins (mean x, rate, n): {s['bins']}")
top_m, top_c = stats["mentions"]["bins"][-1][1], stats["citations"]["bins"][-1][1]
print(f"top-bin ratio mentions/citations = {top_m / top_c:.3f}")
print(f"wrote {OUT}/fig2.png, {OUT}/fig2.pdf")
