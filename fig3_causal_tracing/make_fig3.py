#!/usr/bin/env python3
"""Fig. 3 (fig:causal_trace) of "The selective scientific memory of large language models".

Run from the repository root:
    python fig3_causal_tracing/make_fig3.py
Writes fig3_causal_tracing/output/fig3.png and fig3.pdf, and prints the numbers quoted in the caption and Sec. 2.3.

Panels
  a-c  schematic (static asset cropped from the published figure)
  d    Marie Curie, residual-stream causal trace (window 1)            data/example_traces.npz
  e    Csaba Both, same; static asset unless data/example_traces.npz holds `both_ie` (see README)
  f,g  mean IE by semantic position x layer, top / bottom mention quartile   data/traces_wikidata9k.npz
  h,i  early-site (residual, last subject token) and late-site (attention, last token) AIE by quartile
  j,k  attribute-probe accuracy, recognized vs not, Wilson 95% CIs, two-proportion z-tests
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

HERE = os.path.join("fig3_causal_tracing")
DATA, ASSETS, OUT = (os.path.join(HERE, d) for d in ("data", "assets", "output"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
from matplotlib import font_manager as _fm
_avail = {f.name for f in _fm.fontManager.ttflist}
SANS = next((f for f in ["Arial", "Helvetica", "Liberation Sans"] if f in _avail), "DejaVu Sans")   # headings, letters, site labels

# ---- page geometry: the published PNG is 5194 x 5299 px at 300 dpi; every position below is in those pixels ----
W, H, DPI = 5194, 5299, 300
TICK, LAB, TITLE = 12.5, 14, 16.3                 # pt, as code/make_fig3_v18.py
EX = 1.294                                        # d,e were drawn at 300 dpi and pasted at 1.294x
HM = [(2340, 3197), (3942, 4799)]                 # x extents of d,e and f,g heatmaps
CB = [(3241, 3306), (4843, 4908)]                 # their colorbars
BAR = [(2355, 3404), (3923, 4972)]                # x extents of h,i and j,k
Y_DE, Y_FG, Y_HI, Y_JK = (296, 1152), (1551, 2407), (2781, 3650), (4281, 5132)   # (top, bottom) of each row's axes

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)


def ax_px(x0, x1, y0, y1, **kw):
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], **kw)


def text_px(x, y, s, **kw):
    fig.text(x / W, 1 - y / H, s, **kw)


def letter(x, y, s):
    text_px(x, y, s, fontsize=99 * 72 / DPI, fontweight="bold", family=SANS, va="baseline")


def static(path, x0, y0):
    from matplotlib.image import imread
    im = imread(path)
    a = ax_px(x0, x0 + im.shape[1], y0, y0 + im.shape[0])
    a.imshow(im, interpolation="none", aspect="auto"); a.axis("off")


# ================================ data ================================
tr = np.load(os.path.join(DATA, "traces_wikidata9k.npz"))
sem = tr["residual_sem"].astype(np.float32)               # (9270, 6, 32) residual stream, window 1
att_last = tr["attention_last_token"].astype(np.float32)  # (9270, 32) attention, window 10, last-token row
mentions = pd.read_csv(os.path.join(DATA, "mentions_wikidata9k.csv")).set_index("id").olmo2_mentions.reindex(tr["id"])
# quartiles of OLMo-2 training-data mentions; ties broken by file order, exactly as the original analysis
q = pd.qcut(mentions.rank(method="first"), 4, labels=False).values
early = np.nanmean(sem[:, 2, 5:10], axis=1)               # last subject token, layers 5-9 (python slice 5:10)
late = np.nanmean(att_last[:, 15:20], axis=1)             # last token, layers 15-19 (python slice 15:20)
ex = np.load(os.path.join(DATA, "example_traces.npz"))
probes = pd.read_csv(os.path.join(DATA, "attribute_probes_wikidata9k.csv"))
rec, unrec = probes[probes.recognized == 1], probes[probes.recognized == 0]

# ================================ a-c: schematic ================================
static(os.path.join(ASSETS, "schematic_abc_from_v18.png"), 0, 0)

# ================================ headings + letters ================================
for y, s in [(131, "Localizing and probing recognition"), (4057, "Auditing hallucinations")]:
    text_px(1975, y, s, fontsize=24.4, family=SANS, va="baseline")
for s, x, y in [("d", 2044, 264), ("e", 3662, 264), ("f", 2044, 1474), ("g", 3676, 1474),
                ("h", 2044, 2704), ("i", 3676, 2704), ("j", 2044, 4224), ("k", 3676, 4224)]:
    letter(x, y, s)


# ================================ d,e: single-scientist traces ================================
def example(i, ie, tokens, cmap, title, cbar_label):
    (x0, x1), (c0, c1) = HM[i], CB[i]
    ax = ax_px(x0, x1, *Y_DE)
    hm = ax.pcolor(ie, cmap=cmap, vmin=0, vmax=0.9)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(tokens)) + 0.5)
    ax.set_yticklabels(tokens, fontsize=12 * EX, ha="center")
    ax.tick_params(axis="y", length=0, pad=(HM[0][0] - 2117) * 72 / DPI)  # token column centred 2117 - HM[0][0] px left of each axis
    ax.set_xticks(np.arange(0, 32, 5) + 0.5); ax.set_xticklabels(range(0, 32, 5), fontsize=11 * EX)
    ax.tick_params(axis="x", length=3.5 * EX, width=0.8 * EX)
    ax.set_xlabel("Layer", fontsize=12 * EX)
    ax.grid(axis="y", ls="--", alpha=0.3, lw=0.8 * EX); ax.set_axisbelow(False)
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom"); sp.set_linewidth(0.8 * EX)
    cax = ax_px(c0, c1, *Y_DE)
    cb = fig.colorbar(hm, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8])
    cb.ax.tick_params(labelsize=10 * EX, length=3.5 * EX, width=0.8 * EX); cb.outline.set_linewidth(0.8 * EX)
    if cbar_label:
        cb.set_label("IE (prob)", fontsize=11 * EX)
    text_px((x0 + x1) / 2, 263, title, fontsize=TITLE, ha="center", va="baseline")
    return ax


axd = example(0, ex["curie_ie"], list(ex["curie_tokens"]), "Purples",
              f"Recognized, {int(ex['curie_mentions']):,} mentions", False)
axd.text(9.75, 2.75, "Early site", fontsize=15.2, family=SANS, ha="center", va="baseline")
axd.text(25.8, 6.69, "Late site", fontsize=15.2, family=SANS, ha="center", va="baseline")
if "both_ie" in ex.files:
    example(1, ex["both_ie"], list(ex["both_tokens"]), "Greens", f"Not recognized, {int(ex['both_mentions']):,} mentions", True)
else:                                                  # raw residual trace for Csaba Both not shipped: static panel
    static(os.path.join(ASSETS, "panel_e_from_v18.png"), 3540, 288)
    text_px(sum(HM[1]) / 2, 263, f"Not recognized, {int(ex['both_mentions']):,} mentions",
            fontsize=TITLE, ha="center", va="baseline")

# ================================ f,g: quartile mean heatmaps ================================
ROWS = ["1st subj", "Mid subj", "Last subj", "1st after", "Further", "Last"]
for i, (m, cmap, title) in enumerate([(q == 3, "Purples", "Most mentioned"), (q == 0, "Greens", "Least mentioned")]):
    (x0, x1), (c0, c1) = HM[i], CB[i]
    ax = ax_px(x0, x1, *Y_FG)
    h = np.nanmean(sem[m], axis=0)
    im = ax.imshow(h, aspect="auto", cmap=cmap, vmin=0, vmax=0.15, interpolation="nearest")
    for y in range(6):                                 # dashed row guides over pale cells
        for x in range(32):
            if h[y, x] < 0.03:
                ax.plot([x - 0.5, x + 0.5], [y, y], color="#D9D9D9", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(range(6)); ax.set_yticklabels(ROWS if i == 0 else [], fontsize=TICK)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.tick_params(labelsize=TICK, length=4.5, width=0.8)
    ax.tick_params(axis="y", length=0, pad=8)
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom"); sp.set_linewidth(0.8)
    ax.set_xlabel("Layer", fontsize=LAB, labelpad=4)
    ax.set_title(f"{title} (n = {m.sum():,})", fontsize=TITLE, pad=14)
    cb = fig.colorbar(im, cax=ax_px(c0, c1, *Y_FG), ticks=[0, 0.05, 0.10, 0.15])
    cb.ax.tick_params(labelsize=TICK, length=5, width=1.0); cb.outline.set_linewidth(1.0)
    cb.ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
    if i == 1:
        cb.set_label("AIE (prob)", fontsize=LAB)
text_px(1930, sum(Y_FG) / 2, "Semantic position", rotation=90, va="center", ha="left", fontsize=LAB)

# ================================ h,i: site AIE by quartile ================================
med = [np.median(mentions.values[q == k]) for k in range(4)]
QCOL = ["#C9E3C4", "#A8CFA1", "#B8B0DA", "#8E7FC6"]
site = {}
for (x0, x1), v, title in [(BAR[0], early, "Early site"), (BAR[1], late, "Late site")]:
    ax = ax_px(x0, x1, *Y_HI)
    mu = np.array([v[q == k].mean() for k in range(4)])
    se = np.array([v[q == k].std(ddof=1) / np.sqrt((q == k).sum()) for k in range(4)])
    site[title] = (mu, se)
    ax.bar(range(4), mu, 0.62, yerr=1.96 * se, color=QCOL, zorder=2, error_kw=dict(lw=1.3, capsize=4, capthick=1.3))
    ax.set_xticks(range(4)); ax.set_xticklabels([f"Q{k + 1}\n({int(x):,})" for k, x in enumerate(med)])
    ax.tick_params(labelsize=TICK, length=4.5, width=0.8); ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(True, color="#D9D9D9", lw=1.0, ls=(0, (4, 3))); ax.set_axisbelow(True)
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom"); sp.set_linewidth(0.8)
    ax.set_ylabel("AIE (prob)", fontsize=LAB)
    ax.set_xlabel("# training data quartiles", fontsize=LAB, labelpad=4)
    ax.set_title(title, fontsize=TITLE, pad=14)

# ================================ j,k: attribute probes ================================
PUR, GRN = "#BFB3DB", "#BAD0B4"
stats = []


def probe_panel(x0, x1, cols, labels, title, ylim, yticks, bracket_gap, legend_loc):
    ax = ax_px(x0, x1, *Y_JK)
    w = 0.3
    for j, (g, lab, col) in enumerate([(rec, f"Recognized (n={len(rec):,})", PUR),
                                       (unrec, f"Not Recognized (n={len(unrec):,})", GRN)]):
        k = np.array([g[f"score_{c}"].sum() for c in cols]); n = len(g)
        lo, hi = proportion_confint(k, n, method="wilson")
        p = k / n
        ax.bar(np.arange(len(cols)) + (j - 0.5) * w, 100 * p, w, color=col, label=lab, zorder=2,
               yerr=[100 * (p - lo), 100 * (hi - p)], error_kw=dict(lw=1.3, capsize=5, capthick=1.3))
    for i, c in enumerate(cols):
        k = [rec[f"score_{c}"].sum(), unrec[f"score_{c}"].sum()]; n = [len(rec), len(unrec)]
        pv = proportions_ztest(k, n)[1]
        his = [proportion_confint(kk, nn, method="wilson")[1] for kk, nn in zip(k, n)]
        stats.append((c, 100 * k[0] / n[0], 100 * k[1] / n[1], (k[0] / n[0]) / (k[1] / n[1]), pv))
        if pv < 0.05:
            y = 100 * max(his) + bracket_gap; t = 0.012 * ylim
            ax.plot([i - w / 2, i - w / 2, i + w / 2, i + w / 2], [y - t, y, y, y - t], color="#333333", lw=1.0)
            ax.text(i, y + 0.006 * ylim, "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*",
                    ha="center", va="bottom", fontsize=10.5, color="#333333")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim); ax.set_yticks(yticks)
    ax.tick_params(labelsize=12.7, length=4.5, width=0.8); ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(True, color="#D9D9D9", lw=1.0, ls=(0, (4, 3))); ax.set_axisbelow(True)
    for k_, sp in ax.spines.items():
        sp.set_visible(k_ == "bottom"); sp.set_linewidth(0.8)
    ax.set_ylabel("% correct on probe", fontsize=LAB)
    ax.set_title(title, fontsize=TITLE, pad=12.5)
    ax.legend(fontsize=10.6, loc=legend_loc, frameon=True, borderaxespad=0.2, handlelength=1.6)


# y-limits and bracket offsets matched to the published axes
probe_panel(*BAR[0], ["field", "country", "career_stage", "gender"], ["Field", "Country", "Career\nstage", "Gender"],
            "Biographical probes", 93.7, range(0, 81, 20), 4.0, "upper left")
probe_panel(*BAR[1], ["top_work", "lead_work", "coauthor"], ["Most-cited\nwork", "Lead-author\nwork", "Frequent\ncoauthor"],
            "Scholarly content probes", 7.41, range(0, 7, 2), 0.2, "upper right")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig3.{ext}"), dpi=DPI)
plt.close(fig)

# ================================ numbers quoted in the paper ================================
(me, se_), (ml, sl) = site["Early site"], site["Late site"]
print(f"cohort n = {len(q):,}; quartile sizes {[int((q == k).sum()) for k in range(4)]}; median mentions {[int(x) for x in med]}")
print(f"early site AIE Q4 {me[3]:.4f} vs Q1 {me[0]:.4f}  ->  {me[3] / me[0]:.2f}x   (paper: 0.026 vs 0.005, 5.7x)")
print(f"late  site AIE Q4 {ml[3]:.4f} vs Q1 {ml[0]:.4f}  ->  {ml[3] / ml[0]:.2f}x   (paper: 0.099 vs 0.029, 3.4x)")
print("early by quartile", np.round(me, 4), " late by quartile", np.round(ml, 4))
c = ex["curie_ie"]
print(f"Marie Curie: p(physicist) clean {float(ex['curie_p_clean']):.3f}, corrupted {float(ex['curie_p_corrupt']):.1e}, "
      f"max IE {c.max():.3f} (restored p = {c.max() + float(ex['curie_p_corrupt']):.3f}) at token {int(c.max(1).argmax())}, "
      f"layer {int(c.max(0).argmax())}")
print(f"attribute probes: recognized n = {len(rec):,}, not recognized n = {len(unrec):,}")
for c_, a, b, r, pv in stats:
    print(f"  {c_:13s} {a:6.2f}% vs {b:6.2f}%   ratio {r:4.2f}   p = {pv:.1e}")
print("wrote", os.path.join(OUT, "fig3.png"), "and fig3.pdf")
