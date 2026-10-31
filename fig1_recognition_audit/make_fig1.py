#!/usr/bin/env python3
"""
Fig. 1 -- "LLM recognition of scientists is limited and uneven" (fig:llm_margins).

Redraws all seven panels from the processed inputs in fig1_recognition_audit/data/
and assembles them into the published layout.

    python fig1_recognition_audit/make_fig1.py          # run from the repo root

Writes
    fig1_recognition_audit/output/fig1.png, fig1.pdf     composite (raster, 300 dpi)
    fig1_recognition_audit/output/panels/fig1{a..g}.pdf  each panel as vector PDF
    fig1_recognition_audit/output/fig1_values.csv        every plotted point + CI

Methods (identical to the paper pipeline):
  * response bias   false-positive rate (FPR) on fabricated names, per model;
  * SDT correction  probit shift to the reference threshold (Claude Sonnet 4.6)
                    p_adj = Phi( z(p) - z(f_model) + z(f_ref) );
  * (a)(b)(c)       delta-method 95% CI on z(p) - z(f); the 20k physics rate is
                    survey-weighted, its CI uses the Kish effective n;
  * (d)             weighted citation ventiles, Wilson CI at Kish n, then shift;
  * (e)(f)(g)       logit of recognition on log(1+pubs), log(1+cites), gender,
                    career stage (+ region for g); counterfactual marginals with
                    delta-method CI, survey-weighted for e/f, unweighted for g;
                    LLMs shifted, Wikipedia never corrected.
"""
import io
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from patsy import dmatrix          # ships with statsmodels
from scipy.stats import norm

logging.getLogger("fontTools").setLevel(logging.ERROR)   # quiet Helvetica.ttc subsetting notes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
DPI = 300

# ───────────────────────────── house style ──────────────────────────────
AX_W_IN, AX_H_IN = 2.66, 2.325            # fixed inner axes box (798 x 698 px)
FIGSIZE, MARGIN_L_IN, MARGIN_B_IN = (5.0, 4.0), 1.45, 1.05
LEGEND_KW = dict(fontsize=7, frameon=True, handlelength=1.2, handletextpad=0.5,
                 labelspacing=0.3, borderpad=0.4, markerscale=1.0)
LABEL_FS, AXLABEL_FS, TICK_FS = 10, 12, 9

HOUSE_RC = {
    "axes.spines.top": False, "axes.spines.left": False, "axes.spines.right": False,
    "ytick.left": False, "axes.grid": True, "axes.grid.which": "major",
    "axes.grid.axis": "y", "grid.linestyle": "--", "grid.alpha": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
}

CLAUDE_COLOR, GEMINI_COLOR, GPT_COLOR, WIKI_COLOR = "#BB8E13", "#28A969", "#E47091", "#2F9DDD"
MODELS = [("claude46", "Claude 4.6 Sonnet", CLAUDE_COLOR),
          ("gemini35", "Gemini 3.5 Flash", GEMINI_COLOR),
          ("gpt54", "GPT-5.4", GPT_COLOR)]
REF = "claude46"
FAMILY = {"claude46": "Claude", "gemini35": "Gemini", "gpt54": "GPT"}
XTICK_COLOR = {"Claude": CLAUDE_COLOR, "Gemini": GEMINI_COLOR, "GPT": GPT_COLOR,
               "Wikipedia": WIKI_COLOR, "Wiki": WIKI_COLOR}

FIELDS = ["Physical Sciences", "Biological Sciences", "Chemical Sciences", "Human Society"]
FIELD_LABELS = dict(zip(FIELDS, ["Physics", "Biology", "Chemistry", "Social sciences"]))
GENDER_ORDER = ["female", "male"]
STAGE_ORDER = ["early-career", "mid-career", "late-career"]
REGIONS = ["Africa", "Asia", "Eastern Europe", "Latin America", "North America",
           "Western Europe"]                  # Oceania sampled but not shown

VALUES = []                                   # every plotted point -> fig1_values.csv


def record(panel, series, group, p, lo, hi, **extra):
    VALUES.append(dict(panel=panel, series=series, group=group, p=100 * p,
                       lo=100 * lo, hi=100 * hi, **extra))


# ───────────────────────────── data ─────────────────────────────────────
SCI = pd.read_csv(os.path.join(DATA, "physicists_20k.csv"))
PROBE = pd.read_csv(os.path.join(DATA, "fake_name_probe_counts.csv")).set_index("model")
FIELDS10K = pd.read_csv(os.path.join(DATA, "other_fields_counts.csv"))
TIMELINE = pd.read_csv(os.path.join(DATA, "timeline_counts.csv"), parse_dates=["release"])
OPENALEX = pd.read_csv(os.path.join(DATA, "openalex_coverage.csv")).set_index("field")[
    "openalex_retrievable_pct"].to_dict()

FPR = {m: PROBE.loc[m, "fake_yes"] / PROBE.loc[m, "fake_n"] for m, _, _ in MODELS}
NFAKE = {m: int(PROBE.loc[m, "fake_n"]) for m, _, _ in MODELS}
REF_F = FPR[REF]
Z_REF = norm.ppf(REF_F)


def frame(outcome):
    """Per-scientist frame in the column layout the estimators expect."""
    d = SCI.rename(columns={"citations": "cum_citations", "publications": "cum_publication_count",
                            "gender": "Gender", "id": "author_id"}).copy()
    d["y"] = d[outcome].astype(float)         # NaN = no parsable answer (left out)
    return d


# ───────────────────────────── statistics ───────────────────────────────
def sdt_shift(p, f_model, f_ref=REF_F):
    c = lambda x: np.clip(x, 1e-6, 1 - 1e-6)
    return float(norm.cdf(norm.ppf(c(p)) - norm.ppf(c(f_model)) + norm.ppf(c(f_ref))))


def sdt_rate_ci(k, n, f, nf, z_ref=Z_REF):
    """Corrected rate and delta-method 95% CI."""
    p = np.clip(k / n, 1e-4, 1 - 1e-4)
    f = np.clip(f, 1e-4, 1 - 1e-4)
    zp, zf = norm.ppf(p), norm.ppf(f)
    delta = zp - zf
    se = np.sqrt(p * (1 - p) / (n * norm.pdf(zp) ** 2) + f * (1 - f) / (nf * norm.pdf(zf) ** 2))
    return (norm.cdf(delta + z_ref), norm.cdf(delta - 1.96 * se + z_ref),
            norm.cdf(delta + 1.96 * se + z_ref))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0, c - h), min(1, c + h)


def kish(w):
    return w.sum() ** 2 / (w ** 2).sum()


def physics_rate(model):
    """Survey-weighted corrected physics rate on the 20k (Kish effective n)."""
    d = frame(model).dropna(subset=["y", "weight"])
    p = float(np.average(d["y"], weights=d["weight"]))
    neff = kish(d["weight"])
    return sdt_rate_ci(int(round(p * neff)), int(round(neff)), FPR[model], NFAKE[model])


def weighted_share(col, weight):
    d = SCI.dropna(subset=[col])
    return float(np.average(d[col], weights=weight.loc[d.index]))


def wikipedia_rate():
    d = SCI.dropna(subset=["weight"])
    p = float(np.average(d["wikipedia"], weights=d["weight"]))
    neff = kish(d["weight"])
    _, lo, hi = wilson(int(round(p * neff)), int(round(neff)))
    return p, lo, hi


def fit_logit(d, extra=""):
    d = d.copy()
    d["log_pubs"] = np.log1p(d["cum_publication_count"])
    d["log_cites"] = np.log1p(d["cum_citations"])
    need = ["y", "log_pubs", "log_cites", "Gender", "career_stage", "author_id"] + ([extra] if extra else [])
    d = d.dropna(subset=need)
    d["y"] = d["y"].astype(int)
    f = "y ~ log_pubs + log_cites + C(Gender) + C(career_stage)" + (f" + C({extra})" if extra else "")
    # one row per scientist, so clustering on the id is the HC-robust sandwich
    fit = smf.logit(f, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["author_id"]})
    return fit, d


def marginal(fit, d_cf, weights=None):
    """Counterfactual average predicted probability + delta-method 95% CI."""
    p = fit.predict(d_cf).values
    X = np.asarray(dmatrix(fit.model.data.design_info, d_cf, return_type="matrix"))
    w = np.ones(len(p)) if weights is None else np.where(np.isfinite(weights), weights, 0.0)
    w = w / w.sum()
    mp = float(np.sum(w * p))
    g = ((w * p * (1 - p))[:, None] * X).sum(axis=0)
    se = float(np.sqrt(g @ fit.cov_params().values @ g))
    return mp, mp - 1.96 * se, mp + 1.96 * se


def demo_rows(kind, weighted, panel):
    groups, col = {"gender": (GENDER_ORDER, "Gender"), "stage": (STAGE_ORDER, "career_stage"),
                   "region": (REGIONS, "region")}[kind]
    rows, fits = [], {}
    for src in [m for m, _, _ in MODELS] + ["wikipedia"]:
        d = frame(src)
        if kind == "region":
            d = d[d["region"].isin(REGIONS)]
        fit, used = fit_logit(d, extra="region" if kind == "region" else "")
        fits[src] = fit
        w = used["weight"].values if weighted else None
        name = FAMILY.get(src, "Wikipedia")
        for g in groups:
            mp, lo, hi = marginal(fit, used.assign(**{col: g}), w)
            if src != "wikipedia":
                mp, lo, hi = (sdt_shift(x, FPR[src]) for x in (mp, lo, hi))
            rows.append(dict(model=name, group=g, p=mp, lo=lo, hi=hi))
            record(panel, name, g, mp, lo, hi)
    return rows, fits


# ───────────────────────────── drawing helpers ──────────────────────────
def new_panel():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes((MARGIN_L_IN / FIGSIZE[0], MARGIN_B_IN / FIGSIZE[1],
                       AX_W_IN / FIGSIZE[0], AX_H_IN / FIGSIZE[1]))
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_FS)
    return fig, ax


def style_model_xaxis(ax, labels):
    for i in range(len(labels)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color="0.92", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=LABEL_FS)
    for tick, m in zip(ax.get_xticklabels(), labels):
        tick.set_color(XTICK_COLOR.get(m, "black"))
    ax.set_xlim(-0.5, len(labels) - 0.5)


def grouped_panel(rows, order, colors, ymax=None):
    fig, ax = new_panel()
    labels = list(dict.fromkeys(r["model"] for r in rows))
    offs = np.linspace(-0.22, 0.22, len(order))
    style_model_xaxis(ax, labels)
    for i, m in enumerate(labels):
        for off, g in zip(offs, order):
            r = next(r for r in rows if r["model"] == m and r["group"] == g)
            ax.errorbar(i + off, r["p"] * 100, yerr=[[(r["p"] - r["lo"]) * 100], [(r["hi"] - r["p"]) * 100]],
                        fmt="o", color=colors[g], ecolor=colors[g], markersize=6, elinewidth=1,
                        capsize=2.5, capthick=1, markeredgecolor="none", zorder=3)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[g],
                              markersize=6, label=g) for g in order], loc="upper right", **LEGEND_KW)
    ax.set_ylim(0, ymax if ymax else max(r["hi"] for r in rows) * 100 * 1.30)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("Recognition rate (%)", fontsize=AXLABEL_FS)
    return fig


# ───────────────────────────── panels ───────────────────────────────────
def panel_a():
    rows = []
    for m, label, color in MODELS:
        p, lo, hi = physics_rate(m)
        rows.append(dict(x=FAMILY[m], label=label, color=color, p=p, lo=lo, hi=hi))
        record("a", label, "physics", p, lo, hi)
    p, lo, hi = wikipedia_rate()
    rows.append(dict(x="Wiki", label="Wikipedia", color=WIKI_COLOR, p=p, lo=lo, hi=hi))
    record("a", "Wikipedia", "physics", p, lo, hi)
    google = 100 * weighted_share("google_research_page", SCI["weight"].fillna(1.0))
    lines = [(OPENALEX["Physical Sciences"], "OpenAlex", (0, (6, 3))),
             (google, "Google Search", (0, (1, 1.6)))]
    VALUES.append(dict(panel="a", series="Google Search", group="physics", p=google))
    VALUES.append(dict(panel="a", series="OpenAlex", group="physics", p=OPENALEX["Physical Sciences"]))

    fig, ax = new_panel()
    style_model_xaxis(ax, [r["x"] for r in rows])
    for val, lab, dash in lines:
        ax.axhline(val, color="0.45", linestyle=dash, linewidth=1.4, zorder=2)
        ax.text(len(rows) - 0.55, val + 1.8, f"{lab} {val:.1f}%", ha="right", va="bottom",
                fontsize=LABEL_FS - 1.5, color="0.45")
    for i, r in enumerate(rows):
        ax.errorbar(i, r["p"] * 100, yerr=[[(r["p"] - r["lo"]) * 100], [(r["hi"] - r["p"]) * 100]],
                    fmt="o", color=r["color"], ecolor=r["color"], markersize=8, elinewidth=1.4,
                    capsize=3, capthick=1.4, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=r["color"],
                              markersize=6, label=r["label"]) for r in rows],
              loc="center right", bbox_to_anchor=(1.0, 0.5), **LEGEND_KW)
    ax.set_ylim(0, 107)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{t:.0f}%" for t in np.arange(0, 101, 20)])
    ax.set_ylabel("Recognition rate (%)", fontsize=AXLABEL_FS)
    return fig


def panel_b():
    dash = {"Physical Sciences": (0, (6, 3)), "Biological Sciences": (0, (1.5, 1.5)),
            "Chemical Sciences": (0, (7, 2, 1, 2)), "Human Society": (0, (4, 1.5, 1, 1.5, 1, 1.5))}
    pts = {}
    for f in FIELDS:
        for m, label, _ in MODELS:
            if f == "Physical Sciences":
                pts[(f, m)] = physics_rate(m)
            else:
                r = FIELDS10K[(FIELDS10K.model == m) & (FIELDS10K.field == f)].iloc[0]
                pts[(f, m)] = sdt_rate_ci(int(r.n_yes), int(r.n_yes + r.n_no + r.n_unknown),
                                          FPR[m], NFAKE[m])
            record("b", label, FIELD_LABELS[f], *pts[(f, m)])

    fig, ax = new_panel()
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.grid(False, axis="y")
    for fi, f in enumerate(FIELDS):
        if fi % 2 == 0:
            ax.axhspan(fi - 0.45, fi + 0.45, color="#f0f0f0", zorder=0)
        for mi, (m, label, color) in enumerate(MODELS):
            p, lo, hi = pts[(f, m)]
            ax.errorbar(p * 100, fi + (mi - 1) * 0.22, xerr=[[(p - lo) * 100], [(hi - p) * 100]],
                        fmt="o", color=color, ecolor=color, markersize=7, elinewidth=1.5, capsize=3,
                        capthick=1.5, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        cov = OPENALEX[f]
        ax.plot([cov, cov], [fi - 0.45, fi + 0.45], color="0.45", linestyle=dash[f], linewidth=1.4,
                zorder=4, solid_capstyle="butt")
        ax.text(cov - 2.2, fi, f"{cov:.1f}%", ha="right", va="center", fontsize=LABEL_FS - 1.5, color="0.45")
    ax.set_yticks(range(len(FIELDS)))
    ax.set_yticklabels([FIELD_LABELS[f] for f in FIELDS], fontsize=LABEL_FS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels([f"{t:.0f}%" for t in np.arange(0, 101, 20)])
    ax.set_xlabel("Recognition rate (%)", fontsize=AXLABEL_FS)
    handles = [Line2D([0], [0], marker="o", color=c, markersize=7, linestyle="none",
                      markeredgecolor="white", markeredgewidth=0.5) for _, _, c in MODELS]
    handles.append(Line2D([0], [0], color="0.45", linewidth=1.4, linestyle=dash["Physical Sciences"]))
    ax.legend(handles, [l for _, l, _ in MODELS] + ["Bibliometric retrieval"], loc="lower center",
              bbox_to_anchor=(0.55, 0.02), ncol=1, **LEGEND_KW)
    return fig


TIMELINE_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
               "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]}
TL_FAMILIES = {"Claude": dict(color="#BB8E13", marker="D"),
               "GPT": dict(color="#E47091", marker="o"),
               "Gemini": dict(color="#28A969", marker="s")}


def timeline_estimates():
    """Each release SDT-corrected to one threshold: Sonnet 4.6's FPR on the timeline probe."""
    t = TIMELINE.copy()
    ref = t.loc[t.model == "claude46s"].iloc[0]
    z_ref = norm.ppf(ref.fake_yes / ref.fake_n)
    est = [sdt_rate_ci(r.n_yes, r.n_yes + r.n_no + r.n_unknown, r.fake_yes / r.fake_n, r.fake_n, z_ref)
           for r in t.itertuples()]
    t["adj_rate"], t["adj_lo"], t["adj_hi"] = map(np.array, zip(*est))
    for r in t.itertuples():
        record("c", r.label, r.family, r.adj_rate, r.adj_lo, r.adj_hi, release=r.release.date())
    return t


def panel_c():
    gs = 2.325 / 3.772                                    # geometry scale of the original panel
    ax_w, ax_h = 12.157 * gs * 0.90, 2.325
    figsize = (9.3, 4.4)
    ms, elw, mew, caps, capw = 10 * gs, 2.4 * gs, 1.2 * gs, 4.0 * gs, 1.8 * gs
    df = timeline_estimates().sort_values("release").reset_index(drop=True)

    fig = plt.figure(figsize=figsize, dpi=DPI)
    ax = fig.add_axes((1.45 / figsize[0], 1.05 / figsize[1], ax_w / figsize[0], ax_h / figsize[1]))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="0.90", alpha=1)
    ax.grid(False, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", length=4, width=0.9, labelsize=9)

    base = df["adj_hi"].max() * 100 + 15.0                # common baseline for the names
    r_lo, r_hi = df["adj_rate"].min() * 100, df["adj_rate"].max() * 100
    ax.axhspan(r_lo, r_hi, color="#cccccc", alpha=0.3, lw=0, zorder=0)
    ytr = ax.get_yaxis_transform()
    ax.text(0.997, r_hi + 1.4, f"{r_hi:.1f}%", transform=ytr, ha="right", va="bottom", fontsize=8,
            color="0.35", zorder=5)
    ax.text(0.997, r_lo - 1.4, f"{r_lo:.1f}%", transform=ytr, ha="right", va="top", fontsize=8,
            color="0.35", zorder=5)

    # markers at true release dates; only the rotated names are pushed apart (>= 25 days)
    df["x"] = mdates.date2num(df["release"])
    xl = df["x"].tolist()
    for _ in range(200):
        moved = False
        for i in range(len(xl) - 1):
            if xl[i + 1] - xl[i] < 25:
                s = (25 - (xl[i + 1] - xl[i])) / 2
                xl[i] -= s
                xl[i + 1] += s
                moved = True
        if not moved:
            break
    df["xlab"] = xl

    for fam, info in TL_FAMILIES.items():
        d = df[df.family == fam]
        ax.errorbar(d["x"], d["adj_rate"] * 100,
                    yerr=[((d["adj_rate"] - d["adj_lo"]) * 100).values, ((d["adj_hi"] - d["adj_rate"]) * 100).values],
                    fmt=info["marker"], color=info["color"], ecolor=info["color"], markersize=ms,
                    markeredgecolor="white", markeredgewidth=mew, elinewidth=elw, capsize=caps,
                    capthick=capw, linestyle="none", zorder=3)
    for _, r in df.iterrows():
        c = TL_FAMILIES[r["family"]]["color"]
        ax.plot([r["x"], r["xlab"]], [r["adj_hi"] * 100 + 0.8, base - 0.7], color=c, lw=0.8,
                alpha=0.55, linestyle=(0, (1, 2.2)), zorder=2)
        ax.text(r["xlab"], base, r["label"], rotation=90, ha="center", va="bottom", fontsize=10.5,
                color=c, zorder=4)

    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-08-15"))
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.tick_params(axis="x", which="minor", length=2.5, width=0.8)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Recognition rate (%)", fontsize=12, labelpad=6)
    ax.set_xlabel("Model release date", fontsize=12, labelpad=4)
    ax.legend(handles=[Line2D([0], [0], marker=i["marker"], color="none", markerfacecolor=i["color"],
                              markeredgecolor="white", markeredgewidth=mew, markersize=ms + 1, label=f)
                       for f, i in TL_FAMILIES.items()],
              loc="upper right", bbox_to_anchor=(1.0, 1.03), ncol=1, frameon=False, fontsize=7,
              handletextpad=0.4, labelspacing=0.45, borderpad=0)
    return fig


def wquantile(v, w, qs):
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, v)


def panel_d():
    ref = frame(REF)
    ref = ref[ref["weight"].fillna(0) > 0]
    med = float(wquantile(ref["cum_citations"], ref["weight"], [0.5])[0])
    edges = wquantile(ref["cum_citations"], ref["weight"], np.linspace(0, 1, 21))
    edges[0], edges[-1] = -0.5, ref["cum_citations"].max() + 1
    edges = np.unique(edges)
    ymin, ymax = 5, 55

    fig, ax = new_panel()
    for m, label, color in MODELS:
        s = frame(m).dropna(subset=["cum_citations", "y"])
        s = s[s["weight"].fillna(0) > 0].copy()
        s["bin"] = pd.cut(s["cum_citations"], bins=edges, labels=False, include_lowest=True)
        x, p_, lo_, hi_ = [], [], [], []
        for b in range(len(edges) - 1):
            g = s[s["bin"] == b]
            w = g["weight"].values
            p = float(np.average(g["y"], weights=w))
            neff = kish(w)
            _, lo, hi = wilson(int(round(p * neff)), int(round(neff)))
            p, lo, hi = (sdt_shift(v, FPR[m]) for v in (p, lo, hi))
            xm = wquantile(g["cum_citations"], w, [0.5])[0]
            x.append(xm); p_.append(p); lo_.append(lo); hi_.append(hi)
            record("d", label, f"ventile {b + 1}", p, lo, hi, median_citations=xm, n=len(g))
        x, p_, lo_, hi_ = map(np.array, (x, p_, lo_, hi_))
        ax.plot(x, p_ * 100, "-o", color=color, ms=4.5, mec="white", mew=0.6, lw=1.5, zorder=3, label=label)
        ax.fill_between(x, lo_ * 100, hi_ * 100, color=color, alpha=0.15, lw=0)
    ax.axvline(med, color="grey", ls=":", lw=0.9, zorder=1)
    ax.text(med * 1.15, ymin + (ymax - ymin) * 0.46, "median citations", color="dimgrey",
            fontsize=6.5, ha="left", va="center", rotation=90)
    ax.set_xscale("log")
    ax.set_xlabel("Citations", fontsize=AXLABEL_FS)
    ax.set_ylabel("Recognition rate (%)", fontsize=AXLABEL_FS)
    ax.set_ylim(ymin, ymax)
    ticks = list(range(10, ymax + 1, 10))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks])
    ax.legend(loc="upper left", **LEGEND_KW)
    VALUES.append(dict(panel="d", series="median citations", group="all", p=np.nan, median_citations=med))
    return fig


FITS = {}


def panel_e():
    rows, FITS["gender"] = demo_rows("gender", True, "e")
    return grouped_panel(rows, GENDER_ORDER, {"female": "#b05a6b", "male": "#7aa3d4"})


def panel_f():
    rows, _ = demo_rows("stage", True, "f")
    return grouped_panel(rows, STAGE_ORDER, {"early-career": "#c6d3ec", "mid-career": "#8a96c9",
                                             "late-career": "#6a3a8c"})


def panel_g():
    rows, _ = demo_rows("region", False, "g")
    return grouped_panel(rows, REGIONS, {r: cm.Set2(i) for i, r in enumerate(REGIONS)}, ymax=40)


# ───────────────────────────── composite ────────────────────────────────
# Layout (3850 px wide at 300 dpi): two panels per row, panel c on its own row.
W, PAD = 3850, 40
C_SCALE = 1.601          # panel c is drawn 1.6x larger than its native 300-dpi size
C_ROW_H, C_TOP = 1350, 3  # panel c's row height and the gap above its content
LABEL_SIZE = 20          # axis-label size as it appears in the composite (pt x render scale)
YLABEL_X = 90            # left edge of the left-column y labels (a, c, d, f), canvas px


def render(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return (plt.imread(buf)[..., :3] * 255).round().astype(np.uint8)


def set_label_size(fig, s):
    """Axis labels at LABEL_SIZE once the panel is scaled by s, so all panels match."""
    for ax in fig.axes:
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text():
                lab.set_fontsize(LABEL_SIZE / s)


def row_render(pair, figs, rc):
    """Render a pair of panels at one scale that fills the row width, so both axes boxes have
    the same size; also return how far to drop each panel so the two x axes line up, and
    where the left panel's y label (its leftmost element) starts."""
    s = 1.0
    for _ in range(3):           # label size and row scale depend on each other; this converges
        widths = {}
        for k in pair:
            with plt.rc_context(rc[k]):
                set_label_size(figs[k], s)
                widths[k] = render(figs[k], DPI).shape[1]
        s = (W - 3 * PAD) / sum(widths.values())
    imgs, drop = {}, {}
    for k in pair:
        with plt.rc_context(rc[k]):
            set_label_size(figs[k], s)
            imgs[k] = render(figs[k], DPI * s)
            top_in = figs[k].get_tightbbox().y1 + plt.rcParams["savefig.pad_inches"]
        drop[k] = (top_in - MARGIN_B_IN) * DPI * s       # image top -> x axis, in px
    lowest = max(drop.values())
    return imgs, {k: round(lowest - drop[k]) for k in pair}, content_box(imgs[pair[0]])[0]


def content_box(img):
    nw = (img < 250).any(axis=2)
    ys, xs = np.where(nw.any(axis=1))[0], np.where(nw.any(axis=0))[0]
    return xs[0], ys[0]


def vector_pdf(figs, rc, labelled, place, width, total_h):
    """fig1.pdf with the PNG's layout, but each panel kept as vector PDF (selectable text)."""
    from pypdf import PdfReader, PdfWriter, Transformation
    base_buf = io.BytesIO()
    fig = labelled(None)
    fig.savefig(base_buf, format="pdf", dpi=DPI, facecolor="white")
    plt.close(fig)
    base = PdfReader(base_buf).pages[0]
    page_h = float(base.mediabox.height)
    px = 72 / DPI                                    # canvas px -> PDF pt
    for k, (x, y) in place.items():
        buf = io.BytesIO()
        with plt.rc_context(rc[k]):
            figs[k].savefig(buf, format="pdf", bbox_inches="tight", transparent=True)
        page = PdfReader(buf).pages[0]
        s = width[k] * px / float(page.mediabox.width)
        ty = page_h - y * px - float(page.mediabox.height) * s
        base.merge_transformed_page(page, Transformation().scale(s).translate(x * px, ty))
    out = PdfWriter()
    out.add_page(base)
    with open(os.path.join(OUT, "fig1.pdf"), "wb") as f:
        out.write(f)


def main():
    os.makedirs(os.path.join(OUT, "panels"), exist_ok=True)
    # tick labels are created at draw time, so every render runs inside its panel's rc context
    rc = {k: HOUSE_RC for k in "abdefg"}
    rc["c"] = TIMELINE_RC
    figs = {}
    for k, f in zip("abcdefg", (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g)):
        with plt.rc_context(rc[k]):
            figs[k] = f()
            figs[k].savefig(os.path.join(OUT, "panels", f"fig1{k}.pdf"), bbox_inches="tight")

    rows = [["a", "b"], ["c"], ["d", "e"], ["f", "g"]]
    imgs, drop, lab_x = {}, {}, {}
    for pair in (["a", "b"], ["d", "e"], ["f", "g"]):
        row_imgs, row_drop, lab_x[pair[0]] = row_render(pair, figs, rc)
        imgs.update(row_imgs)
        drop.update(row_drop)
    # panel c: fixed scale; its content starts C_TOP below the row top, cropped to the row height
    with plt.rc_context(rc["c"]):
        set_label_size(figs["c"], C_SCALE)
        raw = render(figs["c"], DPI * C_SCALE)
    lab_x["c"], y0 = content_box(raw)
    imgs["c"], drop["c"] = raw[y0:y0 + C_ROW_H - C_TOP], C_TOP
    # rendered width of each panel (sets its PDF scale); panel c's PDF page starts y0 above its crop
    width = {k: imgs[k].shape[1] for k in "abdefg"}
    width["c"], shift = raw.shape[1], {"c": -y0}

    heights = [max(drop[k] + imgs[k].shape[0] for k in r) for r in rows]
    heights[1] = C_ROW_H
    total_h = sum(heights) + PAD * (len(rows) + 1)
    canvas = np.full((total_h, W, 3), 255, np.uint8)
    letters, place = [], {}
    y = PAD
    for r, rh in zip(rows, heights):
        for j, k in enumerate(r):
            im = imgs[k]
            # left column: y labels at YLABEL_X; right panels flush right, so rows end together
            x = round(YLABEL_X - lab_x[k]) if j == 0 else W - PAD - im.shape[1]
            im = im[:, max(0, -x):W - max(x, 0)]
            x = max(x, 0)
            yk = y + drop[k]
            canvas[yk:yk + im.shape[0], x:x + im.shape[1]] = im
            place[k] = (round(YLABEL_X - lab_x[k]) if j == 0 else x, yk + shift.get(k, 0))
            letters.append((k, PAD if j == 0 else x, y))
        y += rh + PAD

    def labelled(image):
        fig = plt.figure(figsize=(W / DPI, total_h / DPI), dpi=DPI)
        if image is not None:
            fig.figimage(image, 0, 0, origin="upper")
        for k, x, yt in letters:
            # label top-left at (x, yt - 10) with 110-px Arial Bold, as in the published composite
            fig.text(x / W, 1 - (yt - 10 + 100) / total_h, k, fontsize=110 / DPI * 72, fontweight="bold",
                     family=["Arial", "Helvetica", "DejaVu Sans"], va="baseline", ha="left")
        return fig

    fig = labelled(canvas)
    fig.savefig(os.path.join(OUT, "fig1.png"), dpi=DPI)
    plt.close(fig)
    vector_pdf(figs, rc, labelled, place, width, total_h)
    for f in figs.values():
        plt.close(f)

    pd.DataFrame(VALUES).to_csv(os.path.join(OUT, "fig1_values.csv"), index=False, float_format="%.4f")
    report()


def report():
    """Numbers quoted in the caption and Sec. 2.1."""
    v = pd.DataFrame(VALUES)
    get = lambda pan, s, g: float(v[(v.panel == pan) & (v.series == s) & (v.group == g)].p.iloc[0])
    lab = {m: l for m, l, _ in MODELS}
    print("(a) physics, corrected: " + ", ".join(f"{FAMILY[m]} {get('a', lab[m], 'physics'):.1f}%" for m, _, _ in MODELS)
          + f"; Wikipedia {get('a', 'Wikipedia', 'physics'):.2f}%; Google {get('a', 'Google Search', 'physics'):.1f}%")
    for f in FIELDS[1:]:
        vals = [get("b", lab[m], FIELD_LABELS[f]) for m, _, _ in MODELS]
        print(f"(b) {FIELD_LABELS[f]:16s} {min(vals):.1f}-{max(vals):.1f}%")
    b = v[v.panel == "b"].p
    print(f"(b) all field x model: {b.min():.1f}-{b.max():.1f}%")
    c = v[v.panel == "c"]
    latest = c.sort_values("release").groupby("group").tail(1)
    print(f"(c) range {c.p.min():.1f}-{c.p.max():.1f}%; latest per family: "
          + ", ".join(f"{r.series} {r.p:.1f}%" for r in latest.itertuples()))
    d = v[(v.panel == "d") & v.group.str.startswith("ventile")]
    lo_ = {s: g.p.iloc[0] for s, g in d.groupby("series")}
    hi_ = {s: g.p.iloc[-1] for s, g in d.groupby("series")}
    print("(d) bottom 5%: " + ", ".join(f"{FAMILY[m]} {lo_[lab[m]]:.1f}%" for m, _, _ in MODELS)
          + " | top 5%: " + ", ".join(f"{FAMILY[m]} {hi_[lab[m]]:.1f}%" for m, _, _ in MODELS)
          + f" | unrecognised in top 5%: {100 - max(hi_.values()):.0f}-{100 - min(hi_.values()):.0f}%")
    for pan, a_, b_ in (("e", "male", "female"), ("f", "late-career", "early-career")):
        print(f"({pan}) {a_}/{b_} - 1: " + ", ".join(
            f"{s} {100 * (get(pan, s, a_) / get(pan, s, b_) - 1):.0f}%" for s in ["Claude", "Gemini", "GPT", "Wikipedia"]))
    fw = FITS["gender"]["wikipedia"]
    k = [n for n in fw.params.index if n.startswith("C(Gender)")][0]
    print(f"(e) Wikipedia gender coefficient {k}: z={fw.tvalues[k]:.2f}, p={fw.pvalues[k]:.2f}")
    print("(g) Claude: " + ", ".join(f"{r} {get('g', 'Claude', r):.1f}%" for r in REGIONS))
    print(f"-> {os.path.relpath(os.path.join(OUT, 'fig1.png'))}")


if __name__ == "__main__":
    main()
