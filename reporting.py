"""
The tables and figures of Part 1, from saved metrics only (no retraining, no resimulation).
Imports common, preprocessing, training and postprocessing.

Contents
    1. Tables (was tables.py)
    2. Figures (was figures.py)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")                     # files only, no window: runs on a server too
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import CELLS, SEEDS, TRAINING_CONFIGS, block_dir, cell_levers, log, run_path
from .postprocessing import CALIBRATION_POINTS, CAL_NAMES, TOL_PHI, load_metrics


# =====================================================================================================================
# 1. Tables (was tables.py)
# =====================================================================================================================

# Stage 14 (Part 1, G7): the tables, built from saved metrics only (pipeline.evaluate_step), as CSV and LaTeX booktabs.
#
# Every ablation table has one row per cell A-D (cells not run are left out), the two lever columns (Loss,
# Weighting), then for each metric: the mean over seeds, the difference from cell A, and [min, max] over seeds. Every
# number is computed per seed first; scenarios are never pooled across seeds. Metrics that are themselves an average
# over the B generated series (PCA, correlations) also show +-1.96 standard deviations over the series (mean over seeds).
#
#     tab_admissibility   Table 1 statistics of the daily mean Phi, violation share, Phi parts; the raw vendor grid
#                         (the market) as the reference row first
#     tab_calibration     main: nine-point average, return, surface-VIX;  _full: each of the 11 variables (appendix)
#     tab_dynamics        mean rho_t, corr(return, PC1-3 projections) (Table 6), the reduced correlation matrix (Table 7)
#     tab_pca             explained variance of PC1-3 and |<v_k data, v_k model>| (Tables 4 and 5)
#     tab_reweighting     cells A, C: median ESS, its 5th percentile, share of days with ESS < 100
#     tab_effects         main effect of each lever (Loss, Weighting) on five headline metrics (2^2 contrast, per seed)

STATS = ("mean", "diff_A", "min", "max")
LEVERS = ("Loss", "Weighting")


class Metric:
    """One column group: fn(metrics of one cell and seed) -> number; pm(...) -> 1.96 sd over series (optional)."""

    def __init__(self, label, fn, fmt="%.3f", pm=None, data=None):
        self.label, self.fn, self.fmt, self.pm, self.data = label, fn, fmt, pm, data


def _per_seed(index, block, seeds):
    """{cell: [metrics dict of each available seed]}"""
    out = {}
    for cell in CELLS:
        mets = [m for m in (load_metrics(index, block, cell, s) for s in seeds) if m is not None]
        if mets:
            out[cell] = mets
    return out


def ablation_frame(per_seed, metrics, data_rows=None):
    """Numeric table: rows = data rows then cells; columns (metric label, stat)."""
    rows, index = [], []
    ref = {}
    for cell, mets in per_seed.items():
        for m in metrics:
            if cell == "A":
                ref[m.label] = np.mean([m.fn(x) for x in mets])
    for label, values in (data_rows or {}).items():
        row = {("", lev): "" for lev in LEVERS}          # the row name (first column) says what it is
        for m in metrics:
            v = values.get(m.label, np.nan)
            row.update({(m.label, "mean"): v, (m.label, "diff_A"): np.nan, (m.label, "min"): np.nan,
                        (m.label, "max"): np.nan})
            if m.pm is not None:
                row[(m.label, "pm196")] = np.nan
        rows.append(row)
        index.append(label)
    for cell, mets in per_seed.items():
        row = dict(zip([("", lev) for lev in LEVERS], cell_levers(cell)))
        for m in metrics:
            v = np.array([m.fn(x) for x in mets], dtype=float)
            row.update({(m.label, "mean"): v.mean(), (m.label, "diff_A"): v.mean() - ref.get(m.label, np.nan),
                        (m.label, "min"): v.min(), (m.label, "max"): v.max()})
            if m.pm is not None:
                row[(m.label, "pm196")] = float(np.mean([m.pm(x) for x in mets]))
        row[("", "seeds")] = len(mets)
        rows.append(row)
        index.append(cell)
    df = pd.DataFrame(rows, index=index)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _fmt(x, fmt):
    if isinstance(x, str):
        return x
    if x is None or not np.isfinite(x):
        return ""
    return fmt % x


def to_latex(df, metrics, path, caption, label):
    """booktabs LaTeX: one column group per metric with mean, diff from A, [min, max] (and +-1.96 sd)."""
    groups = []
    for m in metrics:
        sub = [("mean", lambda r, m=m: _fmt(r[(m.label, "mean")], m.fmt)),
               (r"$\Delta$A", lambda r, m=m: _fmt(r[(m.label, "diff_A")], "%+" + m.fmt[1:])),
               ("[min, max]", lambda r, m=m: "" if not np.isfinite(r[(m.label, "min")]) else
                "[%s, %s]" % (_fmt(r[(m.label, "min")], m.fmt), _fmt(r[(m.label, "max")], m.fmt)))]
        if m.pm is not None:
            sub.append((r"$\pm1.96$sd", lambda r, m=m: _fmt(r[(m.label, "pm196")], m.fmt)))
        groups.append((m.label, sub))
    ncol = 3 + sum(len(s) for _, s in groups)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\caption{%s}" % caption, r"\label{%s}" % label,
             r"\begin{tabular}{l ll %s}" % " ".join("r" * len(s) for _, s in groups), r"\toprule",
             " & ".join(["", "", ""] + [r"\multicolumn{%d}{c}{%s}" % (len(s), tex(g)) for g, s in groups]) + r" \\"]
    start = 4
    rules = []
    for _, s in groups:
        rules.append(r"\cmidrule(lr){%d-%d}" % (start, start + len(s) - 1))
        start += len(s)
    lines.append(" ".join(rules))
    lines.append(" & ".join(["Cell", "Loss", "Weighting"] + [h for _, s in groups for h, _ in s]) + r" \\")
    lines.append(r"\midrule")
    for name, r in df.iterrows():
        cells = [tex(str(name))] + [tex(str(r[("", lev)])) for lev in LEVERS]
        cells += [f(r) for _, s in groups for _, f in s]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    assert all(line.count("&") in (0, ncol - 1) for line in lines if r"\\" in line and "multicolumn" not in line)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def tex(s):
    return (s.replace("%", r"\%").replace("_", r"\_").replace("Phi", r"$\Phi$").replace("rho", r"$\rho$")
            .replace("sigma", r"$\sigma$").replace("|<", r"$|\langle$").replace(">|", r"$\rangle|$"))


def simple_table(df, out, name, fmt, caption):
    """A plain table as CSV and booktabs LaTeX (headers escaped, the label left as it is)."""
    df.to_csv(os.path.join(out, name + ".csv"), index=False)
    with open(os.path.join(out, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(df.rename(columns=lambda c: str(c).replace("_", r"\_")).to_latex(
            index=False, float_format=fmt, caption=caption, label="tab:" + name))


def save(df, metrics, out, name, caption):
    df.to_csv(os.path.join(out, name + ".csv"))
    to_latex(df, metrics, os.path.join(out, name + ".tex"), caption, "tab:" + name)


# ---- the tables

def admissibility(per_seed, market):
    metrics = [Metric("mean Phi", lambda x: x["phi_mean"].mean(), "%.2e"),
               Metric("std Phi", lambda x: x["phi_mean"].std(), "%.2e"),
               Metric("median Phi", lambda x: np.median(x["phi_mean"]), "%.2e"),
               Metric("violation share", lambda x: x["viol_share"].mean(), "%.4f"),
               Metric("calendar", lambda x: x["phi_parts"][:, 0].mean(), "%.2e"),
               Metric("call spread", lambda x: x["phi_parts"][:, 1].mean(), "%.2e"),
               Metric("butterfly", lambda x: x["phi_parts"][:, 2].mean(), "%.2e")]
    rows = {}
    if market is not None:
        p = market["phi"]
        rows["Market (raw vendor grid)"] = {"mean Phi": p.mean(), "std Phi": p.std(), "median Phi": np.median(p),
                                       "violation share": (p > TOL_PHI).mean(), "calendar": market["phi_parts"][:, 0].mean(),
                                       "call spread": market["phi_parts"][:, 1].mean(),
                                       "butterfly": market["phi_parts"][:, 2].mean()}
    return metrics, ablation_frame(per_seed, metrics, rows)


def _cov(stat, cols):
    keys = {"below 2.5%": "below_025", "above 97.5%": "above_975", "inside 95%": "inside_95",
            "below 1%": "below_01", "above 99%": "above_99"}
    if stat == "median width":
        return lambda x: np.mean([np.median(x["width_95"][:, c]) for c in cols])
    return lambda x: 100 * x[keys[stat]][:, cols].mean()


def calibration_main(per_seed):
    groups = {"9-point avg": list(range(9)), "return": [9], "surface-VIX": [10]}
    metrics = []
    for g, cols in groups.items():
        for stat, fmt in (("inside 95%", "%.1f"), ("below 2.5%", "%.1f"), ("above 97.5%", "%.1f"),
                          ("median width", "%.4f")):
            metrics.append(Metric("%s: %s" % (g, stat), _cov(stat, cols), fmt))
    return metrics, ablation_frame(per_seed, metrics)


def calibration_full(per_seed, var):
    metrics = [Metric("%s" % stat, _cov(stat, [var]), fmt) for stat, fmt in
               (("below 1%", "%.1f"), ("below 2.5%", "%.1f"), ("inside 95%", "%.1f"), ("above 97.5%", "%.1f"),
                ("above 99%", "%.1f"), ("median width", "%.4f"))]
    return metrics, ablation_frame(per_seed, metrics)


def _series(key, *ix):
    return (lambda x: x[key][(slice(None),) + ix].mean(), lambda x: 1.96 * x[key][(slice(None),) + ix].std())


PAIRS = [(0, 1, "R, level"), (0, 2, "R, ATM 1M"), (0, 3, "R, VIX"), (1, 2, "level, ATM 1M"), (1, 3, "level, VIX"),
         (2, 3, "ATM 1M, VIX")]


def dynamics(per_seed, market):
    metrics = [Metric("mean rho_t", lambda x: x["rho_bar"], "%.3f")]
    for k in range(3):
        fn, pm = _series("corr_ret_pc", k)
        metrics.append(Metric("corr(R, PC%d)" % (k + 1), fn, "%.3f", pm))
    for i, j, name in PAIRS:
        fn, pm = _series("corr_mat", i, j)
        metrics.append(Metric("corr(%s)" % name, fn, "%.3f", pm))
    rows = {}
    if market is not None:
        rows["Data"] = {"corr(R, PC%d)" % (k + 1): market["ret_pc"][k] for k in range(3)}
        rows["Data"].update({"corr(%s)" % name: market["corrmat"][i, j] for i, j, name in PAIRS})
    return metrics, ablation_frame(per_seed, metrics, rows)


def pca(per_seed, market):
    metrics = []
    for k in range(3):
        fn, pm = _series("pca_share", k)
        metrics.append(Metric("PC%d share" % (k + 1), fn, "%.3f", pm))
    for k in range(3):
        fn, pm = _series("pca_inner", k)
        metrics.append(Metric("|<v%d data, v%d model>|" % (k + 1, k + 1), fn, "%.3f", pm))
    rows = {}
    if market is not None:
        rows["Data"] = {"PC%d share" % (k + 1): market["share"][k] for k in range(3)}
        rows["Data"].update({"|<v%d data, v%d model>|" % (k + 1, k + 1): 1.0 for k in range(3)})
    return metrics, ablation_frame(per_seed, metrics, rows)


def reweighting(per_seed):
    keep = {c: v for c, v in per_seed.items() if CELLS[c][1] == "max_penalty"}
    metrics = [Metric("median ESS", lambda x: np.median(x["ess"]), "%.0f"),
               Metric("ESS 5th pct", lambda x: np.percentile(x["ess"], 5), "%.0f"),
               Metric("days ESS < 100 (%)", lambda x: 100 * (x["ess"] < 100).mean(), "%.2f")]
    return metrics, ablation_frame(keep, metrics)


HEADLINE = {"mean daily Phi": lambda x: x["phi_mean"].mean(), "violation share": lambda x: x["viol_share"].mean(),
            "95% coverage, 9 points": lambda x: 100 * x["inside_95"][:, :9].mean(), "mean rho_t": lambda x: x["rho_bar"],
            "PC1 inner product": lambda x: x["pca_inner"][:, 0].mean()}


def effects(index, block, seeds):
    """
    Main effect of each lever: (mean over the 2 cells with the lever 'on') - (mean over the 2 with it 'off'), per
    seed, then the mean and range over seeds. 'On' = Phi in the loss, re-weighted output. Needs all 4 cells.
    A small effect, or one that appears only with the other lever, is a finding, not a failure.
    """
    on = {"Loss (Phi in - out)": lambda c: TRAINING_CONFIGS[CELLS[c][0]]["phi_in_loss"],
          "Weighting (re-weighted - unweighted)": lambda c: CELLS[c][1] == "max_penalty"}
    complete = []                                    # seeds with all four cells, each file read once
    for seed in seeds:
        mets = {c: load_metrics(index, block, c, seed) for c in CELLS}
        if all(v is not None for v in mets.values()):
            complete.append(mets)
    rows = []
    for lever, is_on in on.items():
        row = {"lever": lever}
        for metric, fn in HEADLINE.items():
            per = []
            for mets in complete:
                hi = np.mean([fn(mets[c]) for c in CELLS if is_on(c)])
                lo = np.mean([fn(mets[c]) for c in CELLS if not is_on(c)])
                per.append(hi - lo)
            row.update({(metric, "mean"): np.mean(per) if per else np.nan,
                        (metric, "min"): np.min(per) if per else np.nan,
                        (metric, "max"): np.max(per) if per else np.nan, (metric, "seeds"): len(per)})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("lever")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _load_market(index, block):
    p = os.path.join(block_dir(index, block), "market.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def make_tables(index="SPX", block="val", seeds=SEEDS):
    """Write every table into results/<INDEX>/<block>/tables/ (CSV and .tex), from saved metrics only."""
    out = os.path.join(block_dir(index, block), "tables")
    os.makedirs(out, exist_ok=True)
    per_seed = _per_seed(index, block, seeds)
    market = _load_market(index, block)
    if not per_seed:
        log("no metrics for %s %s: run the evaluate step first" % (index, block))
        return
    tag = "%s, %s block" % (index, "validation" if block == "val" else block)
    m, df = admissibility(per_seed, market)
    save(df, m, out, "tab_admissibility", "Admissibility (%s): daily mean $\\Phi$ and violation share" % tag)
    m, df = calibration_main(per_seed)
    save(df, m, out, "tab_calibration", "Calibration (%s): coverage of 95\\%% bands" % tag)
    frames = []
    for v, name in enumerate(CAL_NAMES):
        m, df = calibration_full(per_seed, v)
        frames.append(df.assign(variable=name))
        save(df, m, out, "tab_calibration_full_%02d" % v, "Calibration (%s): %s" % (tag, tex(name)))
    pd.concat(frames).to_csv(os.path.join(out, "tab_calibration_full.csv"))
    m, df = dynamics(per_seed, market)
    save(df, m, out, "tab_dynamics", "Dynamics (%s): mean $\\rho_t$ and correlations (Tables 6--7 format)" % tag)
    m, df = pca(per_seed, market)
    save(df, m, out, "tab_pca", "PCA of log-IV increments (%s), Tables 4--5 format" % tag)
    m, df = reweighting(per_seed)
    if len(df):
        save(df, m, out, "tab_reweighting", "Re-weighting diagnostics (%s)" % tag)
    if len(per_seed) == len(CELLS):
        eff = effects(index, block, seeds)
        eff.to_csv(os.path.join(out, "tab_effects.csv"))
        eff.columns = pd.MultiIndex.from_tuples([(a, b.replace("_", " ")) for a, b in eff.columns])
        with open(os.path.join(out, "tab_effects.tex"), "w", encoding="utf-8") as f:
            f.write(eff.to_latex(float_format="%.4g", caption="Main effects of the two levers (%s)" % tag,
                                 label="tab:tab_effects", multicolumn_format="c").replace("%", r"\%"))
    log("tables written to %s" % out)


# =====================================================================================================================
# 2. Figures (was figures.py)
# =====================================================================================================================

# Stage 14 (Part 1, G7): the figures, built from saved metrics only (pipeline.evaluate_step), as vector PDF + 300-dpi PNG.
#
# Style: serif text with Computer Modern maths, full-width figures 15.5 cm wide. Fixed colours in every figure:
# market brick #B03838, unweighted navy #1C3864, re-weighted ochre #B4781C, reference bands and grids slate #6C7C98.
# Identity never rests on colour alone: the market is a solid line, unweighted results a filled band or solid line,
# re-weighted results an outline band or dashed line. Time series run over the predicted dates of the block.
# Figures that show a single network use training seed 1, chosen in advance; captions.md says so.
#
# Main text (control networks only):  fig_phi_time, fig_phi_location, fig_bands_1m, fig_rho_time
# Optional:                           fig_pc_loadings
# Appendix:                           fig_bands_3m, fig_bands_12m, fig_grad_norms
# No other figures (brief section 10).


MARKET, UNWEIGHTED, REWEIGHTED, SLATE = "#B03838", "#1C3864", "#B4781C", "#6C7C98"
WIDTH = 15.5 / 2.54                       # inches
FIG_SEED = 1                              # the single network shown, chosen in advance
SEQ = LinearSegmentedColormap.from_list("navy_seq", ["#FFFFFF", "#C9D3E6", "#5B739E", UNWEIGHTED])
DIV = LinearSegmentedColormap.from_list("div", [UNWEIGHTED, "#FFFFFF", MARKET])

STYLE = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": SLATE,
         "axes.linewidth": 0.6, "xtick.color": "#333333", "ytick.color": "#333333", "axes.grid": True,
         "grid.color": SLATE, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
         "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight"}


class Figures:
    """Collects figures and their captions for one index and block."""

    def __init__(self, index, block, seeds):
        self.index, self.block, self.seeds = index, block, seeds
        self.out = os.path.join(block_dir(index, block), "figures")
        os.makedirs(self.out, exist_ok=True)
        self.captions = {}
        p = os.path.join(block_dir(index, block), "market.npz")
        self.market = dict(np.load(p)) if os.path.exists(p) else None
        self.block_name = "validation" if block == "val" else block

    def metrics(self, cell, seed):
        return load_metrics(self.index, self.block, cell, seed)

    def save(self, fig, name, caption):
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(os.path.join(self.out, "%s.%s" % (name, ext)), **kw)
        plt.close(fig)
        self.captions[name] = caption

    def write_captions(self):
        with open(os.path.join(self.out, "captions.md"), "w", encoding="utf-8") as f:
            f.write("# Figure captions (%s, %s block), generated with the figures\n\n" % (self.index, self.block_name))
            for name, cap in self.captions.items():
                f.write("**%s**: %s\n\n" % (name, cap))


def _dates(met):
    return pd.DatetimeIndex(met["dates"])


def _date_axis(ax, step=1):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_phi_time(F):
    per = {c: [m for m in (F.metrics(c, s) for s in F.seeds) if m is not None] for c in ("A", "B")}
    if not per["B"] or F.market is None:
        return log("fig_phi_time: needs cell B metrics and market.npz")
    dates = pd.DatetimeIndex(F.market["dates"])
    series = [F.market["phi"]] + [np.stack([m["phi_mean"] for m in per[c]]) for c in ("B", "A") if per[c]]
    pos = np.concatenate([s.ravel()[s.ravel() > 0] for s in series])
    log_axis = pos.size > 0 and pos.max() / pos.min() > 100
    floor = None
    if log_axis:      # re-weighted means can underflow to 1e-200: cut the axis at the 1st percentile, 6 decades at most
        floor = 10.0 ** np.floor(max(np.log10(np.percentile(pos, 1)), np.log10(pos.max()) - 6))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    mk = np.where(F.market["phi"] > 0, F.market["phi"], np.nan) if log_axis else F.market["phi"]
    ax.plot(dates, mk, color=MARKET, lw=0.8, marker="o", ms=1.6, label="Market (raw vendor grid)")
    for c, color, ls, label in (("B", UNWEIGHTED, "-", "Control networks, unweighted (B)"),
                                ("A", REWEIGHTED, "--", "Control networks, re-weighted (A)")):
        if not per[c]:
            continue
        x = np.stack([m["phi_mean"] for m in per[c]])
        if log_axis:
            x = np.where(x > 0, x, np.nan)
        with warnings.catch_warnings():                  # days where every seed has Phi = 0 stay gaps
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi, mid = np.nanmin(x, 0), np.nanmax(x, 0), np.nanmean(x, 0)
        ax.fill_between(dates, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(dates, mid, color=color, lw=1.0, ls=ls, label=label)
    if log_axis:
        ax.set_yscale("log", nonpositive="mask")
        ax.set_ylim(floor, 10.0 ** (np.ceil(np.log10(pos.max())) + 0.2))
    ax.set_ylabel(r"daily mean $\Phi$")
    _date_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    n = len(per["B"])
    F.save(fig, "fig_phi_time",
           "Daily mean arbitrage penalty $\\Phi$ over the %s block (%s): the raw vendor grid (market), and the "
           "control networks unweighted (cell B) and re-weighted (cell A). Model lines are the mean over %d training "
           "seed(s); the light band is the range across seeds.%s"
           % (F.block_name, F.index, n, " The y-axis is logarithmic (the series span more than two orders of "
              "magnitude) and starts at %.0e; days with $\\Phi = 0$ are not drawn." % floor if log_axis else ""))


def _grid_labels(ax, m, tau, bottom=True, left=True):
    ax.set_xticks(range(len(tau)))
    ax.set_xticklabels(["%d" % round(t * 365) for t in tau] if bottom else [], fontsize=6.5)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(["%.2f" % x for x in m] if left else [], fontsize=6.5)
    ax.tick_params(length=2)
    ax.grid(False)
    if bottom:
        ax.set_xlabel("maturity (days)")
    if left:
        ax.set_ylabel("moneyness $K/S$")


def _heat(ax, z, vmin, vmax, cmap, fmt):
    im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), v in np.ndenumerate(z):
        dark = abs(v - vmin) / (vmax - vmin + 1e-300) > 0.55 if cmap is SEQ else abs(v) / max(abs(vmin), abs(vmax)) > 0.55
        ax.text(j, i, fmt % v, ha="center", va="center", fontsize=5.5, color="white" if dark else "#222222")
    return im


def fig_phi_location(F):
    mb = [m for m in (F.metrics("B", s) for s in F.seeds) if m is not None]
    ma = [m for m in (F.metrics("A", s) for s in F.seeds) if m is not None]
    if not mb or F.market is None:
        return log("fig_phi_location: needs cell B metrics and market.npz")
    panels = [("Market, raw vendor grid\n(share of days)", 100 * F.market["viol_loc"].mean(0)),
              ("Unweighted, B\n(share of scenarios)", 100 * np.mean([m["viol_loc"].mean(0) for m in mb], axis=0))]
    if ma:
        panels.append(("Re-weighted, A\n(weighted share)", 100 * np.mean([m["viol_loc"].mean(0) for m in ma], axis=0)))
    vmax = max(p.max() for _, p in panels) or 1.0
    m, tau = F.market["m"], F.market["tau"]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 3.0), sharey=True)
    for k, (ax, (title, z)) in enumerate(zip(np.atleast_1d(axes), panels)):
        im = _heat(ax, z, 0.0, vmax, SEQ, "%.1f")
        ax.set_title(title)
        _grid_labels(ax, m, tau, left=k == 0)
    fig.colorbar(im, ax=axes, shrink=0.85, label="% taking part in a violation")
    F.save(fig, "fig_phi_location",
           "Where static arbitrage occurs on the grid (%s, %s block): the percentage of days on which each node of the "
           "raw vendor grid (the market) takes part in a violated calendar, call-spread or butterfly term, and the "
           "percentage of scenarios (averaged over days and %d training seed(s)) for the control networks, unweighted "
           "(B) and re-weighted (A, weighted share). One colour scale for all panels."
           % (F.index, F.block_name, len(mb)))


def fig_bands(F, days, name):
    mb, ma = F.metrics("B", FIG_SEED), F.metrics("A", FIG_SEED)
    if mb is None:
        return log("%s: needs cell B, seed %d" % (name, FIG_SEED))
    dates = _dates(mb)
    cols = [k for k, (mm, d) in enumerate(CALIBRATION_POINTS) if d == days]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharex=True)
    fig.subplots_adjust(wspace=0.42)
    for ax, k in zip(axes, cols):
        ax.fill_between(dates, mb["cal_q"][1, :, k], mb["cal_q"][3, :, k], color=UNWEIGHTED, alpha=0.2, lw=0,
                        label="95% band, unweighted (B)")
        if ma is not None:
            for q in (1, 3):
                ax.plot(dates, ma["cal_q"][q, :, k], color=REWEIGHTED, lw=0.6, ls="--",
                        label="95% band, re-weighted (A)" if q == 1 else None)
        ax.plot(dates, mb["cal_realised"][:, k], color=MARKET, lw=0.7, label="Realised implied vol")
        ax.set_title("$m = %.2f$" % CALIBRATION_POINTS[k][0])
        _date_axis(ax, 2)
        ax.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("implied volatility")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    F.save(fig, name,
           "Next-day 95%% bands for the %s implied volatility at moneyness 0.90, 1.00 and 1.10 (%s, %s block), "
           "control networks, training seed %d: realised value (raw vendor grid), the unweighted band (cell B, "
           "filled) and the re-weighted band (cell A, dashed outline)."
           % ({30: "1-month", 90: "3-month", 360: "12-month"}[days], F.index, F.block_name, FIG_SEED))


def fig_rho_time(F, overlay_a=False):
    mb = F.metrics("B", FIG_SEED)
    if mb is None:
        return log("fig_rho_time: needs cell B, seed %d" % FIG_SEED)
    dates = _dates(mb)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    lo, hi = mb["rho_band"]
    ax.axhspan(lo, hi, color=SLATE, alpha=0.3, lw=0, label="constant-correlation band (95%)")
    ax.axhline(mb["rho_bar"], color=SLATE, lw=0.6)
    ax.plot(dates, mb["rho"], color=UNWEIGHTED, lw=0.8, label=r"$\rho_t$, unweighted (B)")
    ma = F.metrics("A", FIG_SEED) if overlay_a else None
    if ma is not None:
        ax.plot(dates, ma["rho"], color=REWEIGHTED, lw=0.8, ls="--", label=r"$\rho_t$, re-weighted (A)")
    ax.set_ylabel(r"$\rho_t$")
    _date_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    F.save(fig, "fig_rho_time",
           "Cross-scenario correlation $\\rho_t$ between the simulated daily return and the generated change in "
           "one-month at-the-money log-volatility (%s, %s block), control networks unweighted (cell B), training seed "
           "%d. Slate: the constant-correlation band around the mean (Bonett and Wright 2000) with n = %d.%s"
           % (F.index, F.block_name, FIG_SEED, round(mb["rho_n"]),
              " Dashed ochre: re-weighted (cell A)." if ma is not None else ""))


def fig_pc_loadings(F):
    mb = F.metrics("B", FIG_SEED)
    if mb is None or F.market is None:
        return log("fig_pc_loadings: needs cell B seed %d and market.npz" % FIG_SEED)
    m, tau = F.market["m"], F.market["tau"]
    lt, lk = len(tau), len(m)
    data = F.market["pcs"][:, 0].reshape(lt, lk).T            # maturity-major vector -> (moneyness, maturity)
    model = mb["pca_v1_mean"].reshape(lt, lk).T
    v = max(abs(data).max(), abs(model).max())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    for k, (ax, title, z) in enumerate(zip(axes, ("Data", "Model, unweighted (B)"), (data, model))):
        im = _heat(ax, z, -v, v, DIV, "%.2f")
        ax.set_title(title)
        _grid_labels(ax, m, tau, left=k == 0)
    fig.colorbar(im, ax=axes, shrink=0.85, label="loading")
    F.save(fig, "fig_pc_loadings",
           "First principal component of daily log-implied-volatility increments (%s, %s block): the data (raw "
           "vendor grid) and the generated increments of the control networks (cell B, training seed %d; the mean over the "
           "generated series of their first eigenvector, each sign-aligned to the data's). One colour scale."
           % (F.index, F.block_name, FIG_SEED))


def fig_grad_norms(F):
    panels = []
    for name in TRAINING_CONFIGS:
        p = os.path.join(run_path(F.index, name, FIG_SEED), "logs.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                panels.append((name, z["match_norms"]))
    if not panels:
        return log("fig_grad_norms: no training logs for seed %d" % FIG_SEED)
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(WIDTH, 2.2 * rows), squeeze=False, sharex=True)
    terms = [("BCE", UNWEIGHTED, "-"), (r"$L_m$", REWEIGHTED, "--"), (r"$L_\tau$", SLATE, "-."), (r"$\Phi$", MARKET, ":")]
    for ax, (name, norms) in zip(axes.ravel(), panels):
        for col in range(norms.shape[1]):
            label, color, ls = terms[col]
            ax.plot(np.arange(1, len(norms) + 1), norms[:, col], color=color, ls=ls, lw=0.8, label=label)
        ax.set_yscale("log")
        ax.set_title(name.replace("_", " "))
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("generator update (match phase)")
    for ax in axes[:, 0]:
        ax.set_ylabel("gradient norm")
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes.ravel()), key=lambda h: len(h[0]))
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    F.save(fig, "fig_grad_norms",
           "Gradient norms with respect to the generator weights during the gradient-norm matching phase (the paper's "
           "Fig. 3 analogue), training seed %d of each training configuration (%s): the BCE term, the smoothness terms "
           "$L_m$ and $L_\\tau$, and $\\Phi$ for the configurations with $\\Phi$ in the loss. Logarithmic y-axis. An "
           "implementation check of the matching." % (FIG_SEED, F.index))


def make_figures(index="SPX", block="val", seeds=SEEDS, rho_overlay_a=False, which=None):
    """Write the figures into results/<INDEX>/<block>/figures/ (PDF + PNG and captions.md), from saved files only."""
    plt.rcParams.update(STYLE)
    F = Figures(index, block, seeds)
    makers = {"fig_phi_time": lambda: fig_phi_time(F), "fig_phi_location": lambda: fig_phi_location(F),
              "fig_bands_1m": lambda: fig_bands(F, 30, "fig_bands_1m"), "fig_rho_time": lambda: fig_rho_time(F, rho_overlay_a),
              "fig_pc_loadings": lambda: fig_pc_loadings(F), "fig_bands_3m": lambda: fig_bands(F, 90, "fig_bands_3m"),
              "fig_bands_12m": lambda: fig_bands(F, 360, "fig_bands_12m"), "fig_grad_norms": lambda: fig_grad_norms(F),
}
    for name, make in makers.items():
        if which is None or name in which:
            make()
    F.write_captions()
    log("figures written to %s" % F.out)
