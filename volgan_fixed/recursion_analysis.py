"""
Part 2, Part A: degradation curves, bootstrap and Holm tests, the pilot, and the tables and figures of the run.
Part 1 never imports this module.

Contents
    1. Degradation, bootstrap, tests (was rec_inference.py)
    2. The pilot (was rec_pilot.py)
    3. Tables and figures of the run (was rec_report.py)
    4. The pilot and report steps (was pipeline.py)
"""

from dataclasses import replace
import json
import os

import matplotlib
matplotlib.use("Agg")                     # files only, no window: runs on a server too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import check_inputs, get_device, log
from .reporting import MARKET, SLATE, STYLE, UNWEIGHTED, WIDTH
from .recursion import (
    Arena, N_VARS, QS, _rec_runs, bench_clouds, benchmark_full, parta_root, rec_margins_step, run_start,
    start_indices)


# =====================================================================================================================
# 1. Degradation, bootstrap, tests (was rec_inference.py)
# =====================================================================================================================

# Part 2, Part A: degradation, bootstrap and tests (brief section 7). Larger always means worse.
#
# Notation: seeds r = 1..5 (the files), starting days i = 1..n, horizon h; "free" is the free-running cloud, "bench" the
# one-step benchmark of the same target day. Per pillar P:
#
#   D_A(h) = mean over (r, i) of  phi_mean_free - phi_mean_bench                                   (signed)
#   D_C(h) = | cbar_free(h) - 0.95 | - | cbar_bench(h) - 0.95 |,  cbar = inside_95 pooled over the nine nodes, all
#            seeds and starts of that run type (the pooling comes first, then the distance)
#   D_D(h) = mean over (r, i) of  | rho_free - rho_bench |        (ED section 6.4, "mean absolute per-start difference";
#            the ED formula wins over the brief's market-referenced variant, as the brief directs when the two differ)
#
# Test statistic T_P = D_P(21) over all seeds and starts; per-seed T_{P,r} are the same over one seed's starts only.
# H0_P: mu_P <= delta_P against mu_P > delta_P. Moving-block bootstrap over the starts, in time order, fixed block
# length L = block_len_starts, blocks drawn with replacement without wrap-around, concatenated and truncated to n; the
# index matrix (B, n) is drawn ONCE and shared by every pillar, seed and horizon; seeds are not resampled.
# p = (1 + #{T* <= delta}) / (B + 1); Holm across (A, C, D); 95% percentile interval of T*.
#
# Clouds that cannot be scored (no valid path) are NaN: they drop out of the means and of the pooled coverage
# (their number is reported through S(h)); the statistics are sums and counts per start so the bootstrap is exact
# and fast.

PILLARS = ("A", "C", "D")
NINE = slice(0, N_VARS - 1)                      # the nine calibration nodes (the return column is last)
TARGET_COVERAGE = 0.95


# ---- reading the stored tables

def load_tables(root, seeds):
    """Stack metrics_{seed}.npz over seeds: dict name -> (S, n, H, ...) for the free_* / bench_* arrays."""
    recs = []
    for s in seeds:
        with np.load(os.path.join(root, "metrics_%d.npz" % s)) as z:
            recs.append({k: z[k] for k in z.files})
    out = {"seeds": np.array(seeds), "starts": recs[0]["starts"], "horizons": recs[0]["horizons"]}
    for r in recs:
        if not np.array_equal(r["starts"], out["starts"]):
            raise ValueError("the seeds do not share the same starting days")
    for k in recs[0]:
        if k.startswith(("free_", "bench_")) or k in ("id_ret", "id_dg", "id_sigma", "seconds"):
            out[k] = np.stack([r[k] for r in recs])
    out["target_dates"] = recs[0]["target_dates"]
    for k in ("real_values", "var_names"):
        if k in recs[0]:
            out[k] = recs[0][k]
    out["model_id"] = str(recs[0]["model_id"])
    return out


def split_runs(T):
    """(free, bench) dicts of the metric arrays, each (S, n, H, ...)."""
    f = {k[5:]: v for k, v in T.items() if k.startswith("free_")}
    b = {k[6:]: v for k, v in T.items() if k.startswith("bench_")}
    return f, b


# ---- sufficient statistics per (seed, start, horizon)

def sufficient(F, B):
    """
    Per pillar, arrays (S, n, H) of sums and counts whose totals give D_P on any selection of seeds and starts:
      A: (sum of differences, count);  C: (free hits, free count, bench hits, bench count);  D: (sum |diff|, count).
    """
    d = F["phi_mean"] - B["phi_mean"]
    ok = np.isfinite(d)
    A = (np.where(ok, d, 0.0), ok.astype(float))
    hf, hb = F["inside"][..., NINE], B["inside"][..., NINE]
    C = (np.nansum(hf, -1), np.isfinite(hf).sum(-1).astype(float), np.nansum(hb, -1), np.isfinite(hb).sum(-1).astype(float))
    e = np.abs(F["rho"] - B["rho"])
    ok = np.isfinite(e)
    D = (np.where(ok, e, 0.0), ok.astype(float))
    return {"A": A, "C": C, "D": D}


def value(P, sums):
    """D_P from totals (the arrays of `sufficient`, already summed over seeds and starts; any trailing axes)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if P == "C":
            fs, fc, bs, bc = sums
            return np.abs(fs / fc - TARGET_COVERAGE) - np.abs(bs / bc - TARGET_COVERAGE)
        num, cnt = sums
        return num / cnt


def curve(S, P, seed_idx=None):
    """D_P(h), h = 1..H, over all starts: (H,). seed_idx: positions of the seeds to include (default all)."""
    arrs = S[P] if seed_idx is None else tuple(a[seed_idx] for a in S[P])
    return value(P, tuple(a.sum(axis=(0, 1)) for a in arrs))


# ---- the moving-block bootstrap

def block_indices(n, L, B, rng):
    """
    (B, n) resampled start positions: ceil(n / L) blocks of L consecutive starts, each block's first start uniform
    on 0..n-L (so blocks stay inside the sample, no wrap-around), concatenated and truncated to n.
    """
    L = min(L, n)
    nb = -(-n // L)
    first = rng.integers(0, n - L + 1, size=(B, nb))
    return (first[:, :, None] + np.arange(L)[None, None, :]).reshape(B, nb * L)[:, :n]


def bootstrap_curves(S, idx, chunk=500):
    """T*_P(h) for every replicate: dict P -> (B, H). The same idx for all pillars, seeds and horizons."""
    per_start = {P: tuple(a.sum(axis=0) for a in S[P]) for P in PILLARS}       # seeds are not resampled: (n, H)
    out = {P: np.empty((len(idx), per_start[P][0].shape[-1])) for P in PILLARS}
    for a in range(0, len(idx), chunk):
        ii = idx[a:a + chunk]
        for P in PILLARS:
            out[P][a:a + chunk] = value(P, tuple(x[ii].sum(axis=1) for x in per_start[P]))
    return out


def p_value(t_star, delta):
    """(1 + #{T* <= delta}) / (B + 1)."""
    t_star = np.asarray(t_star)
    if not np.isfinite(t_star).all():          # every cloud of some replicate is dead: nothing to test (S(21) ~ 0)
        return np.nan
    return (1 + int((t_star <= delta).sum())) / (len(t_star) + 1)


def holm(p):
    """Holm's step-down adjusted p-values for a vector of raw p-values (same order as the input)."""
    p = np.asarray(p, dtype=float)
    nan = np.isnan(p)
    p = np.where(nan, 1.0, p)                  # an untestable pillar counts as p = 1 and stays NaN in the output
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    out[nan] = np.nan
    return out


# ---- survival, power law

def survival(F, n_paths):
    """S(h) per (seed, start): n_valid / n_paths, shape (S, n, H)."""
    return F["n_valid"] / n_paths


def fit_powerlaw(h, d):
    """D(h) ~ a h^b by least squares on log D against log h, over h >= 2 where D > 0. Reports if D <= 0 anywhere."""
    h, d = np.asarray(h, float), np.asarray(d, float)
    use = h >= 2
    pos = use & np.isfinite(d) & (d > 0)
    out = {"a": np.nan, "b": np.nan, "n_positive": int(pos.sum()), "n_used": int(use.sum()),
           "nonpositive_anywhere": bool((d[use] <= 0).any()), "min_D": float(np.nanmin(d[use]))}
    if pos.sum() >= 3:
        b, la = np.polyfit(np.log(h[pos]), np.log(d[pos]), 1)
        out.update(a=float(np.exp(la)), b=float(b))
    return out


# ---- the whole analysis

def analyse(T, cfg, deltas, n_paths=None):
    """
    Everything of brief section 7 from the stacked tables T (load_tables) and the margins deltas = {"A","C","D"}:
    returns {"curves": D_P(h) with bootstrap bands, "tests": one row per pillar, "survival": ..., "powerlaw": ...}.
    """
    F, B = split_runs(T)
    S = sufficient(F, B)
    H = len(T["horizons"])
    h21 = H - 1
    n = len(T["starts"])
    rng = np.random.default_rng(cfg.boot_seed)
    idx = block_indices(n, cfg.block_len_starts, cfg.n_boot, rng)
    star = bootstrap_curves(S, idx)
    seeds = list(T["seeds"])
    res = {"idx_shape": idx.shape}

    curves, tests, pl = [], [], []
    p_raw = np.array([p_value(star[P][:, h21], deltas[P]) for P in PILLARS])
    p_holm = holm(p_raw)
    sens = {}
    for mult in (0.5, 2.0):
        pr = np.array([p_value(star[P][:, h21], mult * deltas[P]) for P in PILLARS])
        sens[mult] = (pr, holm(pr))
    surv = survival(F, n_paths or cfg.n_paths)
    s21 = float(surv[:, :, h21].mean())
    s21_seed = surv[:, :, h21].mean(axis=1)
    outcome = "instability" if s21 < cfg.survival_tol else "drift"
    for q, P in enumerate(PILLARS):
        D = curve(S, P)
        lo, hi = np.nanpercentile(star[P], [2.5, 97.5], axis=0)
        for h in range(H):
            curves.append({"model_id": T["model_id"], "pillar": P, "h": h + 1, "D": D[h], "lo": lo[h], "hi": hi[h]})
        per_seed = np.array([curve(S, P, [r])[h21] for r in range(len(seeds))])
        delta = deltas[P]
        straddle = bool((per_seed > delta).any() and (per_seed <= delta).any())
        mixed = bool((per_seed > 0).any() and (per_seed < 0).any())
        row = {"model_id": T["model_id"], "pillar": P, "T": D[h21], "delta": delta,
               **{"T_seed%d" % s: v for s, v in zip(seeds, per_seed)},
               "ci_lo": lo[h21], "ci_hi": hi[h21], "p_raw": p_raw[q], "p_holm": p_holm[q],
               "reject_holm_5pct": bool(p_holm[q] <= 0.05),    # False if untestable (NaN)
               "p_holm_half_delta": sens[0.5][1][q], "p_holm_double_delta": sens[2.0][1][q],
               "seed_sensitive": straddle or mixed, "straddles_margin": straddle, "mixed_signs": mixed,
               "S21": s21, **{"S21_seed%d" % s: v for s, v in zip(seeds, s21_seed)}, "outcome": outcome,
               "n_starts": n, "n_boot": cfg.n_boot, "boot_seed": cfg.boot_seed, "block_len_starts": cfg.block_len_starts}
        tests.append(row)
        fit = fit_powerlaw(np.arange(1, H + 1), D)
        pl.append({"model_id": T["model_id"], "pillar": P, **fit})
    # survival curve with its bootstrap band, over starts (seeds averaged)
    per_start = surv.mean(axis=0)                                           # (n, H)
    sb = np.stack([per_start[idx[b]].mean(axis=0) for b in range(0, len(idx), max(1, len(idx) // 1000))])
    slo, shi = np.nanpercentile(sb, [2.5, 97.5], axis=0)
    sv = [{"model_id": T["model_id"], "h": h + 1, "S": float(surv[:, :, h].mean()), "lo": slo[h], "hi": shi[h],
           **{"S_seed%d" % s: float(surv[r, :, h].mean()) for r, s in enumerate(seeds)}} for h in range(H)]
    res.update(curves=curves, tests=tests, powerlaw=pl, survival=sv, boot=star, idx=idx, sufficient=S)
    return res


def load_margins(path):
    with open(path) as f:
        m = json.load(f)
    return m, {"A": m["delta_A"], "C": m["delta_C"], "D": m["delta_D"]}


# =====================================================================================================================
# 2. The pilot (was rec_pilot.py)
# =====================================================================================================================

# Part 2, Part A: the pilot (brief section 8), on the TRAINING block only, before any validation result is looked at.
#
#   1. the h = 1 identity check on a handful of starts, all seeds;
#   2. the noise floor: the one-step benchmark computed twice with two different noise seeds, one compared with the other
#      as if it were a "free-running" run, for each pillar; the margins must clearly exceed it (here: floor <= margin / 2);
#   3. Monte Carlo precision: a handful of clouds recomputed with a second noise draw, and the spread reported;
#   4. delta_A, the market's mean Phi on the training block;
#   5. timing: seconds per (seed, start) and the projected total for the validation run.
#
# Nothing here changes a margin, a stride or a bootstrap setting.

ALT_SIM_SEED = 20261013          # the second noise draw of the pilot; never used in the study itself
FLOOR_FRACTION = 0.5             # the noise floor must be below this fraction of the margin


def _pillar_floor(arena_a, arena_b, days, cfg):
    """D_A, D_C, D_D of benchmark B against benchmark A on the same target days (arrays (1, n, 1))."""
    a, b = bench_clouds(arena_a, days), bench_clouds(arena_b, days)
    shape = lambda x: {k: v[None, :, None] for k, v in x.items() if k in ("phi_mean", "inside", "rho")}
    S = sufficient(shape(b), shape(a))
    return {P: float(curve(S, P)[0]) for P in PILLARS}


def pilot(runs, cfg, deltas, root, n_starts=12, log=print):
    """
    runs: {seed: Run} (loaded). cfg: RecursionConfig with block = "train". Returns the report dict and writes
    pilot_report.md under root.
    """
    assert cfg.block == "train", "the pilot runs on the training block"
    cfg_alt = replace(cfg, sim_seed=ALT_SIM_SEED)
    rep = {"identity": {}, "floor": {}, "timing": {}, "mc": {}}
    floors = {P: [] for P in PILLARS}
    mc = {"phi_mean": [], "viol_share": [], "rho": [], "tail_prob_error": []}
    secs = []
    for seed, run in runs.items():
        arena = Arena(run, cfg)
        alt = Arena(run, cfg_alt, check_sim_seed=False)
        all_starts = start_indices(arena.n, cfg)
        starts = all_starts[np.linspace(0, len(all_starts) - 1, min(n_starts, len(all_starts))).astype(int)]
        # 1. identity (run_start raises IdentityError if it fails)
        ident = {"id_ret": 0.0, "id_dg": 0.0, "id_sigma": 0.0}
        recs = []
        for j0 in starts:
            rec = run_start(arena, int(j0))
            recs.append(rec)
            secs.append(float(rec["seconds"]))
            for k in ident:
                ident[k] = max(ident[k], float(rec[k]))
        rep["identity"][seed] = ident
        # 2. noise floor on the target days of these starts (h = 1 and the whole 21 days)
        days = np.unique((starts[:, None] + np.arange(cfg.horizon)[None, :]).ravel())
        fl = _pillar_floor(alt, arena, days, cfg)
        for P in PILLARS:
            floors[P].append(fl[P])
        # 3. Monte Carlo precision: the first three starts again with the second noise draw
        for j0, rec in list(zip(starts, recs))[:3]:
            rec2 = run_start(alt, int(j0))
            for h in (0, cfg.horizon - 1):
                for key in ("phi_mean", "viol_share", "rho"):
                    mc[key].append(float(rec2[key][h] - rec[key][h]))
        for j in starts[:6]:
            r1, dg1, s1 = benchmark_full(arena, [int(j)])
            r2 = benchmark_full(alt, [int(j)])[0]
            for q in (QS[0], QS[-1]):
                x1 = np.quantile(r1[0], q)
                mc["tail_prob_error"].append(float((r2[0] <= x1).mean() - q))
        log("pilot seed %d: identity max diffs %s" % (seed, {k: "%.1e" % v for k, v in ident.items()}))
    rep["identity_ok"] = True                                       # a failure raised above
    for P in PILLARS:
        f = np.array(floors[P])
        rep["floor"][P] = {"per_seed": f.tolist(), "max_abs": float(np.abs(f).max()), "margin": deltas[P],
                           "ratio": float(np.abs(f).max() / deltas[P]),
                           "ok": bool(np.abs(f).max() <= FLOOR_FRACTION * deltas[P])}
    rep["floor_ok"] = all(v["ok"] for v in rep["floor"].values())
    # the spread of the difference of two independent draws is sqrt(2) times one draw's standard error
    fin = {k: np.asarray(v)[np.isfinite(v)] for k, v in mc.items()}          # clouds with no valid path are NaN
    rep["mc"] = {k: {"n": len(fin[k]), "n_nan": len(mc[k]) - len(fin[k]),
                     "se_one_draw": float(np.std(fin[k]) / np.sqrt(2)) if len(fin[k]) > 1 else np.nan}
                 for k in mc}
    per = float(np.mean(secs))
    n_val_starts = len(start_indices(len(next(iter(runs.values())).split.val.dates), replace(cfg, block="val")))
    rep["timing"] = {"seconds_per_seed_start": per, "n_val_starts": int(n_val_starts), "n_seeds": len(runs),
                     "projected_validation_seconds": per * n_val_starts * len(runs)}
    rep["delta_A"] = deltas["A"]
    write_report(rep, root, cfg, deltas)
    return rep


def write_report(rep, root, cfg, deltas):
    L = ["# Part A pilot report (training block)", "",
         "Settings: horizon %d, %d paths, stride %d, sim_seed %d, second noise draw %d. Nothing here changes a "
         "margin, a stride or a bootstrap setting." % (cfg.horizon, cfg.n_paths, cfg.stride, cfg.sim_seed, ALT_SIM_SEED), "",
         "## 1. h = 1 identity (free-running vs benchmark)", "", "| seed | max abs diff returns | dg | sigma |", "|---|---|---|---|"]
    for s, d in rep["identity"].items():
        L.append("| %d | %.2e | %.2e | %.2e |" % (s, d["id_ret"], d["id_dg"], d["id_sigma"]))
    L += ["", "Result: **%s** (tolerance 1e-6 for returns, 1e-5 for dg and sigma)." % ("PASS" if rep["identity_ok"] else "FAIL"),
          "", "## 2. Noise floor (benchmark vs benchmark with another noise draw)", "",
          "| pillar | max abs floor over seeds | margin | floor / margin | floor <= margin/2 |", "|---|---|---|---|---|"]
    for P, v in rep["floor"].items():
        L.append("| %s | %.4g | %.4g | %.3f | %s |" % (P, v["max_abs"], v["margin"], v["ratio"], "yes" if v["ok"] else "**NO**"))
    L += ["", "Result: **%s**. The criterion (floor at most half the margin) is my reading of \"clearly exceed\"; "
          "the brief gives no number." % ("PASS" if rep["floor_ok"] else "FAIL"),
          "Note: the floor for D_D is positive by construction (the mean of an absolute difference).", "",
          "## 3. Monte Carlo precision (standard error of one 1,000-path cloud, from two independent draws)", "",
          "| quantity | n (NaN) | s.e. (one draw) | ED expectation |", "|---|---|---|---|"]
    expect = {"phi_mean": "-", "viol_share": "about 0.006", "rho": "about 0.02", "tail_prob_error": "about 0.005"}
    for k, v in rep["mc"].items():
        L.append("| %s | %d (%d) | %.4g | %s |" % (k, v["n"], v["n_nan"], v["se_one_draw"], expect[k]))
    t = rep["timing"]
    L += ["", "## 4. delta_A", "", "delta_A = %g x the market's mean Phi on the training block = **%.6g**."
          % (cfg.delta_a_mult, deltas["A"]), "",
          "## 5. Timing", "", "%.2f s per (seed, start); %d validation starts x %d seeds => projected %.0f s (%.1f min) "
          "for the free-running part of the validation run (plus the benchmark pass)."
          % (t["seconds_per_seed_start"], t["n_val_starts"], t["n_seeds"], t["projected_validation_seconds"],
             t["projected_validation_seconds"] / 60)]
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "pilot_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# =====================================================================================================================
# 3. Tables and figures of the run (was rec_report.py)
# =====================================================================================================================

# Part 2, Part A: the tables and figures (brief section 12), built from the saved metrics only.
#
# CSV: degradation.csv (D_P(h) with the pointwise percentile band), survival.csv, tests.csv (T_P, per-seed T_{P,r},
# margins, raw and Holm p, interval, sensitivity, seed-sensitive label, outcome), powerlaw.csv.
# Figures (Part 1 house style, no extras): fig_timeseries (each pillar across calendar time at h = 21), fig_degradation (three panels), fig_survival, fig_seed_dots (per-seed T_P per
# pillar), fig_coverage (free versus benchmark by horizon).


NAMES = {"A": "Admissibility", "C": "Calibration", "D": "Dynamics"}
UNITS = {"A": r"$D_A$ (mean $\Phi$, free $-$ benchmark)", "C": r"$D_C$ (coverage distance)",
         "D": r"$D_D$ (mean $|\rho_{free}-\rho_{bench}|$)"}


def write_tables(root, res):
    """The four CSV files; returns the DataFrames."""
    out = {"degradation": pd.DataFrame(res["curves"]), "survival": pd.DataFrame(res["survival"]),
           "tests": pd.DataFrame(res["tests"]), "powerlaw": pd.DataFrame(res["powerlaw"])}
    for name, df in out.items():
        df.to_csv(os.path.join(root, name + ".csv"), index=False)
    return out


def _save(fig, root, name):
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(root, "figures", "%s.%s" % (name, ext)), **kw)
    plt.close(fig)


def make_figures(root, res, T, deltas):
    seeds = list(T["seeds"])
    H = len(T["horizons"])
    h = np.arange(1, H + 1)
    df = pd.DataFrame(res["curves"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH / 3.0), constrained_layout=True)
        for ax, P in zip(axes, PILLARS):
            d = df[df.pillar == P]
            ax.fill_between(d.h, d.lo, d.hi, color=UNWEIGHTED, alpha=0.2, linewidth=0)
            ax.plot(d.h, d.D, color=UNWEIGHTED, lw=1.4)
            ax.axhline(deltas[P], color=MARKET, ls="--", lw=0.9, label="margin")
            ax.axhline(0, color=SLATE, lw=0.6)
            ax.set_title(NAMES[P])
            ax.set_xlabel("horizon $h$ (days)")
            ax.set_ylabel(UNITS[P])
            ax.set_xticks([1, 5, 10, 15, 21])
        axes[0].legend(loc="upper left")
        _save(fig, root, "fig_degradation")

        sv = pd.DataFrame(res["survival"])
        fig, ax = plt.subplots(figsize=(WIDTH / 2, WIDTH / 2.6), constrained_layout=True)
        ax.fill_between(sv.h, sv.lo, sv.hi, color=UNWEIGHTED, alpha=0.2, linewidth=0)
        for s in seeds:
            ax.plot(sv.h, sv["S_seed%d" % s], color=SLATE, lw=0.6)
        ax.plot(sv.h, sv.S, color=UNWEIGHTED, lw=1.4, label="mean over seeds")
        ax.axhline(0.99, color=MARKET, ls="--", lw=0.9, label="0.99")
        ax.set_xlabel("horizon $h$ (days)")
        ax.set_ylabel("survival $S(h)$")
        ax.set_xticks([1, 5, 10, 15, 21])
        ax.legend(loc="lower left")
        _save(fig, root, "fig_survival")

        tests = pd.DataFrame(res["tests"])
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH / 3.2), constrained_layout=True)
        for ax, P in zip(axes, PILLARS):
            row = tests[tests.pillar == P].iloc[0]
            v = [row["T_seed%d" % s] for s in seeds]
            ax.axhline(deltas[P], color=MARKET, ls="--", lw=0.9)
            ax.axhline(0, color=SLATE, lw=0.6)
            ax.plot(seeds, v, "o", color=UNWEIGHTED, ms=4)
            ax.axhspan(row.ci_lo, row.ci_hi, color=UNWEIGHTED, alpha=0.12, linewidth=0)
            ax.axhline(row["T"], color=UNWEIGHTED, lw=0.9)
            ax.set_xticks(seeds)
            ax.set_xlabel("training seed")
            ax.set_ylabel(r"$T_{%s,r}$" % P)
            ax.set_title(NAMES[P])
        _save(fig, root, "fig_seed_dots")

        F, B = split_runs(T)
        cf = np.nanmean(F["inside"][..., NINE], axis=(0, 1, 3))
        cb = np.nanmean(B["inside"][..., NINE], axis=(0, 1, 3))
        fig, ax = plt.subplots(figsize=(WIDTH / 2, WIDTH / 2.6), constrained_layout=True)
        ax.plot(h, cb, color=UNWEIGHTED, ls="--", lw=1.2, label="one-step benchmark")
        ax.plot(h, cf, color=MARKET, lw=1.4, label="free-running")
        ax.axhline(TARGET_COVERAGE, color=SLATE, lw=0.8)
        ax.set_xlabel("horizon $h$ (days)")
        ax.set_ylabel("coverage of the 95% band (nine nodes)")
        ax.set_xticks([1, 5, 10, 15, 21])
        ax.legend(loc="lower left")
        _save(fig, root, "fig_coverage")


def make_timeseries(root, T, window=10):
    """
    fig_timeseries: each pillar's metric at h = 21 across calendar time (the target date of each start), free-running
    against the one-step benchmark of the same days. Seeds: line = mean over seeds, band = min to max over seeds.
    Admissibility: mean Phi. Calibration: coverage of the 95% band pooled over the nine nodes, rolling mean over
    `window` starts (a single start has only nine yes-or-no outcomes). Dynamics: the cross-path correlation rho.
    """
    F, B = split_runs(T)
    dates = pd.DatetimeIndex(T["target_dates"][:, -1])
    h = -1
    roll = lambda x: pd.DataFrame(x.T).rolling(window, min_periods=1).mean().to_numpy().T      # x (S, n)
    series = {
        "A": (F["phi_mean"][:, :, h], B["phi_mean"][:, :, h], r"mean $\Phi$", None),
        "C": (roll(np.nanmean(F["inside"][:, :, h, NINE], axis=-1)), roll(np.nanmean(B["inside"][:, :, h, NINE], axis=-1)),
              "95%% coverage (%d-start mean)" % window, TARGET_COVERAGE),
        "D": (F["rho"][:, :, h], B["rho"][:, :, h], r"$\rho$ (return, 1M ATM log-vol change)", None)}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(WIDTH, WIDTH * 0.95), sharex=True, constrained_layout=True)
        for ax, P in zip(axes, PILLARS):
            free, bench, label, ref = series[P]
            with np.errstate(all="ignore"):
                ax.fill_between(dates, np.nanmin(free, 0), np.nanmax(free, 0), color=MARKET, alpha=0.18, linewidth=0)
                ax.plot(dates, np.nanmean(free, 0), color=MARKET, lw=1.0, label="free-running, $h=21$")
                ax.plot(dates, np.nanmean(bench, 0), color=UNWEIGHTED, lw=1.0, ls="--", label="one-step benchmark")
            if ref is not None:
                ax.axhline(ref, color=SLATE, lw=0.7)
            ax.set_ylabel(label)
            ax.set_title(NAMES[P], loc="left")
        axes[0].legend(loc="upper left", ncol=2)
        axes[-1].set_xlabel("target date of the 21-day path")
        _save(fig, root, "fig_timeseries")


def save_results(root, res, T, deltas):
    """
    results_all.npz: everything needed to compute any further metric afterwards, no re-run. Keys:
      free_<m>, bench_<m>   per-cloud metrics (seeds, starts, horizons, ...): n_valid, phi_mean, viol_share, rho,
                            phi_parts (3), inside / below / above / width (10 = nine nodes then the return),
                            q (3 quantiles 2.5/50/97.5, 10), bench_n_oob; free_viol_loc21 / bench_viol_loc21
      seeds, starts, horizons, target_dates (starts, H), real_values (starts, H, 10), var_names
      deg_<P>, deg_<P>_lo, deg_<P>_hi   D_P(h) with its bootstrap band; boot_<P> (B, H) the replicates; boot_idx (B, n)
      deltaA/C/D, id_* (identity differences), seconds
    Paths themselves are not stored (about 1000 x 42 numbers per cloud); rerun one start to regenerate them.
    """
    out = {k: v for k, v in T.items() if isinstance(v, np.ndarray)}
    out["model_id"] = np.array(T["model_id"])
    df = pd.DataFrame(res["curves"])
    for P in PILLARS:
        d = df[df.pillar == P]
        out["deg_" + P], out["deg_%s_lo" % P], out["deg_%s_hi" % P] = d.D.to_numpy(), d.lo.to_numpy(), d.hi.to_numpy()
        out["boot_" + P] = res["boot"][P]
        out["delta" + P] = np.array(deltas[P])
    out["boot_idx"] = res["idx"]
    np.savez_compressed(os.path.join(root, "results_all.npz"), **out)


# =====================================================================================================================
# 4. The pilot and report steps (was pipeline.py)
# =====================================================================================================================

def rec_pilot_step(cfg, root=None, device=None, n_starts=12):
    """The pilot (training block only): identity, noise floor, Monte Carlo precision, delta_A, timing."""
    from dataclasses import replace as _replace
    root, device = root or parta_root(), device or get_device()
    check_inputs(cfg.index)
    cfg = _replace(cfg, block="train")
    runs = {s: r for s, (_, r) in _rec_runs(cfg, device).items()}
    margins = rec_margins_step(cfg, root)
    deltas = {"A": margins["delta_A"], "C": margins["delta_C"], "D": margins["delta_D"]}
    rep = pilot(runs, cfg, deltas, root, n_starts=n_starts, log=log)
    log("pilot: identity %s, noise floor %s; report in %s" % ("passed", "ok" if rep["floor_ok"] else "NOT BELOW MARGIN/2",
                                                              os.path.join(root, "pilot_report.md")))
    return rep


def rec_report_step(cfg, root=None):
    """Degradation, bootstrap tests, tables and figures from the saved metrics (needs a passed identity check)."""
    root = root or parta_root()
    idp = os.path.join(root, "identity_check.json")
    if not os.path.exists(idp) or not json.load(open(idp)).get("passed"):
        raise RuntimeError("the h = 1 identity check has not passed: no tables are written (%s)" % idp)
    _, deltas = load_margins(os.path.join(root, "margins.json"))
    T = load_tables(root, list(cfg.seeds))
    res = analyse(T, cfg, deltas)
    tabs = write_tables(root, res)
    make_figures(root, res, T, deltas)
    make_timeseries(root, T)
    save_results(root, res, T, deltas)
    t = tabs["tests"]
    for _, r in t.iterrows():
        log("pillar %s: T = %.4g (margin %.4g), 95%% interval [%.4g, %.4g], p = %.4f, Holm p = %.4f, %s%s"
            % (r.pillar, r["T"], r.delta, r.ci_lo, r.ci_hi, r.p_raw, r.p_holm,
               "REJECT" if r.reject_holm_5pct else "retain", ", seed-sensitive" if r.seed_sensitive else ""))
    log("outcome: %s (S(21) = %.4f)" % (t.outcome.iloc[0], t.S21.iloc[0]))
    return tabs
