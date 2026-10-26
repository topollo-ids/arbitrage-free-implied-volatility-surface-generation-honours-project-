"""
The test block (2022-01-03 to 2025-12-31), used once, with the trained networks and every setting frozen.
Run from anywhere; paths are set in _paths.py (one-step results go to 1_one_step/results/SPX/test, the projection run
to 2_recursion/test_results, figures to 3_test_block/figures).

    python 3_test_block/test_eval.py onestep          cells A (control) and C (penalty in loss), and B and D with them:
                                                      main.py's simulate and evaluate steps on the test block
    python 3_test_block/test_eval.py proj 1 2 3 4 5   the 21-day bounded step + projection (Part B code), these seeds
    python 3_test_block/test_eval.py plots            the recursion's analysis (D(h), bootstrap, classes), the two
                                                      figures, and the numbers used in the write-up

Nothing here changes a model, a margin, a seed or a setting: the code is the code that produced the validation results,
and the networks are 1_one_step/trained_networks/SPX/raw_phiout and raw_phiin as trained.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this folder: _paths, style
from _paths import CODE, FIGURES, ONE_STEP, PARTA, PARTB, PROJ_DIR, RESULTS, VAL, open_test_block


def onestep():
    cmd = [sys.executable, os.path.join(ONE_STEP, "main.py"), "--index", "SPX", "--block", "test", "--confirm-final",
           "--steps", "simulate", "evaluate", "--configs", "raw_phiout", "raw_phiin"]
    subprocess.run(cmd, check=True, cwd=CODE)


def proj(seeds):
    import torch
    torch.set_num_threads(1)
    open_test_block()
    from volgan_fixed import remedies as R, recursion as RC
    from volgan_fixed.common import RecursionConfig
    cfg = RecursionConfig(model_id="bounded_proj", block="test")
    device = R.get_device()
    runs = {s: r for s, r in RC._rec_runs(cfg, device).items() if s in seeds}
    noise = RC.NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    arenas = R.model_arenas("bounded_proj", cfg, runs, noise, R.partb_root(), device)
    starts = RC.start_indices(next(iter(arenas.values()))[0].n, cfg)
    os.makedirs(PROJ_DIR, exist_ok=True)
    for s, (folder, run) in runs.items():
        a, remedy = arenas[s]
        RC.identity_gate(a, starts[:3], log=lambda m, s=s: print("seed %d: %s" % (s, m), flush=True))
        sig = RC.seed_signature(run, folder, cfg, remedy)
        bench = RC.get_bench(a, PROJ_DIR, s, RC.seed_signature(run, folder, cfg), log=lambda m: print(m, flush=True))
        RC.run_seed(a, PROJ_DIR, s, sig, starts, log=lambda m: print(m, flush=True))
        out = RC.consolidate(a, PROJ_DIR, s, sig, starts, bench)
        idm = max(float(out[k].max()) for k in ("id_ret", "id_dg", "id_sigma"))
        phi_post = float(out["free_phi_post_max"].max())
        if idm != 0.0 or phi_post != 0.0:
            raise RuntimeError("seed %d: identity %.3e, post-projection Phi %.3e" % (s, idm, phi_post))
        print("seed %d done: identity 0, post-projection Phi 0, witness slack %.3e" % (s, a.repair.witness_min_slack),
              flush=True)


# =====================================================================================================  analysis + plots
def plots():
    import pandas as pd
    from style import plt, mdates, MKT, CTRL, PEN, PROJ
    from matplotlib.lines import Line2D
    from volgan_fixed import recursion_analysis as RA
    from volgan_fixed.common import RecursionConfig
    cfg = RecursionConfig(model_id="bounded_proj", block="test")
    seeds = list(cfg.seeds)

    res_path = os.path.join(PROJ_DIR, "results_all.npz")
    if not os.path.exists(res_path):
        _, deltas = RA.load_margins(os.path.join(PARTA, "margins.json"))
        T = RA.load_tables(PROJ_DIR, seeds)
        res = RA.analyse(T, cfg, deltas)
        RA.write_tables(PROJ_DIR, res)
        RA.save_results(PROJ_DIR, res, T, deltas)
    P = np.load(res_path)
    Pv = np.load(os.path.join(PARTB, "bounded_proj", "results_all.npz"))
    tb = os.path.join(RESULTS, "SPX", "test")
    mk = np.load(os.path.join(tb, "market.npz"))
    mkv = np.load(os.path.join(VAL, "market.npz"))
    one = lambda base, cell: [np.load(f"{base}/{cell}/seed_{s}/metrics.npz") for s in seeds]
    A, C = one(tb, "A"), one(tb, "C")
    Av, Cv = one(VAL, "A"), one(VAL, "C")
    assert np.array_equal(A[0]["dates"], mk["dates"])
    dates = mk["dates"]

    # ---- per-network pillar values: one-step cells per day, the recursion on day 21
    def onestep_vals(Z):
        return {"phi": np.array([z["phi_mean"].mean() for z in Z]),
                "cov": np.array([100 * z["inside_95"][:, :9].mean() for z in Z]),
                "rho": np.array([np.nanmean(z["rho"]) for z in Z])}

    def rec_vals(z, h=20):
        I = z["free_inside"][:, :, h, :9]
        return {"phi": np.array([np.nanmean(z["free_phi_mean"][s, :, h]) for s in range(5)]),
                "cov": np.array([100 * np.nansum(I[s]) / np.isfinite(I[s]).sum() for s in range(5)]),
                "rho": np.array([np.nanmean(z["free_rho"][s, :, h]) for s in range(5)])}

    V = {"test": [onestep_vals(A), onestep_vals(C), rec_vals(P)], "val": [onestep_vals(Av), onestep_vals(Cv), rec_vals(Pv)]}
    names = ["Control (cell A), one day", "Penalty in loss (cell C), one day", "Bounded step + projection, 21 days"]
    cols = [CTRL, PEN, PROJ]

    # ---- figure 1: the daily penalty over the test block
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    pA = np.mean([z["phi_mean"] for z in A], 0)
    pC = np.mean([z["phi_mean"] for z in C], 0)
    x21 = P["target_dates"][:, 20]
    p21 = np.nanmean(P["free_phi_mean"][:, :, 20], 0)
    sci = lambda v: r"%.2f\times10^{%d}" % (v / 10 ** np.floor(np.log10(v)), np.floor(np.log10(v))) if v > 0 else "0"
    ax.scatter(dates, mk["phi"], s=3, color=MKT, linewidths=0, zorder=4, label=r"Market: average $\Phi=%s$" % sci(mk["phi"].mean()))
    ax.scatter(dates, pA, s=3, color=CTRL, linewidths=0, zorder=3, label=r"Control, one day: $%s$" % sci(pA.mean()))
    ax.scatter(dates, pC, s=3, color=PEN, linewidths=0, zorder=3, label=r"Penalty in loss, one day: $%s$" % sci(pC.mean()))
    ax.scatter(x21, p21, s=9, color=PROJ, linewidths=0, zorder=2, label=r"Bounded step + projection, day 21: $%s$" % sci(p21.mean()))
    ax.set_yscale("symlog", linthresh=1e-6, linscale=0.5)
    ax.set_ylim(-2e-7, 20); ax.set_yticks([0, 1e-6, 1e-4, 1e-2, 1])
    ax.set_ylabel(r"Daily mean penalty $\Phi$")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_minor_locator(mdates.MonthLocator((4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(np.datetime64("2022-01-01"), np.datetime64("2026-01-01"))
    ax.grid(axis="y", lw=0.3, color="0.85")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, markerscale=2.5, handletextpad=0.3, ncol=2, columnspacing=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig_test_phi.png"), dpi=400); plt.close(fig)

    # ---- figure 2: the three pillars, validation (hollow) against test (filled)
    mkt_rho = {"test": mk["corrmat"][0, 2], "val": mkv["corrmat"][0, 2]}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for ax, key, title in zip(axes, ("phi", "cov", "rho"), ("Admissibility", "Coverage", "Dynamics")):
        for i, col in enumerate(cols):
            for blk, off, face in (("val", -0.14, "white"), ("test", 0.14, col)):
                v = V[blk][i][key]
                ax.vlines(i + off, v.min(), v.max(), color=col, lw=0.9, alpha=0.6, zorder=1)
                ax.scatter(i + off, v.mean(), s=26, facecolor=face, edgecolor=col, linewidth=1.1, zorder=3)
        if key == "phi":
            ax.axhline(mk["phi"].mean(), color=MKT, lw=0.9, zorder=0)
            ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
            ax.set_ylim(-1e-5, 3 * max(max(v["phi"].max() for v in V[b]) for b in V))
            ax.set_ylabel(r"Mean penalty $\Phi$", fontsize=9)
        elif key == "cov":
            ax.axhline(95, color="0.3", ls=(0, (4, 2)), lw=0.8, zorder=0)
            ax.set_ylabel("Coverage of the 95% band (%)", fontsize=9)
        else:
            ax.axhline(mkt_rho["test"], color=MKT, lw=0.9, zorder=0)
            ax.set_ylabel("Leverage correlation", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Control", "Penalty\nin loss", "Bounded +\nprojection"], fontsize=7)
        ax.set_xlim(-0.6, 2.6); ax.tick_params(axis="y", labelsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", pad=3)
        ax.grid(axis="y", lw=0.3, color="0.85")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.3", label="Validation, 2018 to 2021"),
               Line2D([], [], marker="o", ls="", color="0.3", label="Test, 2022 to 2025"),
               Line2D([], [], color=MKT, lw=0.9, label="Market, test block"),
               Line2D([], [], color="0.3", ls=(0, (4, 2)), lw=0.8, label="Nominal 95%")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.9); fig.savefig(os.path.join(FIGURES, "fig_test_pillars.png"), dpi=400)
    plt.close(fig)

    # ---- numbers for the write-up
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print("test block: %d days, %s to %s; market mean Phi %.3e (val %.3e), violating days %d; market leverage %.3f (val %.3f)"
          % (len(dates), str(dates[0])[:10], str(dates[-1])[:10], mk["phi"].mean(), mkv["phi"].mean(),
             int((mk["phi"] > 0).sum()), mkt_rho["test"], mkt_rho["val"]))
    for i, n in enumerate(names):
        for key in ("phi", "cov", "rho"):
            t, v = V["test"][i][key], V["val"][i][key]
            print("%-36s %-4s test %.4g [%.4g, %.4g]  val %.4g [%.4g, %.4g]" % (n, key, t.mean(), t.min(), t.max(), v.mean(), v.min(), v.max()))
    for z, n in ((A, "A"), (C, "C")):
        print(n, "violating share %.3f%%" % (100 * np.mean([zz["viol_share"].mean() for zz in z])),
              "ESS<100 days %.2f%%" % (100 * np.mean([np.mean(zz["ess"] < 100) for zz in z])),
              "below/above 9 nodes %.1f / %.1f" % (100 * np.mean([zz["below_025"][:, :9].mean() for zz in z]),
                                                   100 * np.mean([zz["above_975"][:, :9].mean() for zz in z])))
    print("one-step Phi relative to market: A %.2f  C %.2f" % (V["test"][0]["phi"].mean() / mk["phi"].mean(), V["test"][1]["phi"].mean() / mk["phi"].mean()))
    d = {k: float(P["delta" + k]) for k in "ACD"}
    cls = lambda lo, hi, m: "material" if lo > m else ("not detected" if hi < m else "indeterminate")
    print("recursion on test: %d starts" % len(P["starts"]))
    for k in "ACD":
        print(" D_%s(21) %.4g [%.4g, %.4g]  classes d/2,d,2d: %s | val %.4g [%.4g, %.4g]" % (
            k, P["deg_" + k][20], P["deg_%s_lo" % k][20], P["deg_%s_hi" % k][20],
            ", ".join(cls(P["deg_%s_lo" % k][20], P["deg_%s_hi" % k][20], f * d[k]) for f in (0.5, 1, 2)),
            Pv["deg_" + k][20], Pv["deg_%s_lo" % k][20], Pv["deg_%s_hi" % k][20]))
    S21 = P["free_n_valid"][:, :, 20] / 1000
    print(" S(21) %.4f by net %s; projected share h21 %.3f; drift 1M ATM %.2f pts; viol share h21 %.4f"
          % (S21.mean(), S21.mean(1), np.nanmean(P["free_proj_share"][:, :, 20]),
             100 * np.nanmean(P["free_q"][:, :, 20, 1, 1] - P["real_values"][None, :, 20, 1]), np.nanmean(P["free_viol_share"][:, :, 20])))
    print(" one-step bench (cell B) on test: phi %.3e, cov %.1f, rho %.3f" % (
        np.nanmean(P["bench_phi_mean"][:, :, 20]), 100 * np.nansum(P["bench_inside"][:, :, 20, :9]) / np.isfinite(P["bench_inside"][:, :, 20, :9]).sum(),
        np.nanmean(P["bench_rho"][:, :, 20])))
    yr = pd.DatetimeIndex(dates).year
    for y in sorted(set(yr)):
        m = yr == y
        print(" %d: market Phi %.2e  A %.2e  C %.2e  A cov %.1f  C cov %.1f  1M ATM mean %.1f%%" % (
            y, mk["phi"][m].mean(), pA[m].mean(), pC[m].mean(), 100 * np.mean([z["inside_95"][m, :9].mean() for z in A]),
            100 * np.mean([z["inside_95"][m, :9].mean() for z in C]), 100 * mk["cal_realised"][m, 1].mean()))
    for nm in ("fig_test_phi.png", "fig_test_pillars.png"):
        print("wrote", os.path.join(FIGURES, nm))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "onestep":
        onestep()
    elif cmd == "proj":
        proj([int(s) for s in sys.argv[2:]] or [1, 2, 3, 4, 5])
    elif cmd == "plots":
        plots()
    else:
        print(__doc__)
