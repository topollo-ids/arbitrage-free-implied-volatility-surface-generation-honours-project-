"""
The qualitative check on the test block: the SPX surface on the day of the market's highest mean penalty
(25 November 2022), beside one draw from each model run on the test block (network 1).

    python 3_test_block/test_surface_day.py find      the candidate days: highest market penalty, volatility and return
    python 3_test_block/test_surface_day.py draws     regenerate the four surfaces, save 3_test_block/results/surface_day_<day>.npz
    python 3_test_block/test_surface_day.py figure    the 2 x 2 figure, 3_test_block/figures/fig_test_surfaces.png, its numbers
    python 3_test_block/test_surface_day.py checks    how typical each draw is within its own ensemble

The draws: cells A (control) and C (penalty in loss), one day ahead, one scenario drawn from the re-weighted law exactly
as the dynamics histories draw them (pca_indices); and the bounded step + projection, the 21-day path ending on the day,
started 20 trading days earlier, one valid path of 1,000 chosen at random (seed 20261019). Settings are frozen; if the
stored work file of the covering start exists (2_recursion/test_results/work), the regenerated path is checked
against it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this folder: _paths, style
from _paths import FIGURES, OWN_RESULTS, PROJ_DIR, RESULTS, open_test_block

DAY = np.datetime64("2022-11-25")
DRAWS = os.path.join(OWN_RESULTS, "surface_day_%s.npz" % str(DAY))
PICK_SEED = 20261019


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def find():
    torch = _torch()
    open_test_block()
    from volgan_fixed.training import load_run
    from _paths import RUNS
    run = load_run(os.path.join(RUNS, "SPX", "raw_phiout", "seed_1"), torch.device("cpu"))
    tb = run.split.block("test")
    mk = np.exp(tb.past_log_surface[:-1] + tb.target[:-1, 1:])
    d = tb.dates[:-1]
    lk = len(tb.m)
    t30, mi = int(np.argmin(abs(tb.tau * 365 - 30))), int(np.argmin(abs(tb.m - 1.0)))
    a = 100 * mk[:, t30 * lk + mi]
    ret = 100 * tb.target[:-1, 0] / tb.ret_scale
    mkz = np.load(os.path.join(RESULTS, "SPX", "test", "market.npz"))
    phi, dd = mkz["phi"], mkz["dates"]
    print("max 1M ATM:", d[a.argmax()], "%.1f" % a.max(), "| max over surface:", d[(100 * mk.max(1)).argmax()],
          "%.1f" % (100 * mk.max()))
    print("max |return|:", d[np.abs(ret).argmax()], "%.2f" % ret[np.abs(ret).argmax()])
    print("max market Phi:", dd[phi.argmax()], "%.3e" % phi.max())
    for yr in (2022, 2023, 2024, 2025):
        s = (d >= np.datetime64("%d-01-01" % yr)) & (d < np.datetime64("%d-01-01" % (yr + 1)))
        print(yr, "max 1M ATM %.1f on" % a[s].max(), d[s][a[s].argmax()], "| mean %.1f" % a[s].mean())
    top = np.argsort(-phi)[:6]
    print("top market Phi days:", [(str(dd[i])[:10], "%.2e" % phi[i]) for i in top])


def _arena():
    from volgan_fixed import remedies as R, recursion as RC
    from volgan_fixed.common import RecursionConfig
    cfg = RecursionConfig(model_id="bounded_proj", block="test")
    device = R.get_device()
    runs = {s: r for s, r in RC._rec_runs(cfg, device).items() if s == 1}
    noise = RC.NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    arena, _ = R.model_arenas("bounded_proj", cfg, runs, noise, R.partb_root(), device)[1]
    return cfg, arena, RC


def _step(RC, cfg, arena, j0, want_k):
    for o in RC.rollout(arena.gen, arena.ds, arena.run.data, j0, arena.noise, cfg, None, None,
                        gen_late=arena.gen_late, clip=arena.clip, repair=arena.repair):
        if o.k == want_k:
            return o


def draws():
    torch = _torch()
    open_test_block()
    from volgan_fixed.training import load_run
    from volgan_fixed.postprocessing import generate, scenario_phi, beta_t, weights, pca_indices
    from _paths import RUNS
    out = {}
    for cell, folder in (("A", "raw_phiout"), ("C", "raw_phiin")):          # one day ahead, network 1
        run = load_run(os.path.join(RUNS, "SPX", folder, "seed_1"), torch.device("cpu"))
        tb = run.split.block("test")
        j = np.array([int(np.where(tb.dates == DAY)[0][0])])
        _, sigma = generate(run.gen, tb, j, run.config.n_scenarios, run.config.sim_seed)
        p = scenario_phi(sigma, tb.m, tb.tau, tb.rates[j])
        beta = beta_t(p.total, "max_penalty")
        w = weights(p.total, beta)
        pick = pca_indices(w, beta, 1, run.config.sim_seed)[:, 0]
        out["draw_" + cell] = sigma[0, pick[0]]
        out["phi_" + cell] = float(p.total[0, pick[0]])
        out["phimean_" + cell] = float((w * p.total).sum())
        print(cell, "drawn Phi %.3e" % out["phi_" + cell], "weighted mean Phi %.3e" % out["phimean_" + cell],
              "ESS %.0f" % (1 / (w ** 2).sum()))
        if cell == "A":
            jT, m, tau = int(j[0]), tb.m, tb.tau
            out["market"] = np.exp(tb.past_log_surface[jT] + tb.target[jT, 1:])
            out["prev"] = np.exp(tb.past_log_surface[jT])
            out["phi_market"] = float(np.ravel(scenario_phi(out["market"][None, None], m, tau, tb.rates[j]).total)[0])
            print("market Phi on the day %.4e" % out["phi_market"])

    cfg, arena, RC = _arena()                                              # bounded step + projection, network 1
    H = cfg.horizon
    starts = RC.start_indices(arena.n, cfg)
    jT = int(np.argwhere(arena.ds.dates.values == DAY)[0][0])
    j0c = int([s for s in starts if 0 <= jT - s <= H - 1][-1])
    kc = jT - j0c + 1
    stored = os.path.join(PROJ_DIR, "work", "bounded_proj", "seed_1", "start_%05d.npz" % j0c)
    if os.path.exists(stored):                                             # the regenerated path equals the stored run
        st = np.load(stored)
        o = _step(RC, cfg, arena, j0c, kc)
        c = arena.cloud(o.j, o.sigma[o.valid], o.r[o.valid], o.dg[o.valid])
        dq = float(np.abs(np.asarray(c["q"]) - st["q"][kc - 1]).max())
        print("check start %d k %d: n_valid %d vs %d, q diff %.1e, rho diff %.1e, phi diff %.1e"
              % (j0c, kc, int(o.valid.sum()), int(st["n_valid"][kc - 1]), dq, abs(float(c["rho"]) - st["rho"][kc - 1]),
                 abs(float(c["phi_mean"]) - st["phi_mean"][kc - 1])))
        assert int(o.valid.sum()) == int(st["n_valid"][kc - 1]) and dq == 0.0
    else:
        print("no stored work file for start %d (work/ was not kept): regeneration not cross-checked" % j0c)
    j0 = jT - (H - 1)
    o = _step(RC, cfg, arena, j0, H)
    assert o.j == jT and arena.ds.dates[o.j] == DAY
    pick = int(np.random.default_rng(PICK_SEED).choice(np.where(o.valid)[0]))
    sig = o.sigma[pick]
    out["draw_P"] = sig
    out["phi_P"] = float(np.ravel(scenario_phi(sig[None, None], arena.ds.m, arena.ds.tau, arena.ds.rates[[jT]]).total)[0])
    out["n_valid_P"] = int(o.valid.sum())
    out["start_date_P"] = str(arena.ds.dates[j0])[:10]
    print("projection: start", out["start_date_P"], "valid paths", out["n_valid_P"], "of", cfg.n_paths,
          "drawn Phi", out["phi_P"])
    np.savez_compressed(DRAWS, m=m, tau=tau, **{k: np.asarray(v) for k, v in out.items()})
    print("saved", DRAWS)


def figure():
    from matplotlib.colors import LinearSegmentedColormap
    from style import plt, MKT, CTRL, PEN, PROJ
    z = np.load(DRAWS)
    m, days = z["m"], np.round(z["tau"] * 365).astype(int)
    lk, lt = len(m), len(days)
    grid = lambda v: (np.asarray(v) * 100).reshape(lt, lk)
    cm = lambda c, lo, hi: LinearSegmentedColormap.from_list("c", [lo, c, hi])
    panels = [("Market", z["market"], cm(MKT, "#f6d5d1", "#7f1d12")),
              ("Control, one day", z["draw_A"], cm(CTRL, "#d4e3f1", "#0d3361")),
              ("Penalty in loss, one day", z["draw_C"], cm(PEN, "#f6dfec", "#6b2a50")),
              ("Bounded step and projection, 21 days", z["draw_P"], cm(PROJ, "#d2eee4", "#05553d"))]
    allv = np.concatenate([grid(p[1]).ravel() for p in panels])
    zmin, zmax = 5 * np.floor(allv.min() / 5), 5 * np.ceil(allv.max() / 5)
    M, T = np.meshgrid(m, days)
    fig = plt.figure(figsize=(7.2, 6.6))
    for i, (title, v, cmap) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.plot_surface(M, T, grid(v), cmap=cmap, vmin=zmin, vmax=zmax, edgecolor="white", linewidth=0.3,
                        antialiased=True, rstride=1, cstride=1, alpha=0.97)
        ax.set_zlim(zmin, zmax)
        ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.set_yticks([30, 120, 240, 360])
        ax.set_xlabel("Moneyness", fontsize=8, labelpad=-1)
        ax.set_ylabel("Maturity (days)", fontsize=8, labelpad=-1)
        ax.set_zlabel("Implied vol. (%)", fontsize=8, labelpad=-2)
        ax.tick_params(labelsize=7, pad=-2)
        ax.view_init(elev=24, azim=-58)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor((1, 1, 1, 0))
            a.pane.set_edgecolor("0.8")
            a._axinfo["grid"].update(linewidth=0.3, color="0.85")
        ax.set_title(title, fontsize=9, pad=0)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.96, bottom=0.03, wspace=0.08, hspace=0.12)
    path = os.path.join(FIGURES, "fig_test_surfaces.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    mi = {k: list(np.round(m, 2)).index(k) for k in (0.8, 0.9, 1.0, 1.1, 1.2)}
    t30, t90, t360 = [list(days).index(d) for d in (30, 90, 360)]
    for t, v, ph in [("market", z["market"], float(z["phi_market"])), ("control A", z["draw_A"], float(z["phi_A"])),
                     ("penalty C", z["draw_C"], float(z["phi_C"])), ("projection", z["draw_P"], float(z["phi_P"]))]:
        g = grid(v)
        print("%-11s max %.1f  1M ATM %.1f  1M 0.8 %.1f  1M 1.2 %.1f  ATM 30d %.1f 90d %.1f 360d %.1f  Phi %.2e"
              % (t, g.max(), g[t30, mi[1.0]], g[t30, mi[0.8]], g[t30, mi[1.2]], g[t30, mi[1.0]], g[t90, mi[1.0]],
                 g[t360, mi[1.0]], ph))
    print("previous-day 1M ATM %.1f" % grid(z["prev"])[t30, mi[1.0]], "| vertical range", zmin, zmax)
    print("wrote", path)


def checks():
    torch = _torch()
    open_test_block()
    from volgan_fixed.training import load_run
    from volgan_fixed.postprocessing import generate, scenario_phi, beta_t, weights
    from _paths import RUNS
    z = np.load(DRAWS)

    def node(tb):
        return int(np.argmin(abs(tb.tau * 365 - 30))) * len(tb.m) + int(np.argmin(abs(tb.m - 1.0)))

    for cell, folder in (("A", "raw_phiout"), ("C", "raw_phiin")):
        run = load_run(os.path.join(RUNS, "SPX", folder, "seed_1"), torch.device("cpu"))
        tb = run.split.block("test")
        j = np.array([int(np.where(tb.dates == DAY)[0][0])])
        _, sg = generate(run.gen, tb, j, run.config.n_scenarios, run.config.sim_seed)
        p = scenario_phi(sg, tb.m, tb.tau, tb.rates[j])
        w = weights(p.total, beta_t(p.total, "max_penalty"))[0]
        n = node(tb)
        a, d = 100 * sg[0][:, n], 100 * z["draw_" + cell][n]
        o = np.argsort(a)
        cw = np.cumsum(w[o])
        q = lambda x: a[o][np.searchsorted(cw, x)]
        print(cell, "draw 1M ATM %.1f" % d, "weighted percentile %.0f" % (100 * w[a <= d].sum()),
              "median %.1f, 2.5 to 97.5: %.1f %.1f" % (q(.5), q(.025), q(.975)),
              "share with Phi>0: unweighted %.1f%%, weighted %.2f%%" % (100 * (p.total[0] > 0).mean(),
                                                                    100 * w[p.total[0] > 0].sum()))
    cfg, arena, RC = _arena()
    jT = int(np.argwhere(arena.ds.dates.values == DAY)[0][0])
    j0 = jT - (cfg.horizon - 1)
    n = node(arena.ds)
    cond = np.exp(arena.ds.past_log_surface[j0])
    print("conditioning day", str(arena.ds.dates[j0 - 1])[:10], "1M ATM %.1f%%, max %.1f%%" % (100 * cond[n], 100 * cond.max()))
    for o in RC.rollout(arena.gen, arena.ds, arena.run.data, j0, arena.noise, cfg, None, None,
                        gen_late=arena.gen_late, clip=arena.clip, repair=arena.repair):
        if o.k in (1, 7, 14, 21):
            a = 100 * o.sigma[o.valid][:, n]
            line = "k %2d valid %d  1M ATM 2.5/50/97.5: %.1f %.1f %.1f  max %.1f" % (
                o.k, o.valid.sum(), *np.percentile(a, [2.5, 50, 97.5]), a.max())
            if o.k == cfg.horizon:
                line += "  | the draw %.1f sits at percentile %.0f" % (100 * z["draw_P"][n], 100 * (a <= 100 * z["draw_P"][n]).mean())
            print(line)


if __name__ == "__main__":
    steps = {"find": find, "draws": draws, "figure": figure, "checks": checks}
    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        print(__doc__)
    else:
        steps[sys.argv[1]]()
