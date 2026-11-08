"""
Post-processing checks, merged by topic from the stage tests (every assertion kept). Run on its own:

    python tests/test_postprocessing.py

--- from test_stage8.py ---
Stage 8 checks: the simulation draws the paper's scenarios correctly, is reproducible day by day,
keeps memory bounded, and a saved run reloads into the same generator.

--- from test_stage10.py ---
Stage 10 checks: scenario re-weighting (eqs 17-20).
  - weights: hand example, no NaN where the released formula underflows (A5), invariance to shifting Phi,
  - the three readings of beta(t) in eq 20 do what they claim,
  - eq 18 means and eq 19 quantiles against a hand example and three independent references
    (NumPy's weighted quantile, eq 19 written as a loop, and the original quantile()),
  - the original functions' bugs B4, B5 and B17 shown on small examples, and fixed here,
  - daily statistics: raw = plain statistics, re-weighting never raises the mean penalty, and the
    rebuilt scenarios are checked to be the simulated ones.

--- from test_stage11.py ---
Stage 11 checks: the evaluation helpers.
  - interpolation: the paper's rule, exact for bilinear functions, equal to the original get_points3,
    and F11 shown (get_points extrapolates flat),
  - VIX: 100 sigma on flat smiles, equal to the continuous formula integrated numerically, sensible wings,
    the original VIX's bugs B2 and B3 shown, and agreement with the published VIX on the market surfaces,
  - realised values and the published index are matched by date (B21), and the extras of daily_statistics.

--- from test_stage13.py ---
Stage 13 checks (Part 1: split, seeds, market values, run folders, Part 2 hooks).
  - the three blocks end exactly on the configured dates, do not overlap, and the test block is refused unless
    allow_test=True; a run folder trained before Part 1 still loads (split by calendar year),
  - common random numbers: two different networks given the same sim_seed and date receive identical noise,
  - the realised values and market_sigma are the raw vendor grid, the same grid the networks see,
  - the Part 2 state update: fed the realised return and log-vol change, it rebuilds the next real condition.
Brief section 12, tests 6, 7 and 8 (rate_source "zero" here: these checks do not depend on the rate values).

--- from test_stage14.py ---
Stage 14 checks (Part 1, the one-pass metrics of section 8).
  - violation location: a hand-built surface with exactly one butterfly violation flags exactly its three nodes,
  - rho_t: with uniform weights it is numpy.corrcoef; with all the weight on two distinct scenarios it is +-1,
  - PCA: on synthetic data with a known covariance the explained variances and the inner products are recovered,
    and the correlation-table helpers agree with numpy on the same data,
  - daily_metrics on a small real case: weighted shares by hand, the per-series PCA moments equal numpy.cov of the
    explicitly built series, and cell bookkeeping: cell B's metrics = cell A's computed with beta = 0.
Brief section 12, tests 9, 10, 11 and 12.
"""

import contextlib
import dataclasses
import functools
import io
import os
import sys
import tempfile

import numpy as np
import original_reference as orig
from scipy.integrate import quad
from scipy.special import ndtr
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from volgan_fixed.common import (
    CELLS, Config, SEEDS, SIM_SEED, TRAINING_CONFIGS, arbitrage_penalty, relative_call, set_seed, training_config,
    variant, violation_nodes)
from volgan_fixed.postprocessing import (
    ATM_1M, VIX_STRIKES, VIX_T, band_stats, beta_t, correlation, correlation_band, daily_metrics, daily_statistics,
    day_key, day_noise, day_scenarios, dynamics_from_cov, effective_sample_size, extras_for, generate,
    generate_full, increments, interpolate, linear_weights, literal_beta, load_vol_index, market_reference,
    market_sigma, max_weight, node_index, pca_indices, principal_components, realised, relative_entropy,
    sample_indices, scenario_phi, simulate, smile_on_strikes, surface_matrices, vix, vix_of_surfaces,
    weighted_corr, weighted_mean, weighted_quantile, weights)
from volgan_fixed.preprocessing import N_SCALARS, prepare, prepare_all
from volgan_fixed.training import build_networks, load_run, train_run


# =====================================================================================================================
# from test_stage8.py
# =====================================================================================================================


CPU = torch.device("cpu")


@functools.lru_cache(maxsize=1)
def setup():
    data, split = prepare("SPX", Config())
    set_seed(0)
    K = data.lk * data.lt
    gen, _ = build_networks(N_SCALARS + K, 1 + K, Config(), CPU)     # any generator will do for these checks
    return data, split.test, gen


def test_noise_is_per_day_and_reproducible():
    a, b = day_noise(0, 5, 100, 32), day_noise(0, 5, 100, 32)
    assert torch.equal(a, b) and a.shape == (100, 32)
    assert not torch.equal(a, day_noise(0, 6, 100, 32)) and not torch.equal(a, day_noise(1, 5, 100, 32))


def test_scenarios_are_built_as_in_the_paper():
    """R_hat = G(a, z)[0] / ret_scale and sigma_hat = exp(g_t + G(a, z)[1:]), with the day's own noise."""
    data, test, gen = setup()
    j, n = 17, 500
    returns, sigma = day_scenarios(gen, test, j, n, seed=3)
    with torch.no_grad():
        cond = torch.tensor(test.condition[j], dtype=torch.float32).expand(n, -1)
        fake = gen(day_noise(3, day_key(test.dates[j]), n, 32), cond).double().numpy()
    assert np.allclose(returns, fake[:, 0] / test.ret_scale, rtol=1e-12)
    assert np.allclose(sigma, np.exp(test.past_log_surface[j] + fake[:, 1:]), rtol=1e-12)
    assert returns.shape == (n,) and sigma.shape == (n, test.m.size * test.tau.size) and (sigma > 0).all()


def test_phi_matches_stage2_surface_by_surface():
    """A8: the batched reshape gives each scenario's surface as an (m, tau) matrix, and the same Phi as Stage 2."""
    data, test, gen = setup()
    _, sigma = generate(gen, test, [0, 1], 50, seed=0)
    rate = np.array([0.0, 0.03])
    p = scenario_phi(sigma, test.m, test.tau, rate)
    ms, ts = np.meshgrid(test.m, test.tau, indexing="ij")
    for day in (0, 1):
        for s in (0, 13, 49):
            matrix = sigma[day, s].reshape(test.tau.size, test.m.size).T          # [i, j] = sigma(m_i, tau_j)
            one = arbitrage_penalty(relative_call(ms, ts, matrix[None], rate[day]), test.m, test.tau)
            assert np.isclose(p.total[day, s], one.total[0], rtol=1e-12, atol=1e-15)
    lk, lt = test.m.size, test.tau.size
    assert np.array_equal(surface_matrices(sigma, lk, lt)[0], sigma[0, 0].reshape(lt, lk).T)


def test_chunking_and_regeneration_do_not_change_results():
    data, test, gen = setup()
    sub = test.slice(0, 10)
    a = simulate(gen, sub, 300, seed=1, chunk_days=1)
    b = simulate(gen, sub, 300, seed=1, chunk_days=4)
    assert np.allclose(a.phi, b.phi, rtol=1e-6, atol=1e-12) and np.allclose(a.returns, b.returns, rtol=1e-6)
    r, sigma = day_scenarios(gen, sub, 7, 300, seed=1)                       # rebuild one day later
    assert np.allclose(r, a.returns[7], rtol=1e-6)
    p = scenario_phi(sigma[None], sub.m, sub.tau, np.zeros(1))
    assert np.allclose(p.total[0], a.phi[7], rtol=1e-6, atol=1e-12)
    c = simulate(gen, sub, 300, seed=2)
    assert not np.allclose(a.returns, c.returns)
    # a date's scenarios do not depend on where it sits: the same date inside a different slice of the test set
    later = test.slice(5, 12)                                               # its day 2 is sub's day 7
    assert later.dates[2] == sub.dates[7]
    assert np.allclose(day_scenarios(gen, later, 2, 300, seed=1)[0], r, rtol=1e-6)
    assert day_key("1970-01-02") == 1 and day_key(test.dates[0]) == (test.dates[0] - np.datetime64("1970-01-01")).days


def test_outputs_and_rate():
    data, test, gen = setup()
    sub = test.slice(0, 6)
    sim = simulate(gen, sub, 200, seed=0)
    assert sim.phi.shape == sim.returns.shape == (6, 200) and sim.phi_parts.shape == (6, 3)
    assert sim.phi.dtype == np.float64 and (sim.phi >= 0).all() and sim.dates.equals(sub.dates)
    assert np.allclose(sim.phi_parts.sum(1), sim.phi.mean(1))
    # only n_days x N numbers per kept quantity: no full surfaces are stored (A6)
    kept = [v for v in vars(sim).values() if isinstance(v, np.ndarray)]
    assert max(v.size for v in kept) == 6 * 200
    # rate: a constant per-day array equals the scalar; a wrong length is refused
    assert np.array_equal(simulate(gen, sub, 200, 0, rate=np.full(6, 0.02)).phi, simulate(gen, sub, 200, 0, rate=0.02).phi)
    try:
        simulate(gen, sub, 200, 0, rate=np.zeros(5))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-length rate accepted")


def test_market_sigma_is_the_data_on_the_predicted_day():
    data, test, gen = setup()
    rows = data.dates.get_indexer(test.dates)
    assert np.allclose(market_sigma(test), data.vols[rows], rtol=1e-12)


def test_run_folder_round_trip():
    data, test, gen = setup()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = variant("control", n_grad=1, n_epochs=1)
        res = train_run("SPX", cfg, tmp, CPU, progress=False)
        assert {"0_initial.pt", "1_matched.pt", "2_trained.pt", "run.json", "logs.npz"} <= set(os.listdir(tmp))
        run = load_run(tmp, CPU)
        assert all(torch.equal(p, q) for p, q in zip(res.gen.parameters(), run.gen.parameters()))
        assert run.config == cfg and run.info["index"] == "SPX" and run.split.test.dates.equals(test.dates)
        a = simulate(res.gen, test.slice(0, 3), 100, seed=0)
        b = simulate(run.gen, run.split.test.slice(0, 3), 100, seed=0)
        assert np.array_equal(a.phi, b.phi) and np.array_equal(a.returns, b.returns)
        with np.load(os.path.join(tmp, "logs.npz")) as logs:                # 'with' closes the file (Windows)
            assert logs["match_norms"].shape[1] == 3 and len(logs["train_d_loss"]) == 1


def test_variants():
    r = variant("released")
    assert (r.rv_window, r.smoothness, r.match_on, r.restart, r.gen_bce_weight) == ("code", "released", "sigma", False, 1.0)
    assert variant("control") == Config() and variant("control", n_epochs=5).n_epochs == 5


# =====================================================================================================================
# from test_stage10.py
# =====================================================================================================================


QS = (0.025, 0.1, 0.5, 0.9, 0.975)


def raises(fn, error=ValueError):
    try:
        fn()
    except error:
        return True
    return False


def test_weights():
    w = weights(np.array([[0.0, 1.0, 2.0]]), np.log(2))                     # exp(-beta Phi) = 1, 1/2, 1/4
    assert np.allclose(w, [[4 / 7, 2 / 7, 1 / 7]])
    rng = np.random.default_rng(0)
    phi = rng.exponential(0.05, (5, 300))
    assert np.allclose(weights(phi + 7.0, 40.0), weights(phi, 40.0))      # only differences of Phi matter
    assert np.allclose(weights(phi, 0.0), 1 / 300) and np.allclose(weights(phi, 40.0).sum(1), 1)
    assert np.allclose(weights(np.full((2, 4), 0.3), 1e6), 0.25)           # equal penalties: equal weights
    big = np.array([[1000.0, 1001.0]])                                     # beta Phi far beyond exp's range
    with np.errstate(all="ignore"):
        naive = np.exp(-big) / np.exp(-big).sum()                          # the released formula: 0 / 0
    assert np.isnan(naive).all()
    assert np.allclose(weights(big, 1.0), [[1 / (1 + np.exp(-1)), np.exp(-1) / (1 + np.exp(-1))]])
    assert raises(lambda: weights(phi, -1.0)) and raises(lambda: weights(-phi, 1.0))
    assert raises(lambda: weights(np.where(phi > 0.1, np.nan, phi), 1.0)) and raises(lambda: weights(phi[0], 1.0))


def test_beta_rules():
    rng = np.random.default_rng(1)
    phi = rng.exponential(0.02, (6, 2000))
    phi[3] = 0.0                                                           # a day with no arbitrage at all
    b = beta_t(phi, "max_penalty")
    assert np.allclose(np.delete(b * phi.max(1), 3), 500) and b[3] == 0
    assert np.allclose(weights(phi, b)[3], 1 / 2000)
    worst = phi.argmax(1)[0]
    w = weights(phi, b)[0]                                                 # worst scenario: exp(-500) x the best
    assert np.isclose(np.log(w[worst] / w.max()), -500 * (phi[0, worst] - phi[0].min()) / phi[0].max())
    assert np.allclose(beta_t(phi, "fixed", 1000), 1000) and raises(lambda: beta_t(phi, "fixed"))
    assert np.array_equal(beta_t(phi, "raw"), np.zeros(6)) and raises(lambda: beta_t(phi, "nonsense"))
    lit = literal_beta(np.delete(phi, 3, axis=0))
    p = np.delete(phi, 3, axis=0)
    assert np.allclose(lit * max_weight(p, lit), 500, rtol=1e-9)           # beta = 500 / max_i w_i(beta)
    assert ((0.9 * lit) * max_weight(p, 0.9 * lit) < 500).all() and ((1.1 * lit) * max_weight(p, 1.1 * lit) > 500).all()
    assert np.isclose(literal_beta(np.full((1, 2000), 0.1))[0], 500 * 2000)  # all equal: max w = 1/N


def test_diagnostics():
    N = 1000
    uniform, one = np.full((1, N), 1 / N), np.eye(N)[:1]
    assert np.isclose(effective_sample_size(uniform)[0], N) and np.isclose(relative_entropy(uniform)[0], 0, atol=1e-12)
    assert np.isclose(effective_sample_size(one)[0], 1) and np.isclose(relative_entropy(one)[0], np.log(N))
    part = np.r_[np.full(648, 1 / 648), np.zeros(N - 648)][None]           # Cont and Vuletic: 64.8% arbitrage-free
    assert np.isclose(relative_entropy(part)[0], np.log(1 / 0.648))


def test_weighted_mean():
    rng = np.random.default_rng(2)
    x, w = rng.normal(size=(4, 50)), weights(rng.exponential(0.1, (4, 50)), 20.0)
    assert np.allclose(weighted_mean(x, w), [np.average(x[d], weights=w[d]) for d in range(4)])
    s = rng.normal(size=(4, 50, 3))
    assert np.allclose(weighted_mean(s, w)[:, 2], weighted_mean(s[:, :, 2], w))
    assert np.allclose(weighted_mean(x, weights(np.zeros((4, 50)), 0.0)), x.mean(1))


def eq19_loop(x, w, q):
    """Eq 19 written out: sort by value, add the weights in that order, stop at the first sum >= q."""
    order = np.argsort(x, kind="stable")
    total, j = 0.0, -1
    while total < q - 1e-12:
        j += 1
        total += w[order[j]]
    return x[order[max(j, 0)]]


def test_weighted_quantile():
    x, w = np.array([[3.0, 1.0, 2.0]]), np.array([[0.2, 0.5, 0.3]])        # sorted 1, 2, 3 with weights .5, .3, .2
    assert np.allclose(weighted_quantile(x, w, [0.5, 0.6, 0.8, 0.9])[:, 0], [1, 2, 2, 3])
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 501))
    w = weights(rng.exponential(0.05, (20, 501)), rng.uniform(0, 100, 20))
    got = weighted_quantile(x, w, QS)
    for d in range(20):
        for i, q in enumerate(QS):
            assert got[i, d] == np.quantile(x[d], q, weights=w[d], method="inverted_cdf")   # NumPy's own
            assert got[i, d] == eq19_loop(x[d], w[d], q)                                    # eq 19 as a loop
            assert got[i, d] == orig.quantile(x[d].copy(), w[d].copy(), q)                 # the original
    # equal weights: the "inverted cdf" quantile; with an even N the median is the lower middle value
    xe = rng.normal(size=(3, 10))
    u = np.full((3, 10), 0.1)
    assert np.allclose(weighted_quantile(xe, u, [0.5])[0], np.sort(xe, axis=1)[:, 4])
    assert np.allclose(weighted_quantile(xe, u, QS), np.quantile(xe, QS, axis=1, method="inverted_cdf"))
    # surfaces: every grid point separately (B5)
    s = rng.normal(size=(4, 501, 6))
    ws = w[:4]
    gs = weighted_quantile(s, ws, QS)
    for k in range(6):
        assert np.array_equal(gs[:, :, k], weighted_quantile(s[:, :, k], ws, QS))
    assert raises(lambda: weighted_quantile(x, w, [1.5])) and raises(lambda: weighted_quantile(x, w[:, :10], [0.5]))


def test_the_original_bugs_are_fixed():
    quiet = contextlib.redirect_stderr(io.StringIO())
    # B5: the return quantile stops one element late (q = 0.5 of 1, 2, 3 with equal weights is 2)
    assert orig.VolGAN_quantile_ret_day(np.array([1.0, 2.0, 3.0]), np.full(3, 1 / 3), 0.5) == 3.0
    assert weighted_quantile(np.array([[1.0, 2.0, 3.0]]), np.full((1, 3), 1 / 3), [0.5])[0, 0] == 2.0
    # B5: the surface "quantile" sorts by weight and returns one whole scenario, not a value per grid point
    s = np.array([[1.0, 30.0], [2.0, 10.0], [3.0, 20.0]])                  # 3 scenarios, 2 grid points
    old = orig.VolGAN_quantile_surface_day(s, np.full(3, 1 / 3), 0.5)
    assert any(np.array_equal(old, row) for row in s)                      # a whole scenario
    assert np.array_equal(weighted_quantile(s[None], np.full((1, 3), 1 / 3), [0.5])[0, 0], [2.0, 20.0])
    # B4: the weighted median adds weights in the wrong order, and sorts the caller's array in place
    pen = np.array([[3.0, 0.0, 1.0]])
    with quiet:
        _, median_old = orig.reweighting_stats_mean(pen, 1.0)
    assert np.array_equal(pen, [[0.0, 1.0, 3.0]])                          # the input was changed
    pen = np.array([[3.0, 0.0, 1.0]])
    right = weighted_quantile(pen, weights(pen, 1.0), [0.5])[0, 0]
    assert right == 0.0 and median_old[0] == 1.0                           # weights 0.71, 0.26, 0.04 on 0, 1, 3
    assert np.array_equal(pen, [[3.0, 0.0, 1.0]])                          # ours leaves the input alone
    # B17: the original sampler fails on NumPy 2 (a length-1 array into a single number)
    w = weights(np.array([[0.0, 0.1, 0.2]]), 10.0)
    assert raises(lambda: orig.VolGAN_sample(np.zeros((3, 1, 2)), np.zeros((3, 1)), w))
    idx = sample_indices(w, 1, np.random.default_rng(0))
    assert idx.shape == (1, 1) and np.issubdtype(idx.dtype, np.integer)


def test_sampling_frequencies():
    w = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    idx = sample_indices(w, 200_000, np.random.default_rng(4))
    for d in range(2):
        assert np.allclose(np.bincount(idx[d], minlength=4) / 200_000, w[d], atol=0.005)
    assert np.array_equal(idx, sample_indices(w, 200_000, np.random.default_rng(4)))


@functools.lru_cache(maxsize=1)
def setup_10():
    data, split = prepare("SPX", Config())
    set_seed(0)
    gen, _ = build_networks(N_SCALARS + data.lk * data.lt, 1 + data.lk * data.lt, Config(), torch.device("cpu"))
    sub = split.test.slice(0, 6)
    return gen, sub, simulate(gen, sub, 400, seed=5)


def test_daily_statistics():
    gen, sub, sim = setup_10()
    beta = beta_t(sim.phi, "max_penalty")
    out = daily_statistics(gen, sub, sim, {"raw": 0.0, "re-weighted": beta}, qs=QS, chunk_days=4)
    raw, rew = out["raw"], out["re-weighted"]
    _, sigma = generate(gen, sub, np.arange(6), 400, sim.seed)
    # raw = the plain statistics of the scenarios
    assert np.allclose(raw.ret_mean, sim.returns.mean(1)) and np.allclose(raw.phi_mean, sim.phi.mean(1))
    assert np.allclose(raw.surf_mean, sigma.mean(1), rtol=1e-6)
    assert np.allclose(raw.surf_q, np.quantile(sigma, QS, axis=1, method="inverted_cdf"), rtol=1e-6)
    assert np.allclose(raw.ret_q, np.quantile(sim.returns, QS, axis=1, method="inverted_cdf"))
    assert np.allclose(raw.ess, 400) and np.allclose(raw.entropy, 0, atol=1e-12)
    # re-weighted: the weights of eq 17, and tilting towards low Phi can only lower the mean penalty
    w = weights(sim.phi, beta)
    assert np.allclose(rew.phi_mean, (w * sim.phi).sum(1)) and np.array_equal(rew.beta, beta)
    assert (rew.phi_mean <= raw.phi_mean + 1e-15).all() and (rew.ess <= 400 + 1e-9).all()
    assert np.allclose(rew.surf_q[2], weighted_quantile(sigma, w, [0.5])[0], rtol=1e-6)
    # the rebuilt scenarios are checked to be the ones that were weighted
    assert raises(lambda: daily_statistics(gen, sub, dataclasses.replace(sim, seed=6), {"raw": 0.0}), RuntimeError)
    assert raises(lambda: daily_statistics(gen, sub, dataclasses.replace(sim, rate=sim.rate + 0.05), {"raw": 0.0}),
                  RuntimeError)
    assert raises(lambda: daily_statistics(gen, sub.slice(0, 5), sim, {"raw": 0.0}))


# =====================================================================================================================
# from test_stage11.py
# =====================================================================================================================


M = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

TAU = np.array([30, 60, 90, 180, 270, 360]) / 365


def test_interpolation_is_the_papers_rule():
    rng = np.random.default_rng(0)
    pts = [(1.0, 30 / 365), (1.0, 0.25), (0.83, 0.4), (0.75, 1 / 52), (1.25, 1.2), (1.02, 1 / 252)]   # in and outside
    # exact for any bilinear function a + b m + c tau + d m tau, inside the grid and in the extrapolated region
    for _ in range(5):
        a, b, c, d = rng.normal(size=4)
        T, Mg = np.meshgrid(TAU, M, indexing="ij")
        surf = (a + b * Mg + c * T + d * Mg * T).reshape(1, -1)
        assert np.allclose(interpolate(surf, M, TAU, pts)[0], [a + b * mm + c * tt + d * mm * tt for mm, tt in pts])
    # on the grid it returns the grid value
    surf = rng.uniform(0.1, 0.4, (3, 42))
    assert np.allclose(interpolate(surf, M, TAU, [(M[2], TAU[4])]), surf[:, [4 * 7 + 2]])
    # equal to the original get_points3 (linear in m, then in tau, linear extrapolation)
    for s in surf:
        I = s.reshape(6, 7).T                                                # the original's (m, tau) matrix
        for mm, tt in pts:
            assert np.isclose(interpolate(s, M, TAU, [(mm, tt)])[0], orig.get_points3(I, M, TAU, [mm], [tt])[0, 0])
    # F11: get_points extrapolates FLAT, so it disagrees outside the grid and agrees inside
    I = surf[0].reshape(6, 7).T
    assert np.isclose(orig.get_points(I, M, TAU, [0.9], [0.3])[0, 0], interpolate(surf[0], M, TAU, [(0.9, 0.3)])[0])
    assert not np.isclose(orig.get_points(I, M, TAU, [0.7], [0.3])[0, 0], interpolate(surf[0], M, TAU, [(0.7, 0.3)])[0])
    assert np.allclose(linear_weights([0.0, 1.0, 3.0], [-1.0, 0.5, 4.0]).sum(axis=0), 1)


def continuous_vix(sig_fn, r=0.0, T=VIX_T):
    """The VIX formula with infinitely many strikes: K_0 = F, so the second term vanishes (independent of ours)."""
    f = np.exp(r * T)
    def price(m, put):
        s = sig_fn(m)
        d1 = (-np.log(m) + T * (r + s * s / 2)) / (s * np.sqrt(T))
        c = ndtr(d1) - m * np.exp(-r * T) * ndtr(d1 - s * np.sqrt(T))
        return c - 1 + m * np.exp(-r * T) if put else c
    puts = quad(lambda m: price(m, True) / m ** 2, 0.2, f, limit=200)[0]
    calls = quad(lambda m: price(m, False) / m ** 2, f, 2.5, limit=200)[0]
    return 100 * np.sqrt(2 / T * np.exp(r * T) * (puts + calls))


def test_vix_formula():
    # flat smile: Black-Scholes with constant vol has VIX = 100 sigma (up to the strike spacing)
    for r in (0.0, 0.04):
        got = vix(np.full((3, 7), 1.0) * np.array([[0.1], [0.2], [0.5]]), M, r)
        assert np.allclose(got, [10, 20, 50], atol=0.05), (r, got)
    # a skewed smile: our discrete sum equals the continuous formula integrated numerically, on the same smile
    rng = np.random.default_rng(1)
    for _ in range(3):
        smile = np.sort(rng.uniform(0.08, 0.45, 7))[::-1][None]            # falling with moneyness, like SPX
        fn = lambda m: smile_on_strikes(smile, M, np.array([m]))[0, 0]
        for r in (0.0, 0.03):
            assert abs(vix(smile, M, r)[0] - continuous_vix(fn, r)) < 0.06
    # wings: positive, the put wing never falls, the call wing never falls below the last grid vol's variance
    s = smile_on_strikes(np.vstack([smile[0], smile[0][::-1]]), M)
    assert (s > 0).all()
    left, right = VIX_STRIKES < M[0], VIX_STRIKES > M[-1]
    assert (np.diff(s[:, left], axis=1) <= 1e-12).all()                     # moving towards K/S = 0.8, vol falls
    assert (np.diff(s[:, right], axis=1) >= -1e-12).all()                   # beyond K/S = 1.2, vol never falls
    assert (s[:, right] >= s[:, [np.flatnonzero(VIX_STRIKES <= M[-1])[-1]]] - 1e-12).all()
    assert np.allclose(smile_on_strikes(smile, M, M), smile)                 # on the grid: the grid values


def test_the_original_vix_bugs():
    flat = np.full((7, 6), 0.2)                                              # the original's (m, tau) matrix
    ours = vix_of_surfaces(flat.T.reshape(1, -1), M, TAU)[0]
    old = orig.VIX(100.0, flat, M, TAU)
    assert abs(ours - 20) < 0.05 and old < 16                                # B3: only puts are summed
    # B2: only the extrapolated vol at 50% moneyness is used, so changing the smile above it changes nothing
    tilt = flat.copy()
    tilt[4:, :] = 0.12                                                       # cheaper calls at m >= 1.05
    assert np.isclose(orig.VIX(100.0, tilt, M, TAU), old)
    assert not np.isclose(vix_of_surfaces(tilt.T.reshape(1, -1), M, TAU)[0], ours)


@functools.lru_cache(maxsize=1)
def spx():
    data, split = prepare("SPX", Config())
    return data, split.test


def test_published_index_and_dates():
    data, test = spx()
    vix_series = load_vol_index("SPX")
    assert vix_series.index.is_monotonic_increasing and vix_series.index.is_unique
    actual = vix_series.reindex(data.dates)
    assert actual.notna().all()                                              # every surface day has a VIX close
    ours = vix_of_surfaces(data.vols, data.m, data.tau)
    err = ours - actual.to_numpy()
    assert np.corrcoef(ours, actual)[0, 1] > 0.99 and np.sqrt((err ** 2).mean()) < 1.5
    vxn = load_vol_index("NDX")
    nd, _ = prepare("NDX", Config())
    assert vxn.reindex(nd.dates).notna().mean() > 0.99
    # realised values belong to the predicted day test.dates[i] (B21)
    rows = data.dates.get_indexer(test.dates)
    rv = realised(test)
    assert np.allclose(rv["sigma"], data.vols[rows]) and np.allclose(rv["level"], data.spot[rows])
    assert np.allclose(rv["return"], np.log(data.spot[rows] / data.spot[rows - 1]))
    assert np.allclose(rv["vix_from_surface"], ours[rows])
    b = band_stats([1.0, 2.0, 5.0], [1.0, 2.0, 3.0], [0, 0, 0], [2, 3, 4])
    assert np.isclose(b["inside band %"], 200 / 3) and b["median band width"] == 3 and np.isclose(b["mean abs error of mean"], 2 / 3)


def test_extras_in_daily_statistics():
    data, test = spx()
    set_seed(0)
    gen, _ = build_networks(N_SCALARS + data.lk * data.lt, 1 + data.lk * data.lt, Config(), torch.device("cpu"))
    sub = test.slice(0, 5)
    sim = simulate(gen, sub, 300, seed=2, rate=0.02)
    ex = extras_for(sub, sim, points=[(1.0, 0.25)])
    out = daily_statistics(gen, sub, sim, {"raw": 0.0}, qs=(0.025, 0.5, 0.975), extras=ex)["raw"]
    r_hat, sigma = generate(gen, sub, np.arange(5), 300, sim.seed)
    v = vix_of_surfaces(sigma, sub.m, sub.tau, 0.02)
    assert np.allclose(out.extra_mean["VIX"], v.mean(1), rtol=1e-6)
    assert np.allclose(out.extra_q["VIX"], np.quantile(v, (0.025, 0.5, 0.975), axis=1, method="inverted_cdf"), rtol=1e-6)
    lvl = sub.spot_prev[:, None] * np.exp(r_hat)
    assert np.allclose(out.extra_mean["index level"], lvl.mean(1), rtol=1e-9)
    assert np.allclose(out.extra_mean["vol m=1.00, 91d"], interpolate(sigma, sub.m, sub.tau, [(1.0, 0.25)])[..., 0].mean(1), rtol=1e-6)


# =====================================================================================================================
# from test_stage13.py
# =====================================================================================================================

ZERO = Config(rate_source="zero")


class NoiseRecorder(nn.Module):
    """A generator that remembers the noise it was given."""

    def __init__(self, gen):
        super().__init__()
        self.gen, self.noise_dim, self.noise = gen, gen.noise_dim, None

    def forward(self, noise, condition):
        self.noise = noise.clone()
        return self.gen(noise, condition)


def test_common_random_numbers():
    P = prepare_all("SPX", ZERO)
    val = P.inputs.val
    nets = []
    for seed in (1, 2):                                                # two different trained-network stand-ins
        set_seed(seed)
        K = P.data.lk * P.data.lt
        nets.append(NoiseRecorder(build_networks(N_SCALARS + K, 1 + K, Config(), CPU)[0]))
    assert not torch.equal(nets[0].gen.linear1.weight, nets[1].gen.linear1.weight)
    outs = [generate_full(g, val, [5, 6], 100, SIM_SEED) for g in nets]
    assert torch.equal(nets[0].noise, nets[1].noise)                   # the same noise for every model
    assert not np.allclose(outs[0][0], outs[1][0])                     # different networks, different scenarios
    expect = torch.cat([day_noise(SIM_SEED, day_key(val.dates[j]), 100, 32) for j in (5, 6)])
    assert torch.equal(nets[0].noise, expect)
    generate_full(nets[0], val, [5, 6], 100, SIM_SEED + 1)
    assert not torch.equal(nets[0].noise, nets[1].noise)               # another sim seed, other noise
    # the simulation seed is one fixed number, separate from the training seeds, in every training configuration
    assert SIM_SEED == 20261012 and SIM_SEED not in SEEDS and Config().sim_seed == SIM_SEED
    assert all(training_config(n, s).sim_seed == SIM_SEED for n in TRAINING_CONFIGS for s in SEEDS)
    assert sorted(CELLS) == list("ABCD") and CELLS["A"] == ("raw_phiout", "max_penalty")


def test_market_values_are_the_raw_vendor_grid():
    P = prepare_all("SPX", Config(rate_source="zero"))
    inputs, market = P.inputs.val, P.market.val
    assert inputs is market and inputs.dates.equals(market.dates)       # the networks see, and are scored on, one grid
    rows = P.data.dates.get_indexer(market.dates)
    rv = realised(market, 0.0, inputs.dates)
    assert np.array_equal(rv["sigma"], market_sigma(market, inputs.dates))
    assert np.allclose(rv["sigma"], P.data.vols[rows], rtol=1e-12)    # the market as quoted
    assert np.allclose(rv["return"], np.log(P.data.spot[rows] / P.data.spot[rows - 1]))
    try:
        market_sigma(market, inputs.dates[1:])
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned dates were accepted as the market")


# =====================================================================================================================
# from test_stage14.py
# =====================================================================================================================


def test_violation_location():
    i, j = np.meshgrid(np.arange(7), np.arange(6), indexing="ij")
    C = (0.5 + 0.02 * j - 0.4 * (M[:, None] - 0.8))[None].astype(float)   # linear in m: no butterfly; rising in tau
    p = arbitrage_penalty(C, M, TAU)
    assert p.total[0] < 1e-12 and not violation_nodes(C, M, TAU).any()
    C[0, 3, 2] += 0.004                                                 # lift one price: concave at that node only
    p = arbitrage_penalty(C, M, TAU)
    assert p.butterfly[0] > 1e-4 and p.calendar[0] == 0 and p.call[0] == 0
    flagged = violation_nodes(C, M, TAU)[0]
    expect = np.zeros((7, 6), bool)
    expect[[2, 3, 4], 2] = True                                         # (i-1, j), (i, j), (i+1, j)
    assert np.array_equal(flagged, expect)
    C2 = C.copy()
    C2[0, 3, 3] = C2[0, 3, 2] - 0.001                                   # a calendar violation next to it
    assert violation_nodes(C2, M, TAU)[0][3, 2] and violation_nodes(C2, M, TAU)[0][3, 3]


def test_rho_t():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(5, 400)), rng.normal(size=(5, 400))
    y = 0.6 * x + y
    w = np.full((5, 400), 1 / 400)
    assert np.allclose(weighted_corr(x, y, w), [np.corrcoef(x[d], y[d])[0, 1] for d in range(5)], atol=1e-12)
    w = np.zeros((2, 400))
    w[:, [3, 17]] = 0.5                                                 # all the weight on two distinct scenarios
    got = weighted_corr(x[:2], y[:2], w)
    sign = np.sign((x[:2, 3] - x[:2, 17]) * (y[:2, 3] - y[:2, 17]))
    assert np.allclose(got, sign, atol=1e-12) and set(np.abs(got).round(12)) == {1.0}
    lo, hi = correlation_band(-0.7, 10_000)
    assert lo < -0.7 < hi and np.isclose(np.arctanh(hi) - np.arctanh(-0.7), 1.959963984540054 / np.sqrt(9997))


def test_pca_and_correlation_helpers():
    rng = np.random.default_rng(1)
    K = 42
    V, _ = np.linalg.qr(rng.normal(size=(K, K)))
    lam = np.r_[10.0, 4.0, 1.5, np.linspace(0.3, 0.01, K - 3)]
    cov_g = V @ np.diag(lam) @ V.T
    vecs, share = principal_components(cov_g, 3)                        # exact covariance: exact recovery
    assert np.allclose(share, lam[:3] / lam.sum()) and np.allclose(np.abs((vecs * V[:, :3]).sum(0)), 1)
    z_g = rng.multivariate_normal(np.zeros(K), cov_g, size=200_000)
    vs, ss = principal_components(np.cov(z_g, rowvar=False), 3)        # sampled: recovered to sampling error
    assert np.allclose(ss, lam[:3] / lam.sum(), atol=0.005) and (np.abs((vs * V[:, :3]).sum(0)) > 0.999).all()
    # the correlation tables from a covariance equal numpy's correlations of the series themselves
    n = 3000
    g = z_g[:n]
    r = 0.01 * (g @ V[:, 0]) / np.sqrt(lam[0]) + 0.01 * rng.normal(size=n)
    dlv = 0.5 * g[:, 3] + rng.normal(size=n)
    z = increments(r, g, dlv)
    dyn = dynamics_from_cov(np.cov(z, rowvar=False), V[:, :3], atm=3)
    proj = g @ V[:, :3]
    assert np.allclose(dyn["ret_pc"], [np.corrcoef(r, proj[:, k])[0, 1] for k in range(3)], atol=1e-10)
    ref = np.corrcoef(np.column_stack([r, proj[:, 0], g[:, 3], dlv]), rowvar=False)
    assert np.allclose(dyn["corrmat"], ref, atol=1e-10) and np.allclose(correlation(np.cov(z.T))[0, 1:3],
                                                                         np.corrcoef(z.T)[0, 1:3])


def small_case(n_days=12, N=60, seed=4):
    P = prepare_all("SPX", Config(rate_source="zero"))
    inputs, market = P.inputs.val.slice(0, n_days), P.market.val.slice(0, n_days)
    set_seed(seed)
    K = P.data.lk * P.data.lt
    gen = build_networks(N_SCALARS + K, 1 + K, Config(), torch.device("cpu"))[0]
    for p in gen.parameters():                                          # a rough network: some arbitrage to weigh
        p.data.mul_(3.0)
    sim = simulate(gen, inputs, N, Config().sim_seed, rate=inputs.rates)
    return gen, inputs, market, sim


def test_daily_metrics_by_hand_and_cell_bookkeeping():
    gen, inputs, market, sim = small_case()
    n_days, N = sim.phi.shape
    ref = market_reference(market)
    out = daily_metrics(gen, inputs, market, sim, ("raw", "max_penalty"), b_pca=20, chunk_days=5, reference=ref)
    raw, rew = out["raw"], out["max_penalty"]
    assert (sim.phi > 0).mean() > 0.05                                  # re-weighting has something to do
    w = weights(sim.phi, beta_t(sim.phi, "max_penalty"))
    assert np.allclose(rew["viol_share"], (w * (sim.phi > 1e-12)).sum(1)) and np.allclose(raw["phi_mean"], sim.phi.mean(1))
    assert np.allclose(raw["ess"], N) and (rew["ess"] <= N + 1e-9).all()
    assert np.allclose(raw["phi_parts"].sum(1), raw["phi_mean"]) and raw["viol_loc"].shape == (n_days, inputs.m.size, inputs.tau.size)
    # rho_t from the rebuilt scenarios
    r_hat, dg, sigma = generate_full(gen, inputs, np.arange(n_days), N, sim.seed)
    atm = int(node_index(inputs.m, inputs.tau, [ATM_1M])[0])
    assert np.allclose(raw["rho"], [np.corrcoef(r_hat[d], dg[d, :, atm])[0, 1] for d in range(n_days)])
    # the PCA moments accumulated chunk by chunk equal numpy.cov of the explicitly built series
    vix = vix_of_surfaces(sigma, inputs.m, inputs.tau, 0.0)
    dlv = np.log(vix) - np.log(ref["vix_cond"])[:, None]
    for rule, met in out.items():
        idx = pca_indices(weights(sim.phi, met["beta"]), met["beta"], 20, sim.seed)
        shares = []
        for b in range(20):
            k = idx[:, b]
            z = increments(r_hat[np.arange(n_days), k], dg[np.arange(n_days), k], dlv[np.arange(n_days), k])
            shares.append(principal_components(np.cov(z[:, 1:-1], rowvar=False), 3)[1])
        assert np.allclose(met["pca_share"], shares, atol=1e-8), rule
    assert (pca_indices(w, beta_t(sim.phi, "max_penalty"), 20, sim.seed)[beta_t(sim.phi, "max_penalty") == 0]
            == np.arange(20)).all()
    # calibration: realised values come from the market, indicators agree with the quantiles
    assert np.array_equal(raw["cal_realised"], ref["cal_realised"]) and raw["cal_q"].shape == (5, n_days, 11)
    assert np.array_equal(raw["inside_95"], (ref["cal_realised"] >= raw["cal_q"][1]) & (ref["cal_realised"] <= raw["cal_q"][3]))
    # cell bookkeeping (test 12): cell B = cell A computed with beta = 0, from the same networks
    zero = daily_metrics(gen, inputs, market, sim, ("max_penalty",), b_pca=20, reference=ref,
                         beta_override={"max_penalty": np.zeros(n_days)})["max_penalty"]
    for key, v in raw.items():
        if key != "rho_n":
            assert np.allclose(np.asarray(v, float), np.asarray(zero[key], float), rtol=1e-10, atol=1e-12), key
    assert np.isclose(zero["rho_n"], raw["rho_n"])
    try:
        daily_metrics(gen, inputs, inputs.slice(0, n_days - 1), sim)
    except ValueError:
        pass
    else:
        raise AssertionError("a market of other dates was accepted")


def _run(tests, title):
    """Run every test; one that needs the market workbooks is skipped, with a note, when they are not found."""
    skipped = []
    for fn in tests:
        try:
            fn()
        except FileNotFoundError as e:
            skipped.append(fn.__name__)
            print("SKIP %s: needs the market workbooks (%s)" % (fn.__name__, str(e).split(":")[-1].strip()[:80]))
    print("%s checks passed%s" % (title, " (%d skipped: market workbooks not found)" % len(skipped) if skipped else ""))


if __name__ == "__main__":
    _run([
        test_noise_is_per_day_and_reproducible,
        test_scenarios_are_built_as_in_the_paper,
        test_phi_matches_stage2_surface_by_surface,
        test_chunking_and_regeneration_do_not_change_results,
        test_outputs_and_rate,
        test_market_sigma_is_the_data_on_the_predicted_day,
        test_run_folder_round_trip,
        test_variants,
        test_weights,
        test_beta_rules,
        test_diagnostics,
        test_weighted_mean,
        test_weighted_quantile,
        test_the_original_bugs_are_fixed,
        test_sampling_frequencies,
        test_daily_statistics,
        test_interpolation_is_the_papers_rule,
        test_vix_formula,
        test_the_original_vix_bugs,
        test_published_index_and_dates,
        test_extras_in_daily_statistics,
        test_common_random_numbers,
        test_market_values_are_the_raw_vendor_grid,
        test_violation_location,
        test_rho_t,
        test_pca_and_correlation_helpers,
        test_daily_metrics_by_hand_and_cell_bookkeeping,
    ], "Post-processing")
