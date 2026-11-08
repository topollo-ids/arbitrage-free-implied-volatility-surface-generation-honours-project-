"""
Recursion (Part 2, Part A) checks, merged by topic from the stage tests (every assertion kept). Run on its own:

    python tests/test_recursion.py

--- from rec_synth.py ---
Synthetic fixtures for the Part A tests: a small random market and a randomly initialised generator (no workbook).

--- from test_rec_a.py ---
Part A checks, part 1 (brief section 10, tests 1 to 4, and the guards): the state update, the calendar noise, the
h = 1 identity, validity and survival, the test-block guard. CPU, synthetic data, no workbook needed.

--- from test_rec_b.py ---
Part A checks, part 2 (brief section 10, tests 5 to 8): cloud metrics against closed forms, the moving-block
bootstrap, Holm, the sufficient-statistics identities, and resumability end to end on synthetic data.
"""

from dataclasses import replace
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from volgan_fixed import recursion, recursion_analysis as inf
from volgan_fixed.common import Config, RecursionConfig, SIM_SEED, set_seed
from volgan_fixed.postprocessing import (
    ATM_1M, CALIBRATION_POINTS, TOL_PHI, day_key, day_noise, generate_full, node_index, scenario_phi, step)
from volgan_fixed.preprocessing import N_SCALARS, Split, SurfaceData, data_preprocessing, prepare_all
from volgan_fixed.recursion import (
    Arena, FIXED_NOISE_DATES, NoiseTable, State, batched_next_state, cloud_metrics, identity_diffs, initial_state,
    next_state, noise_hash, noise_hashes, rollout, run_start, start_indices)
from volgan_fixed.training import Generator, build_networks


# =====================================================================================================================
# from rec_synth.py
# =====================================================================================================================


M = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

TAU = np.array([30, 60, 90, 180, 270, 360]) / 365

K = len(M) * len(TAU)


def synthetic(n_days=150, seed=0, rv_window="paper"):
    """(SurfaceData, Dataset of all samples, Split with val = the middle part)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    spot = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n_days)))
    vols = 0.2 * np.exp(0.1 * rng.standard_normal((n_days, K)))
    rates = np.full((n_days, len(TAU)), 0.02)
    data = SurfaceData("SYN", dates, spot, vols, M, TAU, r=rates)
    ds = data_preprocessing(data, rv_window, np.sqrt(252), rates)
    n = len(ds.dates)
    split = Split(train=ds.slice(0, 20), test=ds.slice(n - 20, n), val=ds.slice(20, n - 20))
    return data, ds, split


def generator(seed=0):
    torch.manual_seed(seed)
    gen = Generator(32, 3 + K, 16, 1 + K)
    with torch.no_grad():
        gen.linear3.weight *= 0.05             # a stable random network: the 21-day recursion stays inside the bounds
        gen.linear3.bias *= 0.05
    return gen.eval()


def fake_run(gen=None, rv_window="paper", n_days=150, seed=0):
    data, ds, split = synthetic(n_days, seed, rv_window)
    return SimpleNamespace(gen=gen or generator(seed), config=SimpleNamespace(sim_seed=SIM_SEED, noise_dim=32),
                           split=split, market=split, data=data, info={"signature": "synthetic"})


def arena(cfg=None, **kw):
    cfg = cfg or RecursionConfig()
    return Arena(fake_run(**kw), cfg)


# =====================================================================================================================
# from test_rec_a.py
# =====================================================================================================================


# computed once on the reference machine (torch 2.14.1 CPU); a different machine / torch that changes the noise fails here
REFERENCE_NOISE_HASHES = {"2018-01-02": "6fb4decb627467bd", "2019-07-15": "5e247ab3cdf4b57e",
                          "2021-12-31": "04c94274979b4839"}


def test_batched_next_state():
    rng = np.random.default_rng(1)
    P, rs = 7, np.sqrt(252)
    for rvw in ("paper", "code"):
        cond = np.column_stack([rng.normal(size=(P, 2)), 0.2 + rng.random((P, 1)), rng.normal(-1.5, 0.2, (P, K))])
        buf = 0.01 * rng.standard_normal((P, 22))
        states = [State(cond[p].copy(), buf[p].copy(), rs, rvw) for p in range(P)]
        for _ in range(21):
            r_hat, dg = rng.normal(size=P), 0.02 * rng.standard_normal((P, K))
            cond, buf = batched_next_state(cond, buf, r_hat, dg, rs, rvw)
            states = [next_state(s, r_hat[p], dg[p]) for p, s in enumerate(states)]
            for p, s in enumerate(states):
                assert np.allclose(cond[p], s.condition, rtol=1e-13, atol=1e-14), rvw
                assert np.allclose(buf[p], s.returns, rtol=0, atol=1e-15)


def test_calendar_noise():
    t1, t2 = NoiseTable(SIM_SEED), NoiseTable(SIM_SEED)
    d = "2019-03-04"
    z = t1.get(d)
    assert z.shape == (1000, 32) and z.dtype == torch.float32
    assert torch.equal(z, day_noise(SIM_SEED, day_key(d), 10_000, 32)[:1000])           # (a) the first 1,000 rows
    t2.get("2020-01-02"), t2.get("2018-06-01")                                          # evaluation order is irrelevant
    assert torch.equal(t2.get(d), z) and torch.equal(t1.get(d), z)
    assert not torch.equal(t1.get("2019-03-05"), z)                                     # different date, different noise
    assert noise_hash(z) == noise_hash(t2.get(d))
    assert not torch.equal(NoiseTable(SIM_SEED + 1).get(d), z)                          # the simulation seed matters
    # (drawing 1000 rows directly may or may not give the same numbers on another torch build: that is why the
    # recipe draws 10,000 and slices, and why nothing here depends on the direct draw)
    # (c) fixed dates equal the stored constants
    assert noise_hashes(NoiseTable(SIM_SEED), FIXED_NOISE_DATES) == REFERENCE_NOISE_HASHES


def test_benchmark_uses_the_same_noise_rows():
    a = arena()
    j = 4
    r, dg, sigma = generate_full(a.gen, a.ds, [j], 1000, SIM_SEED, n_draw=10_000)       # (b) generate_full ...
    z = NoiseTable(SIM_SEED).get(a.ds.dates[j])
    cond = torch.as_tensor(np.tile(a.ds.condition[j], (1000, 1)), dtype=torch.float32)
    with torch.no_grad():
        r_hat, dg_hat = step(a.gen, cond, z)                                             # ... and a hand-built pass
    assert np.array_equal(r[0], r_hat.double().numpy() / a.ds.ret_scale)
    assert np.array_equal(dg[0], dg_hat.double().numpy())
    assert np.array_equal(sigma[0], np.exp(a.ds.condition[j, None, N_SCALARS:] + dg_hat.double().numpy()))
    # the default is the Part 1 behaviour: n drawn, n kept
    r0, _, _ = generate_full(a.gen, a.ds, [j], 50, SIM_SEED)
    z50 = day_noise(SIM_SEED, day_key(a.ds.dates[j]), 50, 32)
    with torch.no_grad():
        r50, _ = step(a.gen, torch.as_tensor(np.tile(a.ds.condition[j], (50, 1)), dtype=torch.float32), z50)
    assert np.array_equal(r0[0], r50.double().numpy() / a.ds.ret_scale)


def test_h1_identity_and_recursion_departs():
    for rvw in ("paper", "code"):
        a = arena(rv_window=rvw)
        rec = run_start(a, 3)                                                            # raises if h = 1 differs
        assert rec["id_ret"] == 0 and rec["id_dg"] == 0 and rec["id_sigma"] == 0
        outs = list(rollout(a.gen, a.ds, a.run.data, 3, a.noise, a.cfg))
        assert [o.k for o in outs] == list(range(1, 22)) and [o.j for o in outs] == list(range(3, 24))
        assert max(identity_diffs(a, 3, outs[0]).values()) == 0
        r, dg, sg = generate_full(a.gen, a.ds, [4], 1000, SIM_SEED, n_draw=10_000)
        assert np.abs(outs[1].sigma - sg[0]).max() > 1e-6                                # h = 2 is fed back: it differs
        assert rec["n_valid"][0] == 1000


class Fake(nn.Module):
    """Returns zeros, except on call number `at`: path 0 gets NaN, path 1 a 400% vol, path 2 an infinite return."""
    noise_dim = 32

    def __init__(self, at=3):
        super().__init__()
        self.p, self.at, self.calls = nn.Parameter(torch.zeros(1)), at, 0

    def forward(self, noise, cond):
        self.calls += 1
        out = torch.zeros(noise.shape[0], 1 + K)
        if self.calls == self.at:
            out[0, 1:] = float("nan")
            out[1, 1:] = 3.0                      # sigma ~ 0.2 e^3 = 4.0 > 3.00
            out[2, 0] = float("inf")
        return out


def test_validity_is_absorbing():
    a = arena()
    a.gen = Fake(at=3)
    seen = []

    def post(k, cond, r_hat, dg_hat, valid, ctx):
        seen.append((k, bool(np.isfinite(cond).all()), int(valid.sum())))
        return cond, k
    outs = list(rollout(a.gen, a.ds, a.run.data, 2, a.noise, a.cfg, post_step=post))
    n_valid = [int(o.valid.sum()) for o in outs]
    assert n_valid[:2] == [1000, 1000] and all(n == 997 for n in n_valid[2:]), n_valid      # stays dead at k > 3
    assert all(n1 <= n0 for n0, n1 in zip(n_valid, n_valid[1:]))
    assert not outs[3].valid[:3].any() and outs[3].valid[3:].all()
    assert [s[0] for s in seen] == list(range(2, 22))                                      # the hook skips step 1
    assert all(s[1] for s in seen)                                                         # dead paths stay finite
    assert [o.extra for o in outs[1:4]] == [2, 3, 4] and outs[0].extra is None
    # a cloud with no valid path gives NaN and a flag, not an exception
    empty = a.cloud(5, outs[3].sigma[:0], outs[3].r[:0], outs[3].dg[:0], with_nodes=True)
    assert empty["ok"] is False and empty["n_valid"] == 0 and np.isnan(empty["phi_mean"])
    assert np.isnan(empty["inside"]).all() and np.isnan(empty["viol_loc"]).all()
    live = a.cloud(5, outs[3].sigma[outs[3].valid], outs[3].r[outs[3].valid], outs[3].dg[outs[3].valid])
    assert live["ok"] and live["n_valid"] == 997


def test_guards_and_starts():
    cfg = RecursionConfig(block="test")
    try:
        Arena(fake_run(), cfg)
        raise AssertionError("the test block must be refused")
    except PermissionError:
        pass
    assert Arena(fake_run(), cfg, allow_test=True).ds is not None
    try:
        Arena(fake_run(), RecursionConfig(sim_seed=1))
        raise AssertionError("a sim_seed other than the run's must be refused")
    except ValueError:
        pass
    s = start_indices(87, RecursionConfig())
    assert s[0] == 0 and s[-1] + 20 <= 86 and np.all(np.diff(s) == 5) and s[-1] + 5 + 20 > 86
    assert RecursionConfig().block_len_starts == 5 and RecursionConfig(stride=3).block_len_starts == 7


# =====================================================================================================================
# from test_rec_b.py
# =====================================================================================================================


def real_for(values, n_days=3):
    cal = node_index(M, TAU, CALIBRATION_POINTS)
    return SimpleNamespace(m=M, tau=TAU, cal=cal, atm=int(node_index(M, TAU, [ATM_1M])[0]),
                           values=np.tile(np.asarray(values, float), (n_days, 1)), rates=np.full((n_days, len(TAU)), 0.02))


def test_metrics_closed_form():
    n = 50
    sigma = np.full((n, K), 0.2)
    r, dg = np.zeros(n), np.zeros(n)
    inside = cloud_metrics(sigma, r, dg, real_for([0.2] * 9 + [0.0]), 1)
    assert inside["ok"] and inside["n_valid"] == n
    assert (inside["inside"] == 1).all() and (inside["below"] == 0).all() and (inside["above"] == 0).all()
    assert (inside["width"] == 0).all() and np.isnan(inside["rho"])                    # identical paths: rho undefined
    assert np.allclose(inside["q"], np.array([[0.2] * 9 + [0.0]] * 3))
    p = scenario_phi(sigma[None], M, TAU, np.full((1, len(TAU)), 0.02))
    assert np.isclose(inside["phi_mean"], p.total.mean()) and inside["viol_share"] == float((p.total > TOL_PHI).mean())
    assert np.allclose(inside["phi_parts"], [p.calendar.mean(), p.call.mean(), p.butterfly.mean()])
    above = cloud_metrics(sigma, r, dg, real_for([0.3] * 9 + [0.01]), 1)
    assert (above["inside"] == 0).all() and (above["above"] == 1).all() and (above["below"] == 0).all()
    below = cloud_metrics(sigma, r, dg, real_for([0.1] * 9 + [-0.01]), 1)
    assert (below["inside"] == 0).all() and (below["below"] == 1).all()
    # rho: perfectly correlated increments give 1; a single path or two give NaN rather than an error
    rng = np.random.default_rng(0)
    r = rng.standard_normal(n)
    c = cloud_metrics(sigma, r, 2 * r, real_for([0.2] * 10), 0, with_nodes=True)
    assert np.isclose(c["rho"], 1.0, atol=1e-12) and c["viol_loc"].shape == (len(M), len(TAU))
    assert np.isnan(cloud_metrics(sigma[:2], r[:2], r[:2], real_for([0.2] * 10), 0)["rho"])
    # the 95% interval of a known sample: equal weights, eq 19 (the smallest value with cumulative weight >= q)
    x = np.arange(1, 101, dtype=float)
    s = np.full((100, K), 0.2)
    c = cloud_metrics(s, x, x, real_for([0.2] * 9 + [50.0]), 0)
    assert c["q"][0, 9] == 3.0 and c["q"][2, 9] == 98.0 and c["inside"][9] == 1.0 and c["width"][9] == 95.0


def test_bootstrap_blocks():
    rng = np.random.default_rng(5)
    n, L, B = 23, 5, 400
    idx = inf.block_indices(n, L, B, rng)
    assert idx.shape == (B, n) and idx.min() >= 0 and idx.max() <= n - 1
    # inside every block of 5 the starts are consecutive and the blocks never wrap around
    for row in idx[:50]:
        for a in range(0, n - L + 1, L):
            assert np.array_equal(np.diff(row[a:a + L]), np.ones(L - 1))
    # blocks are drawn with replacement: repeats occur; the last block is truncated, not wrapped
    assert any(len(np.unique(row[::L])) < len(row[::L]) for row in idx)
    assert (idx[:, -3:] - idx[:, -3][:, None] == np.arange(3)).all()
    # every first start of a block is a valid block start: 0 .. n - L
    assert idx[:, ::L].max() <= n - L
    # a sample shorter than a block: one block of length n
    assert (inf.block_indices(3, 5, 4, rng) == np.arange(3)).all()
    # the same seed gives the same indices
    assert np.array_equal(inf.block_indices(n, L, 10, np.random.default_rng(2)), inf.block_indices(n, L, 10, np.random.default_rng(2)))


def test_pvalue_degenerate_and_holm():
    B = 10_000
    assert inf.p_value(np.full(B, 0.5), 0.1) == 1 / (B + 1)                  # all replicates above the margin
    assert inf.p_value(np.full(B, 0.05), 0.1) == 1.0                         # all at or below it
    assert inf.p_value(np.r_[np.zeros(30), np.ones(70)], 0.5) == 31 / 101
    # a degenerate input through the whole chain: every start has the same degradation > margin
    n, H = 40, 3
    ones = np.ones((2, n, H))
    S = {"A": (0.5 * ones, ones), "C": (0.99 * 9 * ones, 9 * ones, 0.95 * 9 * ones, 9 * ones),
         "D": (0.4 * ones, ones)}
    idx = inf.block_indices(n, 5, 200, np.random.default_rng(0))
    star = inf.bootstrap_curves(S, idx)
    assert np.allclose(star["A"], 0.5) and np.allclose(star["D"], 0.4)
    assert np.allclose(star["C"], abs(0.99 - 0.95) - 0.0)
    assert inf.p_value(star["A"][:, -1], 0.1) == 1 / 201
    # Holm on a known vector (the ED's worked example), and in a scrambled order
    assert np.allclose(inf.holm([0.004, 0.030, 0.200]), [0.012, 0.060, 0.200])
    assert np.allclose(inf.holm([0.200, 0.004, 0.030]), [0.200, 0.012, 0.060])
    assert np.allclose(inf.holm([0.01, 0.011, 0.5]), [0.03, 0.03, 0.5])      # monotone: 3*0.01, max(2*0.011, 0.03)
    assert np.allclose(inf.holm([0.6, 0.7, 0.8]), [1.0, 1.0, 1.0])


def test_powerlaw():
    h = np.arange(1, 22)
    fit = inf.fit_powerlaw(h, 0.01 * h ** 1.5)
    assert np.isclose(fit["a"], 0.01) and np.isclose(fit["b"], 1.5) and not fit["nonpositive_anywhere"]
    d = 0.01 * h ** 1.5
    d[5] = -0.1
    assert inf.fit_powerlaw(h, d)["nonpositive_anywhere"]


def test_resume_and_end_to_end():
    cfg = replace(RecursionConfig(), n_boot=300)
    a = arena(cfg)
    starts = recursion.start_indices(a.n, cfg)[:8]
    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "net")
        os.makedirs(folder)
        open(os.path.join(folder, "2_trained.pt"), "wb").write(b"weights")
        sig = recursion.seed_signature(a.run, folder, cfg)
        calls = []
        real_run_start = recursion.run_start
        recursion.run_start = lambda *args, **kw: (calls.append(args[1]), real_run_start(*args, **kw))[1]
        try:
            bench = recursion.get_bench(a, root, 1, sig, log=lambda m: None)
            recursion.run_seed(a, root, 1, sig, starts, log=lambda m: None)
            assert calls == list(starts)
            recursion.run_seed(a, root, 1, sig, starts, log=lambda m: None)                # rerun: a no-op
            assert calls == list(starts)
            recursion.run_seed(a, root, 1, sig, recursion.start_indices(a.n, cfg)[:10], log=lambda m: None)
            assert len(calls) == 10 and calls[8:] == list(recursion.start_indices(a.n, cfg)[8:10])   # only the new ones
            # a changed setting changes the signature, and a rerun refuses to mix
            sig2 = recursion.seed_signature(a.run, folder, replace(cfg, vol_hi=2.0))
            assert sig2 != sig and recursion.seed_signature(a.run, folder, replace(cfg, stride=3, n_boot=5)) == sig
            try:
                recursion.run_seed(a, root, 1, sig2, starts, log=lambda m: None)
                raise AssertionError("mixed settings must be refused")
            except RuntimeError:
                pass
            recursion.get_bench(a, root, 1, sig, log=lambda m: None)                       # cached, same signature
        finally:
            recursion.run_start = real_run_start
        # consolidate two "seeds" (the same network twice) and analyse
        starts = starts[:8]
        for s in (1, 2):
            if s == 2:
                bench2 = recursion.get_bench(a, root, 2, sig, log=lambda m: None)
                recursion.run_seed(a, root, 2, sig, starts, log=lambda m: None)
            recursion.consolidate(a, root, s, sig, starts, bench if s == 1 else bench2)
        T = inf.load_tables(root, [1, 2])
        F, Bn = inf.split_runs(T)
        assert F["inside"].shape == (2, 8, 21, 10) and F["q"].shape == (2, 8, 21, 3, 10)
        assert np.array_equal(F["n_valid"][:, :, 0], np.full((2, 8), 1000.0))
        res = inf.analyse(T, cfg, {"A": 1e-4, "C": 0.05, "D": 0.10})
        S = res["sufficient"]
        for P in inf.PILLARS:                                                          # at h = 1 free equals benchmark
            assert abs(inf.curve(S, P)[0]) < 1e-12, P
        assert res["boot"]["A"].shape == (300, 21)
        t = {r["pillar"]: r for r in res["tests"]}
        assert set(t) == {"A", "C", "D"} and all("T_seed1" in r and "T_seed2" in r for r in t.values())
        assert np.isclose(t["A"]["T"], np.mean([t["A"]["T_seed1"], t["A"]["T_seed2"]]))        # seeds share the starts
        assert t["A"]["outcome"] in ("drift", "instability") and 0 <= t["C"]["p_holm"] <= 1
        sv = np.array([r["S"] for r in res["survival"]])
        assert sv[0] == 1.0 and (np.diff(sv) <= 1e-12).all()                              # survival never increases


# =====================================================================================================================
# from test_stage13.py
# =====================================================================================================================


CPU = torch.device("cpu")


def test_state_update_rebuilds_the_next_condition():
    """Part 2 hook: with the realised return and log-vol change as the 'simulated' step, the next condition is real."""
    for window in ("paper", "code"):
        P = prepare_all("SPX", Config(rate_source="zero", rv_window=window))
        ds = P.inputs.val
        state = initial_state(P.data, ds, 10)
        assert np.allclose(state.condition, ds.condition[10])
        for j in range(10, 15):                                        # five steps along the realised path
            assert np.allclose(state.condition, ds.condition[j], rtol=0, atol=1e-12)
            state = next_state(state, ds.target[j, 0], ds.target[j, 1:])
    set_seed(0)
    K = P.data.lk * P.data.lt
    gen = build_networks(N_SCALARS + K, 1 + K, Config(), CPU)[0]
    r, dg = step(gen, torch.zeros(4, N_SCALARS + K), torch.zeros(4, 32))
    assert r.shape == (4,) and dg.shape == (4, K)


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
        test_batched_next_state,
        test_calendar_noise,
        test_benchmark_uses_the_same_noise_rows,
        test_h1_identity_and_recursion_departs,
        test_validity_is_absorbing,
        test_guards_and_starts,
        test_metrics_closed_form,
        test_bootstrap_blocks,
        test_pvalue_degenerate_and_holm,
        test_powerlaw,
        test_resume_and_end_to_end,
        test_state_update_rebuilds_the_next_condition,
    ], "Recursion (Part 2, Part A)")
