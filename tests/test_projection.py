"""
Part 2, Part B, second remedy: the in-loop projection onto the arbitrage-free set (projection brief, section 5).

    python tests/test_projection.py

  1. L built from penalty_terms reproduces the terms on random price surfaces (1e-13);
  2. 1,000 perturbed Black-Scholes surfaces with injected violations: Phi is exactly 0 after the round trip
     price -> volatility -> price, and the inversion error is below 1e-12 in price;
  3. a surface with Phi = 0 is returned bit for bit;
  4. rollout with repair=None equals the Part A rollout; bounded_proj with nothing projected equals bounded;
  5. step 1 is never projected (the h = 1 identity);
  6. determinism: two runs of the same start give identical arrays;
  7. sparsity (report only): how many nodes change for a surface with a single butterfly violation.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_recursion import M, TAU, K, fake_run
from test_remedies import _outs, _parta_rollout, _same
from volgan_fixed.common import RecursionConfig, arbitrage_penalty, penalty_terms, relative_call
from volgan_fixed.projection import Projector, stacked_terms, to_mat, to_vec
from volgan_fixed.recursion import Arena, rollout, run_start

PR = Projector(M, TAU)
MS, TS = np.meshgrid(M, TAU, indexing="ij")


def bs_surfaces(n, rng):
    """Smooth smiles: 18% at the money, skewed, a mild term structure, one rate curve per surface."""
    a = 0.14 + 0.08 * rng.random((n, 1, 1))
    skew = 0.1 + 0.3 * rng.random((n, 1, 1))
    sig = a + skew * (1 - MS[None]) ** 2 + 0.2 * skew * (1 - MS[None]) + 0.02 * np.sqrt(TS[None])
    return sig


def test_L_from_penalty_terms():
    rng = np.random.default_rng(0)
    C = rng.random((1000, len(M), len(TAU)))
    assert PR.L.shape == (101, 42)
    assert np.abs(C.reshape(1000, -1) @ PR.L.T - stacked_terms(C, M, TAU)).max() <= 1e-13


def test_projection_round_trip():
    rng = np.random.default_rng(1)
    n_done, oor = 0, 0
    for r0 in np.linspace(-0.0017, 0.05, 10):
        rate = r0 + 0.002 * rng.random(len(TAU))
        sig = bs_surfaces(400, rng)
        bump = np.zeros_like(sig)
        for b in range(len(sig)):                                       # bumps of up to 5 vol points at random nodes
            k = rng.integers(1, 6)
            bump[b].flat[rng.choice(sig[b].size, k, replace=False)] = rng.uniform(-0.05, 0.05, k)
        sig = np.clip(sig + bump, 0.02, None)
        C = relative_call(MS, TS, sig, np.repeat(rate[None], len(sig), 0))
        bad = arbitrage_penalty(C, M, TAU).total > 0
        sig = sig[bad][:100]
        vec = to_vec(sig)
        out, info = PR(vec, np.ones(len(vec), bool), rate)
        assert info["projected"].all() and info["retries"] == 0
        oor += int(info["oor"].sum())
        ok = ~info["oor"]
        Crt = relative_call(MS, TS, to_mat(out[ok], len(M), len(TAU)), np.repeat(rate[None], ok.sum(), 0))
        assert (stacked_terms(Crt, M, TAU) <= 0).all() and (arbitrage_penalty(Crt, M, TAU).total == 0).all()
        assert info["phi_post_max"] == 0.0 and info["inv_err_max"] < 1e-12
        n_done += len(vec)
    assert n_done == 1000 and oor == 0, (n_done, oor)


def test_clean_surface_unchanged():
    rng = np.random.default_rng(2)
    rate = np.full(len(TAU), 0.02)
    sig = bs_surfaces(50, rng)
    C = relative_call(MS, TS, sig, np.repeat(rate[None], 50, 0))
    clean = arbitrage_penalty(C, M, TAU).total == 0
    assert clean.sum() > 10
    vec = to_vec(sig[clean])
    out, info = PR(vec, np.ones(len(vec), bool), rate)
    assert np.array_equal(out, vec) and not info["projected"].any() and not info["changed"].any()
    dead = vec.copy()
    dead[0, 3] = np.nan                                                 # non-finite: not projected, returned as is
    out, info = PR(dead, np.ones(len(dead), bool), rate)
    assert np.array_equal(out, dead, equal_nan=True) and not info["projected"][0]


def _arena(repair, clip=(np.full(K, -0.05), np.full(K, 0.05))):
    return Arena(fake_run(), RecursionConfig(), clip=clip, repair=repair)


def test_default_and_switched_off():
    a = _arena(None, clip=None)
    _same(_outs(rollout, a.gen, a, repair=None), _outs(_parta_rollout, a.gen, a))
    clip = (np.full(K, -0.05), np.full(K, 0.05))
    off = lambda sigma, alive, rate: PR(sigma, np.zeros_like(alive), rate)          # projection switched off
    _same(_outs(rollout, a.gen, a, clip=clip, repair=off), _outs(rollout, a.gen, a, clip=clip))


def test_step1_never_projected():
    a = _arena(None, clip=None)
    calls = []

    def spy(sigma, alive, rate):
        calls.append(1)
        return PR(sigma, alive, rate)
    outs = _outs(rollout, a.gen, a, repair=spy)
    assert len(calls) == 20 and outs[0].repair_info is None
    _same(outs[:1], _outs(rollout, a.gen, a)[:1])
    assert sum(o.repair_info["projected"].sum() for o in outs[1:]) > 0      # the projection does act at k >= 2


def test_deterministic():
    a = _arena(PR)
    r1, r2 = run_start(a, 3), run_start(a, 3)
    for k in r1:
        if k != "seconds":
            assert np.array_equal(r1[k], r2[k], equal_nan=True), k
    assert r1["proj_share"][1:].sum() > 0 and (r1["proj_share"][0] == 0) and (r1["phi_post_max"] == 0).all()


def test_sparsity_single_butterfly():
    rate = np.full(len(TAU), 0.02)
    sig = bs_surfaces(1, np.random.default_rng(3))
    i, j = 3, 2                                                         # m = 1.00, 90 days
    for bump in np.linspace(0.005, 0.1, 40):
        s = sig.copy()
        s[0, i, j] += bump
        C = relative_call(MS, TS, s, rate[None])
        cal, slope, fly = penalty_terms(C, M, TAU)
        if (fly > 0).sum() == 1 and (cal > 0).sum() == 0 and (slope > 0).sum() == 0:
            break
    else:
        raise AssertionError("no single-butterfly surface found")
    out, info = PR(to_vec(s), np.ones(1, bool), rate)
    print("sparsity: a single butterfly violation (bump %.3f at m = 1.00, 90 d) changes %d node(s), total %.4f vol points"
          % (bump, info["nodes"][0], 100 * info["dvol_sum"][0]))
    assert info["nodes"][0] >= 1 and info["phi_post_max"] == 0.0


if __name__ == "__main__":
    for fn in (test_L_from_penalty_terms, test_projection_round_trip, test_clean_surface_unchanged,
               test_default_and_switched_off, test_step1_never_projected, test_deterministic,
               test_sparsity_single_butterfly):
        fn()
    print("Projection (Part 2, Part B, second remedy) checks passed")
