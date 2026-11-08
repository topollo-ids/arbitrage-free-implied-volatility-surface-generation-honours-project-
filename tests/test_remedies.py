"""
Part 2, Part B checks (PART2_PartB_BUILD_BRIEF section 4.3), on synthetic data (no workbook needed):

    python tests/test_remedies.py

  1. rollout with gen_late=None, clip=None gives the same arrays as the Part A rollout (a frozen copy below);
  2. clip = (-inf, +inf) gives the same arrays as no clip;
  3. gen_late = gen gives the same arrays as gen_late = None;
  4. step 1 is never clipped and never uses gen_late;
  5. clipped increments lie in [lo, hi] at k >= 2, and sigma and the state update use the clipped values;
  6. pool targets rebuild the real surface (cond_syn[3:] + target_syn[1:] == g_real(j)), the return slot is ds.target[j, 0];
  7. the pool uses only training-block dates, and POOL_SEED is not the simulation seed;
  8. the noise passed at step k is the same calendar-indexed block for G0 and G1;
  plus a smoke test of one fine-tuning round (two epochs) from a saved checkpoint.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_recursion import K, arena, generator, synthetic          # the Part A synthetic fixtures
from volgan_fixed import remedies as R
from volgan_fixed.common import Config, RecursionConfig, SIM_SEED, save_checkpoint
from volgan_fixed.preprocessing import N_SCALARS, WINDOW, Split
from volgan_fixed.postprocessing import step
from volgan_fixed.recursion import NoiseTable, StepOut, batched_next_state, initial_state, rollout
from volgan_fixed.training import build_networks, make_optimisers


# ---- the Part A rollout's arithmetic, frozen as it was before Part B (its two input checks are left out)

@torch.no_grad()
def _parta_rollout(gen, ds, data, j0, noise, cfg, post_step=None, ctx=None):
    P, H = cfg.n_paths, cfg.horizon
    st = initial_state(data, ds, j0)
    device = next(gen.parameters()).device
    gen.eval()
    cond = np.tile(st.condition, (P, 1))
    buf = np.tile(st.returns, (P, 1))
    valid = np.ones(P, dtype=bool)
    prev = None
    for k in range(1, H + 1):
        j = j0 + k - 1
        extra = None
        if k >= 2 and post_step is not None:
            cond, extra = post_step(k, cond, prev[0], prev[1], valid, ctx)
        z = noise.get(ds.dates[j]).to(device)
        r_scaled, dg = step(gen, torch.as_tensor(cond, dtype=torch.float32, device=device), z)
        r_scaled = r_scaled.double().cpu().numpy()
        dg = dg.double().cpu().numpy()
        r = r_scaled / ds.ret_scale
        with np.errstate(over="ignore", invalid="ignore"):
            sigma = np.exp(cond[:, N_SCALARS:] + dg)
            ok = (np.isfinite(r) & np.isfinite(sigma).all(axis=1)
                  & (sigma >= cfg.vol_lo).all(axis=1) & (sigma <= cfg.vol_hi).all(axis=1))
        valid = valid & ok
        yield StepOut(k, j, r, dg, sigma, valid.copy(), extra)
        with np.errstate(over="ignore", invalid="ignore"):
            new, new_buf = batched_next_state(cond, buf, r_scaled, dg, ds.ret_scale, ds.rv_window)
        cond = np.where(valid[:, None], new, cond)
        buf = np.where(valid[:, None], new_buf, buf)
        prev = (r_scaled, dg)


def _outs(fn, gen, a, j0=3, **kw):
    return list(fn(gen, a.ds, a.run.data, j0, a.noise, a.cfg, **kw))


def _same(xs, ys):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert x.k == y.k and x.j == y.j
        for f in ("r", "dg", "sigma", "valid"):
            assert np.array_equal(getattr(x, f), getattr(y, f), equal_nan=f != "valid"), (x.k, f)


class Zero(nn.Module):
    """A network that returns zeros (return 0, log-vol increment 0)."""
    noise_dim = 32

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, noise, cond):
        return torch.zeros(noise.shape[0], 1 + K)


class Spy(nn.Module):
    """Wraps a generator and records the noise block of every call."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.noise_dim, self.seen = inner, 32, []

    def forward(self, noise, cond):
        self.seen.append(noise.clone())
        return self.inner(noise, cond)


def test_default_is_part_a():
    a = arena()
    old = _outs(_parta_rollout, a.gen, a)
    _same(_outs(rollout, a.gen, a), old)
    _same(_outs(rollout, a.gen, a, gen_late=None, clip=None), old)
    assert all(o.clip_bind is None for o in _outs(rollout, a.gen, a))


def test_infinite_clip_is_no_clip():
    a = arena()
    inf = (np.full(K, -np.inf), np.full(K, np.inf))
    outs = _outs(rollout, a.gen, a, clip=inf)
    _same(outs, _outs(rollout, a.gen, a))
    assert outs[0].clip_bind is None and all(not o.clip_bind.any() for o in outs[1:])


def test_gen_late_equal_gen():
    a = arena()
    _same(_outs(rollout, a.gen, a, gen_late=a.gen), _outs(rollout, a.gen, a))


def test_step1_untouched():
    a = arena()
    tight = (np.full(K, -1e-9), np.full(K, 1e-9))
    base = _outs(rollout, a.gen, a)
    outs = _outs(rollout, a.gen, a, gen_late=Zero(), clip=tight)
    _same(outs[:1], base[:1])                                       # h = 1: G0, unclipped
    assert outs[0].clip_bind is None
    assert np.abs(base[0].dg).max() > 1e-6                          # the tight clip would have bound at h = 1
    assert all((o.dg == 0).all() and (o.r == 0).all() for o in outs[1:])   # k >= 2: the zero network


def test_clip_values_used():
    a = arena()
    lo, hi = np.full(K, -2e-3), np.full(K, 2e-3)
    seen = []

    def capture(k, cond, r_hat, dg_hat, valid, ctx):
        seen.append((k, cond.copy(), dg_hat.copy()))
        return cond, None
    outs = _outs(rollout, a.gen, a, clip=(lo, hi), post_step=capture)
    assert sum(o.clip_bind.sum() for o in outs[1:]) > 0              # the clip binds somewhere
    for o in outs[1:]:
        assert (o.dg >= lo).all() and (o.dg <= hi).all()
    for (k, cond, dg_prev), o_prev, o in zip(seen, outs[:-1], outs[1:]):
        assert o.k == k
        assert np.array_equal(dg_prev, o_prev.dg)                   # the state update was fed the clipped dg
        if k >= 3:
            g_prev = seen[k - 3][1][:, N_SCALARS:]
            assert np.allclose(cond[:, N_SCALARS:], g_prev + o_prev.dg, rtol=0, atol=1e-14)
        assert np.array_equal(o.sigma, np.exp(cond[:, N_SCALARS:] + o.dg))   # sigma uses the clipped dg


def _train_run(n_days=150):
    """A synthetic run whose training block has 60 samples (the pool needs j0 + 20 <= n - 1)."""
    data, ds, _ = synthetic(n_days)
    n = len(ds.dates)
    split = Split(train=ds.slice(0, 60), val=ds.slice(60, n - 20), test=ds.slice(n - 20, n))
    gen = generator(0)
    return SimpleNamespace(gen=gen, config=Config(), split=split, market=split, data=data,
                           info={"signature": "synthetic", "alpha_m": 1e-5, "alpha_tau": 2e-5, "index": "SPX"})


def test_pool_targets():
    run = _train_run()
    out, summ = R.build_pool(run, log=lambda m: None, every=0)
    ds = run.split.train
    g_real = ds.past_log_surface + ds.target[:, 1:]
    n_starts = len(ds.dates) - 20
    assert summ["n_pairs"] == n_starts * 20 * R.RC.pool_paths and summ["dropped_total"] == 0
    assert set(np.unique(out["k"])) == set(range(2, 22))
    err = np.abs(out["cond"][:, N_SCALARS:] + out["target"][:, 1:] - g_real[out["j"]]).max()
    assert err <= R.POOL_TOL and summ["identity_max_error"] == err
    assert np.array_equal(out["target"][:, 0], ds.target[out["j"], 0])
    assert np.array_equal(out["j"], out["j0"] + out["k"] - 1)
    # the states are the rollout's own: k = 2 is built from one simulated day of the observed start
    st = initial_state(run.data, ds, 0)
    first = out["cond"][(out["j0"] == 0) & (out["k"] == 2)]
    assert np.allclose(first[:, 1], st.condition[0]) and not np.allclose(first[:, N_SCALARS:], st.condition[N_SCALARS:])
    clipped, _ = R.build_pool(run, clip=(np.full(K, -1e-3), np.full(K, 1e-3)), log=lambda m: None, every=0)
    assert not np.array_equal(clipped["cond"], out["cond"])          # the bounded variant rolls with the clip


def test_pool_dates_and_seed():
    run = _train_run()
    out, _ = R.build_pool(run, log=lambda m: None, every=0)
    train = run.split.train
    assert out["j"].max() <= len(train.dates) - 1 and out["j0"].max() + 20 <= len(train.dates) - 1
    assert set(train.dates[out["j"]]) <= set(train.dates) and train.dates[out["j"]].max() < run.split.val.dates[0]
    assert R.RC.pool_seed != SIM_SEED and R.pool_cfg().sim_seed == R.RC.pool_seed
    same, _ = R.build_pool(run, noise=NoiseTable(R.RC.pool_seed, R.RC.pool_paths), log=lambda m: None, every=0)
    other, _ = R.build_pool(run, noise=NoiseTable(SIM_SEED, R.RC.pool_paths), log=lambda m: None, every=0)
    assert np.array_equal(same["cond"], out["cond"]) and not np.array_equal(other["cond"], out["cond"])


def test_same_noise_for_g0_and_g1():
    a = arena()
    g0, g1, alone = Spy(a.gen), Spy(generator(7)), Spy(a.gen)
    outs = _outs(rollout, g0, a, gen_late=g1)
    _outs(rollout, alone, a)
    assert len(g0.seen) == 1 and len(g1.seen) == 20 and len(alone.seen) == 21
    for k, o in enumerate(outs, start=1):
        z = (g0.seen + g1.seen)[k - 1]
        assert torch.equal(z, a.noise.get(a.ds.dates[o.j])) and torch.equal(z, alone.seen[k - 1])


def test_finetune_smoke():
    run = _train_run()
    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "net")
        os.makedirs(folder)
        gen, disc = build_networks(N_SCALARS + K, 1 + K, run.config, torch.device("cpu"))
        gen.load_state_dict(run.gen.state_dict())
        g_opt, d_opt = make_optimisers(gen, disc, run.config)
        save_checkpoint(os.path.join(folder, "2_trained.pt"), gen, disc, g_opt, d_opt, 1e-5, 2e-5, 1, run.config.as_dict())
        out, _ = R.build_pool(run, log=lambda m: None, every=0)
        pf = os.path.join(root, "pool.npz")
        np.savez(pf, cond=out["cond"].astype(np.float32), target=out["target"].astype(np.float32))
        res = R.finetune(run, folder, pf, "dad", 1, os.path.join(root, "ft"), torch.device("cpu"), log=lambda m: None,
                         epochs=2)
        assert os.path.exists(os.path.join(root, "ft", "ft_trained.pt")) and res["n_real"] == 60
        assert all(np.isfinite(v) for v in res["final"].values()) and res["ft_seed"] == 30001
        ck = torch.load(os.path.join(root, "ft", "ft_trained.pt"), weights_only=False)
        moved = any(not torch.equal(ck["gen"][k], gen.state_dict()[k]) for k in ck["gen"])
        assert moved and ck["alpha"] == 1e-5 and ck["beta"] == 2e-5                  # trained further, same alphas


if __name__ == "__main__":
    for fn in (test_default_is_part_a, test_infinite_clip_is_no_clip, test_gen_late_equal_gen, test_step1_untouched,
               test_clip_values_used, test_pool_targets, test_pool_dates_and_seed, test_same_noise_for_g0_and_g1,
               test_finetune_smoke):
        fn()
    print("Remedies (Part 2, Part B) checks passed")
