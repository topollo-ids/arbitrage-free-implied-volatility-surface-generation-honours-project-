"""
Training checks, merged by topic from the stage tests (every assertion kept). Run on its own:

    python tests/test_training.py

--- from test_stage0.py ---
Stage 0 checks: same seed -> same numbers; a checkpoint restores a network exactly.

--- from test_stage5.py ---
Stage 5 checks: the networks have the paper's architecture and the expected number of parameters,
and are the SAME networks as the original code's (same weights from the same seed, same outputs).

--- from test_stage6.py ---
Stage 6 checks: the smoothness penalty is the paper's (eqs 11-12), and the released one is reproduced.
  - a hand-computed example, and eqs 11-12 written out as plain loops, on our grid and the paper's,
  - direction: each term sees only its own axis (this is what the released code got wrong),
  - the released version equals the ORIGINAL code's numbers,
  - the tracker's diagnosis (F7 corrected): released = paper penalty on the transposed surface + an lk/lt slip.

--- from test_stage7.py ---
Stage 7 checks: the training loop follows section 3.4 and eqs 13-14, and its released mode is the released code.
  - eq 14 and eq 13 assembled exactly as written (one step compared with a hand-built step),
  - gradient norms, alphas = mean of the norm ratios, restart from the initialisation, no empty batch,
  - released mode reproduces the ORIGINAL GradientMatching + TrainLoopNoVal weight for weight.

--- from test_stage7_phi.py ---
Stage 7 extension checks (Part 1, lever 2: Phi in the generator loss).
  - the torch Phi equals the NumPy Phi (the measuring instrument) to 1e-10 on 1,000 random surfaces with random
    rates, per surface and per maturity, and torch.autograd.gradcheck passes in float64,
  - REGRESSION GUARD: with the control switches, fit() on fixed tensors reproduces, bit for bit, the weights the
    pre-Part-1 fit() produced (tests/fixtures/regression_fit_before.pt, saved before any change),
  - the rates stay aligned with their samples through the per-epoch permutation in train(),
  - the alpha_phi rule (mean of ratios over non-zero Phi gradients; ratio of means below 50%) and its record.
Brief section 12, tests 3, 4 and 5.
"""

import contextlib
from dataclasses import replace
import functools
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import original_reference as orig
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from original_reference import original_smoothness
from volgan_fixed.common import (
    CELLS, Config, INDEX_FILES, RUNS_DIR, TRAINING_CONFIGS, arbitrage_penalty, arbitrage_penalty_torch,
    cell_levers, get_device, load_checkpoint, relative_call, relative_call_torch, save_checkpoint, set_seed,
    training_config)
from volgan_fixed.preprocessing import N_SCALARS, data_preprocessing, load_surfaces, split_by_year, to_tensors
from volgan_fixed.training import (
    Discriminator, Generator, PaperSmoothness, ReleasedSmoothness, TorchPhi, batch_slices, build_networks,
    discriminator_step, fit, generated_log_surface, generator_step, grad_norm, load_run, make_optimisers,
    make_smoothness, match_alpha_phi, n_params, penalised_surface, released_m_block, released_t_block, signature,
    train, train_run)


# =====================================================================================================================
# from test_stage0.py
# =====================================================================================================================


def test_seed_repeats():
    set_seed(123)
    a = (torch.randn(5), np.random.rand(5))
    set_seed(123)
    b = (torch.randn(5), np.random.rand(5))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    set_seed(124)
    c = torch.randn(5)
    assert not torch.equal(a[0], c)


def test_checkpoint_round_trip():
    device = get_device()
    set_seed(0)
    gen, disc = nn.Linear(4, 3), nn.Linear(3, 1)
    go = torch.optim.RMSprop(gen.parameters(), lr=1e-4)
    do = torch.optim.RMSprop(disc.parameters(), lr=1e-4)
    path = os.path.join(tempfile.mkdtemp(), "ck.pt")
    save_checkpoint(path, gen, disc, go, do, alpha=2.5, beta=3.5, seed=0, config={"H": 16})

    set_seed(1)  # fresh networks with different weights
    gen2, disc2 = nn.Linear(4, 3), nn.Linear(3, 1)
    go2 = torch.optim.RMSprop(gen2.parameters(), lr=1e-4)
    do2 = torch.optim.RMSprop(disc2.parameters(), lr=1e-4)
    assert not torch.equal(gen.weight, gen2.weight)
    alpha, beta, seed, config = load_checkpoint(path, gen2, disc2, go2, do2, device)
    assert torch.equal(gen.weight, gen2.weight) and torch.equal(disc.weight, disc2.weight)
    assert (alpha, beta, seed, config) == (2.5, 3.5, 0, {"H": 16})


# =====================================================================================================================
# from test_stage1.py
# =====================================================================================================================


SPX, NDX = INDEX_FILES["SPX"], INDEX_FILES["NDX"]


# =====================================================================================================================
# from test_stage5.py
# =====================================================================================================================


def expected_params(noise, cond, H, out):
    """Count by hand: weights + biases of each dense layer."""
    gen = (noise + cond) * H + H + H * 2 * H + 2 * H + 2 * H * out + out
    disc = (cond + out) * H + H + H * 1 + 1
    return gen, disc


def test_paper_defaults():
    c = Config()
    assert (c.noise_dim, c.hidden_dim, c.n_epochs, c.n_grad) == (32, 16, 10_000, 25)
    assert (c.lr_g, c.lr_d, c.batch_size, c.n_scenarios) == (1e-4, 1e-4, 100, 10_000)
    assert c.rv_window == "paper" and np.isclose(c.ret_scale, np.sqrt(252)) and c.train_fraction == 0.8
    assert c.as_dict()["noise_dim"] == 32


def test_parameter_counts():
    c = Config()
    # our 42-point grid: condition 3 + 42, outcome 1 + 42
    gen, disc = build_networks(45, 43, c, torch.device("cpu"))
    assert (n_params(gen), n_params(disc)) == (3211, 1441) == expected_params(32, 45, 16, 43)
    # the original 80-point grid: Book 1 Figure 3 quotes 5,073 and 2,657
    gen, disc = build_networks(83, 81, c, torch.device("cpu"))
    assert (n_params(gen), n_params(disc)) == (5073, 2657) == expected_params(32, 83, 16, 81)
    # any other sizes follow the same formula
    for noise, cond, H, out in [(8, 10, 4, 7), (16, 52, 8, 50)]:
        g, d = Generator(noise, cond, H, out), Discriminator(cond + out, H)
        assert (n_params(g), n_params(d)) == expected_params(noise, cond, H, out)


def test_layer_shapes():
    gen, disc = build_networks(45, 43, Config(), torch.device("cpu"))
    assert [tuple(l.weight.shape) for l in (gen.linear1, gen.linear2, gen.linear3)] == [(16, 77), (32, 16), (43, 32)]
    assert [tuple(l.weight.shape) for l in (disc.linear1, disc.linear2)] == [(16, 88), (1, 16)]


def test_identical_to_original_networks():
    """Same seed -> same weights in the same places -> same outputs as the released classes."""
    for cond, out in [(45, 43), (83, 81)]:
        set_seed(7)
        g_orig, d_orig = orig.Generator(32, cond, 16, out), orig.Discriminator(cond + out, 16)
        set_seed(7)
        g_new, d_new = Generator(32, cond, 16, out), Discriminator(cond + out, 16)
        for a, b in [(g_orig, g_new), (d_orig, d_new)]:
            sa, sb = a.state_dict(), b.state_dict()
            assert list(sa) == list(sb)                                  # same names, same order
            assert all(torch.equal(sa[k], sb[k]) for k in sa)
        noise, condition = torch.randn(64, 32), torch.randn(64, cond)
        assert torch.equal(g_orig(noise, condition), g_new(noise, condition))
        pair = torch.randn(64, cond + out)
        assert torch.equal(d_orig(pair), d_new(pair))
        # a state dict saved from the original loads straight into ours (checkpoint compatibility)
        Generator(32, cond, 16, out).load_state_dict(g_orig.state_dict())


def test_outputs_and_gradients():
    gen, disc = build_networks(45, 43, Config(), torch.device("cpu"))
    noise, condition = torch.randn(100, 32), torch.randn(100, 45)
    fake = gen(noise, condition)
    assert fake.shape == (100, 43)
    p = disc(torch.cat([condition, fake], dim=-1))
    assert p.shape == (100, 1) and (p > 0).all() and (p < 1).all()
    # the generator's loss reaches every generator weight (no dead layer)
    torch.nn.BCELoss()(p, torch.ones_like(p)).backward()
    assert all(w.grad is not None and w.grad.abs().sum() > 0 for w in gen.parameters())


# =====================================================================================================================
# from test_stage6.py
# =====================================================================================================================


CPU = torch.device("cpu")

OUR_M = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

OUR_TAU = np.array([30, 60, 90, 180, 270, 360]) / 365

PAPER_M = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4])

PAPER_TAU = np.array([1 / 252, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1.0])

ORIG_M = np.linspace(0.6, 1.4, 10)

ORIG_TAU = np.array([30, 60, 91, 122, 152, 182, 273, 365]) / 365


def eq11_12_loops(v, m, tau):
    """The paper's formulas written out literally. g(i, j) = g(m_i, tau_j) = v[j*lk + i]."""
    lk, lt = len(m), len(tau)
    g = lambda i, j: v[j * lk + i]
    L_m = sum((g(i + 1, j) - g(i, j)) ** 2 / (m[i + 1] - m[i]) ** 2 for j in range(lt) for i in range(lk - 1))
    L_tau = sum((g(i, j + 1) - g(i, j)) ** 2 / (tau[j + 1] - tau[j]) ** 2 for i in range(lk) for j in range(lt - 1))
    return L_m, L_tau


def test_hand_computed():
    # m = 0.9, 1.0, 1.2 (uneven); tau = 0.1, 0.3. Surface (rows tau, columns m): [0, 0.1, 0.3] and [0.2, 0.2, 0.2]
    # L_m   = (0.1/0.1)^2 + (0.2/0.2)^2 + 0 = 2
    # L_tau = (0.2/0.2)^2 + (0.1/0.2)^2 + (-0.1/0.2)^2 = 1.5
    pen = PaperSmoothness([0.9, 1.0, 1.2], [0.1, 0.3], CPU)
    L_m, L_tau = pen(torch.tensor([[0.0, 0.1, 0.3, 0.2, 0.2, 0.2]], dtype=torch.float64))
    assert torch.allclose(L_m, torch.tensor([2.0], dtype=torch.float64))
    assert torch.allclose(L_tau, torch.tensor([1.5], dtype=torch.float64))


def test_matches_the_formulas_as_loops():
    rng = np.random.default_rng(0)
    for m, tau in [(OUR_M, OUR_TAU), (PAPER_M, PAPER_TAU), (ORIG_M, ORIG_TAU)]:
        V = rng.normal(size=(5, len(m) * len(tau)))
        L_m, L_tau = PaperSmoothness(m, tau, CPU)(torch.tensor(V))
        for b in range(5):
            Lm_ref, Lt_ref = eq11_12_loops(V[b], m, tau)
            assert np.isclose(L_m[b].item(), Lm_ref, rtol=1e-12) and np.isclose(L_tau[b].item(), Lt_ref, rtol=1e-12)


def test_each_term_sees_only_its_own_axis():
    """g = a*m + b*tau exactly: L_m = a^2 (lk-1) lt and L_tau = b^2 (lt-1) lk on ANY grid, even an uneven one."""
    for m, tau in [(OUR_M, OUR_TAU), (PAPER_M, PAPER_TAU)]:
        lk, lt = len(m), len(tau)
        pen = PaperSmoothness(m, tau, CPU)
        T, M = np.meshgrid(tau, m, indexing="ij")                    # (lt, lk), maturity-major
        for a, b in [(1.0, 0.0), (0.0, 1.0), (-2.0, 3.0), (0.0, 0.0)]:
            v = torch.tensor((a * M + b * T).reshape(1, -1))
            L_m, L_tau = pen(v)
            assert np.isclose(L_m.item(), a ** 2 * (lk - 1) * lt, rtol=1e-10, atol=1e-10)
            assert np.isclose(L_tau.item(), b ** 2 * (lt - 1) * lk, rtol=1e-10, atol=1e-10)


def test_released_equals_original_code():
    rng = np.random.default_rng(1)
    for m, tau in [(OUR_M, OUR_TAU), (ORIG_M, ORIG_TAU)]:
        v = torch.tensor(rng.normal(size=(50, len(m) * len(tau))), dtype=torch.float)
        ours_m, ours_t = ReleasedSmoothness(m, tau, CPU)(v)
        orig_m, orig_t = original_smoothness(v, m, tau, len(m), len(tau))
        assert np.isclose(ours_m.mean().item(), orig_m.item(), rtol=1e-5)
        assert np.isclose(ours_t.mean().item(), orig_t.item(), rtol=1e-5)


def test_released_is_paper_on_the_transposed_surface():
    """Tracker F7 (corrected): one transposition error plus an lk/lt slip in the m-block."""
    rng = np.random.default_rng(2)
    for m, tau in [(OUR_M, OUR_TAU), (ORIG_M, ORIG_TAU)]:
        lk, lt = len(m), len(tau)
        G = rng.normal(size=(lt, lk))                                 # G[j, i] = g(m_i, tau_j)
        v = torch.tensor(G.reshape(1, -1), dtype=torch.float)         # our layout: moneyness fastest
        vT = torch.tensor(G.T.reshape(1, -1), dtype=torch.float)      # transposed: maturity fastest
        paper_m, paper_tau = eq11_12_loops(G.reshape(-1), m, tau)
        matrix_t, tsq = released_t_block(tau, lk, lt, CPU)
        assert np.isclose((((vT @ matrix_t.T) ** 2) @ tsq).item(), paper_tau, rtol=1e-5)
        matrix_m, m_seq = released_m_block(m, lt, lk, CPU)           # lk and lt swapped
        assert np.isclose((((vT @ matrix_m.T) ** 2) @ m_seq).item(), paper_m, rtol=1e-5)
        # ...and on the layout the data actually uses, the released terms look along the wrong axes:
        T, M = np.meshgrid(tau, m, indexing="ij")
        smile = torch.tensor(np.log(0.2 + (M - 1) ** 2).reshape(1, -1), dtype=torch.float)   # varies with m only
        term = torch.tensor(np.log(0.15 + 0.05 * T).reshape(1, -1), dtype=torch.float)       # varies with tau only
        paper, released = PaperSmoothness(m, tau, CPU), ReleasedSmoothness(m, tau, CPU)
        assert paper(smile)[1].item() == 0 and paper(term)[0].item() == 0          # paper: each term sees its own axis
        rel_m, rel_tau = released(smile)
        assert rel_m.item() == 0 and rel_tau.item() > 0          # released "m" term blind to a smile; "tau" term sees it
        rel_m, rel_tau = released(term)
        assert rel_m.item() > 0                                  # released "m" term reacts to term structure
        seam_only = sum(tsq[k] * (term[0, k + 1] - term[0, k]) ** 2 for k in range(lk - 1, lk * lt - 1, lk))
        assert np.isclose(rel_tau.item(), float(seam_only), rtol=1e-5) and rel_tau.item() > 0   # all of it from seams


def test_gradients_dtype_and_switches():
    pen = PaperSmoothness(OUR_M, OUR_TAU, CPU)
    v = torch.randn(3, 42, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: sum(t.sum() for t in pen(x)), (v,))
    assert pen(torch.randn(3, 42)).__getitem__(0).dtype == torch.float32
    assert isinstance(make_smoothness("paper", OUR_M, OUR_TAU, CPU), PaperSmoothness)
    assert isinstance(make_smoothness("released", OUR_M, OUR_TAU, CPU), ReleasedSmoothness)
    # the penalised object: g_hat = g_t + dg_hat (F8: "g" in the control, exp(g) only in the ablation)
    fake, past = torch.randn(4, 43), torch.randn(4, 42)
    assert torch.equal(generated_log_surface(fake, past), past + fake[:, 1:])
    assert torch.equal(penalised_surface(fake, past, "g"), past + fake[:, 1:])
    assert torch.allclose(penalised_surface(fake, past, "sigma"), torch.exp(past + fake[:, 1:]))
    for bad in [lambda: make_smoothness("x", OUR_M, OUR_TAU, CPU), lambda: penalised_surface(fake, past, "x")]:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("bad option accepted")


# =====================================================================================================================
# from test_stage13.py
# =====================================================================================================================


def test_old_run_folder_still_loads():
    old = os.path.join(HERE, "..", "old_unused", "early_trial_runs", "spx_control_quick")
    if not os.path.exists(os.path.join(old, "run.json")):
        print("   (no old_unused/early_trial_runs/spx_control_quick here: old-run check skipped)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("run.json", "2_trained.pt"):
            shutil.copy(os.path.join(old, f), tmp)
        with open(os.path.join(tmp, "run.json")) as f:
            info = json.load(f)
        assert "train_end" not in info["config"]                       # trained before Part 1
        if not os.path.exists(os.environ.get("VOLGAN_RATES_FILE", "")) and "rate_source" not in info["config"]:
            info["config"]["rate_source"] = "zero"                     # rates do not matter for this check
            with open(os.path.join(tmp, "run.json"), "w") as f:
                json.dump(info, f)
        run = load_run(tmp, CPU)
        assert run.split.test_start_year == 2022 and run.split.val is None and len(run.split.test.dates) == info["n_test"]
        ck = torch.load(os.path.join(old, "2_trained.pt"), weights_only=False)
        assert all(torch.equal(p, ck["gen"][k]) for k, p in run.gen.state_dict().items())


# =====================================================================================================================
# from test_stage7.py
# =====================================================================================================================

RELEASED = dict(smoothness="released", match_on="sigma", restart=False, gen_bce_weight=1.0)


@functools.lru_cache(maxsize=1)
def spx_train():
    d = load_surfaces(SPX, "SPX")
    cond, targ = to_tensors(split_by_year(data_preprocessing(d), 0.8).train, CPU)
    return cond, targ, d.m, d.tau


def small(n):
    cond, targ, m, tau = spx_train()
    return cond[:n], targ[:n], m, tau


def quiet(fn, *args, **kwargs):
    """The original prints progress bars; keep the test output clean."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def run_original(cond, targ, m, tau, cfg):
    set_seed(cfg.seed)
    gen, disc = build_networks(cond.shape[1], targ.shape[1], cfg, CPU)          # Stage 5: same as the original's
    g_opt, d_opt = make_optimisers(gen, disc, cfg)
    gen, g_opt, disc, d_opt, crit, a, b = quiet(orig.GradientMatching, gen, g_opt, disc, d_opt, nn.BCELoss(),
                                                cond, targ, m, tau, None, None, cfg.n_grad, cfg.lr_g, cfg.lr_d,
                                                cfg.batch_size, cfg.noise_dim, "cpu", lk=len(m), lt=len(tau))
    gen, g_opt, disc, d_opt, crit = quiet(orig.TrainLoopNoVal, a, b, gen, g_opt, disc, d_opt, crit, cond, targ,
                                          m, tau, None, None, cfg.n_epochs, cfg.lr_g, cfg.lr_d, cfg.batch_size,
                                          cfg.noise_dim, "cpu", lk=len(m), lt=len(tau))
    return gen, disc, a, b


def same_weights(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_released_mode_reproduces_the_original():
    cond, targ, m, tau = small(250)                                              # 3 batches: 100, 100, 50
    cfg = replace(Config(), n_grad=2, n_epochs=3, seed=11, **RELEASED)
    gen_o, disc_o, a, b = run_original(cond, targ, m, tau, cfg)
    res = fit(cond, targ, m, tau, cfg, CPU)
    # alphas: the original adds squared norms parameter by parameter in double precision, ours sums in float32,
    # so they agree to about 8 digits; once multiplied into the float32 loss they round to the same number:
    assert np.isclose(res.alpha_m, a, rtol=1e-6, atol=0) and np.isclose(res.alpha_tau, b, rtol=1e-6, atol=0)
    assert same_weights(gen_o, res.gen) and same_weights(disc_o, res.disc)     # bit for bit


def test_paper_half_in_eq13_halves_alpha():
    """With the 1/2 of eq 13 the BCE gradient norm halves, so the matched alphas are exactly half."""
    cond, targ, m, tau = small(250)
    cfg = replace(Config(), n_grad=1, n_epochs=0, seed=3, **RELEASED)
    full = fit(cond, targ, m, tau, cfg, CPU)
    half = fit(cond, targ, m, tau, replace(cfg, gen_bce_weight=0.5), CPU)
    assert np.isclose(half.alpha_m, full.alpha_m / 2, rtol=1e-5) and np.isclose(half.alpha_tau, full.alpha_tau / 2, rtol=1e-5)


def test_one_update_is_eq14_then_eq13():
    """Each step equals the paper's loss built by hand, in the control setting (paper penalty on g, the 1/2)."""
    cond, targ, m, tau = small(100)
    cfg = Config()
    smooth = PaperSmoothness(m, tau, CPU)
    alpha_m, alpha_tau = 2e-4, 3e-3
    bce = nn.BCELoss()
    set_seed(0)
    gen, disc = build_networks(cond.shape[1], targ.shape[1], cfg, CPU)
    gen2, disc2 = build_networks(cond.shape[1], targ.shape[1], cfg, CPU)
    gen2.load_state_dict(gen.state_dict()); disc2.load_state_dict(disc.state_dict())
    (g_opt, d_opt), (g_opt2, d_opt2) = make_optimisers(gen, disc, cfg), make_optimisers(gen2, disc2, cfg)

    torch.manual_seed(5)
    discriminator_step(gen, disc, d_opt, bce, cond, targ, cfg.noise_dim)
    generator_step(gen, disc, g_opt, bce, smooth, cond, cfg.noise_dim, "train", "g", alpha_m, alpha_tau, 0.5)

    torch.manual_seed(5)                                                         # the same noise, by hand
    z = torch.randn(len(cond), cfg.noise_dim)
    with torch.no_grad():
        fake = gen2(z, cond)
    pr, pf = disc2(torch.cat([cond, targ], -1)), disc2(torch.cat([cond, fake], -1))
    J_D = -0.5 * torch.log(pr).mean() - 0.5 * torch.log(1 - pf).mean()         # eq 14
    d_opt2.zero_grad(); J_D.backward(); d_opt2.step()
    z = torch.randn(len(cond), cfg.noise_dim)
    fake = gen2(z, cond)
    g_hat = cond[:, N_SCALARS:] + fake[:, 1:]                                   # g_t + dg_hat
    L_m, L_tau = smooth(g_hat)
    J_G = (-0.5 * torch.log(disc2(torch.cat([cond, fake], -1))).mean()
           + alpha_m * L_m.mean() + alpha_tau * L_tau.mean())                   # eq 13
    g_opt2.zero_grad(); J_G.backward(); g_opt2.step()

    for a, b in [(gen, gen2), (disc, disc2)]:
        for p, q in zip(a.parameters(), b.parameters()):
            assert torch.allclose(p, q, rtol=1e-6, atol=1e-8)


def test_grad_norm_and_alphas():
    cond, targ, m, tau = small(250)
    set_seed(0)
    gen, _ = build_networks(cond.shape[1], targ.shape[1], Config(), CPU)
    loss = gen(torch.randn(len(cond), 32), cond).pow(2).mean()
    n = grad_norm(loss, list(gen.parameters()))
    loss.backward()
    assert np.isclose(n, torch.cat([p.grad.flatten() for p in gen.parameters()]).norm().item(), rtol=1e-6)
    res = fit(cond, targ, m, tau, replace(Config(), n_grad=2, n_epochs=0), CPU)
    norms = res.match["norms"]
    assert norms.shape == (2 * 3, 3) and (norms > 0).all()                      # one row per generator update
    assert np.isclose(res.alpha_m, np.mean(norms[:, 0] / norms[:, 1]))
    assert np.isclose(res.alpha_tau, np.mean(norms[:, 0] / norms[:, 2]))
    sig = fit(cond, targ, m, tau, replace(Config(), n_grad=2, n_epochs=0, match_on="sigma"), CPU)
    assert sig.alpha_m > 5 * res.alpha_m and sig.alpha_tau > 5 * res.alpha_tau   # F8: calibrating on sigma inflates alpha


def test_restart_and_checkpoints():
    cond, targ, m, tau = small(250)
    with tempfile.TemporaryDirectory() as tmp:
        for restart in (True, False):
            folder = os.path.join(tmp, str(restart))
            cfg = replace(Config(), n_grad=2, n_epochs=0, restart=restart)
            res = fit(cond, targ, m, tau, cfg, CPU, checkpoint_dir=folder)
            assert sorted(os.listdir(folder)) == ["0_initial.pt", "1_matched.pt", "2_trained.pt"]
            states = {}
            for name in ("0_initial.pt", "1_matched.pt"):
                g, dsc = build_networks(cond.shape[1], targ.shape[1], cfg, CPU)
                go, do = make_optimisers(g, dsc, cfg)
                a_m, a_t, seed, saved_cfg = load_checkpoint(os.path.join(folder, name), g, dsc, go, do, CPU)
                states[name] = g
            assert saved_cfg["restart"] == restart and a_m == res.alpha_m and seed == cfg.seed
            start = states["0_initial.pt"] if restart else states["1_matched.pt"]
            assert same_weights(start, res.gen)                                  # n_epochs=0: still where stage 2 began
            assert not same_weights(states["0_initial.pt"], states["1_matched.pt"])


def test_no_empty_batch():
    for n in (1, 99, 100, 101, 200, 250, 3754):
        b = batch_slices(n, 100)
        assert len(b) == -(-n // 100) and all(s.stop > s.start for s in b)
        assert np.array_equal(np.concatenate([np.arange(n)[s] for s in b]), np.arange(n))
    cond, targ, m, tau = small(200)                                              # a multiple of the batch size
    cfg = replace(Config(), n_grad=1, n_epochs=1, **RELEASED)
    try:
        run_original(cond, targ, m, tau, cfg)
    except IndexError:
        pass                                                                     # B10: the original fails here
    else:
        raise AssertionError("expected the original to fail on an empty batch")
    res = fit(cond, targ, m, tau, cfg, CPU)
    assert len(res.match["norms"]) == 2 and np.isfinite(res.trained["log"]["d_loss"]).all()


def test_determinism_and_logs():
    cond, targ, m, tau = small(250)
    cfg = replace(Config(), n_grad=1, n_epochs=2)
    a, b = fit(cond, targ, m, tau, cfg, CPU), fit(cond, targ, m, tau, cfg, CPU)
    c = fit(cond, targ, m, tau, replace(cfg, seed=1), CPU)
    assert same_weights(a.gen, b.gen) and not same_weights(a.gen, c.gen)
    log = a.trained["log"]
    assert all(len(v) == 2 and np.isfinite(v).all() for v in log.values())
    assert ((log["d_real"] > 0) & (log["d_real"] < 1)).all()


def test_bad_settings_rejected():
    cond, targ, m, tau = small(100)
    for bad in [dict(smoothness="x"), dict(match_on="x"), dict(restart=1), dict(gen_bce_weight=0.0),
                dict(n_grad=0), dict(n_epochs=-1)]:
        try:
            fit(cond, targ, m, tau, replace(Config(), **{"n_grad": 1, "n_epochs": 0, **bad}), CPU)
        except ValueError:
            continue
        raise AssertionError("accepted %s" % bad)


# =====================================================================================================================
# from test_stage7_phi.py
# =====================================================================================================================

M = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

TAU = np.array([30, 60, 90, 180, 270, 360]) / 365

FIXTURE = os.path.join(HERE, "fixtures", "regression_fit_before.pt")


def test_torch_phi_equals_numpy_phi():
    rng = np.random.default_rng(7)
    ms, ts = np.meshgrid(M, TAU, indexing="ij")
    sig = rng.uniform(0.05, 0.9, (1000, 7, 6))                        # rough surfaces: plenty of violations
    for r in (rng.uniform(-0.01, 0.06, (1000, 6)), rng.uniform(-0.01, 0.06, 1000), 0.03):
        a = arbitrage_penalty(relative_call(ms, ts, sig, r), M, TAU)
        t = arbitrage_penalty_torch(relative_call_torch(ms, ts, torch.tensor(sig), torch.as_tensor(r, dtype=torch.float64)), M, TAU)
        for x, y in zip(a, t):
            assert np.allclose(x, y.numpy(), rtol=0, atol=1e-10)
        assert (a.total > 0).mean() > 0.9 and (a.calendar > 0).any() and (a.butterfly > 0).any()
    # TorchPhi (the loss term) prices g_t + dg_hat in the training layout exactly like the instrument
    g = np.log(sig[:5]).transpose(0, 2, 1).reshape(5, -1)            # maturity-major vectors
    rates = rng.uniform(0, 0.05, (5, 6))
    phi = TorchPhi(M, TAU, CPU).penalty(torch.zeros(5, 43, dtype=torch.float64), torch.tensor(g), torch.tensor(rates))
    assert np.allclose(phi.numpy(), arbitrage_penalty(relative_call(ms, ts, sig[:5], rates), M, TAU).total, atol=1e-10)


def test_gradcheck():
    ms, ts = np.meshgrid(M, TAU, indexing="ij")
    sig = torch.tensor(np.random.default_rng(3).uniform(0.1, 0.6, (3, 7, 6)), dtype=torch.float64, requires_grad=True)
    r = torch.tensor(np.random.default_rng(4).uniform(0.0, 0.05, (3, 6)), dtype=torch.float64)
    f = lambda s: arbitrage_penalty_torch(relative_call_torch(ms, ts, s, r), M, TAU).total
    assert torch.autograd.gradcheck(f, (sig,), eps=1e-6, atol=1e-6)


def test_regression_guard():
    """Control switch (phi_in_loss=False): the same weights as before Part 1, bit for bit."""
    fx = torch.load(FIXTURE, weights_only=False)
    cfg = Config(**fx["config"])
    assert (cfg.phi_in_loss, cfg.seed, cfg.n_grad, cfg.n_epochs) == (False, 1, 2, 3)
    for rates in (None, torch.full((len(fx["condition"]), 6), 0.03, dtype=torch.float64)):   # rates are ignored
        res = fit(fx["condition"], fx["target"], fx["m"], fx["tau"], cfg, CPU, rates=rates)
        for net, key in ((res.gen, "gen"), (res.disc, "disc")):
            for name, p in net.state_dict().items():
                assert torch.equal(p, fx[key][name]), name
        assert res.alpha_m == fx["alpha_m"] and res.alpha_tau == fx["alpha_tau"] and res.alpha_phi is None
        assert "phi" not in res.trained["log"] and res.match["norms"].shape[1] == 3


class AlignmentProbe:
    """Stands in for TorchPhi: checks that every batch's rates belong to its samples (the id is in both)."""

    def __init__(self):
        self.seen = []

    def __call__(self, fake, past_log_surface, rates):
        ids_cond = past_log_surface[:, 0].round().long()
        ids_rate = rates[:, 0].round().long()
        assert torch.equal(ids_cond, ids_rate), "rates are no longer aligned with their samples"
        self.seen.append(ids_cond.clone())
        return fake.sum() * 0.0


def test_rates_stay_aligned_through_the_permutation():
    n = 250
    ids = torch.arange(n, dtype=torch.float64)
    cond = torch.randn(n, N_SCALARS + 42)
    cond[:, N_SCALARS] = ids.float()                                   # the sample id, exact in float32
    targ = torch.randn(n, 43) * 0.01
    rates = torch.zeros(n, 6, dtype=torch.float64)
    rates[:, 0] = ids
    cfg = replace(Config(), n_grad=1, n_epochs=2, phi_in_loss=True)
    set_seed(0)
    gen, disc = build_networks(cond.shape[1], targ.shape[1], cfg, CPU)
    g_opt, d_opt = make_optimisers(gen, disc, cfg)
    probe = AlignmentProbe()
    train(gen, disc, g_opt, d_opt, cond, targ, make_smoothness("paper", M, TAU, CPU), cfg, "train", 2,
          alpha_m=1e-6, alpha_tau=1e-6, rates=rates, phi=probe, alpha_phi=1.0)
    seen = torch.cat(probe.seen)
    assert len(seen) == 2 * n and not torch.equal(seen[:n], torch.arange(n))     # shuffled, every sample once
    assert torch.equal(seen[:n].sort().values, torch.arange(n)) and torch.equal(seen[n:].sort().values, torch.arange(n))


def test_alpha_phi_rule_and_record():
    b = np.array([1.0, 2.0, 3.0, 4.0])
    out = match_alpha_phi(b, np.array([0.5, 0.0, 1.0, 2.0]))          # 75% non-zero: mean of ratios over those
    assert np.isclose(out["alpha_phi"], np.mean([2.0, 3.0, 2.0])) and out["alpha_phi_n_used"] == 3
    assert np.isclose(out["alpha_phi_zero_share"], 0.25) and out["alpha_phi_rule"].startswith("mean of ratios")
    out = match_alpha_phi(b, np.array([0.0, 0.0, 0.0, 2.0]))          # 25% non-zero: ratio of means, all updates
    assert np.isclose(out["alpha_phi"], b.mean() / 0.5) and out["alpha_phi_rule"].startswith("ratio of means")
    try:
        match_alpha_phi(b, np.zeros(4))
    except ValueError:
        pass
    else:
        raise AssertionError("alpha_phi matched without any Phi gradient")
    fx = torch.load(FIXTURE, weights_only=False)
    rates = torch.full((len(fx["condition"]), 6), 0.03, dtype=torch.float64)
    cfg = replace(Config(**fx["config"]), phi_in_loss=True, n_epochs=2)
    res = fit(fx["condition"], fx["target"], fx["m"], fx["tau"], cfg, CPU, rates=rates)
    norms = res.match["norms"]
    assert norms.shape[1] == 4 and res.alpha_phi == match_alpha_phi(norms[:, 0], norms[:, 3])["alpha_phi"]
    assert len(res.trained["log"]["phi"]) == 2 and np.isfinite(res.trained["log"]["phi"]).all()
    assert res.alpha_m == fx["alpha_m"]                               # the match phase still steps on BCE only
    try:
        fit(fx["condition"], fx["target"], fx["m"], fx["tau"], cfg, CPU)
    except ValueError:
        pass
    else:
        raise AssertionError("Phi in the loss accepted without rates")


# =====================================================================================================================
# new: the 2 x 2 design and the removal of the data lever
# =====================================================================================================================


def test_part1_configs_and_cells():
    """The design is a 2 x 2: two training configurations, four cells (A control, B, C, D), no data lever."""
    assert set(TRAINING_CONFIGS) == {"raw_phiout", "raw_phiin"}
    assert TRAINING_CONFIGS["raw_phiout"] == dict(phi_in_loss=False) and TRAINING_CONFIGS["raw_phiin"] == dict(phi_in_loss=True)
    assert CELLS == {"A": ("raw_phiout", "max_penalty"), "B": ("raw_phiout", "raw"),
                     "C": ("raw_phiin", "max_penalty"), "D": ("raw_phiin", "raw")}
    assert cell_levers("A") == ("out", "re-weighted") and cell_levers("D") == ("in", "unweighted")
    for name in TRAINING_CONFIGS:
        assert training_config(name, 3).seed == 3
    try:
        training_config("smooth_phiout", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a removed configuration was accepted")


def test_nothing_of_the_data_lever_is_left():
    """No module defines or imports anything about the kernel smoothing or the projection (smoothness, the loss
    penalty of the paper, is a different thing and stays). data_smooth / data_repair survive only as the removed
    settings kept so that old run signatures do not change."""
    root = os.path.join(HERE, "..")
    files = [os.path.join(root, "1_one_step", "main.py"), os.path.join(root, "2_recursion", "part2.py")] + \
            [os.path.join(root, "volgan_fixed", f) for f in os.listdir(os.path.join(root, "volgan_fixed")) if f.endswith(".py")]
    for path in files:
        text = open(path, encoding="utf-8").read()
        for word in ("smoothed_data", "bandwidth", "select_bandwidth", "smooth_all", "project_surface", "repaired_data",
                     "SMOOTHED_DIR", "REPAIRED_DIR"):
            assert word not in text, "%s mentions %s" % (os.path.basename(path), word)
        if not path.endswith("training.py"):                 # the one place the removed settings are listed
            for word in ("data_smooth", "data_repair"):
                assert word not in text, "%s mentions %s" % (os.path.basename(path), word)
    assert not [f for f in os.listdir(os.path.join(root, "volgan_fixed")) if f.startswith(("smooth.", "repair."))]
    assert not hasattr(Config(), "data_smooth") and not hasattr(Config(), "data_repair")


def test_signatures_match_the_baseline():
    """Run signatures identify trained runs: removing the data lever must not change them (fixtures/baseline.json,
    recorded before the removal). Depends on the workbooks' hashes, so it skips without them."""
    base = json.load(open(os.path.join(HERE, "fixtures", "baseline.json")))["signatures"]
    assert len(base) == 4
    for key, want in base.items():
        name, seed = key.split("/seed_")
        assert signature("SPX", training_config(name, int(seed))) == want, key


def test_old_run_json_loads_and_a_smoothed_one_is_refused():
    """A run folder whose config carries the removed keys loads; one trained on the smoothed grid raises."""
    cfg = replace(training_config("raw_phiout", 1), n_grad=1, n_epochs=1)
    with tempfile.TemporaryDirectory() as tmp:
        train_run("SPX", cfg, tmp, torch.device("cpu"), progress=False)
        path = os.path.join(tmp, "run.json")
        info = json.load(open(path))
        assert not any(k in info["config"] for k in ("data_smooth", "smooth_h1", "data_repair"))
        info["config"].update({"data_smooth": False, "smooth_h1": 0.002, "smooth_h2": 0.046, "smooth_exclude_days": [270],
                               "data_repair": False, "repair_norm": "l2"})           # the shape of a run made before the removal
        json.dump(info, open(path, "w"))
        run = load_run(tmp, torch.device("cpu"))
        assert run.config == cfg and run.split.val is not None
        info["config"]["data_smooth"] = True
        json.dump(info, open(path, "w"))
        try:
            load_run(tmp, torch.device("cpu"))
        except ValueError as e:
            assert "no longer" in str(e)
        else:
            raise AssertionError("a run trained on the smoothed grid was loaded")


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
        test_seed_repeats,
        test_checkpoint_round_trip,
        test_paper_defaults,
        test_parameter_counts,
        test_layer_shapes,
        test_identical_to_original_networks,
        test_outputs_and_gradients,
        test_hand_computed,
        test_matches_the_formulas_as_loops,
        test_each_term_sees_only_its_own_axis,
        test_released_equals_original_code,
        test_released_is_paper_on_the_transposed_surface,
        test_gradients_dtype_and_switches,
        test_old_run_folder_still_loads,
        test_released_mode_reproduces_the_original,
        test_paper_half_in_eq13_halves_alpha,
        test_one_update_is_eq14_then_eq13,
        test_grad_norm_and_alphas,
        test_restart_and_checkpoints,
        test_no_empty_batch,
        test_determinism_and_logs,
        test_bad_settings_rejected,
        test_torch_phi_equals_numpy_phi,
        test_gradcheck,
        test_regression_guard,
        test_rates_stay_aligned_through_the_permutation,
        test_alpha_phi_rule_and_record,
        test_part1_configs_and_cells,
        test_nothing_of_the_data_lever_is_left,
        test_signatures_match_the_baseline,
        test_old_run_json_loads_and_a_smoothed_one_is_refused,
    ], "Training")
