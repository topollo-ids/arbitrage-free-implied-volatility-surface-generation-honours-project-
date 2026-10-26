"""
Simulation, scenario re-weighting, the one-pass metrics, and the simulate / evaluate steps.
Imports common, preprocessing and training.

Contents
    1. Simulation (was simulate.py)
    2. Re-weighting and the one-pass metrics (was reweight.py)
    3. Evaluation helpers (was evaluate.py)
    4. The simulate and evaluate steps (was pipeline.py)
"""

from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
import torch
from tqdm import tqdm

from .common import (
    CELLS, SEEDS, TRAINING_CONFIGS, VOL_INDEX, _done, arbitrage_penalty, block_dir, cell_of, get_device, log,
    metrics_path, relative_call, sim_path, text_hash, training_config, violation_nodes)
from .preprocessing import N_SCALARS, prepare_all
from .training import _trained, load_run


# =====================================================================================================================
# 1. Simulation (was simulate.py)
# =====================================================================================================================

# Stage 8: one-step simulation (the paper's N scenarios per test day; the one-step template for Book 2).
#
# Tracker rows: 8.1 (B1: nothing to unpack, fit() returns a named result), 8.2-8.3 (B15, A6: bounded memory),
# 8.4 (A8: reshapes, no loops), 8.5 (R9: nothing recomputed), 8.6 (R13: the generator's own device), 8.7 (B21).
#
# For test day j, scenario s is G(a_j, z_s) with a_j the day's condition. It gives
#     R_hat     = generated return / ret_scale         a daily log-return again (F6, Stage 3)
#     g_hat     = g_t + dg_hat                          the generated log-vol surface
#     sigma_hat = exp(g_hat)                            the generated implied-vol surface
# and its arbitrage penalty Phi(sigma_hat) (eqs 2-5, Stage 2), evaluated in float64 (A10).
#
# Memory (A6): only Phi and R_hat are kept for every scenario, n_days x N numbers each. The noise for a day
# comes from its own generator, seeded by (seed, the day's DATE), so the full scenarios of any day can be rebuilt
# exactly with day_scenarios(), alone, in any chunk, or from any slice of the test set that contains that date.
#
# Part 1:
#   - the seed passed here is ALWAYS Config.sim_seed, never the training seed (G10). Every model therefore receives
#     the same noise on a given date: common random numbers across cells and training seeds;
#   - the inputs Dataset is what the network is conditioned on, and the market (realised) surface comes from the
#     same raw vendor grid;
#   - generate_full() also returns the network's increment dg_hat, the model side of PCA and correlations;
#   - Part 2: step() is one network evaluation; the state update that builds on it lives in recursion.py.

SEED_STRIDE = 1_000_000          # the day numbered k (days since 1970) under seed s uses generator seed s * SEED_STRIDE + k


def day_key(date):
    """The number that seeds a date's noise: whole days since 1 January 1970 (about 20,000 for our dates)."""
    return int((pd.Timestamp(date) - pd.Timestamp("1970-01-01")).days)


def day_noise(seed, key, n, noise_dim):
    """The N noise vectors of one day, from that day's own generator (CPU, then moved by the caller)."""
    g = torch.Generator().manual_seed(seed * SEED_STRIDE + int(key))
    return torch.randn(n, noise_dim, generator=g)


def step(gen, condition, noise):
    """
    One network evaluation (Part 2 hook): G(condition, noise) -> (R_hat, dg_hat), tensors (B,) and (B, K).
    R_hat is in the units of the condition's return slots (daily return x ret_scale, F6); divide by ret_scale for a
    daily log-return. dg_hat is the generated change in log-vol at every grid point.
    """
    fake = gen(noise, condition)
    return fake[:, 0], fake[:, 1:]


@torch.no_grad()
def generate_full(gen, test, days, n, seed, n_draw=None):
    """
    Scenarios for the days `days` of the inputs Dataset `test`, float64 arrays:
    R_hat (D, n) daily log-returns, dg_hat (D, n, K) the network's log-vol increments, and
    sigma_hat (D, n, K) = exp(g_t + dg_hat) with g_t the conditioning surface. One forward pass for all of them.
    n_draw (Part 2): draw n_draw noise vectors per day and keep the first n. torch.randn(n, d) does not give the
    same first rows for different n, so n_draw = 10,000 with n = 1,000 reproduces the first 1,000 scenarios of the
    Part 1 simulation exactly. The default (None: n_draw = n) is the Part 1 behaviour, unchanged.
    """
    days = np.asarray(days)
    device = next(gen.parameters()).device
    keys = [day_key(test.dates[j]) for j in days]
    n_draw = n if n_draw is None else n_draw
    noise = torch.stack([day_noise(seed, k, n_draw, gen.noise_dim)[:n] for k in keys]).to(device)    # (D, n, noise)
    cond = torch.as_tensor(test.condition[days], dtype=torch.float32, device=device)       # (D, 3+K)
    cond = cond[:, None, :].expand(-1, n, -1)                                                # (D, n, 3+K)
    gen.eval()
    r_hat, dg = step(gen, cond.reshape(-1, cond.shape[-1]), noise.reshape(-1, noise.shape[-1]))
    returns = r_hat.reshape(len(days), n).double().cpu().numpy() / test.ret_scale
    dg = dg.reshape(len(days), n, -1).double().cpu().numpy()
    sigma = np.exp(test.condition[days, None, N_SCALARS:] + dg)                              # exp(g_t + dg_hat)
    return returns, dg, sigma


def generate(gen, test, days, n, seed):
    """
    Scenarios for the test days `days`: R_hat (len(days), n) and sigma_hat (len(days), n, K), float64.
    One forward pass for all of them.
    """
    returns, _, sigma = generate_full(gen, test, days, n, seed)
    return returns, sigma


def day_scenarios(gen, test, day, n, seed):
    """Rebuild the full scenarios of one test day: R_hat (n,), sigma_hat (n, K). Same numbers as in simulate()."""
    returns, sigma = generate(gen, test, [day], n, seed)
    return returns[0], sigma[0]


def surface_matrices(sigma, lk, lt):
    """(..., K) maturity-major vectors -> (B, lk, lt) matrices, moneyness on axis 1 (A8: a reshape, no loop)."""
    return sigma.reshape(-1, lt, lk).transpose(0, 2, 1)


def per_day(x, n_days, lt, name="rate"):
    """A rate as (n_days, lt): from a scalar, one value per day, or one per day and maturity."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.full((n_days, lt), float(x))
    if x.shape == (n_days,):
        return np.repeat(x[:, None], lt, axis=1)
    if x.shape == (n_days, lt):
        return x
    raise ValueError("%s must be a scalar, (%d,) or (%d, %d); got shape %s" % (name, n_days, n_days, lt, x.shape))


def scenario_phi(sigma, m, tau, rate=0.0, with_nodes=False):
    """
    Arbitrage penalty of each scenario. sigma: (D, n, K); rate: see per_day, one row per test day.
    Returns a Phi of arrays shaped (D, n). with_nodes=True also returns, from the same prices, the (D, n, lk, lt)
    booleans of arbitrage.violation_nodes: which grid nodes take part in a violated term (Part 1, metric 8.1).
    """
    D, n, _ = sigma.shape
    lk, lt = len(m), len(tau)
    r = per_day(rate, D, lt, "rate")
    ms, ts = np.meshgrid(m, tau, indexing="ij")
    C = relative_call(ms, ts, surface_matrices(sigma, lk, lt), np.repeat(r, n, axis=0))
    p = arbitrage_penalty(C, m, tau)
    p = type(p)(*(x.reshape(D, n) for x in p))
    if not with_nodes:
        return p
    return p, violation_nodes(C, m, tau).reshape(D, n, lk, lt)


def market_sigma(market, dates=None):
    """
    The market surface on each predicted day: exp(g_t + dg), the quantity the scenarios try to forecast.
    market: the raw-grid Dataset. dates: the simulated dates, which must be the market's.
    """
    if dates is not None and not market.dates.equals(pd.DatetimeIndex(dates)):
        raise ValueError("the market Dataset's dates are not the simulated dates")
    return np.exp(market.past_log_surface + market.target[:, 1:])


@dataclass
class Simulation:
    dates: pd.DatetimeIndex   # (n_days,) predicted day of each test sample
    phi: np.ndarray           # (n_days, N) arbitrage penalty of every scenario, float64
    phi_parts: np.ndarray     # (n_days, 3) daily mean of the calendar, call and butterfly terms
    returns: np.ndarray       # (n_days, N) generated daily log-return of every scenario
    rate: np.ndarray          # (n_days, lt) rate used in the relative call price, per day and maturity
    seed: int
    n_scenarios: int


def simulate(gen, test, n_scenarios, seed, rate=0.0, chunk_days=4, progress=False):
    """
    N scenarios for every day of the test Dataset, a few days at a time (chunk_days) to bound memory.
    rate: a scalar, one per test day, or one per test day and maturity (Stage 9: rates.py); in Part 1 the inputs
    Dataset's own rates (test.rates). rate = 0 reproduces the released code (F4).
    seed: the simulation seed, Config.sim_seed in Part 1 (G10), the same for every model.
    """
    n_days = len(test.dates)
    rate = per_day(rate, n_days, len(test.tau), "rate")
    phi = np.empty((n_days, n_scenarios))
    returns = np.empty((n_days, n_scenarios))
    parts = np.empty((n_days, 3))
    for start in tqdm(range(0, n_days, chunk_days), desc="simulate", disable=not progress):
        days = np.arange(start, min(start + chunk_days, n_days))
        r_hat, sigma = generate(gen, test, days, n_scenarios, seed)
        p = scenario_phi(sigma, test.m, test.tau, rate[days])
        phi[days], returns[days] = p.total, r_hat
        parts[days] = np.stack([p.calendar.mean(1), p.call.mean(1), p.butterfly.mean(1)], axis=1)
    return Simulation(test.dates, phi, parts, returns, rate, seed, n_scenarios)


def ns(dates):
    """Dates as a datetime64[ns] array: how every .npz file stores them, whatever pandas' own resolution is."""
    return np.asarray(pd.DatetimeIndex(dates), dtype="datetime64[ns]")


def save_simulation(path, sim, **meta):
    """Write a Simulation (and any extra metadata, e.g. the run's config hash) to one compressed .npz file."""
    np.savez_compressed(path, dates=ns(sim.dates), phi=sim.phi, phi_parts=sim.phi_parts, returns=sim.returns,
                        rate=sim.rate, seed=sim.seed, n_scenarios=sim.n_scenarios,
                        **{"meta_" + k: np.asarray(v) for k, v in meta.items()})


def load_simulation(path):
    """(Simulation, metadata dict) from save_simulation's file."""
    with np.load(path) as z:
        sim = Simulation(pd.DatetimeIndex(z["dates"]), z["phi"], z["phi_parts"], z["returns"], z["rate"],
                         int(z["seed"]), int(z["n_scenarios"]))
        meta = {k[5:]: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files if k.startswith("meta_")}
    return sim, meta


# =====================================================================================================================
# 2. Re-weighting and the one-pass metrics (was reweight.py)
# =====================================================================================================================

# Stage 10: scenario re-weighting (paper section 3.3, eqs 15-20; Cont and Vuletic 2023, section 4).
#
# Tracker rows: 10.1 (F10: weights and beta(t) implemented), 10.2 (A5: stable weights, one weighted quantile),
# 10.3-10.4 (B4, B5: replaced by that one routine), 10.6 (B17: sampling returns integer indices),
# 10.7 (R7: no duplicated functions), 10.8 (eq 18 weighted means, vectorised).
# Part 1 (G6): daily_metrics() computes every per-day metric of the ablation in one pass, reusing these helpers.
# Row 10.5 (B6, "quantile has no return") does not apply to our copy of VolGAN.py: its quantile() returns
# and computes eq 19 correctly; tests/test_stage10.py uses it as a reference.
#
# One day t, N scenarios with arbitrage penalties Phi_1 .. Phi_N (Stage 8):
#   weights (eq 17)    w_i = exp(-beta Phi_i) / sum_j exp(-beta Phi_j)
#                      computed as exp(-beta (Phi_i - min_j Phi_j)) / sum_j (...), equal in exact arithmetic. The best
#                      scenario always contributes exp(0) = 1, so the sum is never 0 and no weight is NaN (A5).
#   mean (eq 18)       E_beta[X] = sum_i w_i x_i
#   quantile (eq 19)   x_(k) with k = min{j : w_(1) + ... + w_(j) >= q}, where x_(1) <= ... <= x_(N) and w_(i) is the
#                      weight of the scenario holding x_(i). For a surface, at every grid point separately.
#   sampling (eq 17)   draw scenario i with probability w_i.
# With beta = 0 every weight is 1/N, so the raw (unweighted) statistics come from the same functions.
#
# beta(t), eq 20. As printed, beta(t) = 500 / max_i w_i(t), where the weights w_i(t) themselves depend on beta(t).
# Three readings, chosen with rule= in beta_t (decision log):
#   "max_penalty" (control)  beta(t) = c / max_i Phi_i(t), c = 500. The day's worst scenario gets exp(-500) times
#       the weight of an arbitrage-free one, whatever the scale of that day's penalties. Reasons: beta has the
#       units of 1/Phi, so the denominator should be a penalty; the paper's Figure 7 calls it "adaptive beta";
#       and it is of the order of the fixed beta = 1000 that Cont and Vuletic (2023) used for VolGAN when the
#       worst penalty is about 0.5.
#   "fixed"                  beta(t) = c every day (Cont and Vuletic 2023: c = 1000; the released code: an argument).
#   "literal"                the beta(t) that solves beta = c / max_i w_i(beta) exactly (see literal_beta).

C_BETA = 500.0
QUANTILE_TOL = 1e-12      # eq 19's ">= q", allowing for rounding in the running sum of the weights
RULES = ("max_penalty", "fixed", "literal", "raw")


def _penalties(phi):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be (n_days, N), got shape %s" % (phi.shape,))
    if not np.isfinite(phi).all() or (phi < 0).any():
        raise ValueError("penalties must be finite and non-negative")
    return phi


def _per_day_beta(beta, n_days):
    beta = np.broadcast_to(np.asarray(beta, dtype=float), (n_days,))
    if not np.isfinite(beta).all() or (beta < 0).any():
        raise ValueError("beta must be finite and non-negative")
    return beta


def weights(phi, beta):
    """Eq 17 for every day. phi: (n_days, N); beta: a scalar or one per day. Rows sum to 1."""
    phi = _penalties(phi)
    beta = _per_day_beta(beta, phi.shape[0])
    e = np.exp(-beta[:, None] * (phi - phi.min(axis=1, keepdims=True)))
    return e / e.sum(axis=1, keepdims=True)


def max_weight(phi, beta):
    """max_i w_i(beta) for every day: the weight of the best scenario, 1 / sum_j exp(-beta (Phi_j - min Phi))."""
    phi = _penalties(phi)
    beta = _per_day_beta(beta, phi.shape[0])
    return 1.0 / np.exp(-beta[:, None] * (phi - phi.min(axis=1, keepdims=True))).sum(axis=1)


def literal_beta(phi, c=C_BETA, iterations=60):
    """
    Eq 20 read literally: the beta(t) with beta * max_i w_i(beta) = c. f(beta) = beta * max_i w_i(beta) rises
    strictly (max_i w_i(beta) is the best scenario's weight, which grows with beta), so the solution is unique;
    since 1/N <= max_i w_i <= 1 it lies in [c, c N]. Found by bisection in log(beta), every day at once.
    """
    phi = _penalties(phi)
    lo = np.full(phi.shape[0], np.log(c))
    hi = np.full(phi.shape[0], np.log(c * phi.shape[1]))
    for _ in range(iterations):
        mid = (lo + hi) / 2
        above = np.exp(mid) * max_weight(phi, np.exp(mid)) >= c
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return np.exp(hi)


def beta_t(phi, rule="max_penalty", c=None):
    """beta for every day under one reading of eq 20 (see the module docstring); "raw" gives beta = 0."""
    phi = _penalties(phi)
    if rule == "max_penalty":
        worst = phi.max(axis=1)
        c = C_BETA if c is None else c
        # a day whose scenarios are all arbitrage-free has uniform weights for every beta; use 0
        return np.where(worst > 0, c / np.where(worst > 0, worst, 1.0), 0.0)
    if rule == "fixed":
        if c is None:
            raise ValueError("rule 'fixed' needs c, the value of beta")
        return np.full(phi.shape[0], float(c))
    if rule == "literal":
        return literal_beta(phi, C_BETA if c is None else c)
    if rule == "raw":
        return np.zeros(phi.shape[0])
    raise ValueError("rule must be one of %s, got %r" % (RULES, rule))


def effective_sample_size(w):
    """1 / sum_i w_i^2: N when all weights are equal, 1 when one scenario holds all the weight."""
    return 1.0 / (np.asarray(w) ** 2).sum(axis=-1)


def relative_entropy(w):
    """
    KL(w || uniform) = sum_i w_i ln(N w_i) >= 0: 0 for equal weights, ln(N / n) when the weight is spread evenly
    over n scenarios. Cont and Vuletic's E_N(beta) (their eq 20, plotted in their Figure 10) is N times this.
    """
    w = np.asarray(w, dtype=float)
    terms = np.zeros_like(w)
    np.multiply(w, np.log(w.shape[-1] * w, where=w > 0, out=np.zeros_like(w)), out=terms, where=w > 0)
    return terms.sum(axis=-1)


def _expand(w, x):
    """Weights (n_days, N) shaped to broadcast against x (n_days, N, ...)."""
    w = np.asarray(w, dtype=float)
    if w.shape != x.shape[:2]:
        raise ValueError("weights %s do not match scenarios %s" % (w.shape, x.shape[:2]))
    return w.reshape(w.shape + (1,) * (x.ndim - 2))


def weighted_mean(x, w):
    """Eq 18. x: (n_days, N) or (n_days, N, K), scenarios on axis 1; w: (n_days, N)."""
    x = np.asarray(x, dtype=float)
    return (_expand(w, x) * x).sum(axis=1)


def weighted_quantile(x, w, qs):
    """
    Eq 19 for each level q in qs. x: (n_days, N) or (n_days, N, K), scenarios on axis 1; w: (n_days, N).
    Each day, and each grid point of a surface, is sorted by its own values with the weights carried along
    (A5; B4 and B5 sorted the wrong thing). Returns (len(qs), n_days) or (len(qs), n_days, K).
    """
    x = np.asarray(x, dtype=float)
    wb = np.broadcast_to(_expand(w, x), x.shape)
    order = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, order, axis=1)
    running = np.cumsum(np.take_along_axis(wb, order, axis=1), axis=1)     # w_(1) + ... + w_(j)
    out = []
    for q in np.atleast_1d(qs):
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile levels must lie in [0, 1], got %r" % (q,))
        k = (running < q - QUANTILE_TOL).sum(axis=1, keepdims=True)          # first j with running sum >= q
        k = np.minimum(k, x.shape[1] - 1)
        out.append(np.take_along_axis(xs, k, axis=1)[:, 0])
    return np.stack(out)


def sample_indices(w, size, rng):
    """Eq 17's sampling: for each day, `size` scenario indices drawn with probabilities w (B17: plain integers)."""
    w = np.asarray(w, dtype=float)
    return np.stack([rng.choice(w.shape[1], size=size, p=row) for row in w])


@dataclass
class DailyStats:
    """Weighted next-day statistics of one test set under one choice of beta (raw: beta = 0)."""
    name: str
    qs: tuple
    beta: np.ndarray        # (n_days,)
    ess: np.ndarray         # (n_days,) effective number of scenarios
    max_weight: np.ndarray  # (n_days,)
    entropy: np.ndarray     # (n_days,) KL(w || uniform)
    phi_mean: np.ndarray    # (n_days,) eq 18 applied to the arbitrage penalty
    ret_mean: np.ndarray    # (n_days,) eq 18, daily log-return
    ret_q: np.ndarray       # (len(qs), n_days) eq 19
    surf_mean: np.ndarray   # (n_days, K) eq 18 at every grid point (implied vols)
    surf_q: np.ndarray      # (len(qs), n_days, K) eq 19 at every grid point
    extra_mean: dict = field(default_factory=dict)   # name -> (n_days,) eq 18 of each extra quantity (Stage 11)
    extra_q: dict = field(default_factory=dict)      # name -> (len(qs), n_days) eq 19 of each extra quantity


def daily_statistics(gen, test, sim, betas, qs=(0.025, 0.5, 0.975), chunk_days=4, progress=False, extras=None):
    """
    Next-day statistics for every test day under each beta in betas ({name: beta per day}), from one Simulation.
    The surfaces are rebuilt day by day from the simulation's seed (Stage 8), so each scenario's surface is the one
    whose penalty set its weight. That is checked, not assumed: every chunk's rebuilt returns must equal the
    simulated ones, and the first chunk's penalties are recomputed and compared.
    extras: {name: f(returns (D, N), sigma (D, N, K), days) -> (D, N)}, further per-scenario quantities (e.g. the
    VIX of each scenario, Stage 11) whose weighted means and quantiles are wanted too.
    """
    n_days, N = sim.phi.shape
    if len(test.dates) != n_days or not test.dates.equals(sim.dates):
        raise ValueError("the simulation does not belong to this test set")
    W = {name: weights(sim.phi, beta) for name, beta in betas.items()}
    K = test.condition.shape[1] - N_SCALARS
    surf_mean = {name: np.empty((n_days, K)) for name in W}
    surf_q = {name: np.empty((len(qs), n_days, K)) for name in W}
    extras = extras or {}
    ex_mean = {name: {e: np.empty(n_days) for e in extras} for name in W}
    ex_q = {name: {e: np.empty((len(qs), n_days)) for e in extras} for name in W}
    for start in tqdm(range(0, n_days, chunk_days), desc="statistics", disable=not progress):
        days = np.arange(start, min(start + chunk_days, n_days))
        r_hat, sigma = generate(gen, test, days, N, sim.seed)
        if not np.allclose(r_hat, sim.returns[days], rtol=1e-6, atol=1e-9):
            raise RuntimeError("rebuilt scenarios differ from the simulated ones (another generator or seed?)")
        if start == 0:
            phi = scenario_phi(sigma, test.m, test.tau, sim.rate[days]).total
            if not np.allclose(phi, sim.phi[days], rtol=1e-6, atol=1e-10):
                raise RuntimeError("rebuilt penalties differ from the simulated ones (another rate?)")
        values = {e: fn(r_hat, sigma, days) for e, fn in extras.items()}
        for name, w in W.items():
            surf_mean[name][days] = weighted_mean(sigma, w[days])
            surf_q[name][:, days] = weighted_quantile(sigma, w[days], qs)
            for e, v in values.items():
                ex_mean[name][e][days] = weighted_mean(v, w[days])
                ex_q[name][e][:, days] = weighted_quantile(v, w[days], qs)
    out = {}
    for name, w in W.items():
        out[name] = DailyStats(
            name=name, qs=tuple(qs), beta=np.broadcast_to(np.asarray(betas[name], float), (n_days,)).copy(),
            ess=effective_sample_size(w), max_weight=w.max(axis=1), entropy=relative_entropy(w),
            phi_mean=weighted_mean(sim.phi, w), ret_mean=weighted_mean(sim.returns, w),
            ret_q=weighted_quantile(sim.returns, w, qs), surf_mean=surf_mean[name], surf_q=surf_q[name],
            extra_mean=ex_mean[name], extra_q=ex_q[name])
    return out


# ---- Part 1 (G6): every per-day metric of the ablation in ONE pass over the scenarios (brief section 8)

TOL_PHI = 1e-12          # a scenario "has arbitrage" when its Phi exceeds this (violation share)
B_PCA = 1000             # generated series for PCA and the correlation tables


def pca_indices(w, beta, b_pca, sim_seed):
    """
    (n_days, b_pca) scenario indices of the generated series used for PCA (brief 8.3). On a day with uniform weights
    (beta = 0) series b takes scenario b; otherwise it takes the scenario drawn from that day's weights with the
    uniform number of an RNG seeded by (sim_seed, b): one stream per series, one number per day. The choice is made
    day by day, so the re-weighted rule at beta = 0 picks exactly the unweighted series (cell bookkeeping, test 12).
    """
    n_days, N = w.shape
    if b_pca > N:
        raise ValueError("b_pca (%d) cannot exceed the number of scenarios (%d)" % (b_pca, N))
    u = np.stack([np.random.default_rng([sim_seed, b]).random(n_days) for b in range(b_pca)], axis=1)
    cdf = np.cumsum(w, axis=1)
    drawn = np.stack([np.searchsorted(cdf[t], u[t], side="right") for t in range(n_days)])
    drawn = np.minimum(drawn, N - 1)
    uniform = np.broadcast_to(np.arange(b_pca), (n_days, b_pca))
    return np.where((np.asarray(beta) == 0)[:, None], uniform, drawn)


def daily_metrics(gen, inputs, market, sim, rules=("raw", "max_penalty"), b_pca=B_PCA, chunk_days=4, progress=False,
                  reference=None, beta_override=None):
    """
    The Part 1 metrics of one trained network under each weighting rule, from one Simulation, in one pass per chunk
    of days (brief 7.2, 8). The surfaces are rebuilt from (network, sim.seed, date) and checked against the stored
    returns and penalties, as in daily_statistics.
      gen, inputs  the network and the Dataset it is conditioned on (the raw vendor grid);
      market       the RAW Dataset of the same days: every realised value comes from it (G9);
      sim          simulate(gen, inputs, N, config.sim_seed, inputs.rates);
      rules        beta_t rules: "raw" (beta = 0, unweighted) and "max_penalty" (re-weighted, c = 500);
      reference    evaluate.market_reference(market) if already computed (the data side: PCA eigenvectors etc.);
      beta_override  {rule: beta per day} replacing beta_t for that rule (tests: cell A at beta = 0 is cell B).
    Model increments are the network's output dg_hat, measured from the condition surface the network saw; the model's change in log surface-VIX is likewise measured from that condition surface.
    Returns {rule: {name: array}}; per day unless stated:
      beta, ess, max_weight, entropy                          re-weighting diagnostics
      phi_mean, phi_parts (n, 3), viol_share, viol_loc (n, lk, lt)   admissibility (8.1)
      cal_q (5, n, 11), cal_mean (n, 11), cal_realised (n, 11), and the indicators below_025, above_975,
      inside_95, below_01, above_99, inside_98, width_95 (n, 11)      calibration (8.2)
      rho (n,), rho_bar, rho_n, rho_band (2,)                 rho_t and the constant-correlation band (8.3)
      pca_share (B, 3), pca_inner (B, 3), pca_v1_mean (K,)    per generated series b (8.3)
      corr_ret_pc (B, 3), corr_mat (B, 4, 4)                  per generated series b (8.3)
    """
    n_days, N = sim.phi.shape
    if not (inputs.dates.equals(sim.dates) and market.dates.equals(sim.dates)):
        raise ValueError("inputs, market and simulation must cover the same days")
    if inputs.rates is None or not np.allclose(sim.rate, inputs.rates, rtol=0, atol=1e-15):
        raise ValueError("the simulation's rates are not the inputs' rates")
    ref = reference if reference is not None else market_reference(market)
    m, tau = inputs.m, inputs.tau
    lk, lt, K = len(m), len(tau), len(m) * len(tau)
    cal = node_index(m, tau, CALIBRATION_POINTS)
    atm = int(node_index(m, tau, [ATM_1M])[0])
    betas = {rule: beta_t(sim.phi, rule) for rule in rules}
    betas.update({r: np.broadcast_to(np.asarray(b, float), (n_days,)).copy() for r, b in (beta_override or {}).items()})
    W = {rule: weights(sim.phi, betas[rule]) for rule in rules}
    idx = {rule: pca_indices(W[rule], betas[rule], b_pca, sim.seed) for rule in rules}
    n_var = len(cal) + 2
    vix_cond = ref["vix_cond"]                                       # the VIX of the condition surface (raw grid)

    def empty(*shape):
        return {rule: np.empty(shape) for rule in rules}
    parts, viol, loc = empty(n_days, 3), empty(n_days), empty(n_days, lk, lt)
    cal_q, cal_mean, rho = empty(len(CAL_QS), n_days, n_var), empty(n_days, n_var), empty(n_days)
    S1 = {rule: np.zeros((b_pca, K + 2)) for rule in rules}
    S2 = {rule: np.zeros((b_pca, K + 2, K + 2)) for rule in rules}

    for start in tqdm(range(0, n_days, chunk_days), desc="metrics", disable=not progress):
        days = np.arange(start, min(start + chunk_days, n_days))
        r_hat, dg, sigma = generate_full(gen, inputs, days, N, sim.seed)
        if not np.allclose(r_hat, sim.returns[days], rtol=1e-6, atol=1e-9):
            raise RuntimeError("rebuilt scenarios differ from the simulated ones (another generator or seed?)")
        p, nodes = scenario_phi(sigma, m, tau, sim.rate[days], with_nodes=True)
        if not np.allclose(p.total, sim.phi[days], rtol=1e-6, atol=1e-10):
            raise RuntimeError("rebuilt penalties differ from the simulated ones (another rate?)")
        pp = np.stack([p.calendar, p.call, p.butterfly], axis=-1)                      # (D, N, 3)
        bad = (p.total > TOL_PHI).astype(float)
        flat_nodes = nodes.reshape(len(days), N, -1).astype(float)
        vix = vix_of_surfaces(sigma, m, tau, np.repeat(sim.rate[days, 0][:, None], N, axis=1))   # (D, N)
        X = np.concatenate([sigma[..., cal], r_hat[..., None], vix[..., None]], axis=-1)       # (D, N, 11)
        dlogvix = np.log(vix) - np.log(vix_cond[days])[:, None]
        for rule in rules:
            w = W[rule][days]
            parts[rule][days] = weighted_mean(pp, w)
            viol[rule][days] = (w * bad).sum(axis=1)
            loc[rule][days] = weighted_mean(flat_nodes, w).reshape(len(days), lk, lt)
            cal_q[rule][:, days] = weighted_quantile(X, w, CAL_QS)
            cal_mean[rule][days] = weighted_mean(X, w)
            rho[rule][days] = weighted_corr(r_hat, dg[..., atm], w)
            pick = idx[rule][days]                                                   # (D, B)
            z = increments(np.take_along_axis(r_hat, pick, axis=1),
                           np.take_along_axis(dg, pick[..., None], axis=1),
                           np.take_along_axis(dlogvix, pick, axis=1))                  # (D, B, K + 2)
            S1[rule] += z.sum(axis=0)
            S2[rule] += np.einsum("dbi,dbj->bij", z, z)

    out = {}
    real = ref["cal_realised"]
    for rule in rules:
        w, q = W[rule], cal_q[rule]
        ess = effective_sample_size(w)
        n_band = float(N) if rule == "raw" else float(np.median(ess))       # decision log: n for the band
        rho_bar = float(np.mean(rho[rule]))
        cov = (S2[rule] - np.einsum("bi,bj->bij", S1[rule], S1[rule]) / n_days) / (n_days - 1)
        dyn = dynamics_from_cov(cov, ref["pcs"], atm)
        lo95, hi95 = q[CAL_QS.index(0.025)], q[CAL_QS.index(0.975)]
        lo98, hi98 = q[CAL_QS.index(0.01)], q[CAL_QS.index(0.99)]
        out[rule] = {
            "beta": betas[rule], "ess": ess, "max_weight": w.max(axis=1), "entropy": relative_entropy(w),
            "phi_mean": weighted_mean(sim.phi, w), "phi_parts": parts[rule], "viol_share": viol[rule],
            "viol_loc": loc[rule],
            "cal_q": q, "cal_mean": cal_mean[rule], "cal_realised": real,
            "below_025": real < lo95, "above_975": real > hi95, "inside_95": (real >= lo95) & (real <= hi95),
            "below_01": real < lo98, "above_99": real > hi98, "inside_98": (real >= lo98) & (real <= hi98),
            "width_95": hi95 - lo95,
            "rho": rho[rule], "rho_bar": rho_bar, "rho_n": n_band,
            "rho_band": np.array(correlation_band(rho_bar, n_band)),
            "pca_share": dyn["share"], "pca_inner": dyn["inner"], "pca_v1_mean": dyn["v1"].mean(axis=0),
            "corr_ret_pc": dyn["ret_pc"], "corr_mat": dyn["corrmat"],
        }
    return out


# =====================================================================================================================
# 3. Evaluation helpers (was evaluate.py)
# =====================================================================================================================

# Stage 11: evaluation helpers.
#
# Tracker rows: 11.1 (F11, R2: one interpolator, the paper's rule with linear extrapolation), 11.2 (B21: forecasts and
# realised values belong to the same predicted day), 11.3-11.6 (B2, B3, B20, A7: a VIX built from the whole 30-day
# smile, puts below K_0 and calls above, central strike spacing, forward F = S e^{rT}), 11.7 (B8: not built).
#
# Interpolation, paper section 4.1: "We interpolate sigma_t(m, tau) linearly first in moneyness, and then in time to
# maturity. When extrapolation is necessary, it is linear." On a fixed grid this makes the value at a point (m*, tau*)
# a fixed weighted sum of the grid values, so it is one matrix product for any number of surfaces. Our grid covers
# m in [0.8, 1.2] and tau in [30, 360] days; the paper's m = 0.75 and 1.25 and its 1-day and 1-week maturities lie
# outside it, where the rule extrapolates into a region with no data, so our forecast points stay inside the grid.
#
# VIX (CBOE white paper), applied to a constant-maturity 30-day smile, so no interpolation between two expiries:
#     VIX^2 = (2/T) e^{rT} sum_i (dK_i / K_i^2) Q(K_i) - (1/T) (F / K_0 - 1)^2,     T = 30/365,  F = S e^{rT}
# K_0 is the first strike at or below F; Q(K) is the put below K_0, the call above it and their average at K_0;
# dK_i is half the distance between the neighbouring strikes (one-sided at the two ends). Every term divides by S,
# so the formula runs on moneyness K/S and relative prices. The formula needs the smile on strikes from 20% to 250%
# of spot, far beyond the grid's 80% to 120%. Inside the grid the smile is interpolated linearly in moneyness (the
# paper's rule). Beyond it, total implied variance w = sigma^2 T is extended linearly in log-moneyness from the last
# two grid points, with the slope held between 0 and 2 in absolute value (Lee's moment formula bounds it by 2): the
# variance can never turn negative (B20) and the put wing never falls. Checked on the market surfaces: this tracks the
# CBOE VIX with correlation 0.997 and an RMSE of about 1.1 vol points over 2007-2025 (scripts/show_stage11.py).
#
# Part 1 (G6, G9): realised values always come from the RAW vendor grid (never a projected Dataset), checked; the nine calibration points
# are grid nodes (no interpolation); and the dynamics helpers: a weighted Pearson correlation across scenarios
# (rho_t), principal components of increments, and the constant-correlation band (Bonett and Wright 2000).

VIX_T = 30 / 365
VIX_STRIKES = np.unique(np.round(np.r_[np.arange(0.2, 0.6, 0.02), np.arange(0.6, 1.4, 0.005),
                                       np.arange(1.4, 2.5 + 1e-9, 0.02)], 10))


# ---- interpolation (paper section 4.1)

def linear_weights(x, x_new):
    """
    (len(x), len(x_new)) matrix A with values_new = values @ A: linear interpolation between the neighbouring grid
    points, and linear extrapolation from the first or last interval outside the grid.
    """
    x, x_new = np.asarray(x, dtype=float), np.atleast_1d(np.asarray(x_new, dtype=float))
    i = np.clip(np.searchsorted(x, x_new, side="right") - 1, 0, len(x) - 2)
    a = (x_new - x[i]) / (x[i + 1] - x[i])
    A = np.zeros((len(x), len(x_new)))
    A[i, np.arange(len(x_new))] = 1 - a
    A[i + 1, np.arange(len(x_new))] += a
    return A


def point_weights(m, tau, points):
    """
    (K, P) matrix W with values = surfaces @ W for points [(m*, tau*), ...]: linear in moneyness first, then in
    maturity, as the paper states. Surfaces use the maturity-major layout: entry j * lk + i is (m_i, tau_j).
    """
    m_new, tau_new = np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)
    Am, At = linear_weights(m, m_new), linear_weights(tau, tau_new)          # (lk, P), (lt, P)
    return (At[:, None, :] * Am[None, :, :]).reshape(len(tau) * len(m), len(points))


def interpolate(surfaces, m, tau, points):
    """Implied vol at each point for every surface: (..., K) -> (..., P)."""
    return np.asarray(surfaces, dtype=float) @ point_weights(m, tau, points)


# ---- VIX from a 30-day smile

def smile_on_strikes(sig30, m, strikes=VIX_STRIKES, T=VIX_T):
    """(B, lk) smiles on the grid moneyness -> (B, len(strikes)): linear inside the grid, variance wings outside."""
    sig30, m, M = np.asarray(sig30, dtype=float), np.asarray(m, dtype=float), np.asarray(strikes, dtype=float)
    inside = (M >= m[0]) & (M <= m[-1])
    out = np.empty((sig30.shape[0], len(M)))
    out[:, inside] = sig30 @ linear_weights(m, M[inside])
    k, w = np.log(m), sig30 ** 2 * T
    left = np.clip((w[:, 1] - w[:, 0]) / (k[1] - k[0]), -2.0, 0.0)          # w rises as k falls, at most slope 2
    right = np.clip((w[:, -1] - w[:, -2]) / (k[-1] - k[-2]), 0.0, 2.0)      # w flat or rising, at most slope 2
    L, R = M < m[0], M > m[-1]
    out[:, L] = np.sqrt((w[:, [0]] + left[:, None] * (np.log(M[L]) - k[0])) / T)
    out[:, R] = np.sqrt((w[:, [-1]] + right[:, None] * (np.log(M[R]) - k[-1])) / T)
    return out


def vix(sig30, m, r=0.0, strikes=VIX_STRIKES, T=VIX_T):
    """
    VIX level (in vol points) of each 30-day smile. sig30: (B, lk), the smile on the grid moneyness m.
    r: continuous 30-day rate, a scalar or one per smile (B,). The index pays no dividends (decision log), F = S e^{rT}.
    """
    M = np.asarray(strikes, dtype=float)
    s = smile_on_strikes(sig30, m, M, T)
    r = np.broadcast_to(np.asarray(r, dtype=float), (s.shape[0],))[:, None]
    d1 = (-np.log(M) + T * (r + s ** 2 / 2)) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    call = ndtr(d1) - M * np.exp(-r * T) * ndtr(d2)                         # C / S
    put = call - 1 + M * np.exp(-r * T)                                      # P / S, put-call parity
    f = np.exp(r * T)                                                        # F / S
    i0 = np.searchsorted(M, f[:, 0], side="right") - 1                       # K_0: first strike at or below F
    m0 = M[i0][:, None]
    Q = np.where(M < m0, put, call)
    rows = np.arange(len(i0))
    Q[rows, i0] = (put[rows, i0] + call[rows, i0]) / 2
    dM = np.empty_like(M)
    dM[1:-1] = (M[2:] - M[:-2]) / 2
    dM[0], dM[-1] = M[1] - M[0], M[-1] - M[-2]
    var = 2 / T * np.exp(r[:, 0] * T) * (dM / M ** 2 * Q).sum(axis=1) - (f[:, 0] / m0[:, 0] - 1) ** 2 / T
    return 100 * np.sqrt(var)


def vix_of_surfaces(surfaces, m, tau, r=0.0):
    """VIX of full surfaces (..., K): the 30-day smile is the first maturity block of the maturity-major layout."""
    if not np.isclose(tau[0], VIX_T):
        raise ValueError("the grid's first maturity is %.1f days, the VIX needs 30" % (tau[0] * 365))
    surfaces = np.asarray(surfaces, dtype=float)
    lead = surfaces.shape[:-1]
    sig30 = surfaces.reshape(-1, surfaces.shape[-1])[:, :len(m)]
    r = np.broadcast_to(np.asarray(r, dtype=float), lead).reshape(-1)
    return vix(sig30, m, r).reshape(lead)


# ---- realised values and the published volatility index

def load_vol_index(index):
    """The published volatility index for an equity index (SPX: VIX, NDX: VXN) as a date-indexed series, oldest first."""
    path, sheet = VOL_INDEX[index]
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    head = int(np.flatnonzero(raw.iloc[:, 0].astype(str).str.strip() == "Date")[0])
    body = raw.iloc[head + 1:, :2].dropna()
    series = pd.Series(body.iloc[:, 1].astype(float).to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(body.iloc[:, 0])))
    series = series.sort_index()
    if not series.index.is_unique:
        raise ValueError("%s: repeated dates" % sheet)
    return series.rename(sheet)


def realised(market, rate=0.0, dates=None):
    """
    What each forecast is compared with, all on the PREDICTED day market.dates[i] (B21):
    the market surface, the daily log-return, the index level, and the VIX computed from the market surface.
    market: the raw-grid Dataset. dates: the simulated dates, checked equal.
    rate: the same rates as the simulation, (n_days, lt) or a scalar; the VIX uses the 30-day one.
    """
    sigma = market_sigma(market, dates)
    r30 = np.broadcast_to(np.asarray(rate, dtype=float), (len(market.dates), len(market.tau)))[:, 0]
    return {"sigma": sigma, "return": np.log(market.spot / market.spot_prev), "level": market.spot,
            "vix_from_surface": vix_of_surfaces(sigma, market.m, market.tau, r30)}


def band_stats(actual, mean, lo, hi):
    """Share of days with the actual value inside [lo, hi] (in %), median band width, mean absolute error of the mean."""
    actual = np.asarray(actual, dtype=float)
    return {"inside band %": 100 * ((actual >= lo) & (actual <= hi)).mean(),
            "median band width": float(np.median(np.asarray(hi) - np.asarray(lo))),
            "mean abs error of mean": float(np.abs(np.asarray(mean) - actual).mean())}


def extras_for(test, sim, points=(), with_vix=True):
    """
    Per-scenario quantities for daily_statistics(extras=...), each a function (returns, sigma, days) -> (D, N):
    the index level S_{d-1} e^{R_hat}, the VIX of each scenario surface, and the implied vol at each point.
    """
    out = {"index level": lambda r_hat, sigma, days: test.spot_prev[days, None] * np.exp(r_hat)}
    if with_vix:
        out["VIX"] = lambda r_hat, sigma, days: vix_of_surfaces(
            sigma, test.m, test.tau, np.repeat(sim.rate[days, 0][:, None], sigma.shape[1], axis=1))
    for p in points:
        W = point_weights(test.m, test.tau, [p])[:, 0]
        out["vol m=%.2f, %dd" % (p[0], round(p[1] * 365))] = (lambda W: lambda r_hat, sigma, days: sigma @ W)(W)
    return out


# ---- Part 1: calibration points, and the dynamics helpers (brief 8.2 and 8.3)

CALIBRATION_POINTS = tuple((m, d) for d in (30, 90, 360) for m in (0.90, 1.00, 1.10))   # (moneyness, days)
ATM_1M = (1.00, 30)


def node_index(m, tau, points):
    """
    Position in the maturity-major vector (j * lk + i) of each (moneyness, days) point, which must be a GRID NODE:
    the calibration points are read off the grid, never interpolated. (The 120% column is excluded because
    Bloomberg sometimes fills it flat, the 270-day row because it is interpolated in the source data.)
    """
    out = []
    for mm, days in points:
        i, j = np.flatnonzero(np.isclose(m, mm)), np.flatnonzero(np.isclose(np.asarray(tau) * 365, days))
        if len(i) != 1 or len(j) != 1:
            raise ValueError("(%.2f, %d days) is not a grid node" % (mm, days))
        out.append(int(j[0]) * len(m) + int(i[0]))
    return np.array(out)


def weighted_corr(x, y, w):
    """
    Weighted Pearson correlation across scenarios, one per day: x, y, w of shape (n_days, N), weights summing to 1.
    rho_t of the brief (the paper's Fig. 17 quantity) with x = R_hat, y = dg_hat at (1.00, 30 days).
    """
    x, y, w = (np.asarray(a, dtype=float) for a in (x, y, w))
    xc = x - (w * x).sum(axis=1, keepdims=True)
    yc = y - (w * y).sum(axis=1, keepdims=True)
    return (w * xc * yc).sum(axis=1) / np.sqrt((w * xc ** 2).sum(axis=1) * (w * yc ** 2).sum(axis=1))


def correlation_band(rho_bar, n, level=0.95):
    """
    Constant-correlation band (paper section 4.2.4; Bonett and Wright 2000): tanh(atanh rho_bar -+ z / sqrt(n - 3)),
    z the two-sided normal quantile. n: N for unweighted cells, the median daily ESS for re-weighted ones.
    """
    if n <= 3:                                   # the Fisher-z interval needs n > 3 (an ESS collapsed below 4)
        return np.nan, np.nan
    z = norm.ppf(0.5 + level / 2)
    half = z / np.sqrt(n - 3)
    return np.tanh(np.arctanh(rho_bar) - half), np.tanh(np.arctanh(rho_bar) + half)


def principal_components(cov, k=3):
    """Top-k eigenvectors (columns) and explained-variance shares of covariance matrices (..., K, K)."""
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals, axis=-1)[..., ::-1]
    vals = np.take_along_axis(vals, order, axis=-1)
    vecs = np.take_along_axis(vecs, order[..., None, :], axis=-1)
    share = vals / vals.sum(axis=-1, keepdims=True)
    return vecs[..., :k], share[..., :k]


def correlation(cov):
    """Correlation matrix from a covariance matrix (..., K, K)."""
    sd = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    return cov / (sd[..., :, None] * sd[..., None, :])


CAL_QS = (0.01, 0.025, 0.5, 0.975, 0.99)     # the quantile levels recorded for every calibration variable
CAL_NAMES = tuple("sigma m=%.2f %dd" % p for p in CALIBRATION_POINTS) + ("return", "surface-VIX")


def increments(r, dg, dlogvix):
    """
    The vector whose covariance drives PCA and the correlation tables, per day (and per series):
    z = (daily return, the 42 log-vol increments, the change in log surface-VIX), 44 entries.
    """
    return np.concatenate([np.asarray(r)[..., None], np.asarray(dg), np.asarray(dlogvix)[..., None]], axis=-1)


def dynamics_from_cov(cov, data_pcs, atm, own_pcs=None):
    """
    Everything Tables 4-7 need from the covariance of z = increments(...) (..., 44, 44), given the DATA's top three
    eigenvectors data_pcs (K, 3) and the position atm of (1.00, 30 days) in the K increments:
      share     (..., 3)    explained-variance shares of the top three components of the increments' own covariance
      inner     (..., 3)    |<v_k data, v_k own>|, k = 1, 2, 3
      v1        (..., K)    the own first eigenvector, sign-aligned to the data's
      ret_pc    (..., 3)    corr(return, increments . v_k data): Table 6 format
      corrmat   (..., 4, 4) corr of (return, level-factor projection, d log sigma ATM 1M, d log surface-VIX): Table 7
    """
    cov = np.asarray(cov, dtype=float)
    K = data_pcs.shape[0]
    gg = cov[..., 1:K + 1, 1:K + 1]
    vecs, share = principal_components(gg, 3) if own_pcs is None else own_pcs
    inner = np.abs(np.einsum("...kj,kj->...j", vecs, data_pcs))
    sign = np.sign(np.einsum("...k,k->...", vecs[..., :, 0], data_pcs[:, 0]))
    v1 = vecs[..., :, 0] * sign[..., None]
    var_r = cov[..., 0, 0]
    cov_r_pc = np.einsum("...k,kj->...j", cov[..., 0, 1:K + 1], data_pcs)
    var_pc = np.einsum("kj,...kl,lj->...j", data_pcs, gg, data_pcs)
    ret_pc = cov_r_pc / np.sqrt(var_r[..., None] * var_pc)
    L = np.zeros((4, K + 2))
    L[0, 0] = 1.0
    L[1, 1:K + 1] = data_pcs[:, 0]
    L[2, 1 + atm] = 1.0
    L[3, K + 1] = 1.0
    corrmat = correlation(np.einsum("ik,...kl,jl->...ij", L, cov, L))
    return {"share": share, "inner": inner, "v1": v1, "ret_pc": ret_pc, "corrmat": corrmat}


def market_reference(market):
    """
    The data side of every Part 1 metric for one block, from the RAW market Dataset (G9):
      phi, phi_parts (n, 3), viol_loc (n, lk, lt) 0/1   Phi of the market surface on each predicted day, day's rates
      cal_realised (n, 11)                               the nine calibration vols, the return, the surface-VIX
      vix_cond (n,)                                      surface-VIX of the condition day's market surface
      z (n, 44), cov (44, 44)                            realised increments (return, d log-vol, d log VIX)
      pcs (K, 3), share (3,), ret_pc (3,), corrmat (4, 4)   the data's PCA and correlation tables
    market must carry rates (data_preprocessing(..., rates=...)).
    """
    if market.rates is None or market.cond_rates is None:
        raise ValueError("the market Dataset needs its rates attached")
    m, tau = market.m, market.tau
    sigma = market_sigma(market)
    p, nodes = scenario_phi(sigma[:, None, :], m, tau, market.rates, with_nodes=True)
    cal = node_index(m, tau, CALIBRATION_POINTS)
    atm = int(node_index(m, tau, [ATM_1M])[0])
    vix_d = vix_of_surfaces(sigma, m, tau, market.rates[:, 0])
    vix_c = vix_of_surfaces(np.exp(market.past_log_surface), m, tau, market.cond_rates[:, 0])
    ret = np.log(market.spot / market.spot_prev)
    z = increments(ret, market.target[:, 1:], np.log(vix_d) - np.log(vix_c))
    cov = np.cov(z, rowvar=False)
    pcs, share = principal_components(cov[1:-1, 1:-1], 3)
    pcs = pcs * np.sign(pcs.sum(axis=0))[None, :]                 # a fixed sign: the level factor loads positively
    out = {"dates": market.dates, "phi": p.total[:, 0],
           "phi_parts": np.stack([p.calendar[:, 0], p.call[:, 0], p.butterfly[:, 0]], axis=1),
           "viol_loc": nodes[:, 0].astype(float), "cal_realised": np.column_stack([sigma[:, cal], ret, vix_d]),
           "vix_cond": vix_c, "z": z, "cov": cov, "pcs": pcs, "atm": atm, "cal_index": cal}
    dyn = dynamics_from_cov(cov, pcs, atm, own_pcs=(pcs, share))
    out.update(share=share, ret_pc=dyn["ret_pc"], corrmat=dyn["corrmat"])
    return out


# =====================================================================================================================
# 4. The simulate and evaluate steps (was pipeline.py)
# =====================================================================================================================

EVAL_RULES = ("raw", "max_penalty")


def simulate_step(index="SPX", block="val", names=tuple(TRAINING_CONFIGS), seeds=SEEDS, allow_test=False,
                  n_scenarios=None, device=None, progress=True):
    """One simulation per trained run on the block: N scenarios per day with the shared sim_seed (G10)."""
    device = device or get_device()
    for name, seed, folder in _trained(index, names, seeds):
        run = load_run(folder, device)
        inputs = run.split.block(block, allow_test)
        N = n_scenarios or run.config.n_scenarios
        sig = text_hash("%s|%s|%d|%d" % (run.info.get("signature", folder), block, N, run.config.sim_seed))
        path = sim_path(index, block, name, seed)
        if _done(path, "meta_signature", sig, "simulation %s seed %d" % (name, seed)):
            log("skip simulation %s seed %d: done" % (name, seed))
            continue
        log("simulate %s seed %d on %s: %d days x %d scenarios" % (name, seed, block, len(inputs.dates), N))
        sim = simulate(run.gen, inputs, N, run.config.sim_seed, rate=inputs.rates, progress=progress)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_simulation(path, sim, signature=sig, block=block, run=folder)


# ---- Step 4: evaluation (section 8)

def market_step(index, block, allow_test=False, overrides=None):
    """The data side of every metric for a block (results/<INDEX>/<block>/market.npz): the raw vendor grid."""
    path = os.path.join(block_dir(index, block), "market.npz")
    config = training_config("raw_phiout", seed=SEEDS[0], **(overrides or {}))
    market = prepare_all(index, config).market.block(block, allow_test)
    ref = market_reference(market)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: (ns(v) if k == "dates" else v) for k, v in ref.items()},
                        cal_names=np.array(CAL_NAMES), m=market.m, tau=market.tau, rate_source=config.rate_source)
    return ref


def summary_row(met):
    """Headline numbers of one cell and seed (per-cell CSV)."""
    nine = slice(0, 9)
    return {"mean_phi": met["phi_mean"].mean(), "std_phi": met["phi_mean"].std(), "median_phi": np.median(met["phi_mean"]),
            "viol_share": met["viol_share"].mean(),
            "cov95_nine_points": 100 * met["inside_95"][:, nine].mean(),
            "cov95_return": 100 * met["inside_95"][:, 9].mean(), "cov95_vix": 100 * met["inside_95"][:, 10].mean(),
            "rho_bar": met["rho_bar"], "pc1_inner": met["pca_inner"][:, 0].mean(),
            "median_ess": np.median(met["ess"])}


def evaluate_step(index="SPX", block="val", names=tuple(TRAINING_CONFIGS), seeds=SEEDS, allow_test=False,
                  b_pca=B_PCA, device=None, progress=True, overrides=None):
    """Every section-8 metric for each trained run, unweighted and re-weighted (two cells per run), per seed."""
    device = device or get_device()
    ref = None
    for name, seed, folder in _trained(index, names, seeds):
        path = sim_path(index, block, name, seed)
        if not os.path.exists(path):
            log("no simulation for %s seed %d on %s: run the simulate step first" % (name, seed, block))
            continue
        sim, meta = load_simulation(path)
        sig = text_hash("%s|%d" % (meta["signature"], b_pca))
        cells = [cell_of(name, rule) for rule in EVAL_RULES]
        if all(_done(metrics_path(index, block, c, seed), "signature", sig, "metrics %s seed %d" % (c, seed))
               for c in cells):
            log("skip metrics %s seed %d: done" % ("/".join(cells), seed))
            continue
        if ref is None:
            ref = market_step(index, block, allow_test, overrides)
        run = load_run(folder, device)
        inputs, market = run.split.block(block, allow_test), run.market.block(block, allow_test)
        log("metrics %s seed %d (cells %s) on %s" % (name, seed, "/".join(cells), block))
        out = daily_metrics(run.gen, inputs, market, sim, EVAL_RULES, b_pca, progress=progress, reference=ref)
        for rule, met in out.items():
            p = metrics_path(index, block, cell_of(name, rule), seed)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez_compressed(p, dates=ns(sim.dates), signature=sig, rule=rule, training_config=name, seed=seed,
                                **met)
    write_summaries(index, block, seeds)


def load_metrics(index, block, cell, seed):
    p = metrics_path(index, block, cell, seed)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def write_summaries(index, block, seeds=SEEDS):
    """results/<INDEX>/<block>/<cell>/summary.csv: one row per seed, then the mean, min and max over seeds."""
    for cell in CELLS:
        rows = {}
        for seed in seeds:
            met = load_metrics(index, block, cell, seed)
            if met is not None:
                rows["seed %d" % seed] = summary_row(met)
        if not rows:
            continue
        df = pd.DataFrame(rows).T
        df = pd.concat([df, df.mean().rename("mean over seeds").to_frame().T,
                        df.min().rename("min over seeds").to_frame().T, df.max().rename("max over seeds").to_frame().T])
        df.to_csv(os.path.join(block_dir(index, block), cell, "summary.csv"))
