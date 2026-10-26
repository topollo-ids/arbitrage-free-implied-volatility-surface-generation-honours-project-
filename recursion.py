"""
Part 2, Part A: the 21-day free-running recursion: state update, per-cloud metrics, rollout, noise table,
market margins and the run steps. Part 1 never imports this module.

Contents
    1. The state update (was simulate.py)
    2. Per-cloud metrics (was rec_metrics.py)
    3. The rollout (was recursion.py)
    4. Market margins (was rec_inference.py)
    5. The run steps (was pipeline.py)
"""

from dataclasses import dataclass
import hashlib
import json
import os
import time

import numpy as np
import torch

from .common import (
    CODE_DIR, INDEX_FILES, RATES_FILE, check_inputs, file_hash, get_device, library_versions, log, run_path,
    text_hash, training_config)
from .preprocessing import N_SCALARS, WINDOW, prepare_all
from .training import load_run
from .postprocessing import (
    ATM_1M, CALIBRATION_POINTS, TOL_PHI, day_key, day_noise, generate_full, market_reference, market_sigma,
    node_index, scenario_phi, step, weighted_corr, weighted_quantile)


# =====================================================================================================================
# 1. The state update (was simulate.py)
# =====================================================================================================================

# ---- Part 2 hooks (brief section 14): the state update of a recursion. Not used in Part 1.

@dataclass
class State:
    """What the next condition is built from: the condition (3 + K) and the last daily returns (oldest first)."""
    condition: np.ndarray     # (3 + K,) [R_c x ret_scale, R_c-1 x ret_scale, realised vol, g_c]
    returns: np.ndarray       # (WINDOW + 1,) daily log-returns ending on the condition day c
    ret_scale: float
    rv_window: str            # "paper" or "code" (F5)


def initial_state(data, ds, j):
    """The observed state behind sample j of a Dataset: its condition and the 22 daily returns ending on day c."""
    c = data.dates.get_loc(ds.cond_dates[j])
    if c < WINDOW + 1:
        raise ValueError("need %d returns before the condition day" % (WINDOW + 1))
    returns = np.diff(np.log(data.spot[c - WINDOW - 1:c + 1]))
    return State(ds.condition[j].copy(), returns, ds.ret_scale, ds.rv_window)


def next_state(state, r_hat, dg_hat):
    """
    The condition for the next day from one simulated step: the two return slots shift; the 21-return realised-vol
    buffer takes the simulated return, converted back to daily units (r_hat / ret_scale); g_t+1 = g_t + dg_hat.
    r_hat: the generator's return output (scaled, as step() gives it); dg_hat: (K,).
    """
    daily = float(r_hat) / state.ret_scale
    returns = np.append(state.returns[1:], daily)
    window = returns[-WINDOW:] if state.rv_window == "paper" else returns[-WINDOW - 1:-1]   # F5: "code" is one day stale
    rv = np.sqrt(252 / WINDOW * (window ** 2).sum())
    g = state.condition[N_SCALARS:] + np.asarray(dg_hat, dtype=float)
    condition = np.concatenate([[state.ret_scale * daily, state.condition[0], rv], g])
    return State(condition, returns, state.ret_scale, state.rv_window)


def batched_next_state(cond, buf, r_hat, dg_hat, ret_scale, rv_window):
    """
    next_state for P paths at once (Part 2, Part A). cond (P, 3+K) float64, buf (P, WINDOW+1) float64 daily returns,
    oldest first; r_hat (P,) the generator's return output (scaled, as step() gives it), dg_hat (P, K).
    Returns (new condition (P, 3+K), new buffer (P, WINDOW+1)), the same numbers next_state gives path by path.
    """
    cond, buf = np.asarray(cond, dtype=float), np.asarray(buf, dtype=float)
    daily = np.asarray(r_hat, dtype=float) / ret_scale
    buf_new = np.concatenate([buf[:, 1:], daily[:, None]], axis=1)
    window = buf_new[:, -WINDOW:] if rv_window == "paper" else buf_new[:, -WINDOW - 1:-1]
    rv = np.sqrt(252 / WINDOW * (window ** 2).sum(axis=1))
    g = cond[:, N_SCALARS:] + np.asarray(dg_hat, dtype=float)
    new = np.concatenate([(ret_scale * daily)[:, None], cond[:, :1], rv[:, None], g], axis=1)
    return new, buf_new


# =====================================================================================================================
# 2. Per-cloud metrics (was rec_metrics.py)
# =====================================================================================================================

# Part 2, Part A: the metrics of one cloud (brief section 6).
#
# A cloud is the set of valid paths of one (seed, start, horizon, run type) for one target day. Every weight is uniform
# (raw, unweighted), and every definition is the Part 1 one, reused rather than rewritten:
#     phi_mean    mean of scenario_phi over the cloud                            (reweight.daily_metrics: phi_mean)
#     viol_share  share of paths with Phi > reweight.TOL_PHI                      (viol_share)
#     phi_parts   mean calendar, call-spread and butterfly terms                  (phi_parts)
#     inside/below/above/width   the 95% interval from weighted_quantile at 2.5% and 97.5% with equal weights, the
#                 realised market value inside it / below it / above it, and its width (inside_95, below_025,
#                 above_975, width_95), at the nine calibration nodes and for the daily log-return (10 columns)
#     rho         weighted_corr(R_hat, dg_hat at ATM 1M, uniform)                 (rho): the model's one-step
#                 increment at that step, which is "the dynamics at horizon h"
#     q           the 2.5 / 50 / 97.5 % quantiles of the same ten variables (insurance for the remedy comparison)
#     viol_loc    per-node violation frequency (lk, lt), asked for at h = 21 only
# The VIX column of Part 1 is left out of the series (brief 6).
#
# A cloud with no valid path returns NaN everywhere with ok = False; it never raises.

N_VARS = len(CALIBRATION_POINTS) + 1               # nine nodes, then the return
QS = (0.025, 0.5, 0.975)                           # recorded quantile levels; the interval uses the outer two
VAR_NAMES = tuple("sigma m=%.2f %dd" % p for p in CALIBRATION_POINTS) + ("return",)

SCALARS = {"n_valid": (), "phi_mean": (), "viol_share": (), "rho": ()}
VECTORS = {"phi_parts": (3,), "inside": (N_VARS,), "below": (N_VARS,), "above": (N_VARS,), "width": (N_VARS,),
           "q": (len(QS), N_VARS)}
SHAPES = {**SCALARS, **VECTORS}                     # the per-cloud record, name -> shape


def empty_cloud():
    """The record of a cloud that cannot be scored: NaN, n_valid = 0."""
    out = {k: np.full(s, np.nan) for k, s in SHAPES.items()}
    out["n_valid"] = np.array(0.0)
    return out


class Realised:
    """
    What each cloud is scored against, from the RAW market block (G9): the nine calibration vols and the daily
    log-return on every predicted day, plus the grid and the node positions. market must carry its rates.
    """

    def __init__(self, market):
        self.m, self.tau = market.m, market.tau
        self.cal = node_index(self.m, self.tau, CALIBRATION_POINTS)
        self.atm = int(node_index(self.m, self.tau, [ATM_1M])[0])
        sigma = market_sigma(market)
        ret = np.log(market.spot / market.spot_prev)
        self.values = np.column_stack([sigma[:, self.cal], ret])       # (n, 10)
        self.rates = np.asarray(market.rates, dtype=float)             # (n, lt): the target day's rates
        self.dates = market.dates


def cloud_metrics(sigma, r, dg_atm, real, j, with_nodes=False):
    """
    Metrics of one cloud scored on target sample j. sigma (n, K) simulated surfaces, r (n,) daily log-returns,
    dg_atm (n,) the model's change in log-vol at the 1M ATM node (all of the valid paths only, float64); real: a
    Realised (values (n_days, 10), rates (n_days, lt), grid m, tau, node positions cal).
    Returns a dict of SHAPES (and 'viol_loc' (lk, lt) if with_nodes) and 'ok'.
    """
    n = len(r)
    out = empty_cloud()
    out["ok"] = False
    if with_nodes:
        out["viol_loc"] = np.full((len(real.m), len(real.tau)), np.nan)
    if n == 0:
        return out
    cal = real.cal
    p, nodes = scenario_phi(sigma[None], real.m, real.tau, np.asarray(real.rates[j], float)[None], with_nodes=True)
    phi = p.total[0]
    w = np.full((1, n), 1.0 / n)
    X = np.concatenate([sigma[:, cal], r[:, None]], axis=1)[None]         # (1, n, 10)
    q = weighted_quantile(X, w, QS)[:, 0]                                  # (3, 10)
    lo, hi = q[0], q[-1]
    v = np.asarray(real.values[j])
    out.update(n_valid=np.array(float(n)), phi_mean=phi.mean(), viol_share=float((phi > TOL_PHI).mean()),
               phi_parts=np.array([p.calendar[0].mean(), p.call[0].mean(), p.butterfly[0].mean()]),
               inside=((v >= lo) & (v <= hi)).astype(float), below=(v < lo).astype(float),
               above=(v > hi).astype(float), width=hi - lo, q=q)
    if n >= 3:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["rho"] = weighted_corr(r[None], dg_atm[None], w)[0]       # NaN if a series is constant
    if with_nodes:
        out["viol_loc"] = nodes[0].astype(float).mean(axis=0)
    out["ok"] = True
    return out


def stack(clouds):
    """List of cloud dicts -> dict of arrays with a leading axis (the ones in SHAPES, and viol_loc if present)."""
    keys = [k for k in clouds[0] if k in SHAPES or k in ("viol_loc", "n_oob")]
    return {k: np.stack([np.asarray(c[k], dtype=float) for c in clouds]) for k in keys}


# =====================================================================================================================
# 3. The rollout (was recursion.py)
# =====================================================================================================================

# Part 2, Part A: the free-running (recursive) simulation and its one-step benchmark (brief sections 4 and 5).
#
# For a trained network, a starting sample j0 and 1,000 paths, the free-running run feeds each step's simulated return
# and log-vol increment back as the next step's condition, for 21 steps. Step k predicts the day of sample j0 + k - 1.
# Step 1 uses the OBSERVED condition of sample j0, so at h = 1 it is the one-step model: that is the identity check.
# The benchmark for a target day is the ordinary one-step model: the observed condition of that day's sample.
#
# What is shared with Part 1 and reproduced, not re-implemented:
#   - the noise is simulate.day_noise(sim_seed, day_key(date), 10,000, 32), first 1,000 rows (NoiseTable), a function of
#     the calendar date alone: the same for every start, seed, run type and (later) every remedied model;
#   - one network evaluation is simulate.step; the state update is simulate.batched_next_state (tested against the
#     scalar simulate.next_state); the benchmark is simulate.generate_full with n_draw = 10,000;
#   - the metrics are rec_metrics.cloud_metrics (Part 1 definitions).
#
# Nothing is repaired, clipped, filtered or re-weighted inside the recursion. The only intervention is the validity
# mask (brief 4.5): a path is valid at step k if its 42 vols are finite and inside [vol_lo, vol_hi] and its return is
# finite; invalid is absorbing. A dead path is held at its last valid condition so the arithmetic stays finite, and is
# ignored by every metric. This is bookkeeping, not repair.
#
# Part B hook: rollout(..., post_step=f) calls f(k, cond, r_hat, dg_hat, valid, ctx) -> (cond, extra) for k >= 2 on the
# condition that is about to feed step k (built from step k-1's output r_hat, dg_hat). The first step is never touched.
#
# Part B remedies (both None by default, and then the rollout is the Part A one, bit for bit):
#   gen_late  the network used at steps k >= 2 (the network trained on its own rollouts); step 1 always uses gen;
#   clip      (lo, hi), shape (K,): the bounded step. At k >= 2 the log-vol increment is clipped node by node right
#             after the network call, so the scored surface, the validity check, rho and the next input all use the
#             clipped value; the return is never clipped. StepOut.clip_bind marks the (path, node) entries it bound.
#   repair    a projection.Projector (second remedy): at k >= 2, after the clip, every alive, finite surface with
#             Phi > 0 (target day's rates) is projected onto the arbitrage-free set; the step actually taken,
#             dg_eff = log(sigma) - g at the nodes the projection changed (dg elsewhere, bit for bit), is what is scored
#             and fed back. StepOut.repair_info holds the projection's records for the step.

REC_VERSION = "partA-1"
TOL_IDENT_SIGMA = 1e-5         # brief 5: float32-level comparison of sigma (and of dg)
TOL_IDENT_RET = 1e-6           # float64 returns
FIXED_NOISE_DATES = ("2018-01-02", "2019-07-15", "2021-12-31")   # dates whose noise hashes are stored in tests/


class IdentityError(RuntimeError):
    """The free-running run at h = 1 differs from the benchmark: nothing else may be reported (brief 5)."""


# ---- noise (brief 4.3)

class NoiseTable:
    """Z(d) = day_noise(sim_seed, day_key(d), n_draw, noise_dim)[:n_paths], built lazily and cached for the job."""

    def __init__(self, sim_seed, n_paths=1000, n_draw=10_000, noise_dim=32):
        self.sim_seed, self.n_paths, self.n_draw, self.noise_dim = int(sim_seed), n_paths, n_draw, noise_dim
        self._cache = {}

    def get(self, date):
        key = day_key(date)
        if key not in self._cache:
            self._cache[key] = day_noise(self.sim_seed, key, self.n_draw, self.noise_dim)[:self.n_paths].clone()
        return self._cache[key]


def noise_hash(z):
    """First 16 hex digits of the SHA-256 of a float32 noise block: identifies it across machines and torch versions."""
    return hashlib.sha256(np.ascontiguousarray(z.numpy().astype(np.float32)).tobytes()).hexdigest()[:16]


def noise_hashes(table, dates):
    return {str(d): noise_hash(table.get(d)) for d in dates}


# ---- starting days

def start_indices(n_samples, cfg):
    """j0 = offset, offset + stride, ... with j0 + horizon - 1 <= n_samples - 1 (brief 4.1)."""
    last = n_samples - cfg.horizon
    return np.arange(cfg.start_offset, last + 1, cfg.stride)


# ---- the rollout

@dataclass
class StepOut:
    k: int              # horizon, 1..H
    j: int              # target sample j0 + k - 1
    r: np.ndarray       # (P,) daily log-returns (float64)
    dg: np.ndarray      # (P, K) the network's log-vol increments
    sigma: np.ndarray   # (P, K) simulated surfaces exp(g + dg)
    valid: np.ndarray   # (P,) bool: valid at this step and every earlier one (absorbing)
    extra: object = None
    clip_bind: np.ndarray = None   # (P, K) bool: where the bounded step bound (Part B); None when no clip acted
    repair_info: dict = None       # the projection's records for this step (second remedy); None when it did not act


@torch.no_grad()
def rollout(gen, ds, data, j0, noise, cfg, post_step=None, ctx=None, gen_late=None, clip=None, repair=None):
    """
    Generator of StepOut for k = 1..cfg.horizon, free-running from the observed condition of sample j0 of the inputs
    Dataset ds (data: the SurfaceData behind it, for the 22 observed daily returns). The state is carried in float64
    and cast to float32 only at the network input. See the module docstring for post_step, gen_late, clip, repair.
    """
    P, H = cfg.n_paths, cfg.horizon
    st = initial_state(data, ds, j0)
    if not (np.isclose(st.returns[-1] * ds.ret_scale, ds.condition[j0, 0], rtol=1e-9, atol=1e-12)
            and np.isclose(st.returns[-2] * ds.ret_scale, ds.condition[j0, 1], rtol=1e-9, atol=1e-12)):
        raise AssertionError("the return buffer does not match the condition of sample %d" % j0)
    if ds.rv_window == "paper":
        rv = np.sqrt(252 / WINDOW * (st.returns[-WINDOW:] ** 2).sum())
        if not np.isclose(rv, ds.condition[j0, 2], rtol=1e-9, atol=1e-12):
            raise AssertionError("the return buffer does not reproduce the realised vol of sample %d" % j0)
    device = next(gen.parameters()).device
    gen.eval()
    if gen_late is not None:
        if next(gen_late.parameters()).device != device:
            raise ValueError("gen_late must be on the same device as gen")
        gen_late.eval()
    if clip is not None:
        lo, hi = (np.asarray(x, dtype=float) for x in clip)
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
        net = gen if (k == 1 or gen_late is None) else gen_late            # step 1 is always the control's step
        r_scaled, dg = step(net, torch.as_tensor(cond, dtype=torch.float32, device=device), z)
        r_scaled = r_scaled.double().cpu().numpy()
        dg = dg.double().cpu().numpy()
        bind = None
        if clip is not None and k >= 2:                                    # the bounded step: never at k = 1
            bind = (dg < lo) | (dg > hi)
            dg = np.clip(dg, lo, hi)
        r = r_scaled / ds.ret_scale
        rinfo = None
        with np.errstate(over="ignore", invalid="ignore"):
            sigma = np.exp(cond[:, N_SCALARS:] + dg)
            if repair is not None and k >= 2:                              # the projection: never at k = 1
                sigma, rinfo = repair(sigma, valid, ds.rates[j])
                ch = rinfo["changed"]
                dg = np.where(ch, np.log(sigma) - cond[:, N_SCALARS:], dg)  # the step actually taken
            ok = (np.isfinite(r) & np.isfinite(sigma).all(axis=1)
                  & (sigma >= cfg.vol_lo).all(axis=1) & (sigma <= cfg.vol_hi).all(axis=1))
        valid = valid & ok
        yield StepOut(k, j, r, dg, sigma, valid.copy(), extra, bind, rinfo)
        with np.errstate(over="ignore", invalid="ignore"):
            new, new_buf = batched_next_state(cond, buf, r_scaled, dg, ds.ret_scale, ds.rv_window)
        cond = np.where(valid[:, None], new, cond)          # a dead path is held at its last valid condition
        buf = np.where(valid[:, None], new_buf, buf)
        prev = (r_scaled, dg)


# ---- one trained network, ready to roll

class Arena:
    """One trained network with its inputs, realised values and noise: everything run_start / bench_clouds need."""

    def __init__(self, run, cfg, noise=None, allow_test=False, check_sim_seed=True, gen_late=None, clip=None,
                 repair=None):
        self.gen_late, self.clip, self.repair = gen_late, clip, repair  # Part B remedies; None = Part A
        if check_sim_seed and run.config.sim_seed != cfg.sim_seed:     # off only for the pilot's second noise draw
            raise ValueError("the run's sim_seed %d is not the recursion sim_seed %d (G10)"
                             % (run.config.sim_seed, cfg.sim_seed))
        self.run, self.cfg, self.gen = run, cfg, run.gen
        self.ds = run.split.block(cfg.block, allow_test)           # the test block is refused unless allow_test
        self.market = run.market.block(cfg.block, allow_test)
        if not self.ds.dates.equals(self.market.dates):
            raise ValueError("inputs and market cover different days")
        self.real = Realised(self.market)
        self.noise = noise or NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, run.config.noise_dim)

    @property
    def n(self):
        return len(self.ds.dates)

    def cloud(self, j, sigma, r, dg, with_nodes=False):
        """Metrics of the cloud made of the given (already selected) paths, scored on target sample j."""
        return cloud_metrics(sigma, r, dg[:, self.real.atm], self.real, j, with_nodes)


def benchmark_full(arena, days):
    """The raw one-step outputs for sample(s) `days`: simulate.generate_full with the Part 2 noise (n_draw = 10,000)."""
    c = arena.cfg
    return generate_full(arena.gen, arena.ds, np.atleast_1d(days), c.n_paths, c.sim_seed, n_draw=c.n_draw)


def bench_clouds(arena, days, chunk=8):
    """
    One-step benchmark metrics for each sample in `days`: arrays with a leading axis over days, plus 'days' and
    'n_oob' (finite paths outside the validity bounds: reported, never masked; only non-finite paths are dropped).
    """
    clouds, n_oob = [], []
    c = arena.cfg
    for a in range(0, len(days), chunk):
        dd = np.asarray(days[a:a + chunk])
        r, dg, sigma = benchmark_full(arena, dd)
        for i, j in enumerate(dd):
            fin = np.isfinite(r[i]) & np.isfinite(sigma[i]).all(axis=1)
            oob = fin & ~((sigma[i] >= c.vol_lo).all(axis=1) & (sigma[i] <= c.vol_hi).all(axis=1))
            clouds.append(arena.cloud(j, sigma[i][fin], r[i][fin], dg[i][fin], with_nodes=True))
            n_oob.append(float(oob.sum()))
    out = stack(clouds)
    out["n_oob"] = np.array(n_oob)
    out["days"] = np.asarray(days)
    return out


def run_start(arena, j0, post_step=None, ctx=None):
    """
    The free-running metrics of one start: arrays with a leading axis over horizons 1..H (stack of cloud dicts),
    'viol_loc21' (the h = H violation-node frequencies), the h = 1 identity differences against generate_full, and
    the seconds taken. Raises IdentityError if h = 1 differs from the benchmark outputs.
    """
    c, t0 = arena.cfg, time.time()
    clouds, ident, share, share_any = [], None, [], []
    proj = {k: [] for k in PROJ_KEYS}
    for out in rollout(arena.gen, arena.ds, arena.run.data, j0, arena.noise, c, post_step, ctx,
                       gen_late=arena.gen_late, clip=arena.clip, repair=getattr(arena, "repair", None)):
        v = out.valid
        last = out.k == c.horizon
        clouds.append(arena.cloud(out.j, out.sigma[v], out.r[v], out.dg[v], with_nodes=last))
        if out.k == 1:
            ident = identity_diffs(arena, j0, out)
        if out.clip_bind is None:                                     # no clip at this step (k = 1, or no remedy)
            share.append(0.0)
            share_any.append(0.0)
        elif v.any():                                                 # Part B: shares over the valid paths
            share.append(float(out.clip_bind[v].mean()))
            share_any.append(float(out.clip_bind[v].any(axis=1).mean()))
        else:
            share.append(np.nan)
            share_any.append(np.nan)
        for k, v in proj_record(out.repair_info).items():
            proj[k].append(v)
    rec = stack(clouds[:-1] + [{k: v for k, v in clouds[-1].items() if k != "viol_loc"}])
    rec["viol_loc21"] = clouds[-1]["viol_loc"]
    rec.update(ident)
    rec["clip_share"], rec["clip_any"] = np.array(share), np.array(share_any)
    if getattr(arena, "repair", None) is not None:
        rec.update({k: np.array(v, dtype=float) for k, v in proj.items()})
    rec["seconds"] = np.array(time.time() - t0)
    check_identity(rec, j0)
    return rec


PROJ_KEYS = ("proj_share", "proj_dvol_mean", "proj_dvol_max", "proj_nodes_mean", "proj_out_of_range", "proj_retries",
             "phi_post_max")


def proj_record(info):
    """
    The projection's per-cloud records (second remedy), zeros when it did not act (k = 1, or no projection):
    share of the paths alive before the step that were projected; over projected paths that stayed representable,
    the mean and the maximum of sum_i |sigma_i - sigma_pre_i| and the largest single-node change, in VOLATILITY POINTS
    (x 100); the mean number of nodes changed by more than 1e-8; out-of-range paths; re-solves; max Phi after it.
    """
    if info is None:
        return dict.fromkeys(PROJ_KEYS, 0.0)
    p, oor = info["projected"], info["oor"]
    good = p & ~oor
    return {"proj_share": p.sum() / info["n_alive"] if info["n_alive"] else np.nan,
            "proj_dvol_mean": 100 * info["dvol_sum"][good].mean() if good.any() else 0.0,
            "proj_dvol_max": 100 * info["dvol_max"][good].max() if good.any() else 0.0,
            "proj_nodes_mean": info["nodes"][good].mean() if good.any() else 0.0,
            "proj_out_of_range": float(oor.sum()), "proj_retries": float(info["retries"]),
            "phi_post_max": float(info["phi_post_max"])}


def identity_diffs(arena, j0, out):
    """Largest absolute difference between the free-running h = 1 outputs and generate_full's, over all paths."""
    r, dg, sigma = benchmark_full(arena, [j0])
    return {"id_ret": np.array(np.abs(out.r - r[0]).max()), "id_dg": np.array(np.abs(out.dg - dg[0]).max()),
            "id_sigma": np.array(np.abs(out.sigma - sigma[0]).max())}


def check_identity(rec, j0=None):
    bad = [k for k, tol in (("id_ret", TOL_IDENT_RET), ("id_dg", TOL_IDENT_SIGMA), ("id_sigma", TOL_IDENT_SIGMA))
           if not rec[k] <= tol]
    if bad:
        raise IdentityError("h = 1 identity fails%s: %s" % (" at start %s" % j0 if j0 is not None else "",
                                                           {k: float(rec[k]) for k in bad}))


# ---- files: per-start results flushed as they complete, signatures, resume

def seed_signature(run, folder, cfg, remedy=None):
    """
    Hash of everything one network's Part A results depend on (the start range is NOT part of it).
    remedy (Part B): a dict such as {"model_id", "clip_hash", "late_hash"}, added ONLY when a remedy is active, so the
    control's signature, and with it every Part A resume file, is unchanged.
    """
    keep = ("horizon", "n_paths", "block", "index", "network_cfg", "sim_seed", "n_draw", "vol_lo", "vol_hi", "model_id")
    parts = {"version": REC_VERSION, "cfg": {k: getattr(cfg, k) for k in keep},
             "network": file_hash(os.path.join(folder, "2_trained.pt")), "run_signature": run.info.get("signature"),
             "data": [run.info.get("data_hash"), run.info.get("rates_hash")]}
    if remedy is not None:
        parts["remedy"] = remedy
    return text_hash(json.dumps(parts, sort_keys=True, default=str))


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def is_done(path, sig, what):
    """True if `path` holds finished output with signature sig; RuntimeError if it holds a different one."""
    if not os.path.exists(path):
        return False
    with np.load(path) as z:
        have = z["signature"].item() if "signature" in z.files else None
    if have == sig:
        return True
    raise RuntimeError("%s exists with a different configuration (signature %s, now %s): %s\nDelete it, or point "
                       "VOLGAN_PARTA_DIR elsewhere." % (what, have, sig, path))


def load_npz(path):
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 and z[k].dtype.kind in "USO" else z[k]) for k in z.files}


def work_dir(root, cfg, seed):
    return os.path.join(root, "work", cfg.model_id, "seed_%d" % seed)


def start_path(root, cfg, seed, j0):
    return os.path.join(work_dir(root, cfg, seed), "start_%05d.npz" % j0)


def bench_path(root, cfg, seed):
    return os.path.join(root, "work", cfg.model_id, "bench_seed_%d.npz" % seed)


def get_bench(arena, root, seed, sig, log=print):
    """The benchmark metrics of every day of the block for this network: computed once, kept on disk per seed."""
    path = bench_path(root, arena.cfg, seed)
    days = np.arange(arena.n)
    if is_done(path, sig, "benchmark seed %d" % seed):
        log("benchmark seed %d: loaded" % seed)
        return load_npz(path)
    t0 = time.time()
    b = bench_clouds(arena, days)
    save_npz(path, signature=sig, **b)
    log("benchmark seed %d: %d days in %.0f s" % (seed, arena.n, time.time() - t0))
    return b


def run_seed(arena, root, seed, sig, starts, log=print, post_step=None):
    """All starts of one network, resumable: a finished start file with the same signature is skipped."""
    cfg = arena.cfg
    todo = [j0 for j0 in starts if not is_done(start_path(root, cfg, seed, j0), sig, "start %d seed %d" % (j0, seed))]
    log("seed %d: %d of %d starts to run" % (seed, len(todo), len(starts)))
    t0 = time.time()
    for i, j0 in enumerate(todo):
        rec = run_start(arena, j0, post_step)
        save_npz(start_path(root, cfg, seed, j0), signature=sig, j0=j0, **rec)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            log("seed %d: start %d/%d (j0 = %d), %.1f s per start" % (seed, i + 1, len(todo), j0,
                                                                      (time.time() - t0) / (i + 1)))


def identity_gate(arena, starts, log=print):
    """Brief 5: the h = 1 identity on a handful of starts, in memory, before anything else is written."""
    for j0 in starts:
        run_start_identity(arena, j0)
    log("identity gate passed on starts %s" % list(starts))


def run_start_identity(arena, j0):
    gen = rollout(arena.gen, arena.ds, arena.run.data, j0, arena.noise, arena.cfg, gen_late=arena.gen_late,
                  clip=arena.clip, repair=getattr(arena, "repair", None))
    out = next(gen)
    gen.close()
    check_identity(identity_diffs(arena, j0, out), j0)


def consolidate(arena, root, seed, sig, starts, bench):
    """
    metrics_{seed}.npz: free_* and bench_* arrays shaped (n_starts, H, ...), the bench rows picked by target day.
    Also the identity differences per start and the h = H violation-node frequencies, free and bench.
    """
    cfg = arena.cfg
    recs = [load_npz(start_path(root, cfg, seed, j0)) for j0 in starts]
    out = {"signature": sig, "model_id": cfg.model_id, "seed": seed, "starts": np.asarray(starts),
           "horizons": np.arange(1, cfg.horizon + 1), "target_dates": np.array(
               [[np.datetime64(arena.ds.dates[j0 + k]) for k in range(cfg.horizon)] for j0 in starts], "datetime64[ns]")}
    for name in ("n_valid", "phi_mean", "viol_share", "rho", "phi_parts", "inside", "below", "above", "width", "q"):
        out["free_" + name] = np.stack([r[name] for r in recs])
        days = np.asarray(starts)[:, None] + np.arange(cfg.horizon)[None, :]
        pos = np.searchsorted(bench["days"], days)
        out["bench_" + name] = bench[name][pos]
    out["bench_n_oob"] = bench["n_oob"][np.searchsorted(bench["days"], np.asarray(starts)[:, None]
                                                         + np.arange(cfg.horizon)[None, :])]
    out["real_values"] = arena.real.values[days]                    # (n_starts, H, 10): realised nodes and return
    out["var_names"] = np.array(VAR_NAMES)
    out["free_viol_loc21"] = np.stack([r["viol_loc21"] for r in recs])
    out["bench_viol_loc21"] = bench["viol_loc"][np.searchsorted(bench["days"], np.asarray(starts) + cfg.horizon - 1)]
    for k in ("id_ret", "id_dg", "id_sigma", "seconds"):
        out[k] = np.array([r[k] for r in recs])
    for k in ("clip_share", "clip_any"):                             # Part B; zeros for a model without clipping
        out["free_" + k] = np.stack([r[k] if k in r else np.zeros(cfg.horizon) for r in recs])
    if any("proj_share" in r for r in recs):                         # second remedy only: the projection records
        for k in PROJ_KEYS:
            out["free_" + k] = np.stack([r[k] if k in r else np.zeros(cfg.horizon) for r in recs])
    save_npz(os.path.join(root, "metrics_%d.npz" % seed), **out)
    return out


# =====================================================================================================================
# 4. Market margins (was rec_inference.py)
# =====================================================================================================================

# ---- the admissibility anchor (brief 1a.4): the market's mean Phi on the training block, model-free

def market_margins(cfg):
    """
    delta_A = delta_a_mult x the mean Phi of the market surfaces over the TRAINING block, delta_C, delta_D as set.
    From market data only (no network), so it is recomputed identically on every machine.
    """
    P = prepare_all(cfg.index, training_config(cfg.network_cfg, cfg.seeds[0]))
    block = P.market.block("train")
    phi = market_reference(block)["phi"]
    return {"index": cfg.index, "block": "train", "n_days": int(len(phi)),
            "first": str(block.dates[0].date()), "last": str(block.dates[-1].date()),
            "market_mean_phi_train": float(phi.mean()), "delta_a_mult": cfg.delta_a_mult,
            "delta_A": float(cfg.delta_a_mult * phi.mean()), "delta_C": cfg.delta_c, "delta_D": cfg.delta_d,
            "data_hash": file_hash(INDEX_FILES[cfg.index]), "rates_hash": file_hash(RATES_FILE)}


# =====================================================================================================================
# 5. The run steps (was pipeline.py)
# =====================================================================================================================

# ---- Part 2, Part A: degradation under 21-day recursion (brief: Part A build brief). Same resumable style.
# Layout (VOLGAN_PARTA_DIR, default <code>/outputs/part2/partA):
#   margins.json, identity_check.json, pilot_report.md, run.json
#   work/<model_id>/seed_<k>/start_<j0>.npz    one file per finished start (resume)
#   work/<model_id>/bench_seed_<k>.npz         the one-step benchmark of every day, per seed
#   metrics_<k>.npz, degradation.csv, survival.csv, tests.csv, powerlaw.csv, figures/
# The multi-step code reads only the trained run folders, the data files and the RecursionConfig: no Part 1 output.

def parta_root():
    return os.path.abspath(os.environ.get("VOLGAN_PARTA_DIR") or os.path.join(CODE_DIR, "outputs", "part2", "partA"))


def rec_margins_step(cfg, root=None):
    """margins.json: delta_A (the market's mean Phi on the training block), delta_C, delta_D. Model-free; written once."""
    root = root or parta_root()
    path = os.path.join(root, "margins.json")
    new = market_margins(cfg)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != new:
            raise RuntimeError("margins.json exists with different numbers (%s, now %s): %s" % (old, new, path))
        return new
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(new, f, indent=2)
    log("margins: market mean Phi on the training block %.6g -> delta_A = %.6g (delta_C %g, delta_D %g)"
        % (new["market_mean_phi_train"], new["delta_A"], new["delta_C"], new["delta_D"]))
    return new


def _rec_runs(cfg, device):
    runs = {}
    for seed in cfg.seeds:
        folder = run_path(cfg.index, cfg.network_cfg, seed)
        if not os.path.exists(os.path.join(folder, "run.json")):
            raise FileNotFoundError("no trained run %s: Part A needs all of %s" % (folder, list(cfg.seeds)))
        runs[seed] = (folder, load_run(folder, device))
    return runs


def rec_run_step(cfg, root=None, device=None, allow_test=False, starts=None):
    """
    The free-running experiment for one model: all seeds, all starts, resumable. The h = 1 identity is a gate: it is
    checked on a few starts before anything is written, and on every start as it is computed.
    """
    root, device = root or parta_root(), device or get_device()
    check_inputs(cfg.index)
    runs = _rec_runs(cfg, device)
    noise = NoiseTable(cfg.sim_seed, cfg.n_paths, cfg.n_draw, next(iter(runs.values()))[1].config.noise_dim)
    arenas = {s: Arena(r, cfg, noise, allow_test) for s, (_, r) in runs.items()}
    n = next(iter(arenas.values())).n
    starts = start_indices(n, cfg) if starts is None else np.asarray(starts)
    for s, a in arenas.items():                                           # the gate, before any file is written
        identity_gate(a, starts[:3], log=lambda m, s=s: log("seed %d: %s" % (s, m)))
    margins = rec_margins_step(cfg, root)
    log("%s: %d starts (stride %d) on the %s block of %d samples, %d seeds" % (cfg.model_id, len(starts), cfg.stride,
                                                                           cfg.block, n, len(runs)))
    ident, secs = {}, {}
    for s, (folder, run) in runs.items():
        sig = seed_signature(run, folder, cfg)
        a = arenas[s]
        bench = get_bench(a, root, s, sig, log)
        run_seed(a, root, s, sig, starts, log)
        out = consolidate(a, root, s, sig, starts, bench)
        ident[s] = {k: float(out[k].max()) for k in ("id_ret", "id_dg", "id_sigma")}
        secs[s] = float(out["seconds"].mean())
    with open(os.path.join(root, "identity_check.json"), "w") as f:
        json.dump({"passed": True, "tolerance": {"returns": 1e-6, "dg_and_sigma": 1e-5}, "max_abs_diff_by_seed": ident,
                   "n_starts": int(len(starts))}, f, indent=2)
    dates = a.ds.dates
    fixed = noise_hashes(noise, [dates[0], dates[len(dates) // 2], dates[-1]])
    fixed.update(noise_hashes(noise, FIXED_NOISE_DATES))
    info = {"model_id": cfg.model_id, "settings": cfg.as_dict(), "margins": margins, "sim_seed": cfg.sim_seed,
            "boot_seed": cfg.boot_seed, "noise_hashes": fixed, "n_starts": int(len(starts)),
            "network_run_json_hashes": {str(s): file_hash(os.path.join(f, "run.json")) for s, (f, _) in runs.items()},
            "seconds_per_seed_start": secs, "versions": library_versions(), "device": str(device),
            "block": cfg.block, "first_start": int(starts[0]), "last_start": int(starts[-1])}
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    log("done: %s" % root)
    return info
