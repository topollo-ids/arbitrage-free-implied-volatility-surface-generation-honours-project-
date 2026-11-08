"""
Part 2, Part B (second remedy): projection of a simulated surface onto the arbitrage-free set that Phi measures.

The set. common.penalty_terms(C, m, tau) returns the calendar (Nm, Nt-1), call-spread (Nm-1, Nt) and butterfly
(Nm-2, Nt) terms, each LINEAR in the relative call prices with no constant. A surface is arbitrage-free in the thesis's
sense iff every term is <= 0 (Phi = 0). So L, the 101 x 42 matrix of those terms, is built FROM penalty_terms (column
k = the terms of the k-th unit price surface), never by hand. Price vectors are C[i, j] flattened row-major (i * Nt + j),
moneyness i, maturity j. Added bounds (they make the set smaller, so every output is still inside Phi's set):
intrinsic + DELTA_B <= c <= 1 - DELTA_B with intrinsic = max(0, 1 - m e^{-r tau}), and slope >= -e^{-r tau_j}.

The repair (Cohen, Reisinger and Wang 2020, AMF): the smallest weighted l1 change of the prices that satisfies the
constraints, tightened by EPS_ARB:
    minimise  sum_i w_i (u_i + d_i)   subject to  L (c0 + u - d) <= -EPS_ARB,  slope(c0 + u - d) >= -e^{-r tau} + EPS_ARB,
              intrinsic + DELTA_B <= c0 + u - d <= 1 - DELTA_B,  u, d >= 0;   w_i = 1 / max(vega_i, V_FLOOR)
with vega_i = sqrt(tau_i) phi(d1_i) at the pre-projection volatility, so the objective is, to first order, the total
change in implied volatility. scipy.optimize.linprog, method "highs", default tolerances; the projected surfaces of
one cloud, in path order, in fixed chunks of 200 per LP (block-diagonal, scipy.sparse.kron).

Then c* = c0 + u - d is inverted node by node (only nodes the LP changed; an unchanged node keeps its volatility bit for
bit), re-priced with common.relative_call, and Phi is computed with common.arbitrage_penalty: every term must be <= 0.
A surface that fails is re-solved once with EPS_ARB x 100; if it still fails the run stops (ProjectionError).
"""

from dataclasses import asdict, dataclass
import json

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import norm

from .common import arbitrage_penalty, penalty_terms, relative_call, text_hash

# The brief fixed both at 1e-10. That is below HiGHS's default primal feasibility tolerance (1e-7): on real training
# surfaces 24 of 45,981 LP solutions still had a Phi term above 0 (up to 3.8e-8) and 11 of 16 starts would have stopped.
# DECIDED (the user, 2026-10-18): 1e-6 for both, default solver tolerances kept; tested on the same 16 starts, no term
# above -9.0e-7 and no failure. Recorded as a deviation in REPORT.md section 9.
EPS_ARB = 1e-6           # on every Phi term and every slope lower bound, in its own units
DELTA_B = 1e-6           # price bounds
V_FLOOR = 1e-6           # vega floor in the weights
CHUNK = 200              # projected surfaces per LP
SIG_LO, SIG_HI = 1e-6, 50.0
N_BISECT, N_NEWTON = 200, 2
PRICE_TOL = 1e-14
WITNESS_SIGMA, WITNESS_SLACK = 0.5, 1e-6
CHANGE_TOL = 1e-8        # a node "changed" when its volatility moved by more than this (sparsity record)


@dataclass(frozen=True)
class ProjSettings:
    eps_arb: float = EPS_ARB
    delta_b: float = DELTA_B
    v_floor: float = V_FLOOR
    chunk: int = CHUNK
    norm: str = "weighted l1, w = 1 / max(vega, V_FLOOR) at the pre-projection volatility"
    solver: str = "scipy.optimize.linprog method=highs, default tolerances (scipy %s)" % scipy.__version__
    sigma_range: tuple = (SIG_LO, SIG_HI)
    inversion: str = "bisection on log(sigma), 200 iterations, then 2 Newton steps; price tolerance 1e-14"
    projected_steps: str = "k = 2 ... 21, after the clip"


class ProjectionError(RuntimeError):
    """The solver failed, or a projected surface still carries arbitrage after one retry: stop and report."""


def _call(m, tau, s, r):
    """common.relative_call's formula, elementwise on matching 1-d arrays (the inversion works node by node)."""
    d1 = (-np.log(m) + tau * (r + 0.5 * s ** 2)) / (s * np.sqrt(tau))
    d2 = d1 - s * np.sqrt(tau)
    return norm.cdf(d1) - m * np.exp(-r * tau) * norm.cdf(d2)


def _vega(m, tau, s, r):
    d1 = (-np.log(m) + tau * (r + 0.5 * s ** 2)) / (s * np.sqrt(tau))
    return np.sqrt(tau) * norm.pdf(d1)


def build_L(m, tau):
    """(L (101, 42), slope rows (36, 42), sizes) from penalty_terms applied to unit price surfaces."""
    Nm, Nt = len(m), len(tau)
    n = Nm * Nt
    cols = []
    for k in range(n):
        e = np.zeros((1, Nm, Nt))
        e.flat[k] = 1.0
        cols.append(np.concatenate([t[0].ravel() for t in penalty_terms(e, m, tau)]))
    L = np.array(cols).T
    n_cal, n_slope = Nm * (Nt - 1), (Nm - 1) * Nt
    return L, L[n_cal:n_cal + n_slope], (n_cal, n_slope, (Nm - 2) * Nt)


def stacked_terms(C, m, tau):
    """penalty_terms of (B, Nm, Nt) prices, stacked as (B, 101) in L's row order."""
    return np.concatenate([t.reshape(len(C), -1) for t in penalty_terms(C, m, tau)], axis=1)


def to_mat(v, lk, lt):
    """(B, K) maturity-major vectors -> (B, lk, lt) matrices (postprocessing.surface_matrices' layout)."""
    return v.reshape(-1, lt, lk).transpose(0, 2, 1)


def to_vec(M):
    return M.transpose(0, 2, 1).reshape(len(M), -1)


class Projector:
    """The projection of section 2 for one grid; call it as repair(sigma (P, K), alive (P,), rate (lt,))."""

    def __init__(self, m, tau, settings=ProjSettings()):
        self.m, self.tau = np.asarray(m, float), np.asarray(tau, float)
        self.lk, self.lt = len(m), len(tau)
        self.s = settings
        self.L, self.S, self.sizes = build_L(self.m, self.tau)
        self.ms, self.ts = np.meshgrid(self.m, self.tau, indexing="ij")          # (Nm, Nt)
        self.m_flat, self.t_flat = self.ms.ravel(), self.ts.ravel()
        n = self.lk * self.lt
        Ls, Ss, I = sparse.csr_matrix(self.L), sparse.csr_matrix(self.S), sparse.identity(n, format="csr")
        self.A_unit = sparse.vstack([sparse.hstack([Ls, -Ls]), sparse.hstack([-Ss, Ss]),
                                     sparse.hstack([-I, I]), sparse.hstack([I, -I])]).tocsr()   # (221, 84)
        self.slope_j = np.tile(np.arange(self.lt), self.lk - 1)                  # maturity index of each slope row

    def settings_hash(self):
        return text_hash(json.dumps(asdict(self.s), sort_keys=True, default=str))

    # ---- the pieces

    def bounds(self, rate):
        r = np.asarray(rate, float)
        intrinsic = np.maximum(0.0, 1.0 - self.ms * np.exp(-r[None, :] * self.ts)).ravel()
        slope_lb = -np.exp(-r[self.slope_j] * self.tau[self.slope_j])
        return intrinsic + self.s.delta_b, 1.0 - self.s.delta_b, slope_lb

    def prices(self, sig_mat, rate):
        B = len(sig_mat)
        return relative_call(self.ms, self.ts, sig_mat, np.repeat(np.asarray(rate, float)[None], B, axis=0))

    def witness_slack(self, rate):
        """Smallest slack of every (tightened) constraint for Black-Scholes prices at a constant 50% volatility."""
        c = self.prices(np.full((1, self.lk, self.lt), WITNESS_SIGMA), rate)[0].ravel()
        lb, ub, slope_lb = self.bounds(rate)
        e = self.s.eps_arb
        return float(min((-self.L @ c - e).min(), (self.S @ c - slope_lb - e).min(), (c - lb).min(), (ub - c).min()))

    def solve(self, c0, w, rate, eps):
        """c* (B, 42) for prices c0 (B, 42) with weights w (B, 42): the LP in chunks of `chunk` surfaces."""
        lb, ub, slope_lb = self.bounds(rate)
        out = np.empty_like(c0)
        for a in range(0, len(c0), self.s.chunk):
            cc, ww = c0[a:a + self.s.chunk], w[a:a + self.s.chunk]
            b = len(cc)
            A = sparse.kron(sparse.identity(b, format="csr"), self.A_unit, format="csr")
            rhs = np.concatenate([-eps - cc @ self.L.T, -slope_lb[None] - eps + cc @ self.S.T,
                                  cc - lb[None], ub - cc], axis=1)                # (b, 221), the unit's row order
            obj = np.concatenate([ww, ww], axis=1).ravel()
            res = linprog(obj, A_ub=A, b_ub=rhs.ravel(), bounds=(0, None), method="highs")
            if res.status != 0:
                raise ProjectionError("linprog failed (status %d): %s" % (res.status, res.message))
            x = res.x.reshape(b, 2, -1)
            out[a:a + b] = cc + x[:, 0] - x[:, 1]
        return out

    def invert(self, c, m, tau, r):
        """Implied volatility of 1-d prices c at (m, tau, r): NaN where c is outside [price(1e-6), price(50)]."""
        p_lo, p_hi = _call(m, tau, np.full_like(c, SIG_LO), r), _call(m, tau, np.full_like(c, SIG_HI), r)
        oor = (c < p_lo) | (c > p_hi)
        lo, hi = np.full_like(c, np.log(SIG_LO)), np.full_like(c, np.log(SIG_HI))
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            below = _call(m, tau, np.exp(mid), r) < c
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        s = np.exp(0.5 * (lo + hi))
        for _ in range(N_NEWTON):
            p = _call(m, tau, s, r)
            v = _vega(m, tau, s, r)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_new = np.clip(s - (p - c) / v, SIG_LO, SIG_HI)
            better = np.isfinite(s_new) & (np.abs(_call(m, tau, s_new, r) - c) < np.abs(p - c))
            s = np.where(better & (np.abs(p - c) > PRICE_TOL), s_new, s)
        s[oor] = np.nan
        return s, oor

    # ---- one cloud

    def __call__(self, sigma, alive, rate):
        """
        sigma (P, K) maturity-major vols, alive (P,) the paths still valid before this step, rate (lt,) the target day.
        Projects the finite, alive surfaces with Phi > 0; returns (sigma out, info). Everything else is returned
        unchanged, bit for bit.
        """
        P, K = sigma.shape
        rate = np.asarray(rate, float)
        out = sigma.copy()
        info = {"n_alive": int(alive.sum()), "projected": np.zeros(P, bool), "oor": np.zeros(P, bool),
                "dvol_sum": np.zeros(P), "dvol_max": np.zeros(P), "nodes": np.zeros(P), "retries": 0,
                "phi_post_max": 0.0, "inv_err_max": 0.0, "changed": np.zeros((P, K), bool)}
        fin = alive & np.isfinite(sigma).all(axis=1)
        if not fin.any():
            return out, info
        idx0 = np.flatnonzero(fin)
        mats = to_mat(sigma[idx0], self.lk, self.lt)
        C0 = self.prices(mats, rate)
        need = arbitrage_penalty(C0, self.m, self.tau).total > 0
        if not need.any():
            return out, info
        idx = idx0[need]
        mats, C0 = mats[need], C0[need]
        c0 = C0.reshape(len(idx), -1)
        r_flat = rate[np.tile(np.arange(self.lt), self.lk)]                        # (42,) rate of each node, row-major
        w = 1.0 / np.maximum(_vega(self.m_flat[None], self.t_flat[None], mats.reshape(len(idx), -1), r_flat[None]),
                             self.s.v_floor)
        new_mats, oor_path, err, tries = self._project(c0, w, mats, rate, r_flat, self.s.eps_arb)
        if tries.any():                                                          # one re-solve with EPS_ARB x 100
            info["retries"] = int(tries.sum())
            t = np.flatnonzero(tries)
            m2, o2, e2, still = self._project(c0[t], w[t], mats[t], rate, r_flat, 100 * self.s.eps_arb)
            if still.any():
                raise ProjectionError("%d projected surface(s) still carry arbitrage after a re-solve with "
                                      "EPS_ARB x 100" % still.sum())
            new_mats[t], oor_path[t], err = m2, o2, max(err, e2)
        ok = ~oor_path
        if ok.any():
            post = arbitrage_penalty(self.prices(new_mats[ok], rate), self.m, self.tau).total
            info["phi_post_max"] = float(post.max())
            if info["phi_post_max"] != 0.0:
                raise ProjectionError("post-projection Phi %.3e is not 0" % info["phi_post_max"])
        new_vec = to_vec(new_mats)
        out[idx] = new_vec
        dv = np.abs(new_vec - sigma[idx])
        info["projected"][idx] = True
        info["oor"][idx] = oor_path
        info["dvol_sum"][idx] = np.where(oor_path, 0.0, np.nansum(dv, axis=1))
        info["dvol_max"][idx] = np.where(oor_path, 0.0, np.nanmax(np.where(np.isnan(dv), 0.0, dv), axis=1))
        info["nodes"][idx] = np.where(oor_path, 0.0, (dv > CHANGE_TOL).sum(axis=1))
        info["changed"][idx] = new_vec != sigma[idx]
        info["inv_err_max"] = err
        return out, info

    def _project(self, c0, w, mats, rate, r_flat, eps):
        """LP, inversion of the changed nodes, re-pricing; returns (new vols, out-of-range paths, max price error,
        paths whose re-priced surface still has a positive term)."""
        cs = self.solve(c0, w, rate, eps)
        changed = cs != c0
        flat = mats.reshape(len(c0), -1).copy()
        pi, ni = np.nonzero(changed)
        s, oor = self.invert(cs[pi, ni], self.m_flat[ni], self.t_flat[ni], r_flat[ni])
        flat[pi, ni] = s
        oor_path = np.zeros(len(c0), bool)
        oor_path[pi[oor]] = True
        new = flat.reshape(mats.shape)
        bad = np.zeros(len(c0), bool)
        err = 0.0
        ok = ~oor_path
        if ok.any():
            Crt = self.prices(new[ok], rate)
            okp = np.flatnonzero(ok)
            terms = stacked_terms(Crt, self.m, self.tau)
            bad[okp] = (terms > 0).any(axis=1)
            d = np.abs(Crt.reshape(len(okp), -1) - cs[okp])
            err = float(d.max()) if d.size else 0.0
        return new, oor_path, err, bad
