"""
Settings, locations, reproducibility, the arbitrage penalty and the small helpers every other module uses.
Imports nothing from the rest of the package.

Contents
    1. Settings (was config.py)
    2. Locations of the workbooks, runs and results (was paths.py)
    3. Reproducibility, checkpoints, versions (was repro.py)
    4. The arbitrage penalty, numpy and torch (was arbitrage.py)
    5. Shared step helpers: logging, locations, resumability (was pipeline.py)
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import platform
import random
import time
from typing import NamedTuple

import numpy as np
from scipy.stats import norm
import torch




# =====================================================================================================================
# 1. Settings (was config.py)
# =====================================================================================================================

# One place for every setting. Defaults are the paper's values (tracker 5.1, F12): the released
# VolGAN() defaulted to noise 16, H 8, 1,000 epochs and 100 matching epochs instead.
#
# Later stages add their switches here (penalty, calibration object, restart, rate), each with the
# paper setting as the default (the control) and the released-code setting as the ablation.
#
# Part 1 (the 2x2 arbitrage ablation: Loss x Weighting) adds, all defaulting to the control:
#   - the three-way split by fixed dates (G1): train_end, val_end, test_end, as ISO strings so run.json keeps them;
#   - the rate source (G4): "rates" (control) or "zero" (released code, F4);
#   - the loss lever (Stage 7): phi_in_loss;
#   - the simulation seed (G10): sim_seed, one integer shared by every model and training seed.
# The output lever (re-weighting) is chosen at evaluation time, not here (Stage 10).

SEEDS = (1, 2, 3, 4, 5)          # Part 1 training seeds, fixed in advance; never chosen among (section 2.2)
SIM_SEED = 20261012              # Part 1 simulation seed, the same for every model (common random numbers)


@dataclass(frozen=True)
class Config:
    # networks (paper section 3.4)
    noise_dim: int = 32          # d: dimension of the Gaussian noise
    hidden_dim: int = 16         # H: generator layers H and 2H, discriminator layer H
    # training (paper section 3.4)
    n_epochs: int = 10_000       # full training epochs
    n_grad: int = 25             # gradient-matching epochs that set alpha and beta
    lr_g: float = 1e-4           # RMSprop learning rate, generator
    lr_d: float = 1e-4           # RMSprop learning rate, discriminator
    batch_size: int = 100
    # smoothness penalty (Stage 6)
    smoothness: str = "paper"    # F7: "paper" eqs 11-12 (control) or "released" (ablation)
    match_on: str = "g"          # F8: object penalised during gradient matching, "g" (control) or "sigma" (ablation)
    # training procedure (Stage 7)
    restart: bool = True         # F9: restart stage 2 from the initial weights (control) or continue (ablation)
    gen_bce_weight: float = 0.5  # eq 13's 1/2 on the generator's BCE term (control); the released code uses 1.0.
                                 # Not a pure rescaling: RMSprop's epsilon makes the two train slightly differently.
    # sampling (paper section 3.4)
    n_scenarios: int = 10_000    # N raw scenarios per day
    # data conventions (Stages 3 and 4)
    rv_window: str = "paper"     # F5: "paper" (control) or "code" (ablation)
    ret_scale: float = float(np.sqrt(252))   # F6: kept, see Stage 3
    train_fraction: float = 0.8  # the old split by calendar year (Stage 4); kept only so old run folders still load
    seed: int = 0
    # Part 1: three-way split on the predicted day (G1). Train <= train_end < val <= val_end < test <= test_end.
    train_end: str = "2017-12-31"
    val_end: str = "2021-12-31"
    test_end: str = "2025-12-31"
    # Part 1: rates (G4), used by Phi in the loss, the simulation and the evaluation alike
    rate_source: str = "rates"   # F4: "rates" (control, data/rates.xlsx) or "zero" (released code)
    # Part 1 levers (section 3); the defaults are the control, cells A/B
    phi_in_loss: bool = False    # the Loss lever: add alpha_phi E[Phi(sigma_hat)] to eq 13 (Stage 7)
    sim_seed: int = SIM_SEED     # G10: passed to simulate() / generate() / daily_metrics() for EVERY model

    def as_dict(self):
        """Plain dict, for saving inside a checkpoint (Stage 0)."""
        return asdict(self)


def variant(name, **overrides):
    """
    "control":  the paper (every default above).
    "released": the released code's choices for every fidelity switch so far (F4, F5, F7, F8, F9, eq 13's 1/2).
    Further keyword arguments override single settings, e.g. variant("control", n_epochs=100).
    """
    if name == "control":
        base = Config()
    elif name == "released":
        base = Config(rv_window="code", smoothness="released", match_on="sigma", restart=False, gen_bce_weight=1.0,
                      rate_source="zero")
    else:
        raise ValueError("variant must be 'control' or 'released', got %r" % (name,))
    return replace(base, **overrides)


# ---- Part 1: the 2x2 ablation, Loss x Weighting. Re-weighting is post-processing, so two TRAINING configurations
# ---- give the four cells: each trained network is evaluated unweighted and re-weighted. The networks train on, and
# ---- are scored against, the raw vendor grid.

TRAINING_CONFIGS = {
    "raw_phiout": dict(phi_in_loss=False),    # cells A, B: the control networks (the paper)
    "raw_phiin": dict(phi_in_loss=True),      # cells C, D: Phi added to the generator loss
}

WEIGHTINGS = {"max_penalty": "re-weighted", "raw": "unweighted"}   # reweight.beta_t rules: the output lever

CELLS = {                       # cell -> (training configuration, weighting rule)
    "A": ("raw_phiout", "max_penalty"),     # the control (the paper)
    "B": ("raw_phiout", "raw"),             # the same networks, unweighted: the Part 2 benchmark
    "C": ("raw_phiin", "max_penalty"),
    "D": ("raw_phiin", "raw"),
}


def cell_levers(cell):
    """(Loss, Weighting) labels of a cell, for tables."""
    name, rule = CELLS[cell]
    return ("in" if TRAINING_CONFIGS[name]["phi_in_loss"] else "out", WEIGHTINGS[rule])


def training_config(name, seed, **overrides):
    """The Config of one training configuration and training seed: the control plus that configuration's levers."""
    if name not in TRAINING_CONFIGS:
        raise ValueError("training configuration must be one of %s, got %r" % (list(TRAINING_CONFIGS), name))
    return variant("control", seed=seed, **{**TRAINING_CONFIGS[name], **overrides})


# ---- Part 2, Part A: degradation under 21-day recursion. A separate dataclass, so no Part 1 run signature changes.

@dataclass(frozen=True)
class RecursionConfig:
    """
    Settings of the free-running experiment (Part A brief, section 3). Fixed in advance, never tuned on results.
    The margins on correlation and coverage are fixed numbers; the admissibility margin delta_A = delta_a_mult x the
    market's mean Phi on the training block comes from margins.json (rec_inference.market_margins), not from here.
    """
    horizon: int = 21
    n_paths: int = 1000
    stride: int = 5                  # starting days: every 5th validation sample
    start_offset: int = 0            # j0 in {offset, offset + stride, ...}
    block: str = "val"
    index: str = "SPX"
    network_cfg: str = "raw_phiout"  # the control networks (cells A, B); B = unweighted
    seeds: tuple = SEEDS             # training seeds, as Part 1
    sim_seed: int = SIM_SEED         # the SAME value as Part 1 (G10): never the training seed
    n_draw: int = 10_000             # rows drawn per date; the first n_paths are kept (= Part 1's first scenarios)
    n_boot: int = 10_000
    boot_seed: int = 20261014        # recorded in run.json
    block_len_starts: int = None     # None: ceil(horizon / stride) = 5
    vol_lo: float = 0.01             # validity bounds on implied vol (1% to 300%)
    vol_hi: float = 3.00
    delta_a_mult: float = 1.0        # delta_A = delta_a_mult x the market's mean Phi on the training block
    delta_c: float = 0.05            # 5 percentage points of coverage
    delta_d: float = 0.10            # 0.10 in correlation
    survival_tol: float = 0.99       # instability if S(21) < survival_tol
    model_id: str = "control"        # remedied models get their own id; their tables go under this id

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        if self.block_len_starts is None:
            object.__setattr__(self, "block_len_starts", int(-(-self.horizon // self.stride)))

    def as_dict(self):
        return asdict(self)


# =====================================================================================================================
# 2. Locations of the workbooks, runs and results (was paths.py)
# =====================================================================================================================

# Where the input workbooks, the training runs and the results live.
#
# Every location can be moved without editing code, so the package runs on another machine (Part 1, "runs externally"):
#     VOLGAN_DATA_DIR     folder holding the Bloomberg workbooks
#     VOLGAN_SPX_FILE     the SPX implied-vol workbook   (default <data>/spx_grid42_2007_2025_clean.xlsx)
#     VOLGAN_NDX_FILE     the NDX implied-vol workbook   (default <data>/ndx_grid42_2010_2025_clean.xlsx)
#     VOLGAN_RATES_FILE   the rates workbook             (default <data>/rates_2008_2025_clean.xlsx)
#     VOLGAN_RUNS_DIR     trained networks, one folder per run           (default <code>/runs)
#     VOLGAN_RESULTS_DIR  simulations, metrics, tables and figures        (default <code>/results)
# Without VOLGAN_DATA_DIR the data folder is <code>/data when it exists, and otherwise the folder next to <code>
# (this laptop's layout: project/data beside project/code). <code> is the folder holding volgan_fixed/ and main.py.
# The variables are read once, at import: set them before importing volgan_fixed (main.py does this for --data-dir).

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _data_dir():
    if os.environ.get("VOLGAN_DATA_DIR"):
        return os.path.abspath(os.environ["VOLGAN_DATA_DIR"])
    inside = os.path.join(CODE_DIR, "data")
    return inside if os.path.isdir(inside) else os.path.abspath(os.path.join(CODE_DIR, "..", "data"))


DATA_DIR = _data_dir()
RUNS_DIR = os.path.abspath(os.environ.get("VOLGAN_RUNS_DIR") or os.path.join(CODE_DIR, "runs"))
RESULTS_DIR = os.path.abspath(os.environ.get("VOLGAN_RESULTS_DIR") or os.path.join(CODE_DIR, "results"))


def _file(env, default):
    """A workbook location: the environment variable if set (a bare name is looked up in DATA_DIR), else the default."""
    name = os.environ.get(env) or default
    return os.path.abspath(name if os.path.isabs(name) else os.path.join(DATA_DIR, name))


RATES_FILE = _file("VOLGAN_RATES_FILE", "rates_2008_2025_clean.xlsx")      # r (and q) per day and maturity

VOL_INDEX = {                                                    # Stage 11: published volatility indices
    "SPX": (os.path.join(DATA_DIR, "Project Data 2.xlsx"), "VIX Index"),
    "NDX": (os.path.join(DATA_DIR, "Nasdaq Final Data.xlsx"), "Nasdaq Volatility Index"),   # the VXN
}

INDEX_FILES = {                       # the implied-vol surfaces: the 42-point Bloomberg grid (0.8-1.2 x 30-360 days)
    "SPX": _file("VOLGAN_SPX_FILE", "spx_grid42_2007_2025_clean.xlsx"),
    "NDX": _file("VOLGAN_NDX_FILE", "ndx_grid42_2010_2025_clean.xlsx"),
}


def check_inputs(index, need_rates=True):
    """Fail early, with the expected locations, when an input workbook is missing (before hours of work start)."""
    missing = [p for p in [INDEX_FILES[index]] + ([RATES_FILE] if need_rates else []) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("missing input file(s):\n  %s\nPut them there, or point VOLGAN_DATA_DIR / "
                                "VOLGAN_RATES_FILE at them (see README_run_instructions.pdf)." % "\n  ".join(missing))


# =====================================================================================================================
# 3. Reproducibility, checkpoints, versions (was repro.py)
# =====================================================================================================================

# Stage 0: reproducibility and recoverability.
#
# Tracker rows: 0.3 (B16 seeds), 0.4 (A9 checkpoints), 0.5 (B12 device).
# Part 1 (G8): set_seed also switches on deterministic algorithms; library versions and input-file hashes are
# recorded in every run.json, so a result can be traced to the exact code, libraries and data that produced it.

# cuBLAS needs this before its first use for deterministic matrix products on CUDA (harmless on CPU).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed):
    """Seed Python, NumPy and torch together so a run can be repeated exactly, with deterministic algorithms (G8)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_device():
    """Pick the device once; every tensor in later code is created on it."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def library_versions():
    """Versions of everything that can change a number: torch, numpy, scipy, the QP and LP solvers (run.json)."""
    import scipy
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "scipy": scipy.__version__, "lp_solver": "scipy.optimize.linprog HiGHS (scipy %s)" % scipy.__version__,
           "cuda": torch.version.cuda if torch.cuda.is_available() else None, "platform": platform.platform()}
    try:
        import clarabel
        out["qp_solver"] = "clarabel %s" % clarabel.__version__
    except ImportError:
        out["qp_solver"] = None
    return out


def file_hash(path, n=16):
    """First n hex digits of the SHA-256 of a file's bytes: identifies an input workbook in run.json and caches."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def text_hash(text, n=16):
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def save_checkpoint(path, gen, disc, gen_opt, disc_opt, alpha, beta, seed, config, extra=None):
    """
    Save everything needed to resume or re-simulate without retraining:
    the four state dicts, the penalty weights alpha and beta, the seed and the config.
    Call it before stage 1, after stage 1 and after stage 2 (A9).
    extra: further values to keep (Part 1: alpha_phi and its rule when Phi is in the loss).
    """
    torch.save(
        {
            "gen": gen.state_dict(),
            "disc": disc.state_dict(),
            "gen_opt": gen_opt.state_dict(),
            "disc_opt": disc_opt.state_dict(),
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
            "config": config,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, gen, disc, gen_opt, disc_opt, device):
    """Load a checkpoint into existing networks and optimisers; returns the rest."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["gen"])
    disc.load_state_dict(ckpt["disc"])
    gen_opt.load_state_dict(ckpt["gen_opt"])
    disc_opt.load_state_dict(ckpt["disc_opt"])
    return ckpt["alpha"], ckpt["beta"], ckpt["seed"], ckpt["config"]


# =====================================================================================================================
# 4. The arbitrage penalty, numpy and torch (was arbitrage.py)
# =====================================================================================================================

# Stage 2: the arbitrage penalty as a stand-alone function (the measuring instrument).
#
# Tracker rows: 2.2 (R4: market phi moved out of the loader), 2.3 (R11: duplicates gone);
# Stage 9: 9.1 (F4: the rate is an input, per day and per maturity), 9.2 (F14: decided, dividends not modelled),
# 9.4-9.6 (B19, A10, A1: float64, axis differences), 9.8 (no clamping, so no lost gradients), 9.3 (F13: summary()).
# Part 1: penalty_terms() exposes each term before its positive part, so violation_nodes() can say WHERE on the grid
# violations occur (metric 8.1); the rate shape is checked against the batch (brief 4.2); and a torch copy of eqs 1-5
# (relative_call_torch, arbitrage_penalty_torch) lets Phi enter the generator loss (lever 2, Stage 7).
#
# Paper eqs 1-5, for an index treated as paying no dividends, as the paper does (decision log, F14):
#     c(m, tau) = C / S = N(d1) - m exp(-r tau) N(d2),   d1 = (-ln m + (r + sigma^2/2) tau) / (sigma sqrt tau).
# Under this model the calendar condition at fixed spot moneyness (eq 3) is valid for any r >= 0. Adding a
# dividend yield q > r would make eq 3 flag arbitrage that is not there (tests/test_stage9.py shows it).
#
# C holds relative call prices with shape (B, Nm, Nt): axis 1 = moneyness, axis 2 = maturity,
# so each C[b] is one surface as a matrix.

TOL_TERM = 1e-12      # a term counts as violated when its value before the positive part exceeds this (metric 8.1)


class Phi(NamedTuple):
    """The three penalty terms, one value per surface."""
    calendar: np.ndarray   # p1, eq 3: call price must rise with maturity
    call: np.ndarray       # p2, eq 4: call price must fall with moneyness
    butterfly: np.ndarray  # p3, eq 5: call price must be convex in moneyness

    @property
    def total(self):
        return self.calendar + self.call + self.butterfly


def rate_array(r, batch=None, n_tau=None):
    """
    A rate in a shape that broadcasts against sigma of shape (B, Nm, Nt): a scalar stays a scalar,
    one rate per surface (B,) becomes (B, 1, 1), one per surface and maturity (B, Nt) becomes (B, 1, Nt).
    batch, n_tau: B and Nt when sigma is such a batch; a rate whose length does not match is refused, because
    numpy would otherwise broadcast it silently to the wrong shape. The trap (brief 4.2): one day's surface with
    its six maturity rates must be passed as sigma[None] and r[None, :]; r of shape (Nt,) would be read as one rate
    per SURFACE.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 0:
        return r
    if r.ndim == 1:
        if batch is not None and len(r) != batch:
            raise ValueError("a 1-d rate is one rate per surface: got %d rates for %d surfaces. For one surface "
                             "with a rate per maturity, pass sigma[None] and r[None, :]" % (len(r), batch))
        return r[:, None, None]
    if r.ndim == 2:
        if batch is not None and (r.shape[0] != batch or (n_tau is not None and r.shape[1] != n_tau)):
            raise ValueError("a 2-d rate must be (surfaces, maturities) = (%d, %s), got %s" % (batch, n_tau, r.shape))
        return r[:, None, :]
    raise ValueError("a rate must be a scalar, (B,) or (B, Nt); got shape %s" % (r.shape,))


def relative_call(m, tau, sigma, r=0.0):
    """
    Paper eq 1: c(m, tau) = N(d1) - m exp(-r tau) N(d2).
    m, tau: grids broadcastable against sigma. r: continuously compounded, per year; see rate_array.
    """
    sigma = np.asarray(sigma)
    shape = sigma.shape if sigma.ndim == 3 else (None, None, None)
    r = rate_array(r, shape[0], shape[2])
    d1 = (-np.log(m) + tau * (r + 0.5 * sigma**2)) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return norm.cdf(d1) - m * np.exp(-r * tau) * norm.cdf(d2)


def _checked(C, m, tau):
    C = np.asarray(C, dtype=float)
    m, tau = np.asarray(m, dtype=float), np.asarray(tau, dtype=float)
    if C.ndim != 3 or C.shape[1:] != (len(m), len(tau)):
        raise ValueError("C must have shape (B, %d, %d), got %s" % (len(m), len(tau), C.shape))
    return C, m, tau


def penalty_terms(C, m, tau):
    """
    The terms of eqs 3-5 BEFORE their positive parts; a term is a violation when it is above 0:
      calendar  (B, Nm, Nt-1): tau_j (c(m_i, tau_j) - c(m_i, tau_j+1)) / (tau_j+1 - tau_j)
      call      (B, Nm-1, Nt): slope[i] = (c(m_i+1) - c(m_i)) / (m_i+1 - m_i), the call-spread term
      butterfly (B, Nm-2, Nt): slope[i-1] - slope[i] at interior node i (index i-1 here)
    """
    C, m, tau = _checked(C, m, tau)
    calendar = tau[:-1] * (C[:, :, :-1] - C[:, :, 1:]) / np.diff(tau)
    slope = (C[:, 1:, :] - C[:, :-1, :]) / np.diff(m)[:, None]
    return calendar, slope, slope[:, :-1, :] - slope[:, 1:, :]


def arbitrage_penalty(C, m, tau):
    """Paper eqs 3-5 on a batch of relative call price matrices C, shape (B, Nm, Nt)."""
    calendar, slope, butterfly = penalty_terms(C, m, tau)
    p1 = np.maximum(0, calendar).sum(axis=(1, 2))     # calendar: positive part, summed over the grid
    p2 = np.maximum(0, slope).sum(axis=(1, 2))        # call spread
    p3 = np.maximum(0, butterfly).sum(axis=(1, 2))    # butterfly: (slope to the left) - (slope to the right)
    return Phi(p1, p2, p3)


def violation_nodes(C, m, tau, tol=TOL_TERM):
    """
    (B, Nm, Nt) booleans: node (i, j) takes part in at least one violated term (metric 8.1, "violation location").
    A violated calendar term involves (i, j) and (i, j+1); a call-spread term (i, j) and (i+1, j); a butterfly at
    node i involves (i-1, j), (i, j) and (i+1, j).
    """
    calendar, slope, butterfly = penalty_terms(C, m, tau)
    nodes = np.zeros(np.shape(C), dtype=bool)
    v = calendar > tol
    nodes[:, :, :-1] |= v
    nodes[:, :, 1:] |= v
    v = slope > tol
    nodes[:, :-1, :] |= v
    nodes[:, 1:, :] |= v
    v = butterfly > tol
    nodes[:, :-2, :] |= v
    nodes[:, 1:-1, :] |= v
    nodes[:, 2:, :] |= v
    return nodes


def market_phi(data, r=0.0):
    """Φ of every day's market surface in a SurfaceData; one value per day for each term."""
    ms, ts = np.meshgrid(data.m, data.tau, indexing="ij")            # (Nm, Nt)
    sigma = data.surfaces().transpose(0, 2, 1)                       # (n, Nm, Nt)
    return arbitrage_penalty(relative_call(ms, ts, sigma, r), data.m, data.tau)


def summary(daily):
    """
    The paper's Table 1 statistics for one row (F13: computed on OUR grid and test period, never compared
    with the paper's numbers). daily: one value per day, e.g. the market's Phi, or a model's mean Phi over
    its scenarios (Table 1 caption: "standard deviation and median ... of (re-weighted) average outputs").
    """
    daily = np.asarray(daily, dtype=float)
    return {"mean": daily.mean(), "std": daily.std(), "median": np.median(daily)}


# ---- Part 1, lever 2: the same instrument in torch, differentiable, for the generator loss (Stage 7).
# ---- Mirrors relative_call and arbitrage_penalty exactly: same equations, same rate conventions, same layout.

def _as_tensor(x, like):
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def rate_tensor(r, sigma):
    """rate_array for tensors: a scalar, (B,) -> (B, 1, 1) or (B, Nt) -> (B, 1, Nt), checked against sigma (B, Nm, Nt)."""
    r = _as_tensor(r, sigma)
    batch, n_tau = (sigma.shape[0], sigma.shape[2]) if sigma.dim() == 3 else (None, None)
    if r.dim() == 0:
        return r
    if r.dim() == 1:
        if batch is not None and r.shape[0] != batch:
            raise ValueError("a 1-d rate is one rate per surface: got %d for %d surfaces" % (r.shape[0], batch))
        return r[:, None, None]
    if r.dim() == 2:
        if batch is not None and tuple(r.shape) != (batch, n_tau):
            raise ValueError("a 2-d rate must be (%d, %d), got %s" % (batch, n_tau, tuple(r.shape)))
        return r[:, None, :]
    raise ValueError("a rate must be a scalar, (B,) or (B, Nt); got shape %s" % (tuple(r.shape),))


def relative_call_torch(m, tau, sigma, r=0.0):
    """Eq 1 in torch, differentiable in sigma (and r). m, tau broadcast against sigma (B, Nm, Nt), as in relative_call."""
    m, tau = _as_tensor(m, sigma), _as_tensor(tau, sigma)
    r = rate_tensor(r, sigma)
    d1 = (-torch.log(m) + tau * (r + 0.5 * sigma ** 2)) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    return torch.special.ndtr(d1) - m * torch.exp(-r * tau) * torch.special.ndtr(d2)


def arbitrage_penalty_torch(C, m, tau):
    """Eqs 3-5 in torch on C of shape (B, Nm, Nt); returns a Phi of (B,) tensors (torch.relu for the positive parts)."""
    m, tau = _as_tensor(m, C), _as_tensor(tau, C)
    if C.dim() != 3 or tuple(C.shape[1:]) != (len(m), len(tau)):
        raise ValueError("C must have shape (B, %d, %d), got %s" % (len(m), len(tau), tuple(C.shape)))
    calendar = tau[:-1] * (C[:, :, :-1] - C[:, :, 1:]) / torch.diff(tau)
    slope = (C[:, 1:, :] - C[:, :-1, :]) / torch.diff(m)[:, None]
    butterfly = slope[:, :-1, :] - slope[:, 1:, :]
    return Phi(torch.relu(calendar).sum(dim=(1, 2)), torch.relu(slope).sum(dim=(1, 2)),
               torch.relu(butterfly).sum(dim=(1, 2)))


# =====================================================================================================================
# 5. Shared step helpers: logging, locations, resumability (was pipeline.py)
# =====================================================================================================================

def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---- locations and signatures

def run_path(index, name, seed):
    return os.path.join(RUNS_DIR, index, name, "seed_%d" % seed)


def block_dir(index, block):
    return os.path.join(RESULTS_DIR, index, block)


def sim_path(index, block, name, seed):
    return os.path.join(block_dir(index, block), "sims", name, "seed_%d.npz" % seed)


def metrics_path(index, block, cell, seed):
    return os.path.join(block_dir(index, block), cell, "seed_%d" % seed, "metrics.npz")


def cell_of(name, rule):
    return next(c for c, v in CELLS.items() if v == (name, rule))


def _done(path, key, value, what):
    """True if `path` holds finished output with the same signature; an error if it holds a different one."""
    if not os.path.exists(path):
        return False
    if path.endswith(".json"):
        with open(path) as f:
            have = json.load(f).get(key)
    else:
        with np.load(path) as z:
            have = z[key].item() if key in z.files else None
    if have == value:
        return True
    raise RuntimeError("%s exists with a different configuration (signature %s, now %s): %s\nDelete it, or point "
                       "VOLGAN_RUNS_DIR / VOLGAN_RESULTS_DIR elsewhere." % (what, have, value, path))


def _plan(index, names, seeds, overrides):
    for name in names:
        for seed in seeds:
            config = training_config(name, seed, **overrides)
            yield name, seed, config, run_path(index, name, seed)
