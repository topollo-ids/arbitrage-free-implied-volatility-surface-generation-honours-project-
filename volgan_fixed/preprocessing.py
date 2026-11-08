"""
Data in, Datasets out: the workbooks and the one data call, the condition and target of the paper's
eqs 6-10, the date split and the preparation of one index. Imports only common.

Contents
    1. Surfaces and rates in: the data call (was data.py)
    2. The older rates loader (was rates.py)
    3. Condition and target (was preprocess.py)
    4. The date split and tensors (was split.py)
    5. Preparing a study for one index (was runs.py)
"""

from dataclasses import dataclass
import functools
import os
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import torch

from .common import DATA_DIR, INDEX_FILES, RATES_FILE


# =====================================================================================================================
# 1. Surfaces and rates in: the data call (was data.py)
# =====================================================================================================================

# Stage 1: THE data call. load(index) reads the two workbooks the model uses and returns one checked object:
#
#     data = load("SPX")      # data.dates, data.spot, data.vols (n, K), data.m, data.tau, data.r (n, lt)
#
#   1. the implied-vol workbook of the index (sheets "Surfaces" and "Grid"): dates, index level PX_LAST, and the vols,
#      converted from percent to decimals; the grid (moneyness m = K/S, maturities tau = days/365) is read from the
#      Grid sheet, never assumed;
#   2. the risk-free rate from the rates workbook: r_1M ... r_1Y, decimals, continuously compounded, at 30, 60, 90,
#      180, 270 and 360 days, exactly the grid's maturities (US Treasury bills: one curve for both indices);
#   3. only the days present in BOTH workbooks are kept (the SPX rates start on 4 June 2008: the US Treasury issued no
#      1-year bills before then); r is put on the grid's maturities (linear in maturity, so exact at matching nodes);
#   4. everything is validated once (validate below).
# The result is cached, so every later step that asks for the same index gets the same object without re-reading Excel.
# Dividends are not used (the workbook's q columns are ignored): calls are priced without dividends, as in the paper.
#
# Tracker rows: 1.1 (A4 one file), 1.2 (F3 index level), 1.3 (B13 no positional join: rates are joined by DATE),
# 1.4 (F7 layout), 1.5 (B18 units), 1.6 (A11 validation), 1.7 (B14 grid from the file).
#
# Layout convention used everywhere in volgan_fixed (maturity-major):
#     vols[d, j*lk + i] = sigma_d(m_i, tau_j)
# so moneyness varies fastest and the surface for day d is vols[d].reshape(lt, lk).

@dataclass
class SurfaceData:
    name: str
    dates: pd.DatetimeIndex  # (n,) trading days, strictly increasing
    spot: np.ndarray         # (n,) index level PX_LAST, from the same row as the vols
    vols: np.ndarray         # (n, lk*lt) implied vols as decimals, maturity-major
    m: np.ndarray            # (lk,) moneyness K/S, ascending
    tau: np.ndarray          # (lt,) time to maturity in years (days/365), ascending
    r: np.ndarray = None     # (n, lt) risk-free rate per day and grid maturity (load(); None without a rates file)
    source: str = ""         # the workbook(s) it was read from

    @property
    def lk(self):
        return len(self.m)

    @property
    def lt(self):
        return len(self.tau)

    def surfaces(self):
        """(n, lt, lk) view: surfaces()[d, j, i] = sigma_d(m_i, tau_j). No copy."""
        return self.vols.reshape(len(self.dates), self.lt, self.lk)

    def describe(self):
        """One line for the screen: what was loaded."""
        rates = "no rates" if self.r is None else "r %.2f%% to %.2f%%" % (100 * self.r.min(), 100 * self.r.max())
        return ("%s: %d days %s to %s | grid %d moneyness (%.3g-%.3g) x %d maturities (%d-%d days) = %d points | %s"
                % (self.name, len(self.dates), self.dates[0].date(), self.dates[-1].date(), self.lk, self.m[0],
                   self.m[-1], self.lt, round(self.tau[0] * 365), round(self.tau[-1] * 365), self.lk * self.lt, rates))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(dates, spot, vols, m, tau):
    """A11: every assumption the later stages rely on, checked once, at load."""
    _require(dates.is_unique and dates.is_monotonic_increasing, "dates must be unique and increasing")
    _require((dates.dayofweek < 5).all(), "dates contain a weekend")
    _require(np.isfinite(spot).all() and (spot > 0).all(), "spot must be finite and positive")
    _require(np.isfinite(vols).all(), "vols contain NaN or inf")
    _require(vols.min() > 0.01 and vols.max() < 3.0,
             "vols outside (0.01, 3.0) as decimals: range %.4f to %.4f. Percent not converted?"
             % (vols.min(), vols.max()))
    _require(np.all(np.diff(m) > 0) and np.all(np.diff(tau) > 0), "grids must be strictly ascending")
    _require(vols.shape == (len(dates), len(m) * len(tau)), "vols shape does not match dates and grid")
    _require(len(spot) == len(dates), "spot and dates differ in length")


def load_surfaces(path, name, percent=True):
    """
    Read the 'Surfaces' and 'Grid' sheets of a Bloomberg implied-vol workbook (surfaces only, no rates).
    The Grid sheet names each Surfaces column in a 'Field' (or, in the older workbooks, 'Grid point') column, with its
    'Days' and 'Moneyness m = K/S'; the columns must be maturity-major (moneyness fastest), in that order.
    percent=True: the workbook quotes vols in percent (12.03 = 12.03%), so divide by 100.
    """
    with pd.ExcelFile(path) as xl:
        surf = xl.parse("Surfaces")
        grid = xl.parse("Grid")
    point = "Field" if "Field" in grid.columns else "Grid point"
    grid = grid.dropna(subset=[point])
    _require(list(surf.columns[:2]) == ["Date", "PX_LAST"], "expected columns Date, PX_LAST first")
    _require(list(surf.columns[2:]) == list(grid[point]), "Surfaces columns do not match the Grid sheet")
    days = grid["Days"].to_numpy(float)
    mono = grid["Moneyness m = K/S"].to_numpy(float)
    m, tau = np.unique(mono), np.unique(days) / 365
    _require(len(days) == len(m) * len(tau), "the Grid sheet is not a full moneyness x maturity grid")
    _require(np.allclose(mono.reshape(len(tau), len(m)), m[None, :]), "grid is not moneyness-fastest")
    _require(np.allclose(days.reshape(len(tau), len(m)), np.unique(days)[:, None]), "grid is not maturity-major")
    vols = surf.iloc[:, 2:].to_numpy(float) / (100 if percent else 1)
    data = SurfaceData(name=name, dates=pd.DatetimeIndex(pd.to_datetime(surf["Date"])),
                       spot=surf["PX_LAST"].to_numpy(float), vols=vols, m=m, tau=tau, source=os.path.basename(path))
    validate(data.dates, data.spot, data.vols, data.m, data.tau)
    return data


GRID_M = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)        # the grid the model is built for: the 42-point workbooks
GRID_DAYS = (30, 60, 90, 180, 270, 360)                    # (also the maturities of the rates workbook's r columns)
RATES_SHEET = "SPX"                        # one risk-free curve for every index (see load)
RATE_COLUMNS = ["r_1M", "r_2M", "r_3M", "r_6M", "r_9M", "r_1Y"]
RATE_DAYS = np.array([30, 60, 90, 180, 270, 360])         # the maturities of those columns (the workbook's Method sheet)


@functools.lru_cache(maxsize=8)
def load(index, surfaces=None, rates=None, with_rates=True):
    """
    The one data call: the index's implied-vol surfaces and the risk-free rate, on the days both workbooks have.
    index "SPX" or "NDX"; surfaces, rates: workbook paths (default: paths.INDEX_FILES[index], paths.RATES_FILE; a bare
    file name is looked up in the data folder); with_rates=False reads the surfaces only (r = None), for r = 0 runs.
    The risk-free rate is the US Treasury bill curve, the same for every index: the workbook's SPX and NDX sheets hold
    identical r columns, so one sheet (RATES_SHEET, which covers every trading day from 4 June 2008) serves both.
    """
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(DATA_DIR, p)
    surfaces, rates = resolve(surfaces or INDEX_FILES[index]), resolve(rates or RATES_FILE)
    data = load_surfaces(surfaces, index)
    days = np.round(data.tau * 365).astype(int)
    if data.lk != len(GRID_M) or len(days) != len(GRID_DAYS) or not np.allclose(data.m, GRID_M)             or tuple(days) != GRID_DAYS:
        raise ValueError("%s has the wrong grid: moneyness %s and maturities %s days, but the model needs moneyness %s "
                         "and %s days (the *_grid42_* workbooks). Point SPX_FILE / NDX_FILE in main.py at those."
                         % (surfaces, data.m.round(4).tolist(), days.tolist(), list(GRID_M), list(GRID_DAYS)))
    if not with_rates:
        return data

    table = pd.read_excel(rates, sheet_name=RATES_SHEET, index_col="Date", parse_dates=True)
    _require(table.index.is_unique, "%s: repeated dates" % rates)
    keep = data.dates.isin(table.index)                        # the days both workbooks have
    _require(keep.any(), "no common dates between %s and %s" % (surfaces, rates))
    dates, spot, vols = data.dates[keep], data.spot[keep], data.vols[keep]
    r_nodes = table.loc[dates, RATE_COLUMNS].to_numpy(float)
    _require(np.isfinite(r_nodes).all(), "%s: a missing rate on a surface day" % rates)
    _require(np.abs(r_nodes).max() < 0.25, "rates look like percent; expected decimals (0.043 for 4.3%)")
    r = np.stack([np.interp(data.tau, RATE_DAYS / 365, row) for row in r_nodes])    # linear in maturity, flat outside
    validate(dates, spot, vols, data.m, data.tau)
    return SurfaceData(index, dates, spot, vols, data.m, data.tau, r=r,
                       source="%s + %s" % (os.path.basename(surfaces), os.path.basename(rates)))


# =====================================================================================================================
# 2. The older rates loader (was rates.py)
# =====================================================================================================================

# Stage 9: the interest rate r used to price surfaces for the arbitrage penalty (F4).
#
# The paper backs out a daily rate from put-call parity on option mid-prices. Our workbooks hold implied vols
# and index levels only, so that is not possible; rates come from a workbook pulled from Bloomberg,
# data\rates.xlsx, with one sheet:
#
#     sheet "rates":  Date | 1M | 3M | 6M | 12M   yields in percent, e.g. US Treasury bills
#                                                 (Bloomberg USGG1M, USGG3M, USGG6M, USGG12M Index, PX_LAST).
#                                                 Any columns named <n>M work, at least two of them. Part 1: the
#                                                 Bloomberg names "USGG3M Index", and <n>Y for years, work too;
#                                                 a workbook with a single sheet may give it any name.
#
# Dividends are not modelled (decision log, F14): the index is treated as paying none, as in the paper.
#
# Conventions (recorded in the decision log):
#   - tenor <n>M is n*30 days, the same convention as the surface grid (Bloomberg's 30, 60, 90, 180, 360 days);
#   - a yield y for tenor T years is simple interest over T: continuous r = ln(1 + y T) / T;
#   - r is interpolated linearly in maturity onto the grid, flat beyond the first and last tenor;
#   - a day missing from the workbook (a bond-market holiday) takes the previous value, for at most
#     max_fill rows; a longer gap is an error, never filled silently;
#   - a surface dated d is priced with day d's rates, the market surface and the scenarios for d alike.
#
# Two sources:
#   "zero"   r = 0: the released code (F4), kept for the ablation;
#   "rates"  r from the workbook, per day and per grid maturity: the control.

SOURCES = ("zero", "rates")


@dataclass
class Rates:
    dates: pd.DatetimeIndex
    r: np.ndarray        # (n, lt) continuous rate, per day and grid maturity
    source: str

    def on(self, dates):
        """The rows for the given dates, in that order; every date must be present."""
        rows = self.dates.get_indexer(dates)
        if (rows < 0).any():
            raise ValueError("no rate for %s" % list(pd.DatetimeIndex(dates)[rows < 0][:5].date))
        return Rates(pd.DatetimeIndex(dates), self.r[rows], self.source)


def zero_rates(dates, tau):
    return Rates(pd.DatetimeIndex(dates), np.zeros((len(dates), len(tau))), "zero")


def _aligned(frame, dates, max_fill):
    """frame indexed by date -> the rows for `dates`, forward-filling at most max_fill rows; error otherwise."""
    if not frame.index.is_unique:
        raise ValueError("rates: dates are repeated")
    frame = frame.sort_index()
    filled = frame.reindex(frame.index.union(dates)).ffill(limit=max_fill).loc[dates]
    missing = filled.isna().any(axis=1)
    if missing.any():
        raise ValueError("rates: no value within %d rows for %d dates, first %s"
                         % (max_fill, missing.sum(), list(filled.index[missing][:3].date)))
    return filled.to_numpy(float)


TENOR = re.compile(r"(?:USGG)?(\d+)\s*([MY])(?:\s+INDEX)?")     # 3M, 12M, 1Y, USGG3M Index (any case)


def tenor_months(name):
    """Months of a tenor column name (3M -> 3, 1Y -> 12, 'USGG6M Index' -> 6), or None for any other column."""
    hit = TENOR.fullmatch(str(name).strip().upper())
    if hit is None:
        return None
    return int(hit.group(1)) * (12 if hit.group(2) == "Y" else 1)


def _read_sheet(path):
    """Sheet "rates", or the only sheet of a one-sheet workbook; indexed by the Date column."""
    with pd.ExcelFile(path) as xl:                      # closed at once: Windows cannot delete an open workbook
        names = xl.sheet_names
        if "rates" in names:
            name = "rates"
        elif len(names) == 1:
            name = names[0]
        else:
            raise ValueError("%s: no sheet named 'rates' among %s" % (path, names))
        sheet = xl.parse(name)
    date_col = [c for c in sheet.columns if str(c).strip().lower() in ("date", "dates")]
    if len(date_col) != 1:
        raise ValueError("sheet %r needs one column named Date, got columns %s" % (name, list(sheet.columns)))
    sheet = sheet.set_index(date_col[0])
    sheet.index = pd.to_datetime(sheet.index, errors="coerce")
    sheet = sheet[sheet.index.notna()]
    # Bloomberg writes "#N/A N/A" for a missing print: such a cell counts as missing, like a bond-market holiday
    return sheet.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def load_rates(path, dates, tau, max_fill=3):
    dates, tau = pd.DatetimeIndex(dates), np.asarray(tau, dtype=float)
    sheet = _read_sheet(path)
    tenors = {c: tenor_months(c) for c in sheet.columns if tenor_months(c) is not None}
    if len(tenors) < 2:
        raise ValueError("sheet 'rates' needs at least two tenor columns named like 1M, 3M, 12M")
    if len(set(tenors.values())) != len(tenors):
        raise ValueError("two columns give the same tenor: %s" % tenors)
    cols = sorted(tenors, key=tenors.get)
    T = np.array([tenors[c] * 30 / 365 for c in cols])                       # years, as the surface grid
    y = _aligned(sheet[cols], dates, max_fill)
    if np.nanmax(np.abs(y)) < 1.0:
        raise ValueError("rates look like decimals; expected percent (e.g. 4.3 for 4.3%)")
    r_tenor = np.log1p(y / 100 * T) / T                                       # simple over T -> continuous
    r = np.stack([np.interp(tau, T, row) for row in r_tenor])                # linear in maturity, flat outside
    if not (-0.05 < r.min() and r.max() < 0.25):
        raise ValueError("rates out of range: %.4f to %.4f" % (r.min(), r.max()))
    return Rates(dates, r, "rates")


def get_rates(source, dates, tau, path=RATES_FILE):
    """The rates for a source name (see SOURCES)."""
    if source == "zero":
        return zero_rates(dates, tau)
    if source == "rates":
        if not os.path.exists(path):
            raise FileNotFoundError("rate_source 'rates' needs the rates workbook at %s (sheet 'rates': Date, 1M, 3M, "
                                    "6M, 12M in percent). Put it there, set VOLGAN_RATES_FILE, or use "
                                    "rate_source='zero' for the released-code ablation." % path)
        return load_rates(path, dates, tau)
    raise ValueError("source must be one of %s, got %r" % (SOURCES, source))


# =====================================================================================================================
# 3. Condition and target (was preprocess.py)
# =====================================================================================================================

# Stage 3: build the condition and the target the networks see (paper eqs 6-10).
#
# Tracker rows: 3.1 (renamed, no triple s), 3.2 (F5 realised-vol window as a switch),
# 3.3 (F6 return scaling, kept and recorded), 3.4 (B9 dormant normalisation statistics: not built).
#
# Vocabulary. ret[k] = log(S_k / S_{k-1}) is the return that ENDS on row k of the loaded data.
# Sample j predicts day d (a row p), and is conditioned on day c = p - 1:
#     condition = [ ret[c], ret[c-1], realised vol, g_c (all K log-vols on day c) ]   length 3 + K
#     target    = [ ret[p],           g_p - g_c (K changes in log-vol)             ]   length 1 + K
# Paper eq 9 has a_t = (R_{t-1}, R_{t-2}, gamma_{t-1}, g_t) with t = c, and eq 10 the outputs (R_t, dg_t).
#
# Part 1: a Dataset also carries the rates of each sample's predicted day d and condition day c (Stage 9, per
# maturity), so Phi in the loss prices the generated surface for d with d's rates (Stage 7).

WINDOW = 21                    # trading days in the realised-vol window (paper eq 8)
N_SCALARS = 3                  # the condition is [R1, R2, realised vol], then the surface (B14: slice offset)
FIRST_SAMPLE = WINDOW + 2      # first predicted row. The code-window ablation needs one day more than the
                               # paper window, so both variants start here and use identical days.
DEFAULT_RET_SCALE = np.sqrt(252)   # F6: the released code multiplies returns by sqrt(252); kept, see ret_scale


@dataclass
class Dataset:
    dates: pd.DatetimeIndex   # (n,) the predicted day d of each sample
    cond_dates: pd.DatetimeIndex  # (n,) the condition day d-1: plot anything from the condition against these (B21)
    spot_prev: np.ndarray     # (n,) index level on the condition day d-1
    spot: np.ndarray          # (n,) index level on the predicted day d
    condition: np.ndarray     # (n, 3+K)
    target: np.ndarray        # (n, 1+K)
    m: np.ndarray
    tau: np.ndarray
    ret_scale: float          # returns in condition and target are multiplied by this; divide generated returns by it
    rv_window: str            # "paper" or "code"
    rates: np.ndarray = None       # (n, lt) rates of the predicted day d, per grid maturity (None: not attached)
    cond_rates: np.ndarray = None  # (n, lt) rates of the condition day c = d-1

    @property
    def past_log_surface(self):
        """g_c, the condition-day log-vols: the part of the condition after the three scalars."""
        return self.condition[:, N_SCALARS:]

    def slice(self, start, stop):
        """Samples start..stop-1 as a new Dataset (views, no copy). Used by the Stage 4 split."""
        return self.take(slice(start, stop))

    def take(self, rows):
        """The samples selected by a slice or a boolean mask, as a new Dataset; every per-sample array is cut alike."""
        cut = lambda x: None if x is None else x[rows]
        return Dataset(dates=self.dates[rows], cond_dates=self.cond_dates[rows], spot_prev=self.spot_prev[rows],
                       spot=self.spot[rows], condition=self.condition[rows], target=self.target[rows], m=self.m,
                       tau=self.tau, ret_scale=self.ret_scale, rv_window=self.rv_window, rates=cut(self.rates),
                       cond_rates=cut(self.cond_rates))


def log_returns(spot):
    """ret[k] = log(S_k / S_{k-1}); ret[0] is nan because the file has no close before its first day."""
    ret = np.full(len(spot), np.nan)
    ret[1:] = np.diff(np.log(spot))
    return ret


def realised_vol_ending_at(ret):
    """
    rv[e] = sqrt(252/21 * sum of the 21 squared returns that END on days e-20, ..., e).
    nan for e < 21, where fewer than 21 returns exist.
    """
    rv = np.full(len(ret), np.nan)
    windows = sliding_window_view(ret[1:], WINDOW)          # window w covers ret[1+w : 22+w], ending on day 21+w
    rv[WINDOW:] = np.sqrt(252 / WINDOW * (windows**2).sum(axis=1))
    return rv


def data_preprocessing(data, rv_window="paper", ret_scale=DEFAULT_RET_SCALE, rates=None):
    """
    data: a SurfaceData from Stage 1 (the condition and target come from its surfaces; the returns and the realised
          vol come from spot).
    rv_window "paper": returns ending on days d-21 ... d-1 (eq 8 evaluated on the condition day d-1).
              "code":  returns ending on days d-22 ... d-2 (the released code, one day stale: F5).
    ret_scale: the factor on returns (F6). The released code uses sqrt(252); 1.0 gives plain daily returns.
    rates: None, or (n_days, lt) rates aligned with data.dates (rates.get_rates(...).r); each sample keeps the rows
           of its predicted day and its condition day.
    """
    if rv_window not in ("paper", "code"):
        raise ValueError("rv_window must be 'paper' or 'code', got %r" % (rv_window,))
    n_days = len(data.dates)
    if n_days <= FIRST_SAMPLE:
        raise ValueError("need more than %d days, got %d" % (FIRST_SAMPLE, n_days))
    lag = 0 if rv_window == "paper" else 1

    ret = log_returns(data.spot)
    rv = realised_vol_ending_at(ret)
    g = np.log(data.vols)

    p = np.arange(FIRST_SAMPLE, n_days)    # predicted days
    c = p - 1                              # condition days
    condition = np.column_stack([ret_scale * ret[c], ret_scale * ret[c - 1], rv[c - lag], g[c]])
    target = np.column_stack([ret_scale * ret[p], g[p] - g[c]])
    if not (np.isfinite(condition).all() and np.isfinite(target).all()):
        raise ValueError("non-finite values in the condition or target")

    if rates is not None:
        rates = np.asarray(rates, dtype=float)
        if rates.shape != (n_days, len(data.tau)) or not np.isfinite(rates).all():
            raise ValueError("rates must be finite with shape (%d, %d), got %s" % (n_days, len(data.tau), rates.shape))

    return Dataset(
        dates=data.dates[p], cond_dates=data.dates[c], spot_prev=data.spot[c], spot=data.spot[p],
        condition=condition, target=target, m=data.m, tau=data.tau,
        ret_scale=float(ret_scale), rv_window=rv_window,
        rates=None if rates is None else rates[p], cond_rates=None if rates is None else rates[c],
    )


# =====================================================================================================================
# 4. The date split and tensors (was split.py)
# =====================================================================================================================

# Stage 4: chronological split and conversion to tensors.
#
# Tracker rows: 4.1 (F2: split by date, not fraction; recorded in the decision log),
# 4.2 (B21: the dates of each part are returned), 4.3 (R5: no validation split is built, by decision).
#
# Part 1 (G1) replaces the split by calendar year with a THREE-way split by fixed dates on the predicted day of
# each sample (split_by_dates, the dates in Config):
#     train       first usable day .. train_end   (2017-12-31)
#     validation  train_end + 1    .. val_end     (2018-01-01 .. 2021-12-31): every Part 1 result
#     test        val_end + 1      .. test_end    (2022-01-01 .. 2025-12-31): used once, at the very end
# The realised-vol window of the first validation day may reach back into training returns: that is conditioning,
# not fitting. split_by_year is kept only so that run folders trained before Part 1 still load (runs.load_run).
#
# Nothing is shuffled here: samples stay in time order, and shuffling inside each training epoch is the training
# loop's job.

BLOCKS = ("train", "val", "test")


@dataclass
class Split:
    train: object                  # Dataset: dates_train is train.dates
    test: object                   # Dataset: dates_test is test.dates (the PREDICTED day of each sample)
    test_start_year: int = None    # split_by_year only
    train_fraction: float = None   # share of all samples that ended up in train
    val: object = None             # Dataset of the validation block (split_by_dates only)
    bounds: dict = None            # split_by_dates: the configured last day of each block, ISO strings

    def block(self, name, allow_test=False):
        """
        The Dataset of one block. The test block is refused unless allow_test=True (CLI flag --confirm-final):
        Part 1 is computed on the validation block, and the test block is used once, at the very end of the study.
        """
        if name not in BLOCKS:
            raise ValueError("block must be one of %s, got %r" % (BLOCKS, name))
        if name == "test" and not allow_test:
            raise PermissionError("the test block is used once, at the end of the study: pass allow_test=True "
                                  "(--confirm-final) to evaluate on it")
        out = getattr(self, name)
        if out is None:
            raise ValueError("this split has no %r block (a split by calendar year from before Part 1)" % name)
        return out

    def counts(self):
        return {b: (None if getattr(self, b) is None else len(getattr(self, b).dates)) for b in BLOCKS}


def split_by_year(ds, train_fraction=0.8, test_start_year=None):
    """
    Train = samples whose predicted day is before 1 January of test_start_year; test = the rest.
    test_start_year=None picks the calendar year whose boundary gives a train share closest to train_fraction.
    """
    years = ds.dates.year.to_numpy()
    n = len(years)
    candidates = np.unique(years)[1:]                       # a test set cannot start in the first year
    if test_start_year is None:
        shares = np.array([(years < y).sum() / n for y in candidates])
        test_start_year = int(candidates[np.argmin(np.abs(shares - train_fraction))])
    elif test_start_year not in candidates:
        raise ValueError("test_start_year %r must be one of %s" % (test_start_year, list(candidates)))
    k = int((years < test_start_year).sum())                # number of training samples
    return Split(train=ds.slice(0, k), test=ds.slice(k, n), test_start_year=int(test_start_year),
                 train_fraction=k / n)


def block_bounds(config):
    """The configured last day of each block, as Timestamps, checked to be increasing."""
    ends = [pd.Timestamp(getattr(config, b + "_end")) for b in BLOCKS]
    if not ends[0] < ends[1] < ends[2]:
        raise ValueError("block ends must increase: train_end < val_end < test_end, got %s" % ends)
    return dict(zip(BLOCKS, ends))


def split_by_dates(ds, config):
    """
    Part 1 (G1): Split(train, val, test) by the PREDICTED day of each sample and the fixed dates in config
    (train_end, val_end, test_end). Samples after test_end are not used. Every block must be non-empty.
    """
    ends = block_bounds(config)
    d = ds.dates
    masks = {"train": d <= ends["train"], "val": (d > ends["train"]) & (d <= ends["val"]),
             "test": (d > ends["val"]) & (d <= ends["test"])}
    parts = {b: ds.take(np.asarray(mask)) for b, mask in masks.items()}
    for b, part in parts.items():
        if len(part.dates) == 0:
            raise ValueError("the %s block is empty for data from %s to %s" % (b, d[0].date(), d[-1].date()))
    return Split(train=parts["train"], val=parts["val"], test=parts["test"],
                 train_fraction=len(parts["train"].dates) / len(d),
                 bounds={b: ends[b].strftime("%Y-%m-%d") for b in BLOCKS})


def to_tensors(ds, device, with_rates=False):
    """
    (condition, target) as float32 tensors on the device; the networks train in single precision.
    with_rates=True adds the predicted-day rates (n, lt) in float64, for Phi in the loss (Part 1, lever 2).
    """
    condition = torch.from_numpy(ds.condition).to(torch.float32).to(device)
    target = torch.from_numpy(ds.target).to(torch.float32).to(device)
    if not with_rates:
        return condition, target
    if ds.rates is None:
        raise ValueError("this Dataset carries no rates: build it with data_preprocessing(..., rates=...)")
    return condition, target, torch.from_numpy(np.ascontiguousarray(ds.rates)).to(torch.float64).to(device)


# =====================================================================================================================
# 5. Preparing a study for one index (was runs.py)
# =====================================================================================================================

@dataclass
class Prepared:
    data: object          # SurfaceData, the raw vendor grid (Stage 1)
    rates: np.ndarray     # (n_days, lt) rates of every data day (Stage 9)
    inputs: object        # Split of the model's inputs (the raw grid)
    market: object        # Split of the raw market: realised values come from here (the same object as inputs)


def split_of(ds, config, legacy=False):
    """The split of a Dataset: by fixed dates (Part 1), or by calendar year for a run trained before Part 1."""
    return split_by_year(ds, config.train_fraction) if legacy else split_by_dates(ds, config)


def prepare_all(index, config, legacy=False, progress=False, surfaces=None):
    """
    Stages 1, 3, 4 for one index. P.inputs feeds the networks, P.market is what they are scored against: both are the
    raw vendor grid.
    surfaces: another implied-vol workbook than the index's default (load_run passes the one a run was trained on).
    """
    with_rates = config.rate_source == "rates"
    if config.rate_source not in ("rates", "zero"):
        raise ValueError("rate_source must be 'rates' or 'zero', got %r" % config.rate_source)
    data = load(index, surfaces, with_rates=with_rates)                  # THE data call (cached)
    rates = data.r if with_rates else np.zeros((len(data.dates), data.lt))
    market = split_of(data_preprocessing(data, config.rv_window, config.ret_scale, rates), config, legacy)
    return Prepared(data, rates, market, market)


def prepare(index, config, legacy=False):
    """Stages 1, 3 and 4 for one index: the loaded (raw) data and the split of the model's inputs."""
    p = prepare_all(index, config, legacy)
    return p.data, p.inputs
