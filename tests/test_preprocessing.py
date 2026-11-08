"""
Preprocessing checks, merged by topic from the stage tests (every assertion kept). Run on its own:

    python tests/test_preprocessing.py

--- from test_stage1.py ---
Stage 1 checks: the loader reads both workbooks correctly, the layout is right,
and validate() rejects bad input.
The data call that adds the rates (data.load) is checked by tests/test_stage1_load.py.

--- from test_stage1_load.py ---
Stage 1 checks for THE data call, data.load(index), on the model's workbooks (paths.INDEX_FILES, paths.RATES_FILE):
  - the 42-point grid is read from the Grid sheet (7 moneyness 0.8-1.2 x 6 maturities 30-360 days, maturity-major),
    vols in decimals,
  - only the days present in both workbooks are kept, and the rates are joined by DATE, not by row,
  - r at the grid maturities is exactly the workbook's r_1M ... r_1Y (the same six maturities); one risk-free curve
    serves both indices (read from one sheet, equal to each index's own sheet),
  - surfaces only (with_rates=False) keeps the whole surface window; the result is cached; bad rates are refused.

--- from test_stage2.py ---
Stage 2 checks: the stand-alone arbitrage penalty is correct.
  - hand-computed answers on tiny surfaces,
  - identical to the ORIGINAL matrix-based code on real market surfaces,
  - the rate argument works.

--- from test_stage3.py ---
Stage 3 checks: the condition and target are built from the right days, with the right windows.
  - every number re-derived by slow explicit loops straight from the price and vol levels,
  - the "code" window equals the ORIGINAL preprocessing, the "paper" window differs only in the vol column,
  - the return scale switch, and an alignment guard (leverage effect).

--- from test_stage4.py ---
Stage 4 checks: the split is chronological, by calendar year, loses nothing, and keeps dates aligned.

--- from test_stage9.py ---
Stage 9 checks: the measuring instrument with interest rates.
  - pricing satisfies put-call parity, and is the paper's eq 1 (and the original code) for any rate,
  - rates per day and per maturity reach the right surface and maturity,
  - B19: a float32 pricing pipeline invents arbitrage on arbitrage-free surfaces, ours does not,
  - why dividends are left out (F14): with a dividend yield above the rate, eq 3 flags arbitrage that is not there,
  - the benchmark and the scenarios use the same pricing, and the rate loader is exact.
"""

import functools
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from original_reference import arbitrage_penalty as original_phi, original_preprocessing, penalty_mutau, smallBS
from volgan_fixed.common import (
    Config, INDEX_FILES, RATES_FILE, arbitrage_penalty, market_phi, rate_array, relative_call, set_seed, summary)
from volgan_fixed.postprocessing import day_scenarios, market_sigma, scenario_phi, simulate
from volgan_fixed.preprocessing import (
    FIRST_SAMPLE, N_SCALARS, WINDOW, data_preprocessing, get_rates, load, load_rates, load_surfaces, prepare,
    prepare_all, split_by_year, to_tensors, validate)
from volgan_fixed.training import build_networks


# =====================================================================================================================
# from test_stage1.py
# =====================================================================================================================


SPX, NDX = INDEX_FILES["SPX"], INDEX_FILES["NDX"]


def test_shapes_and_ranges():
    for path, name, n, first, last in [(SPX, "SPX", 4780, "2007-01-03", "2025-12-31"),
                                       (NDX, "NDX", 3963, "2010-04-01", "2025-12-31")]:
        d = load_surfaces(path, name)
        assert (d.lk, d.lt) == (7, 6)
        assert d.vols.shape == (n, 42) and d.surfaces().shape == (n, 6, 7)
        assert d.dates[0] == pd.Timestamp(first) and d.dates[-1] == pd.Timestamp(last)
        assert 0.01 < d.vols.min() and d.vols.max() < 1.0  # decimals, not percent
        assert np.allclose(d.m, [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
        assert np.allclose(d.tau * 365, [30, 60, 90, 180, 270, 360])


def test_reshape_puts_values_where_expected():
    """surfaces()[d, j, i] must be the Bloomberg value for maturity j, moneyness i."""
    d = load_surfaces(SPX, "SPX")
    raw = pd.read_excel(SPX, sheet_name="Surfaces")
    row = 1234
    S = d.surfaces()[row]
    assert np.isclose(S[0, 0], raw.loc[row, "1M_m0.80"] / 100)   # 1M, m=0.8
    assert np.isclose(S[0, 3], raw.loc[row, "1M_m1.00"] / 100)   # 1M, ATM
    assert np.isclose(S[5, 6], raw.loc[row, "1Y_m1.20"] / 100)   # 1Y, m=1.2
    assert np.isclose(S[3, 2], raw.loc[row, "6M_m0.95"] / 100)   # 6M, m=0.95


def test_layout_market_penalty():
    """Row 1.4: market phi is ~0 under the right reshape and large under the transposed one."""
    for path, name in [(SPX, "SPX"), (NDX, "NDX")]:
        d = load_surfaces(path, name)
        ms, ts = np.meshgrid(d.m, d.tau, indexing="ij")             # (Nm, Nt)
        right = d.surfaces().transpose(0, 2, 1)                     # (n, Nm, Nt)
        wrong = d.vols.reshape(len(d.dates), d.lk, d.lt)            # moneyness as the slow axis
        phi_right = arbitrage_penalty(relative_call(ms, ts, right), d.m, d.tau).total
        phi_wrong = arbitrage_penalty(relative_call(ms, ts, wrong), d.m, d.tau).total
        assert np.median(phi_right) < 1e-6 and phi_right.mean() < 0.01, (name, phi_right.mean())
        assert phi_wrong.mean() > 1.0 and (phi_wrong > 0).all(), (name, phi_wrong.mean())


def _good_inputs():
    dates = pd.bdate_range("2020-01-06", periods=5)
    spot = np.linspace(3000, 3100, 5)
    m, tau = np.array([0.9, 1.0, 1.1]), np.array([0.1, 0.5])
    vols = np.full((5, 6), 0.2)
    return dates, spot, vols, m, tau


def _rejects(**changes):
    args = dict(zip(["dates", "spot", "vols", "m", "tau"], _good_inputs()))
    args.update(changes)
    try:
        validate(**args)
    except ValueError:
        return True
    return False


def test_validate_rejects_bad_input():
    dates, spot, vols, m, tau = _good_inputs()
    validate(dates, spot, vols, m, tau)                              # good input passes
    assert _rejects(vols=vols * 100)                                 # percent, not converted
    assert _rejects(vols=np.where(np.arange(30).reshape(5, 6) == 7, np.nan, vols))
    assert _rejects(vols=np.where(np.arange(30).reshape(5, 6) == 7, 0.0, vols))   # zero vol
    assert _rejects(dates=dates[::-1])                               # unsorted dates
    assert _rejects(dates=pd.DatetimeIndex(["2020-01-04"] + list(dates[1:])))      # a Saturday
    assert _rejects(m=m[::-1])                                       # descending grid
    assert _rejects(spot=np.where(np.arange(5) == 2, -1.0, spot))    # bad price
    assert _rejects(vols=vols[:, :5])                                # wrong shape


# =====================================================================================================================
# from test_stage1_load.py
# =====================================================================================================================


R_COLS = ["r_1M", "r_2M", "r_3M", "r_6M", "r_9M", "r_1Y"]


def test_grid_dates_and_units():
    for index, n, first, n_file in (("SPX", 4423, "2008-06-04", 4780), ("NDX", 3963, "2010-04-01", 3963)):
        d = load(index)
        assert np.allclose(d.m, [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
        assert np.array_equal(np.round(d.tau * 365).astype(int), [30, 60, 90, 180, 270, 360])
        assert d.vols.shape == (n, 42) and d.r.shape == (n, 6)
        assert d.dates[0] == pd.Timestamp(first) and d.dates[-1] == pd.Timestamp("2025-12-31")
        raw = pd.read_excel(INDEX_FILES[index], sheet_name="Surfaces").set_index("Date")
        day = d.dates[1234]
        assert np.isclose(d.vols[1234, 0], raw.loc[day, "1M_m0.80"] / 100)          # 1M, m = 0.80
        assert np.isclose(d.vols[1234, 7 * 5 + 3], raw.loc[day, "1Y_m1.00"] / 100)  # 1Y ATM
        assert np.isclose(d.spot[1234], raw.loc[day, "PX_LAST"])
        assert 0.01 < d.vols.min() and d.vols.max() < 3.0 and "rates" in d.source
        assert len(load(index, with_rates=False).dates) == n_file                 # surfaces only: the whole file
    full = load("SPX", with_rates=False)
    assert full.r is None and full.dates[0] == pd.Timestamp("2007-01-03")
    assert len(full.dates) - len(load("SPX").dates) == 357     # SPX days before the first 1-year bill (4 June 2008)
    assert load("SPX") is load("SPX")                           # read once, cached


def test_rates_joined_by_date():
    for index in ("SPX", "NDX"):
        d = load(index)
        own = pd.read_excel(RATES_FILE, sheet_name=index, index_col="Date", parse_dates=True)   # the index's own sheet
        assert np.array_equal(d.r, own.loc[d.dates, R_COLS].to_numpy(float))
        assert np.abs(d.r).max() < 0.1


def test_bad_rates_refused():
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "rates.xlsx")
        sheet = pd.read_excel(RATES_FILE, sheet_name="SPX")
        for bad in (sheet.assign(**{c: sheet[c] * 100 for c in R_COLS}),          # percent instead of decimals
                    sheet.assign(r_3M=np.nan),                                       # a missing rate on surface days
                    sheet.assign(Date=pd.Timestamp("1990-01-01"))):                 # no common dates (and repeated)
            with pd.ExcelWriter(f) as xl:
                bad.to_excel(xl, sheet_name="SPX", index=False)
            try:
                load("SPX", rates=f)
            except ValueError:
                continue
            raise AssertionError("bad rates workbook accepted")


# =====================================================================================================================
# from test_stage2.py
# =====================================================================================================================


M = np.array([0.9, 1.0, 1.1])

TAU = np.array([0.1, 0.2])


def phi_of(C):
    p = arbitrage_penalty(np.array(C, dtype=float)[None], M, TAU)
    return float(p.calendar[0]), float(p.call[0]), float(p.butterfly[0])


def test_hand_computed_violations():
    # rows = moneyness 0.9, 1.0, 1.1; columns = maturity 0.1, 0.2
    # arbitrage-free: falls in m, convex in m, rises in tau -> Φ = 0
    assert phi_of([[0.30, 0.31], [0.20, 0.21], [0.12, 0.13]]) == (0.0, 0.0, 0.0)
    # concave in m (slopes -1 then -1.5): butterfly = (-1) - (-1.5) = 0.5 per column, 2 columns
    cal, call, bfly = phi_of([[0.30, 0.30], [0.20, 0.20], [0.05, 0.05]])
    assert np.allclose([cal, call, bfly], [0.0, 0.0, 1.0])
    # call price falls with maturity: tau_0 * (c_0 - c_1)/(tau_1 - tau_0) = 0.1*0.02/0.1 = 0.02 per row, 3 rows
    cal, call, bfly = phi_of([[0.30, 0.28], [0.20, 0.18], [0.10, 0.08]])
    assert np.allclose([cal, call, bfly], [0.06, 0.0, 0.0])
    # call price rises with m (slope +1): 2 gaps x 2 columns = 4
    cal, call, bfly = phi_of([[0.10, 0.10], [0.20, 0.20], [0.30, 0.30]])
    assert np.allclose([cal, call, bfly], [0.0, 4.0, 0.0])


def test_relative_call_known_values():
    # ATM, r = 0: c = 2 N(sigma sqrt(tau) / 2) - 1
    from scipy.stats import norm
    assert np.isclose(relative_call(1.0, 1.0, 0.2), 2 * norm.cdf(0.1) - 1)
    # same as the original smallBS, including with a rate
    rng = np.random.default_rng(0)
    m, tau, sig = rng.uniform(0.8, 1.2, 50), rng.uniform(0.05, 1, 50), rng.uniform(0.1, 0.5, 50)
    for r in (0.0, 0.03):
        assert np.allclose(relative_call(m, tau, sig, r), smallBS(m, tau, sig, r), rtol=1e-12, atol=0)


def test_matches_original_on_market_surfaces():
    """Our difference-based Φ equals the original matrix-based Φ, term by term, on real days."""
    for path, name in [(SPX, "SPX"), (NDX, "NDX")]:
        d = load_surfaces(path, name)
        ours = market_phi(d)
        P_T, P_K, PB_K = penalty_mutau(d.m, d.tau * 365)          # the original works in days
        taus, ms = np.meshgrid(d.tau, d.m)                          # as in the original SPXData
        worst = np.argsort(ours.total)[-30:]                        # include the most violated days
        days = np.unique(np.concatenate([np.arange(0, len(d.dates), 23), worst]))
        for k in days:
            BS = smallBS(ms, taus, d.surfaces()[k].T, 0)
            P1, P2, P3, tot = original_phi(BS, P_T, P_K, PB_K)
            assert np.isclose(ours.calendar[k], np.sum(P1), rtol=1e-9, atol=1e-12), (name, k)
            assert np.isclose(ours.call[k], np.sum(P2), rtol=1e-9, atol=1e-12), (name, k)
            assert np.isclose(ours.butterfly[k], np.sum(P3), rtol=1e-9, atol=1e-12), (name, k)
            assert np.isclose(ours.total[k], tot, rtol=1e-9, atol=1e-12), (name, k)


def test_market_phi_properties_and_rate():
    for path, name, mean in [(SPX, "SPX", 0.0005), (NDX, "NDX", 0.0019)]:
        d = load_surfaces(path, name)
        p = market_phi(d)
        n = len(d.dates)
        assert p.total.shape == (n,) and p.total.dtype == np.float64
        assert (p.calendar >= 0).all() and (p.call >= 0).all() and (p.butterfly >= 0).all()
        assert np.allclose(p.total, p.calendar + p.call + p.butterfly)
        assert abs(p.total.mean() - mean) < 3e-4, (name, p.total.mean())   # value seen in Stage 1
        # rate argument: scalar 0 == default; a constant per-day array == the scalar; r > 0 changes the answer
        assert np.array_equal(market_phi(d, 0.0).total, p.total)
        assert np.allclose(market_phi(d, np.full(n, 0.02)).total, market_phi(d, 0.02).total)
        assert not np.allclose(market_phi(d, 0.02).total, p.total)


def test_bad_shape_rejected():
    try:
        arbitrage_penalty(np.zeros((1, 2, 3)), M, TAU)
    except ValueError:
        return
    raise AssertionError("wrong shape was accepted")


# =====================================================================================================================
# from test_stage13.py
# =====================================================================================================================

ZERO = Config(rate_source="zero")


def test_blocks_are_the_configured_dates():
    for index in ("SPX", "NDX"):
        P = prepare_all(index, ZERO)
        s = P.inputs
        tr, va, te = s.train.dates, s.val.dates, s.test.dates
        assert tr[-1] <= pd.Timestamp("2017-12-31") < va[0] and va[-1] <= pd.Timestamp("2021-12-31") < te[0]
        assert te[-1] <= pd.Timestamp("2025-12-31")
        assert va[0] == pd.Timestamp("2018-01-02") and te[0] == pd.Timestamp("2022-01-03")      # first trading days
        assert len(tr) + len(va) + len(te) == len(P.data.dates) - 23      # every usable sample, once
        assert tr.intersection(va).empty and va.intersection(te).empty
        assert s.bounds == {"train": "2017-12-31", "val": "2021-12-31", "test": "2025-12-31"}
        # rates travel with their samples; the first validation day is conditioned on the last training day
        assert s.val.rates.shape == (len(va), P.data.lt) and s.val.cond_dates[0] == tr[-1]
        try:
            s.block("test")
        except PermissionError:
            pass
        else:
            raise AssertionError("the test block was handed out without allow_test")
        assert s.block("test", allow_test=True) is s.test and s.block("val") is s.val
    cfg = Config(train_end="2016-12-31")
    assert prepare_all("SPX", replace_rate(cfg)).inputs.val.dates[0].year == 2017
    try:
        prepare_all("SPX", replace_rate(Config(val_end="2016-01-01")))
    except ValueError:
        pass
    else:
        raise AssertionError("block ends out of order accepted")


def replace_rate(cfg):
    from dataclasses import replace
    return replace(cfg, rate_source="zero")


# =====================================================================================================================
# from test_stage3.py
# =====================================================================================================================


def explicit_sample(d, p, rv_window, ret_scale):
    """One sample built with plain loops from the levels. p = predicted row, c = p - 1 = condition row."""
    S, V, c = d.spot, d.vols, p - 1
    r = lambda k: np.log(S[k] / S[k - 1])                    # return ending on row k
    end = c if rv_window == "paper" else c - 1                # last day of the realised-vol window
    rv = np.sqrt(252 / 21 * sum(r(k) ** 2 for k in range(end - 20, end + 1)))
    cond = np.concatenate([[ret_scale * r(c), ret_scale * r(c - 1), rv], np.log(V[c])])
    targ = np.concatenate([[ret_scale * r(p)], np.log(V[p]) - np.log(V[c])])
    return cond, targ


def test_shapes_and_alignment():
    for path, name in [(SPX, "SPX"), (NDX, "NDX")]:
        d = load_surfaces(path, name)
        ds = data_preprocessing(d)
        n, K = len(d.dates) - FIRST_SAMPLE, d.lk * d.lt
        assert ds.condition.shape == (n, N_SCALARS + K) and ds.target.shape == (n, 1 + K)
        assert ds.dates[0] == d.dates[FIRST_SAMPLE] and ds.dates[-1] == d.dates[-1]
        assert np.array_equal(ds.spot, d.spot[FIRST_SAMPLE:]) and np.array_equal(ds.spot_prev, d.spot[FIRST_SAMPLE - 1:-1])
        assert ds.past_log_surface.shape == (n, K)
        # condition-day dates: one trading day before each predicted day (B21: plot condition data against these)
        assert ds.cond_dates.equals(d.dates[FIRST_SAMPLE - 1:-1]) and ds.cond_dates[1:].equals(ds.dates[:-1])
        # alignment guard: return and the change in 1-month ATM vol are strongly negatively correlated
        # on the same day, and uncorrelated if one series is shifted by a day
        atm = int(np.argmin(np.abs(d.m - 1.0)))              # 1M column of the maturity-major layout
        ret, dg = ds.target[:, 0], ds.target[:, 1 + atm]
        assert np.corrcoef(ret, dg)[0, 1] < -0.5
        assert abs(np.corrcoef(ret[1:], dg[:-1])[0, 1]) < 0.2 and abs(np.corrcoef(ret[:-1], dg[1:])[0, 1]) < 0.2


def test_matches_explicit_loops():
    for path, name in [(SPX, "SPX"), (NDX, "NDX")]:
        d = load_surfaces(path, name)
        rng = np.random.default_rng(1)
        for rv_window in ("paper", "code"):
            for scale in (np.sqrt(252), 1.0):
                ds = data_preprocessing(d, rv_window=rv_window, ret_scale=scale)
                for j in list(rng.integers(0, len(ds.dates), 20)) + [0, len(ds.dates) - 1]:
                    cond, targ = explicit_sample(d, FIRST_SAMPLE + j, rv_window, scale)
                    assert np.allclose(ds.condition[j], cond, rtol=1e-11, atol=1e-13), (name, rv_window, j)
                    assert np.allclose(ds.target[j], targ, rtol=1e-11, atol=1e-13), (name, rv_window, j)


def test_code_window_equals_original_preprocessing():
    """rv_window='code' reproduces the released code's arrays (rows after the first, which needs a close we lack)."""
    for path, name in [(SPX, "SPX"), (NDX, "NDX")]:
        d = load_surfaces(path, name)
        log_rtn = np.concatenate([[0.0], np.diff(np.log(d.spot))])   # index 0 is a placeholder, unused after row 0
        true, cond = original_preprocessing(d.vols, log_rtn)
        ds = data_preprocessing(d, rv_window="code")
        assert len(true) - 1 == len(ds.dates)                         # original starts one row earlier (p = 22)
        assert np.allclose(ds.condition, cond[1:], rtol=1e-12, atol=1e-14), name
        assert np.allclose(ds.target, true[1:], rtol=1e-12, atol=1e-14), name


def test_paper_vs_code_window_differ_only_in_the_vol_column():
    d = load_surfaces(SPX, "SPX")
    a, b = data_preprocessing(d, "paper"), data_preprocessing(d, "code")
    assert np.array_equal(a.target, b.target) and np.array_equal(a.dates, b.dates)
    other = [k for k in range(a.condition.shape[1]) if k != 2]
    assert np.array_equal(a.condition[:, other], b.condition[:, other])
    assert not np.allclose(a.condition[:, 2], b.condition[:, 2])
    # paper window = code window + the return ending on d-1 - the return ending on d-22
    ret = np.log(d.spot[1:] / d.spot[:-1])                            # ret[k-1] is the return ending on row k
    p = np.arange(FIRST_SAMPLE, len(d.dates))
    gap = (a.condition[:, 2] ** 2 - b.condition[:, 2] ** 2) * WINDOW / 252
    assert np.allclose(gap, ret[p - 2] ** 2 - ret[p - 23] ** 2, atol=1e-13)


def test_ret_scale_switch():
    d = load_surfaces(SPX, "SPX")
    a, b = data_preprocessing(d), data_preprocessing(d, ret_scale=1.0)
    assert np.isclose(a.ret_scale, np.sqrt(252)) and b.ret_scale == 1.0
    for col in (0, 1):
        assert np.allclose(a.condition[:, col], np.sqrt(252) * b.condition[:, col])
    assert np.allclose(a.target[:, 0], np.sqrt(252) * b.target[:, 0])
    assert np.array_equal(a.target[:, 1:], b.target[:, 1:]) and np.array_equal(a.condition[:, 2:], b.condition[:, 2:])


def test_bad_arguments_rejected():
    d = load_surfaces(SPX, "SPX")
    try:
        data_preprocessing(d, rv_window="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("bad rv_window accepted")


# =====================================================================================================================
# from test_stage4.py
# =====================================================================================================================


def datasets():
    for path, name in [(SPX, "SPX"), (NDX, "NDX")]:
        d = load_surfaces(path, name)
        yield name, d, data_preprocessing(d)


def test_chronological_complete_and_by_year():
    for name, d, ds in datasets():
        sp = split_by_year(ds, 0.8)
        tr, te = sp.train, sp.test
        assert len(tr.dates) + len(te.dates) == len(ds.dates)
        assert tr.dates.max() < te.dates.min()                           # train entirely before test
        assert te.dates[0].year == sp.test_start_year and tr.dates[-1].year == sp.test_start_year - 1
        assert tr.dates.is_monotonic_increasing and te.dates.is_monotonic_increasing
        # nothing reordered or dropped: the two parts glue back into the original arrays
        assert np.array_equal(np.concatenate([tr.condition, te.condition]), ds.condition)
        assert np.array_equal(np.concatenate([tr.target, te.target]), ds.target)
        assert tr.dates.append(te.dates).equals(ds.dates)


def test_year_is_the_closest_to_the_requested_fraction():
    for name, d, ds in datasets():
        years = ds.dates.year.to_numpy()
        shares = {int(y): (years < y).sum() / len(years) for y in np.unique(years)[1:]}
        best = min(shares, key=lambda y: abs(shares[y] - 0.8))
        sp = split_by_year(ds, 0.8)
        assert sp.test_start_year == best and np.isclose(sp.train_fraction, shares[best])
        assert 0.75 < sp.train_fraction < 0.85, (name, sp.train_fraction)
    assert split_by_year(data_preprocessing(load_surfaces(SPX, "SPX")), 0.8).test_start_year == 2022
    assert split_by_year(data_preprocessing(load_surfaces(NDX, "NDX")), 0.8).test_start_year == 2023


def test_explicit_year_and_bad_year():
    name, d, ds = next(datasets())
    sp = split_by_year(ds, test_start_year=2020)
    assert sp.test_start_year == 2020 and sp.test.dates[0].year == 2020 and sp.train.dates[-1].year == 2019
    for bad in (2007, 2026, 1999):                                       # first year, after the last, before the first
        try:
            split_by_year(ds, test_start_year=bad)
        except ValueError:
            continue
        raise AssertionError("year %d accepted" % bad)


def test_test_dates_line_up_with_the_data():
    """forecast i of the test set belongs to dates_test[i]: look each date up in the loaded data and recompute."""
    for name, d, ds in datasets():
        te = split_by_year(ds, 0.8).test
        rows = d.dates.get_indexer(te.dates)                             # the row of each predicted day
        assert (rows >= 1).all()
        assert np.array_equal(te.spot, d.spot[rows]) and np.array_equal(te.spot_prev, d.spot[rows - 1])
        assert np.allclose(te.target[:, 0], te.ret_scale * np.log(d.spot[rows] / d.spot[rows - 1]))
        assert np.allclose(te.target[:, 1:], np.log(d.vols[rows]) - np.log(d.vols[rows - 1]))


def test_to_tensors():
    name, d, ds = next(datasets())
    te = split_by_year(ds, 0.8).test
    cond, targ = to_tensors(te, torch.device("cpu"))
    assert cond.dtype == torch.float32 and targ.dtype == torch.float32
    assert tuple(cond.shape) == te.condition.shape and tuple(targ.shape) == te.target.shape
    assert np.allclose(cond.numpy(), te.condition, rtol=1e-6, atol=1e-7)
    assert np.allclose(targ.numpy(), te.target, rtol=1e-6, atol=1e-7)


# =====================================================================================================================
# from test_stage9.py
# =====================================================================================================================


OUR = (np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]), np.array([30, 60, 90, 180, 270, 360]) / 365)

ORIG = (np.linspace(0.6, 1.4, 10), np.array([30, 60, 91, 122, 152, 182, 273, 365]) / 365)

PAPER = (np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4]),
         np.array([1 / 252, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1]))


def flat(m, tau, sig):
    return np.full((1, len(m), len(tau)), sig)


def test_pricing_parity_and_eq1():
    rng = np.random.default_rng(0)
    # 200 one-point surfaces, shape (200, 1, 1), each with its own rate (one per surface, shape (200,))
    m, tau, sig = (rng.uniform(lo, hi, (200, 1, 1)) for lo, hi in [(0.7, 1.3), (0.02, 1.0), (0.05, 0.8)])
    r_ = rng.uniform(-0.01, 0.06, 200)
    c = relative_call(m, tau, sig, r_)
    r = r_[:, None, None]
    # an independently written put, then put-call parity: c - p = 1 - m exp(-r tau), whatever sigma is
    d1 = (-np.log(m) + (r + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
    p = m * np.exp(-r * tau) * norm.cdf(-(d1 - sig * np.sqrt(tau))) - norm.cdf(-d1)
    assert np.allclose(c - p, 1 - m * np.exp(-r * tau), rtol=0, atol=1e-13)
    # the paper's eq 1, i.e. the original smallBS, with a rate
    assert np.allclose(relative_call(m, tau, sig, 0.03), smallBS(m, tau, sig, 0.03), rtol=1e-12, atol=0)


def test_rate_shapes():
    m, tau = OUR
    ms, ts = np.meshgrid(m, tau, indexing="ij")
    sig = np.random.default_rng(1).uniform(0.1, 0.4, (4, 7, 6))
    r = np.random.default_rng(2).uniform(0, 0.05, (4, 6))                    # per surface and maturity
    C = relative_call(ms, ts, sig, r)
    for b in range(4):
        for j in range(6):
            assert np.allclose(C[b, :, j], relative_call(m, tau[j], sig[b, :, j], r[b, j]))
    assert np.allclose(relative_call(ms, ts, sig, np.full(4, 0.02)), relative_call(ms, ts, sig, 0.02))
    try:
        rate_array(np.zeros((2, 2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("3-d rate accepted")


def test_b19_float64_does_not_invent_arbitrage():
    def price32(m, tau, sig):                   # the whole price in float32, as the released smallBS_tensor
        m_, t_, s_ = (torch.tensor(x, dtype=torch.float32) for x in (m, tau, sig))
        d1 = (-torch.log(m_) + t_ * 0.5 * s_ ** 2) / (s_ * torch.sqrt(t_))
        return (torch.special.ndtr(d1) - m_ * torch.special.ndtr(d1 - s_ * torch.sqrt(t_))).double().numpy()
    worst32 = 0.0
    for m, tau in (OUR, ORIG, PAPER):
        ms, ts = np.meshgrid(m, tau, indexing="ij")
        for sig in np.linspace(0.05, 0.8, 16):                               # flat vol: no arbitrage at all
            for r in (0.0, 0.05):
                assert arbitrage_penalty(relative_call(ms, ts, flat(m, tau, sig), r), m, tau).total[0] < 1e-14
            worst32 = max(worst32, arbitrage_penalty(price32(ms[None], ts[None], flat(m, tau, sig)), m, tau).total[0])
    assert worst32 > 1e-7                                                    # float32 invents some


def test_why_dividends_are_left_out():
    """
    Evidence for the decision (F14). With a dividend yield q, c = exp(-q tau) N(d1) - m exp(-r tau) N(d2).
    A flat-vol surface has no arbitrage, yet the paper's calendar term at fixed spot moneyness flags it
    whenever q > r. Without dividends (the paper's model, and ours) it never does, for any r >= 0.
    """
    m, tau = OUR
    ms, ts = np.meshgrid(m, tau, indexing="ij")

    def call_with_dividends(sig, r, q):
        d1 = (-np.log(ms) + ts * (r - q + sig ** 2 / 2)) / (sig * np.sqrt(ts))
        return (np.exp(-q * ts) * norm.cdf(d1) - ms * np.exp(-r * ts) * norm.cdf(d1 - sig * np.sqrt(ts)))[None]

    for r, q, flagged in [(0.0, 0.0, False), (0.0, 0.015, True), (0.01, 0.03, True), (0.04, 0.015, False)]:
        p = arbitrage_penalty(call_with_dividends(0.2, r, q), m, tau)
        assert p.call[0] == 0 and p.butterfly[0] < 1e-14                     # strike conditions unaffected
        assert (p.calendar[0] > 1e-6) == flagged, (r, q, p.calendar[0])
    for r in (0.0, 0.02, 0.05):                                              # no dividends: never flagged
        assert arbitrage_penalty(relative_call(ms, ts, flat(m, tau, 0.2), r), m, tau).calendar[0] == 0


@functools.lru_cache(maxsize=1)
def spx():
    data, split = prepare("SPX", Config())
    set_seed(0)
    gen, _ = build_networks(N_SCALARS + data.lk * data.lt, 1 + data.lk * data.lt, Config(), torch.device("cpu"))
    return data, split.test, gen


def test_benchmark_and_scenarios_priced_the_same_way():
    data, test, gen = spx()
    # benchmark (F13): the market surfaces of the test days, through the same function as the scenarios
    rows = data.dates.get_indexer(test.dates)
    bench = scenario_phi(market_sigma(test)[:, None, :], test.m, test.tau).total[:, 0]
    # the two paths differ only by the log/exp round trip of market_sigma (3e-16 in sigma); on the 49-point grid's finer
    # moneyness spacing that noise can leave a ~1e-14 term at a positive-part kink, hence atol 1e-12
    assert np.allclose(bench, market_phi(data).total[rows], rtol=1e-12, atol=1e-12)
    s = summary([1.0, 2.0, 6.0])
    assert np.isclose(s["mean"], 3.0) and np.isclose(s["median"], 2.0) and np.isclose(s["std"], np.std([1, 2, 6]))
    # a per-day, per-maturity rate reaches every scenario of its own day
    sub = test.slice(0, 4)
    r = np.linspace(0.0, 0.05, 4 * sub.tau.size).reshape(4, sub.tau.size)
    sim = simulate(gen, sub, 300, seed=0, rate=r)
    _, sigma = day_scenarios(gen, sub, 2, 300, seed=0)
    one = scenario_phi(sigma[None], sub.m, sub.tau, r[2:3]).total[0]
    assert np.allclose(sim.phi[2], one, rtol=1e-6, atol=1e-12) and np.array_equal(sim.rate, r)


def write_rates(path, frame):
    with pd.ExcelWriter(path) as xl:
        frame.to_excel(xl, sheet_name="rates", index=False)


def test_rate_loader():
    m, tau = OUR
    days = pd.bdate_range("2023-01-02", periods=6)
    rates = pd.DataFrame({"Date": days, "1M": 4.0, "3M": 4.6, "12M": 5.2})
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "rates.xlsx")
        write_rates(f, rates)
        got = load_rates(f, days, tau)
        # by hand: tenor nM = 30n days; simple yield -> continuous; linear in maturity between tenors
        T = np.array([30, 90, 360]) / 365
        r_t = np.log1p(np.array([0.040, 0.046, 0.052]) * T) / T
        assert np.allclose(got.r[0], np.interp(tau, T, r_t)) and np.isclose(got.r[0, 0], r_t[0])
        assert np.isclose(got.r[0, 1], (r_t[0] + r_t[1]) / 2)                # 60 days: halfway between 1M and 3M
        assert got.r.shape == (6, 6) and got.source == "rates"
        assert np.array_equal(got.on(days[[4, 1]]).r, got.r[[4, 1]])
        assert np.array_equal(get_rates("rates", days, tau, path=f).r, got.r)
        # a bond-market holiday is filled from the day before; a long gap is an error
        write_rates(f, rates.drop(index=2))
        assert np.allclose(load_rates(f, days, tau).r[2], got.r[1])
        bad = [rates.drop(index=[1, 2, 3, 4]),                                          # 4-day gap
               rates.assign(**{"1M": 0.040, "3M": 0.046, "12M": 0.052}),               # decimals, not percent
               rates[["Date", "1M"]]]                                                   # a single tenor
        for frame in bad:
            write_rates(f, frame)
            try:
                load_rates(f, days, tau)
            except ValueError:
                continue
            raise AssertionError("bad workbook accepted")
    z = get_rates("zero", days, tau)
    assert z.source == "zero" and not z.r.any()
    try:
        get_rates("parity", days, tau)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown source accepted")


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
        test_shapes_and_ranges,
        test_reshape_puts_values_where_expected,
        test_layout_market_penalty,
        test_validate_rejects_bad_input,
        test_grid_dates_and_units,
        test_rates_joined_by_date,
        test_bad_rates_refused,
        test_hand_computed_violations,
        test_relative_call_known_values,
        test_matches_original_on_market_surfaces,
        test_market_phi_properties_and_rate,
        test_bad_shape_rejected,
        test_blocks_are_the_configured_dates,
        test_shapes_and_alignment,
        test_matches_explicit_loops,
        test_code_window_equals_original_preprocessing,
        test_paper_vs_code_window_differ_only_in_the_vol_column,
        test_ret_scale_switch,
        test_bad_arguments_rejected,
        test_chronological_complete_and_by_year,
        test_year_is_the_closest_to_the_requested_fraction,
        test_explicit_year_and_bad_year,
        test_test_dates_line_up_with_the_data,
        test_to_tensors,
        test_pricing_parity_and_eq1,
        test_rate_shapes,
        test_b19_float64_does_not_invent_arbitrage,
        test_why_dividends_are_left_out,
        test_benchmark_and_scenarios_priced_the_same_way,
        test_rate_loader,
    ], "Preprocessing")
