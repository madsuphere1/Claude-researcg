"""Feature and label engineering for XAUUSD 15m bars.

Conventions
-----------
* Input index is bar *start* time in fixed EST (UTC-5). A bar indexed t is
  closed (fully known) at t+15min. Every feature at row t uses only data up
  to and including bar t; every label at row t uses only bars t+1 onward.
* Structure features that need future confirmation (swings, FVG fills) are
  lagged by their confirmation delay, never centred.
* All ratio-like features are normalised by price or ATR so they are
  stationary across a 4x price range (900 -> 4000+).

Output: research/data/features_15m.parquet (float32 features + labels).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EPS = 1e-12


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df.close.shift(1)
    return pd.concat([df.high - df.low, (df.high - pc).abs(),
                      (df.low - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def rolling_slope(s: pd.Series, n: int) -> pd.Series:
    """OLS slope of s on time over trailing n bars, vectorised."""
    x = np.arange(n) - (n - 1) / 2
    denom = (x**2).sum()
    return s.rolling(n).apply(lambda v: float(np.dot(v - v.mean(), x)) / denom,
                              raw=True)


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - m) / (sd + EPS)


def pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Rolling percentile rank of the latest value within trailing n bars."""
    arr = s.to_numpy(dtype=np.float64)
    out = np.full(len(arr), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if len(arr) >= n:
        w = sliding_window_view(arr, n)
        out[n - 1:] = (w[:, :-1] < w[:, -1:]).mean(axis=1)
    return pd.Series(out, index=s.index)


# --------------------------------------------------------------------------
# feature families
# --------------------------------------------------------------------------

def f_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    lr = np.log(df.close).diff()
    for n in (1, 2, 4, 8, 16, 32, 96, 192, 480):
        out[f"ret_{n}"] = np.log(df.close / df.close.shift(n))
    for n in (16, 96, 480):
        out[f"ret_z_{n}"] = zscore(lr, n)
    out["ret_skew_32"] = lr.rolling(32).skew()
    out["ret_skew_96"] = lr.rolling(96).skew()
    out["ret_kurt_96"] = lr.rolling(96).kurt()
    out["ret_autocorr1_96"] = lr.rolling(96).apply(
        lambda v: np.corrcoef(v[:-1], v[1:])[0, 1] if v.std() > 0 else 0.0,
        raw=True)
    # variance ratio (trendiness): var of 4-bar sums vs 4x var of 1-bar
    v1 = lr.rolling(96).var()
    v4 = lr.rolling(4).sum().rolling(96).var()
    out["var_ratio_4_96"] = v4 / (4 * v1 + EPS)
    return pd.DataFrame(out, index=df.index)


def f_candles(df: pd.DataFrame, a14: pd.Series) -> pd.DataFrame:
    o, h, l, c = df.open, df.high, df.low, df.close
    rng = (h - l).replace(0, np.nan)
    out = {
        "body_atr": (c - o) / (a14 + EPS),
        "range_atr": (h - l) / (a14 + EPS),
        "upwick_frac": (h - np.maximum(o, c)) / rng,
        "dnwick_frac": (np.minimum(o, c) - l) / rng,
        "close_pos": (c - l) / rng,
        "body_frac": (c - o).abs() / rng,
        "gap_atr": (o - c.shift(1)) / (a14 + EPS),
    }
    up = (c > o).astype(np.int8)
    for n in (3, 8, 20):
        out[f"up_frac_{n}"] = up.rolling(n).mean()
    # consecutive same-direction closes
    dirs = np.sign(c.diff()).fillna(0).to_numpy()
    streak = np.zeros(len(dirs))
    for i in range(1, len(dirs)):
        streak[i] = streak[i - 1] + dirs[i] if dirs[i] == np.sign(streak[i - 1]) or streak[i - 1] == 0 else dirs[i]
    out["close_streak"] = pd.Series(streak, index=df.index)
    # 3-bar patterns
    out["engulf"] = (((c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))).astype(np.int8)
                     - ((c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))).astype(np.int8))
    out["inside_bar"] = ((h <= h.shift(1)) & (l >= l.shift(1))).astype(np.int8)
    out["outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(np.int8)
    return pd.DataFrame(out, index=df.index)


def f_trend(df: pd.DataFrame, a14: pd.Series) -> pd.DataFrame:
    c = df.close
    out = {}
    for n in (8, 20, 50, 100, 200):
        e = ema(c, n)
        out[f"ema{n}_dist"] = (c - e) / (a14 + EPS)
        out[f"ema{n}_slope"] = e.diff(4) / (a14 + EPS)
    s = c.rolling(20).mean()
    out["sma20_dist"] = (c - s) / (a14 + EPS)
    out["ema_ribbon"] = (
        (ema(c, 8) > ema(c, 20)).astype(np.int8) + (ema(c, 20) > ema(c, 50)).astype(np.int8)
        + (ema(c, 50) > ema(c, 100)).astype(np.int8) + (ema(c, 100) > ema(c, 200)).astype(np.int8))
    # ADX(14)
    upm = df.high.diff()
    dnm = -df.low.diff()
    plus_dm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=df.index)
    trn = true_range(df).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / (trn + EPS)
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / (trn + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    out["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["di_diff"] = pdi - mdi
    # Aroon(25)
    n = 25
    out["aroon_up"] = df.high.rolling(n + 1).apply(np.argmax, raw=True) / n * 100
    out["aroon_dn"] = df.low.rolling(n + 1).apply(np.argmin, raw=True) / n * 100
    for n in (20, 96):
        out[f"lr_slope_{n}"] = rolling_slope(np.log(c), n) * 1e4
        mean_ = c.rolling(n).mean()
        resid = c - mean_
        out[f"lr_r2_{n}"] = (rolling_slope(c, n)**2 * ((np.arange(n) - (n - 1) / 2)**2).sum()
                             / (resid.rolling(n).var() * n + EPS)).clip(0, 1)
    return pd.DataFrame(out, index=df.index)


def f_momentum(df: pd.DataFrame) -> pd.DataFrame:
    c = df.close
    out = {}
    for n in (7, 14, 28):
        out[f"rsi{n}"] = rsi(c, n)
    macd = ema(c, 12) - ema(c, 26)
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd_n"] = macd / c * 1e4
    out["macd_hist_n"] = (macd - sig) / c * 1e4
    out["macd_hist_slope"] = (macd - sig).diff(3) / c * 1e4
    lo, hi = df.low.rolling(14).min(), df.high.rolling(14).max()
    k = 100 * (c - lo) / (hi - lo + EPS)
    out["stoch_k"] = k
    out["stoch_d"] = k.rolling(3).mean()
    out["willr14"] = -100 * (hi - c) / (hi - lo + EPS)
    tp = (df.high + df.low + c) / 3
    out["cci20"] = ((tp - tp.rolling(20).mean())
                    / (0.015 * tp.rolling(20).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True) + EPS))
    # momentum divergence proxies: price slope vs rsi slope disagreement
    ps = rolling_slope(c, 14)
    rs = rolling_slope(rsi(c, 14), 14)
    out["div_rsi"] = np.sign(ps) * np.sign(-rs) * (ps.abs() > ps.rolling(96).std()).astype(np.int8)
    out["rsi14_z96"] = zscore(rsi(c, 14), 96)
    return pd.DataFrame(out, index=df.index)


def f_volatility(df: pd.DataFrame, a14: pd.Series) -> pd.DataFrame:
    c = df.close
    out = {}
    a96 = atr(df, 96)
    out["atr14_p"] = a14 / c * 1e4
    out["atr96_p"] = a96 / c * 1e4
    out["atr_ratio"] = a14 / (a96 + EPS)
    out["atr14_rank_1y"] = pct_rank(a14 / c, 96 * 252 // 4)
    for n in (20,):
        m = c.rolling(n).mean()
        sd = c.rolling(n).std()
        out["bb_pos"] = (c - m) / (2 * sd + EPS)
        out["bb_width"] = 4 * sd / (m + EPS) * 1e4
        kc_mid = ema(c, 20)
        out["kc_pos"] = (c - kc_mid) / (2 * a14 + EPS)
        out["squeeze"] = (2 * sd < 1.5 * a14).astype(np.int8)
    for n in (20, 96):
        dh, dl = df.high.rolling(n).max(), df.low.rolling(n).min()
        out[f"donch_pos_{n}"] = (c - dl) / (dh - dl + EPS)
        out[f"donch_width_{n}"] = (dh - dl) / (a14 + EPS)
    # realized vol from intra-bar M1 proxies
    rv = df.rv
    out["rv_1"] = np.sqrt(rv) * 1e3
    out["rv_8"] = np.sqrt(rv.rolling(8).sum()) * 1e3
    out["rv_z_96"] = zscore(np.sqrt(rv.rolling(8).sum()), 96)
    out["rv_rank_1y"] = pct_rank(rv.rolling(96).sum(), 96 * 252 // 4)
    lr = np.log(c).diff()
    out["vol_expansion"] = (lr.abs().rolling(4).mean()
                            / (lr.abs().rolling(96).mean() + EPS))
    out["vol_of_vol"] = a14.pct_change().rolling(96).std()
    # Parkinson over 20 bars
    out["park_20"] = np.sqrt((np.log(df.high / df.low)**2).rolling(20).mean()
                             / (4 * np.log(2))) * 1e3
    out["compression_20"] = ((df.high.rolling(20).max() - df.low.rolling(20).min())
                             / (a96 * 20 + EPS))
    return pd.DataFrame(out, index=df.index)


def f_activity(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["n_m1"] = df.n_m1
    out["absret_z"] = zscore(df.absret, 96)
    imb = (df.up_m1 - df.down_m1) / (df.up_m1 + df.down_m1 + EPS)
    out["m1_imb"] = imb
    out["m1_imb_8"] = imb.rolling(8).mean()
    out["m1_imb_32"] = imb.rolling(32).mean()
    eff = np.log(df.close / df.close.shift(1)).abs() / (df.absret + EPS)
    out["efficiency_1"] = eff  # straight-line vs path length (Kaufman-like)
    out["efficiency_8"] = (np.log(df.close / df.close.shift(8)).abs()
                           / (df.absret.rolling(8).sum() + EPS))
    out["range_sum_z"] = zscore(df.range_sum, 96)
    return pd.DataFrame(out, index=df.index)


# ---- market structure ----------------------------------------------------

def swing_flags(df: pd.DataFrame, k: int = 3) -> tuple[pd.Series, pd.Series]:
    """Swing high/low confirmed k bars later; flag placed at CONFIRMATION bar
    (t+k) to avoid lookahead, with the swing price attached."""
    h, l = df.high, df.low
    sh = (h.shift(k) == h.rolling(2 * k + 1).max())
    sl = (l.shift(k) == l.rolling(2 * k + 1).min())
    return sh.fillna(False), sl.fillna(False)


def f_structure(df: pd.DataFrame, a14: pd.Series) -> pd.DataFrame:
    c, h, l, o = df.close, df.high, df.low, df.open
    n = len(df)
    out = pd.DataFrame(index=df.index)
    k = 3
    sh_conf, sl_conf = swing_flags(df, k)
    swing_high_px = pd.Series(np.where(sh_conf, h.shift(k), np.nan), index=df.index).ffill()
    swing_low_px = pd.Series(np.where(sl_conf, l.shift(k), np.nan), index=df.index).ffill()
    prev_swing_high = pd.Series(np.where(sh_conf, h.shift(k), np.nan), index=df.index).shift(1).ffill()
    out["dist_swing_high"] = (swing_high_px - c) / (a14 + EPS)
    out["dist_swing_low"] = (c - swing_low_px) / (a14 + EPS)
    # higher-high / lower-low structure over last 2 swings
    sh_px = pd.Series(np.where(sh_conf, h.shift(k), np.nan), index=df.index)
    sl_px = pd.Series(np.where(sl_conf, l.shift(k), np.nan), index=df.index)
    out["hh"] = (sh_px > sh_px.dropna().shift(1).reindex(df.index)).reindex(df.index).ffill().fillna(False).astype(np.int8)
    hh_seq = sh_px.dropna()
    out["hh"] = pd.Series(np.where(hh_seq > hh_seq.shift(1), 1, -1), index=hh_seq.index).reindex(df.index).ffill().fillna(0)
    ll_seq = sl_px.dropna()
    out["ll"] = pd.Series(np.where(ll_seq < ll_seq.shift(1), 1, -1), index=ll_seq.index).reindex(df.index).ffill().fillna(0)
    out["structure"] = out["hh"] - out["ll"]  # +: bullish HH+HL, -: bearish

    # break of structure / change of character:
    # close crosses the most recent confirmed swing level
    bos_up = (c > swing_high_px) & (c.shift(1) <= swing_high_px.shift(1))
    bos_dn = (c < swing_low_px) & (c.shift(1) >= swing_low_px.shift(1))
    out["bos_up"] = bos_up.astype(np.int8)
    out["bos_dn"] = bos_dn.astype(np.int8)
    bos_dir = pd.Series(np.select([bos_up, bos_dn], [1, -1], 0), index=df.index)
    last_bos = bos_dir.replace(0, np.nan).ffill().fillna(0)
    out["last_bos_dir"] = last_bos
    out["choch"] = ((bos_dir != 0) & (bos_dir == -last_bos.shift(1))).astype(np.int8)
    # bars since last BOS
    idx = np.arange(n)
    mark = np.where(bos_dir.to_numpy() != 0, idx, -1)
    mark = pd.Series(mark).replace(-1, np.nan).ffill().fillna(0).to_numpy()
    out["bars_since_bos"] = np.minimum(idx - mark, 500)

    # liquidity sweep: high pokes above prior swing high but close back below
    out["sweep_high"] = ((h > swing_high_px) & (c < swing_high_px)).astype(np.int8)
    out["sweep_low"] = ((l < swing_low_px) & (c > swing_low_px)).astype(np.int8)
    out["sweep_high_8"] = out["sweep_high"].rolling(8).sum()
    out["sweep_low_8"] = out["sweep_low"].rolling(8).sum()

    # fair value gaps (3-bar): bullish when low(t) > high(t-2)
    bull_fvg = (l > h.shift(2))
    bear_fvg = (h < l.shift(2))
    out["fvg_bull"] = bull_fvg.astype(np.int8)
    out["fvg_bear"] = bear_fvg.astype(np.int8)
    out["fvg_bull_32"] = bull_fvg.rolling(32).sum()
    out["fvg_bear_32"] = bear_fvg.rolling(32).sum()
    # distance to mid of most recent bullish/bearish FVG
    bull_mid = pd.Series(np.where(bull_fvg, (l + h.shift(2)) / 2, np.nan), index=df.index).ffill()
    bear_mid = pd.Series(np.where(bear_fvg, (h + l.shift(2)) / 2, np.nan), index=df.index).ffill()
    out["dist_fvg_bull"] = (c - bull_mid) / (a14 + EPS)
    out["dist_fvg_bear"] = (bear_mid - c) / (a14 + EPS)

    # order block proxy: last down candle before an up-BOS (and vice versa)
    down_candle_low = pd.Series(np.where(c < o, l, np.nan), index=df.index).ffill()
    up_candle_high = pd.Series(np.where(c > o, h, np.nan), index=df.index).ffill()
    ob_bull = pd.Series(np.where(bos_up, down_candle_low, np.nan), index=df.index).ffill()
    ob_bear = pd.Series(np.where(bos_dn, up_candle_high, np.nan), index=df.index).ffill()
    out["dist_ob_bull"] = (c - ob_bull) / (a14 + EPS)
    out["dist_ob_bear"] = (ob_bear - c) / (a14 + EPS)
    return out.astype(np.float32, errors="ignore")


def f_levels(df: pd.DataFrame, a14: pd.Series) -> pd.DataFrame:
    c = df.close
    est = df.index
    out = pd.DataFrame(index=est)
    # trading day = 18:00 EST roll (futures convention)
    tday = (est - pd.Timedelta(hours=18)).date
    tday = pd.Series(tday, index=est)
    g = df.groupby(tday.values)
    day_open = g.open.transform("first")
    day_high = g.high.cummax()
    day_low = g.low.cummin()
    out["dist_day_open"] = (c - day_open) / (a14 + EPS)
    out["dist_day_high"] = (day_high - c) / (a14 + EPS)
    out["dist_day_low"] = (c - day_low) / (a14 + EPS)
    day_range = (day_high - day_low)
    out["day_range_atr"] = day_range / (a14 * 8 + EPS)
    out["pos_in_day_range"] = (c - day_low) / (day_range + EPS)

    # previous day levels
    stats = df.assign(_d=tday.values).groupby("_d").agg(
        pd_high=("high", "max"), pd_low=("low", "min"), pd_close=("close", "last"))
    prev = stats.shift(1)
    m = pd.DataFrame({"_d": tday.values}, index=est).join(prev, on="_d")
    out["dist_pdh"] = (m.pd_high - c) / (a14 + EPS)
    out["dist_pdl"] = (c - m.pd_low) / (a14 + EPS)
    out["dist_pdc"] = (c - m.pd_close) / (a14 + EPS)
    out["above_pdh"] = (c > m.pd_high).astype(np.int8)
    out["below_pdl"] = (c < m.pd_low).astype(np.int8)
    # classic pivots from previous day
    pp = (m.pd_high + m.pd_low + m.pd_close) / 3
    out["dist_pp"] = (c - pp) / (a14 + EPS)
    out["dist_r1"] = (2 * pp - m.pd_low - c) / (a14 + EPS)
    out["dist_s1"] = (c - (2 * pp - m.pd_high)) / (a14 + EPS)

    # weekly / monthly anchors (weeks roll Sunday 18:00 EST)
    week = (est - pd.Timedelta(hours=18)).isocalendar()
    wkey = week["year"].astype(str) + "-" + week["week"].astype(str)
    gw = df.groupby(wkey.values)
    out["dist_week_open"] = (c - gw.open.transform("first")) / (a14 + EPS)
    wstats = df.assign(_w=wkey.values).groupby("_w", sort=False).agg(
        pw_high=("high", "max"), pw_low=("low", "min"))
    wprev = wstats.shift(1)
    mw = pd.DataFrame({"_w": wkey.values}, index=est).join(wprev, on="_w")
    out["dist_pwh"] = (mw.pw_high - c) / (a14 + EPS)
    out["dist_pwl"] = (c - mw.pw_low) / (a14 + EPS)
    mkey = pd.Series((est - pd.Timedelta(hours=18)).strftime("%Y-%m"), index=est)
    gm = df.groupby(mkey.values)
    out["dist_month_open"] = (c - gm.open.transform("first")) / (a14 + EPS)
    mstats = df.assign(_m=mkey.values).groupby("_m", sort=False).agg(
        pm_high=("high", "max"), pm_low=("low", "min"))
    mprev = mstats.shift(1)
    mm = pd.DataFrame({"_m": mkey.values}, index=est).join(mprev, on="_m")
    out["dist_pmh"] = (mm.pm_high - c) / (a14 + EPS)
    out["dist_pml"] = (c - mm.pm_low) / (a14 + EPS)

    # round numbers
    for grid, name in ((10.0, "r10"), (50.0, "r50")):
        nearest = (c / grid).round() * grid
        out[f"dist_{name}"] = (c - nearest) / (a14 + EPS)

    # session TWAP (volume unavailable -> time-weighted) anchored at day roll
    tpx = (df.high + df.low + df.close) / 3
    csum = tpx.groupby(tday.values).cumsum()
    cnt = tpx.groupby(tday.values).cumcount() + 1
    twap = csum / cnt
    out["dist_vwap_day"] = (c - twap) / (a14 + EPS)
    out["dist_vwap_96"] = (c - tpx.rolling(96).mean()) / (a14 + EPS)
    return out


def f_time(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    est = df.index
    utc = df.utc.dt.tz_localize("UTC")
    ny = utc.dt.tz_convert("America/New_York")
    ldn = utc.dt.tz_convert("Europe/London")
    out = pd.DataFrame(index=est)
    hour = est.hour + est.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["hour_est"] = hour
    out["dow"] = est.dayofweek
    out["day_of_month"] = est.day
    out["month"] = est.month
    out["bar_of_day"] = ((hour - 18) % 24 * 4).astype(int)

    nyh = ny.dt.hour + ny.dt.minute / 60
    ldnh = ldn.dt.hour + ldn.dt.minute / 60
    out["sess_tokyo"] = ((hour >= 19) | (hour < 4)).astype(np.int8)
    out["sess_london"] = ((ldnh >= 8) & (ldnh < 16.5)).to_numpy().astype(np.int8)
    out["sess_ny"] = ((nyh >= 8) & (nyh < 17)).to_numpy().astype(np.int8)
    out["sess_overlap"] = (out.sess_london & out.sess_ny).astype(np.int8)
    out["mins_since_ny_open"] = ((nyh - 9.5) % 24 * 60).to_numpy()
    out["mins_to_ny_close"] = ((17 - nyh) % 24 * 60).to_numpy()
    out["is_ny_open_hour"] = ((nyh >= 9.5) & (nyh < 10.5)).to_numpy().astype(np.int8)
    out["is_london_open_hour"] = ((ldnh >= 8) & (ldnh < 9)).to_numpy().astype(np.int8)
    out["is_830_window"] = ((nyh >= 8.25) & (nyh < 9)).to_numpy().astype(np.int8)
    out["is_1000_window"] = ((nyh >= 10) & (nyh < 10.5)).to_numpy().astype(np.int8)
    out["is_1400_window"] = ((nyh >= 14) & (nyh < 15)).to_numpy().astype(np.int8)
    out["is_friday_pm"] = ((est.dayofweek == 4) & (hour >= 12)).astype(np.int8)
    out["is_sunday_open"] = ((est.dayofweek == 6) | ((est.dayofweek == 0) & (hour < 2))).astype(np.int8)

    # weekend / holiday gaps: time jump between consecutive bars
    gap_min = est.to_series().diff().dt.total_seconds() / 60
    out["bar_gap_min"] = gap_min.clip(upper=4000).fillna(15)
    out["after_gap"] = (gap_min > 15).astype(np.int8)
    out["weekend_gap_atr"] = np.where(gap_min > 2000,
                                      (df.open - df.close.shift(1)).abs()
                                      / (atr(df, 14) + EPS), 0.0)

    # event calendar distances
    ny_date = ny.dt.date
    for ev in ("FOMC", "NFP"):
        evd = set(pd.to_datetime(cal.loc[cal.event == ev, "date"]).dt.date)
        is_ev = pd.Series([d in evd for d in ny_date], index=est)
        out[f"is_{ev.lower()}_day"] = is_ev.astype(np.int8)
        ev_hour = 14 if ev == "FOMC" else 8.5
        hrs_to = np.where(is_ev, ev_hour - nyh.to_numpy(), np.nan)
        out[f"hrs_to_{ev.lower()}"] = pd.Series(
            np.where(is_ev, np.clip(hrs_to, -12, 12), 24), index=est)
    days = pd.Series(pd.to_datetime(ny_date), index=est)
    ev_days = np.sort(pd.to_datetime(cal.date.unique()))
    nxt = np.searchsorted(ev_days, days.to_numpy())
    prev_i = np.clip(nxt - 1, 0, len(ev_days) - 1)
    nxt_i = np.clip(nxt, 0, len(ev_days) - 1)
    out["days_to_event"] = ((ev_days[nxt_i] - days.to_numpy()) / np.timedelta64(1, "D")).clip(0, 30)
    out["days_since_event"] = ((days.to_numpy() - ev_days[prev_i]) / np.timedelta64(1, "D")).clip(0, 30)
    return out


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def make_labels(df: pd.DataFrame, a14: pd.Series) -> pd.DataFrame:
    """All labels describe the FUTURE relative to the close of bar t.

    Entry is assumed at the open of bar t+1. Triple-barrier levels are set
    off the entry price with TP = 1.5*ATR14, SL = 1.0*ATR14, horizon 16 bars
    (4 hours). If TP and SL are both touched within one 15m bar the outcome
    is counted as SL (pessimistic tie-break).
    """
    c = df.close
    out = pd.DataFrame(index=df.index)
    entry = df.open.shift(-1)
    for n in (1, 5, 10):
        out[f"fwd_ret_{n}"] = np.log(c.shift(-n) / c)
        out[f"y_dir_{n}"] = (out[f"fwd_ret_{n}"] > 0).astype(np.int8)

    h_ = df.high.to_numpy()
    l_ = df.low.to_numpy()
    e_ = entry.to_numpy()
    a_ = a14.to_numpy()
    n = len(df)
    H = 16
    tp_first_long = np.full(n, np.nan)
    tp_first_short = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    tt = np.full(n, np.nan)
    # bar time gaps: do not let a trade span a weekend
    gap_min = df.index.to_series().diff().dt.total_seconds().to_numpy() / 60

    for i in range(n - H - 1):
        e = e_[i]
        a = a_[i]
        if not np.isfinite(e) or not np.isfinite(a) or a <= 0:
            continue
        tp_l, sl_l = e + 1.5 * a, e - 1.0 * a
        tp_s, sl_s = e - 1.5 * a, e + 1.0 * a
        res_l = res_s = 0.0
        done_l = done_s = False
        hi_max, lo_min = -np.inf, np.inf
        t_used = H
        for j in range(1, H + 1):
            if gap_min[i + j] > 120:  # weekend/holiday break -> stop scan
                t_used = j - 1
                break
            hi, lo = h_[i + j], l_[i + j]
            hi_max = max(hi_max, hi)
            lo_min = min(lo_min, lo)
            if not done_l:
                if lo <= sl_l:
                    res_l, done_l = 0.0, True
                elif hi >= tp_l:
                    res_l, done_l = 1.0, True
                if done_l:
                    tt[i] = j
            if not done_s:
                if hi >= sl_s:
                    res_s, done_s = 0.0, True
                elif lo <= tp_s:
                    res_s, done_s = 1.0, True
            if done_l and done_s:
                break
        if t_used >= 1 and np.isfinite(hi_max):
            tp_first_long[i] = res_l if done_l else np.nan
            tp_first_short[i] = res_s if done_s else np.nan
            mfe[i] = (hi_max - e) / a
            mae[i] = (e - lo_min) / a
    out["y_tp_long"] = tp_first_long   # NaN = neither barrier hit (timeout)
    out["y_tp_short"] = tp_first_short
    out["mfe_16"] = mfe
    out["mae_16"] = mae
    out["time_to_hit"] = tt
    return out


# --------------------------------------------------------------------------

def main() -> None:
    bars = pd.read_parquet(HERE / "data" / "xauusd_15m.parquet")
    cal = pd.read_csv(HERE / "data" / "event_calendar.csv", parse_dates=["date"])
    a14 = atr(bars, 14)

    fams = {
        "ret": f_returns(bars),
        "cdl": f_candles(bars, a14),
        "trd": f_trend(bars, a14),
        "mom": f_momentum(bars),
        "vol": f_volatility(bars, a14),
        "act": f_activity(bars),
        "str": f_structure(bars, a14),
        "lvl": f_levels(bars, a14),
        "tim": f_time(bars, cal),
    }
    feats = pd.concat(fams.values(), axis=1)

    # short lags of fast-moving state features: lets models see 1-2 bars of
    # recent dynamics without full sequence modelling
    lag_cols = ["ret_1", "body_atr", "close_pos", "rsi14", "bb_pos", "m1_imb",
                "stoch_k", "macd_hist_n", "vol_expansion", "efficiency_1",
                "range_atr", "gap_atr", "sweep_high", "sweep_low",
                "bos_up", "bos_dn", "dist_vwap_day", "adx14", "di_diff",
                "absret_z"]
    lagged = {}
    for col in lag_cols:
        for lag_n in (1, 2):
            lagged[f"{col}_l{lag_n}"] = feats[col].shift(lag_n)
    feats = pd.concat([feats, pd.DataFrame(lagged, index=feats.index)], axis=1)
    labels = make_labels(bars, a14)
    print(f"features: {feats.shape[1]}  labels: {labels.shape[1]}  rows: {len(feats):,}")

    dupes = feats.columns[feats.columns.duplicated()].tolist()
    assert not dupes, f"duplicate feature names: {dupes}"

    full = pd.concat([feats.astype(np.float32), labels,
                      bars[["open", "high", "low", "close"]]], axis=1)
    full.to_parquet(HERE / "data" / "features_15m.parquet")
    nan_frac = feats.isna().mean().sort_values(ascending=False)
    print("worst NaN fractions:\n", nan_frac.head(10).to_string())


if __name__ == "__main__":
    main()
