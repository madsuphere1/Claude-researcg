"""Merge raw HistData M1 archives, quality-check, and resample to 15m bars.

HistData timestamps are US Eastern Standard Time (UTC-5, fixed, no DST).
We keep everything in that clock ("EST") because it makes the futures-style
trading day natural: the day runs 18:00 -> 17:00 EST. We additionally store
a UTC column for session labelling.

Outputs
-------
research/data/xauusd_m1.parquet    merged, deduplicated 1-minute bars
research/data/xauusd_15m.parquet   15-minute OHLC + tickvol + n_m1 bars
research/data/qc_report.txt        data quality findings
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "raw"


def load_m1() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(RAW.glob("*.parquet"))]
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def qc(df: pd.DataFrame) -> str:
    lines = ["XAUUSD M1 data quality report", "=" * 40]
    lines.append(f"rows: {len(df):,}")
    lines.append(f"range: {df.index[0]} .. {df.index[-1]} (EST, UTC-5 fixed)")

    bad_ohlc = df[(df.high < df.low) | (df.high < df.open) | (df.high < df.close)
                  | (df.low > df.open) | (df.low > df.close)]
    lines.append(f"OHLC violations: {len(bad_ohlc)}")

    nonpos = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    lines.append(f"non-positive prices: {len(nonpos)}")

    r = df.close.pct_change().abs()
    lines.append(f"1-min |return| > 2%: {(r > 0.02).sum()} bars (checked as outliers)")

    # monthly bar counts to expose coverage holes
    monthly = df.groupby(df.index.to_period("M")).size()
    med = monthly.median()
    holes = monthly[monthly < 0.55 * med]
    lines.append(f"median bars/month: {med:.0f}")
    lines.append("months with <55% of median coverage:")
    for per, n in holes.items():
        lines.append(f"  {per}: {n:,} bars ({n / med:.0%})")

    # weekend leakage: bars during Fri 17:00 .. Sun 18:00 EST
    dow = df.index.dayofweek
    hour = df.index.hour
    wk = ((dow == 5)
          | ((dow == 4) & (hour >= 17))
          | ((dow == 6) & (hour < 18))).sum()
    lines.append(f"bars inside weekend window: {wk}")
    return "\n".join(lines)


def resample_15m(m1: pd.DataFrame) -> pd.DataFrame:
    # HistData M1 tick volume is always 0, so build activity proxies from
    # intra-bar M1 movement instead: realized variance, absolute-move sum,
    # and up/down M1 candle counts (order-flow proxy).
    m1 = m1.copy()
    dlog = np.log(m1.close).diff()
    m1["rv"] = dlog**2
    m1["absret"] = dlog.abs()
    m1["up"] = (m1.close > m1.open).astype(np.int32)
    m1["down"] = (m1.close < m1.open).astype(np.int32)
    m1["m1range"] = (m1.high - m1.low) / m1.close

    bars = m1.resample("15min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        n_m1=("close", "size"),
        rv=("rv", "sum"),
        absret=("absret", "sum"),
        up_m1=("up", "sum"),
        down_m1=("down", "sum"),
        range_sum=("m1range", "sum"),
    )
    bars = bars.dropna(subset=["open"])
    # a 15m bar built from very few M1 bars around holidays is unreliable;
    # keep it but record n_m1 so downstream code can filter.
    bars["utc"] = bars.index + pd.Timedelta(hours=5)
    return bars


def main() -> None:
    m1 = load_m1()
    report = qc(m1)

    # drop impossible bars rather than repair them
    ok = ~((m1.high < m1.low) | (m1[["open", "high", "low", "close"]] <= 0).any(axis=1))
    m1 = m1[ok]

    bars = resample_15m(m1)
    m1.to_parquet(HERE / "xauusd_m1.parquet")
    bars.to_parquet(HERE / "xauusd_15m.parquet")
    (HERE / "qc_report.txt").write_text(report + "\n")
    print(report)
    print(f"\n15m bars: {len(bars):,}  {bars.index[0]} .. {bars.index[-1]}")
    print(bars.tail(3).to_string())


if __name__ == "__main__":
    main()
