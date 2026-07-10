"""Fetch daily macro series from FRED (public fredgraph.csv endpoint).

Series (all daily):
  DFII10    10Y TIPS real yield (gold's canonical macro driver)
  DTWEXBGS  Broad trade-weighted dollar index (from 2006)
  VIXCLS    CBOE VIX
  DCOILWTICO WTI crude spot

Known caveats recorded here per A5 duty:
* FRED daily observations carry publication lags that vary by series
  (DFII10 same evening; DTWEXBGS ~1-2 business days). Downstream features
  must shift by >= 2 business days - enforced in the feature builder, not
  here.
* Missing values on market holidays; forward-fill is applied by the
  feature builder with a staleness cap of 5 days.

Output: research/data/fred_daily.parquet (date-indexed, one column per id).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
SERIES = ["DFII10", "DTWEXBGS", "VIXCLS", "DCOILWTICO"]
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_csv(sid: str) -> str:
    """requests times out through this proxy for FRED; curl works, so a
    cached curl download (research/data/fred_<sid>.csv) is preferred and
    requests is the fallback."""
    cached = HERE / f"fred_{sid}.csv"
    if cached.exists():
        return cached.read_text()
    r = requests.get(URL.format(sid=sid), timeout=120,
                     headers={"User-Agent": "Mozilla/5.0 (research)"})
    r.raise_for_status()
    return r.text


def main() -> None:
    frames = []
    for sid in SERIES:
        df = pd.read_csv(io.StringIO(fetch_csv(sid)))
        df.columns = ["date", sid]
        df["date"] = pd.to_datetime(df["date"])
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        frames.append(df.set_index("date"))
        print(f"{sid}: {df[sid].notna().sum():,} obs  "
              f"{df['date'].min().date()} .. {df['date'].max().date()}")
    out = pd.concat(frames, axis=1).sort_index()
    out.to_parquet(HERE / "fred_daily.parquet")
    print("saved fred_daily.parquet", out.shape)


if __name__ == "__main__":
    main()
