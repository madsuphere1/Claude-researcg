"""Acquire BTCUSDT 15m klines from Binance public data dumps
(data.binance.vision) — free, complete, monthly CSV zips from 2017-08.

Unlike the gold HistData set, crypto klines carry REAL microstructure:
base volume, quote volume, trade count, and taker-buy (aggressive-buy)
volume — enabling order-flow features gold lacked. BTC also trades 24/7,
so there are no weekend gaps and no session structure.

Output: research/btc/data/btcusdt_15m.parquet
Columns: open, high, low, close, volume, quote_volume, trades,
         taker_buy_base  (UTC-indexed).
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
CACHE = HERE / "data" / "btcusdt_15m.parquet"
RAW = HERE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{ym}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def months(start=(2017, 8), end=None):
    if end is None:
        now = datetime.utcnow()
        end = (now.year, now.month)
    y, m = start
    out = []
    while (y, m) <= end:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fetch_month(ym: str) -> pd.DataFrame | None:
    cached = RAW / f"{ym}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    try:
        r = requests.get(URL.format(ym=ym), timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200 or len(r.content) < 500:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    # some months have a header row, some don't
    first = raw.split("\n", 1)[0]
    header = 0 if first.lower().startswith("open_time") else None
    df = pd.read_csv(io.StringIO(raw), header=header, names=COLS)
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.iloc[0] > 1_000_000_000_000_000 else "ms"
    df.index = pd.to_datetime(ot.values, unit=unit, utc=True).tz_localize(None)
    out = df[["open", "high", "low", "close", "volume", "quote_volume",
              "trades", "taker_buy_base"]].astype(float)
    out.to_parquet(cached)
    return out


def main() -> None:
    frames = []
    got = 0
    for ym in months():
        d = fetch_month(ym)
        if d is not None and len(d):
            frames.append(d)
            got += 1
            if got % 12 == 0:
                print(f"{ym}: {got} months, {sum(len(f) for f in frames):,} bars", flush=True)
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full.to_parquet(CACHE)
    print(f"DONE {got} months, {len(full):,} bars "
          f"{full.index.min()} .. {full.index.max()}", flush=True)
    print(f"saved {CACHE}")


if __name__ == "__main__":
    main()
