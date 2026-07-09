"""Download XAUUSD 1-minute bars from histdata.com.

HistData serves free ASCII M1 bar archives: one zip per year for past years
(``/ascii/1-minute-bar-quotes/xauusd/<year>``) and one zip per month for the
current year (``.../xauusd/<year>/<month>``). Each download page embeds a
one-time token ``tk`` that must be POSTed to ``get.php`` together with the
form fields, with the page URL as Referer.

Timestamps in the archives are US Eastern Standard Time (GMT-5, no DST).
"""

from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.histdata.com"
PAGE = BASE + "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xauusd/{path}"
GET = BASE + "/get.php"
OUT = Path(__file__).parent / "raw"

FIELD_RE = re.compile(r'id="(tk|date|datemonth|platform|timeframe|fxpair)" value="([^"]*)"')


def fetch_month_or_year(session: requests.Session, path: str) -> pd.DataFrame | None:
    """Download one archive (path like '2022' or '2025/6'); return M1 frame."""
    page_url = PAGE.format(path=path)
    r = session.get(page_url, timeout=60)
    r.raise_for_status()
    fields = dict(FIELD_RE.findall(r.text))
    if "tk" not in fields:
        return None  # page exists but no data for this period
    r = session.post(
        GET,
        data=fields,
        headers={"Referer": page_url},
        timeout=120,
    )
    r.raise_for_status()
    if not r.content[:2] == b"PK":
        raise RuntimeError(f"{path}: response is not a zip ({r.content[:80]!r})")
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(
        zf.open(csv_name),
        sep=";",
        header=None,
        names=["dt", "open", "high", "low", "close", "tickvol"],
    )
    df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    return df.set_index("dt").sort_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research; XAUUSD study)"

    now = pd.Timestamp.utcnow()
    jobs: list[str] = [str(y) for y in range(2009, now.year)]
    jobs += [f"{now.year}/{m}" for m in range(1, now.month + 1)]

    for path in jobs:
        dest = OUT / (path.replace("/", "_") + ".parquet")
        if dest.exists():
            print(f"skip {path} (cached)")
            continue
        try:
            df = fetch_month_or_year(session, path)
        except Exception as exc:  # noqa: BLE001 - log and move on
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print(f"no data {path}")
            continue
        df.to_parquet(dest)
        print(f"ok {path}: {len(df):,} rows  {df.index[0]} .. {df.index[-1]}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
