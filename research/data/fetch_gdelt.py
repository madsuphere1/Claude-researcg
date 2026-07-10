"""Fetch weekly GDELT GKG 2.1 tone/theme snapshots (public, free,
data.gdeltproject.org). See C3-007 registry entry for the frozen design.

Coverage: 2015-02-18 onward only (GKG 2.1 does not exist before that;
GKG 1.0, 2013-Feb2015, uses a different daily schema and is not used
here — documented limitation, not engineered around).

One file per week (Monday 12:00:00 UTC, or the nearest available 15-min
slot that week) is downloaded — a bandwidth-bounded proxy for continuous
coverage. Each raw zip (~6-11MB) is parsed immediately for the six
frozen aggregate values and discarded; only the small parsed table is
cached, so disk usage stays bounded regardless of how many weeks are
fetched.

Output: research/data/gdelt_weekly.parquet (date-indexed, one row/week,
columns: avg_tone, tone_negativity, econ_inflation_share,
epu_policy_share, econ_debt_share, armedconflict_share, n_records).
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
BASE = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
START = datetime(2015, 2, 23, tzinfo=timezone.utc)  # first Monday with coverage
THEMES = ["ECON_INFLATION", "EPU_POLICY", "ECON_DEBT", "ARMEDCONFLICT"]
CACHE = HERE / "gdelt_weekly.parquet"
PROGRESS = HERE / "gdelt_weekly_progress.txt"


def week_stamps(end: datetime) -> list[datetime]:
    out, d = [], START
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def fetch_one(ts: datetime) -> dict | None:
    """Try the exact 12:00 slot, then nearby 15-min slots same day."""
    for minute_offset in (0, 15, -15, 30, -30, 45, -45, 60, -60):
        cand = ts + timedelta(minutes=minute_offset)
        stamp = cand.strftime("%Y%m%d%H%M%S")
        url = BASE.format(ts=stamp)
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException:
            continue
        if r.status_code != 200 or len(r.content) < 1000:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]
                raw = z.read(name).decode("utf-8", errors="replace")
        except zipfile.BadZipFile:
            continue
        return parse_gkg(raw)
    return None


def parse_gkg(raw: str) -> dict:
    n = 0
    tone_sum = 0.0
    neg_sum = 0.0
    theme_hits = {t: 0 for t in THEMES}
    for line in raw.splitlines():
        cols = line.split("\t")
        if len(cols) < 16:
            continue
        themes_field = cols[7]
        tone_field = cols[15]
        tone_parts = tone_field.split(",")
        if len(tone_parts) < 3:
            continue
        try:
            tone = float(tone_parts[0])
            neg = float(tone_parts[2])
        except ValueError:
            continue
        n += 1
        tone_sum += tone
        neg_sum += neg
        for t in THEMES:
            if t in themes_field:
                theme_hits[t] += 1
    if n == 0:
        return None
    out = {"avg_tone": tone_sum / n, "tone_negativity": neg_sum / n,
           "n_records": n}
    for t in THEMES:
        out[f"{t.lower()}_share"] = theme_hits[t] / n
    return out


def main() -> None:
    end = datetime.now(timezone.utc) - timedelta(days=2)
    weeks = week_stamps(end)
    rows = {}
    if CACHE.exists():
        existing = pd.read_parquet(CACHE)
        rows = {ts: existing.loc[ts].to_dict() for ts in existing.index}
        print(f"resuming: {len(rows)} weeks already cached", flush=True)

    done = 0
    for ts in weeks:
        if ts in rows:
            continue
        rec = fetch_one(ts)
        if rec is not None:
            rows[ts] = rec
        done += 1
        if done % 20 == 0:
            df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
            df.to_parquet(CACHE)
            PROGRESS.write_text(f"{len(rows)}/{len(weeks)} weeks fetched\n")
            print(f"{len(rows)}/{len(weeks)} weeks", flush=True)

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.to_parquet(CACHE)
    PROGRESS.write_text(f"DONE {len(rows)}/{len(weeks)} weeks fetched\n")
    print(f"DONE. {len(rows)}/{len(weeks)} weeks, saved {CACHE}", flush=True)


if __name__ == "__main__":
    main()
