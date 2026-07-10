"""Build a scheduled-macro-event calendar for XAUUSD research.

Two precisely-dated event families dominate gold's scheduled-news risk:

* FOMC rate decisions (14:00 ET statement) - scraped from federalreserve.gov
  (current calendar page + historical year pages).
* US Non-Farm Payrolls (08:30 ET) - approximated as the first Friday of each
  month, the BLS's standard slot. Occasional shifts (holidays) are accepted
  as label noise and noted in the report.

Other 08:30 ET releases (CPI, PCE, retail sales, GDP) are captured downstream
by time-of-day features rather than exact dates.

Output: research/data/event_calendar.csv  (date, event, hour_est)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


MONTH_RE = ("January|February|March|April|May|June|July|August|September|"
            "October|November|December")


def scrape_fomc_historical() -> list[pd.Timestamp]:
    """Decision (last) day of each meeting from the per-year archive pages.

    Headings look like ``<h5 ...>January 27-28 Meeting - 2015</h5>`` or, for
    cross-month meetings, ``October 31-November 1 Meeting - 2017``. Headings
    for unscheduled events say "Conference Call"/"Notation Vote" and are
    intentionally excluded.
    """
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (research)"
    dates: set[pd.Timestamp] = set()
    for year in range(2009, 2022):
        url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        try:
            resp = sess.get(url, timeout=60)
            if resp.status_code != 200:
                continue
            html = resp.text
        except Exception as exc:  # noqa: BLE001
            print(f"skip {url}: {exc}")
            continue
        for h5 in re.findall(r"<h5[^>]*>([^<]*Meeting[^<]*)</h5>", html):
            m = re.search(
                rf"({MONTH_RE})\s+(\d{{1,2}})"
                rf"(?:\s*-\s*(?:({MONTH_RE})\s+)?(\d{{1,2}}))?", h5)
            if not m:
                continue
            mon1, _, mon2, d2 = m.groups()
            mon = MONTHS[mon2] if mon2 else MONTHS[mon1]
            day = int(d2) if d2 else int(m.group(2))
            dates.add(pd.Timestamp(year=year, month=mon, day=day))
    return sorted(dates)


def fomc_from_calendar_page_panels() -> list[pd.Timestamp]:
    """Fallback parser for the modern calendar page panel markup."""
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    html = requests.get(url, timeout=60,
                        headers={"User-Agent": "Mozilla/5.0"}).text
    dates: set[pd.Timestamp] = set()
    for panel in re.finditer(r"(\d{4}) FOMC Meetings(.*?)(?=\d{4} FOMC Meetings|$)",
                             html, re.S):
        year = int(panel.group(1))
        body = panel.group(2)
        for m in re.finditer(
                r'fomc-meeting__month[^>]*>\s*<strong>\s*(\w+)(?:/(\w+))?\s*'
                r"</strong>\s*</div>\s*<div[^>]*fomc-meeting__date[^>]*>"
                r"\s*([\d\-\*]+)", body, re.S):
            mon_name, mon2_name, daytxt = m.groups()
            mon = MONTHS.get(mon_name)
            if mon is None:
                continue
            nums = re.findall(r"\d{1,2}", daytxt)
            if not nums:
                continue
            day = int(nums[-1])
            # "Jan/Feb 31-1": decision day falls in the second month
            if len(nums) >= 2 and int(nums[-1]) < int(nums[0]):
                mon = MONTHS.get(mon2_name.capitalize(), mon % 12 + 1) \
                    if mon2_name else mon % 12 + 1
            try:
                dates.add(pd.Timestamp(year=year, month=mon, day=day))
            except ValueError:
                continue
    return sorted(dates)


def nfp_dates(start: str, end: str) -> list[pd.Timestamp]:
    out = []
    for per in pd.period_range(start, end, freq="M"):
        d = per.to_timestamp()  # first day of month
        offset = (4 - d.dayofweek) % 7
        out.append(d + pd.Timedelta(days=offset))
    return out


def main() -> None:
    fomc = set(scrape_fomc_historical()) | set(fomc_from_calendar_page_panels())
    fomc = sorted(fomc)
    per_year = pd.Series([d.year for d in fomc]).value_counts().sort_index()
    print("FOMC dates per year:\n", per_year.to_string())

    rows = [{"date": d.date(), "event": "FOMC", "hour_est": 14} for d in fomc]
    rows += [{"date": d.date(), "event": "NFP", "hour_est": 8}
             for d in nfp_dates("2009-01", "2026-07")]
    cal = pd.DataFrame(rows).sort_values("date")
    cal.to_csv(HERE / "event_calendar.csv", index=False)
    print(f"calendar rows: {len(cal)}")


if __name__ == "__main__":
    main()
