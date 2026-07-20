"""
Ingestion of natural-event data from the NASA EONET v3 API.

EONET (Earth Observatory Natural Event Tracker) curates geolocated, dated
natural events (wildfires, storms, volcanoes, floods, icebergs, ...) from many
authoritative sources. The v3 API returns, per event, a list of `geometry`
points — each an observation with its own date, coordinates and (sometimes) a
magnitude. One storm therefore yields many time-stamped track points.

We page the API by calendar year (`start`/`end` params) to avoid any single
oversized response, then flatten every geometry point into one tidy row:

    event_id | title | category | source | date | lon | lat | magnitude | unit | closed

Endpoint verified July 2026: https://eonet.gsfc.nasa.gov/api/v3/events

Run:
    python src/ingest.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV = RAW_DIR / "eonet_events.csv"

API = "https://eonet.gsfc.nasa.gov/api/v3/events"
FIRST_YEAR = 2012          # sparse before this; 2015+ is dense
LAST_YEAR = 2026
TIMEOUT = 60


def _fetch_year(year: int, session: requests.Session) -> list[dict]:
    """Fetch every event (open + closed) whose track falls inside `year`."""
    params = {
        "status": "all",
        "start": f"{year}-01-01",
        "end": f"{year}-12-31",
    }
    resp = session.get(API, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("events", [])


def _flatten(events: list[dict]) -> list[dict]:
    """Explode each event's geometry list into one row per observation point."""
    rows: list[dict] = []
    for ev in events:
        cats = ev.get("categories") or [{}]
        category = cats[0].get("id")
        category_title = cats[0].get("title")
        sources = ev.get("sources") or [{}]
        source = sources[0].get("id")
        for geom in ev.get("geometry") or []:
            coords = geom.get("coordinates")
            # Points are [lon, lat]; polygons nest deeper — take the centroid-ish
            # first vertex so every geometry contributes a locatable row.
            if geom.get("type") == "Point":
                lon, lat = coords[0], coords[1]
            else:
                flat = coords
                while isinstance(flat[0], list):
                    flat = flat[0]
                lon, lat = flat[0], flat[1]
            rows.append({
                "event_id": ev.get("id"),
                "title": ev.get("title"),
                "category": category,
                "category_title": category_title,
                "source": source,
                "date": geom.get("date"),
                "lon": lon,
                "lat": lat,
                "magnitude": geom.get("magnitudeValue"),
                "magnitude_unit": geom.get("magnitudeUnit"),
                "closed": ev.get("closed"),
            })
    return rows


def download(save: bool = True) -> pd.DataFrame:
    """Page the API year by year and return the flattened observation table."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "portfolio-eonet/1.0"})

    all_rows: list[dict] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        events = _fetch_year(year, session)
        rows = _flatten(events)
        all_rows.extend(rows)
        print(f"[ingest] {year}: {len(events):>4} events -> {len(rows):>5} points")
        time.sleep(0.3)  # be polite to the public API

    df = pd.DataFrame(all_rows)
    # A track point can appear under two adjacent years (spans New Year); dedupe.
    df = df.drop_duplicates(subset=["event_id", "date", "lon", "lat"])
    df = df.reset_index(drop=True)
    print(f"[ingest] total {len(df):,} unique observation points "
          f"across {df.event_id.nunique():,} events")

    if save:
        df.to_csv(RAW_CSV, index=False)
        print(f"[ingest] saved -> {RAW_CSV.relative_to(DATA_DIR.parent)}")
    return df


def load_raw() -> pd.DataFrame:
    """Return the cached raw table, downloading it on first run."""
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    return download()


if __name__ == "__main__":
    frame = download()
    print(frame.category_title.value_counts())
    print(frame.head())
