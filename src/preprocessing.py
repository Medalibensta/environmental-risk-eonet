"""
Cleaning and spatio-temporal structuring of the EONET observation table.

Turns the raw geometry points into an analysis-ready frame:

    * parse ISO dates -> datetime, derive year / month / season;
    * drop points with impossible coordinates;
    * assign each point to a coarse **continent/ocean region** (bounding boxes)
      and to a **grid cell** (default 5°) for hotspot & risk aggregation;
    * keep a `is_open` flag from the `closed` field.

No geopandas dependency — regions are assigned with simple lon/lat boxes so the
project installs cleanly on any platform.

Run:
    python src/preprocessing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ingest import load_raw

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "eonet_clean.csv"

GRID_DEG = 5.0

# Coarse region boxes (lon_min, lon_max, lat_min, lat_max). Checked in order;
# first match wins, so more specific land boxes precede the ocean catch-alls.
REGION_BOXES: list[tuple[str, float, float, float, float]] = [
    ("North America", -168, -52, 15, 84),
    ("South America", -82, -34, -56, 15),
    ("Europe", -25, 45, 36, 72),
    ("Africa", -18, 52, -36, 37),
    ("Middle East / C. Asia", 45, 75, 12, 45),
    ("South Asia", 60, 97, 5, 39),
    ("East Asia", 97, 146, 18, 54),
    ("Southeast Asia", 95, 141, -11, 18),
    ("Oceania", 110, 180, -48, -10),
    ("Russia / N. Asia", 45, 180, 45, 82),
    ("Antarctica", -180, 180, -90, -60),
    ("Arctic", -180, 180, 72, 90),
]

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def _assign_region(lon: float, lat: float) -> str:
    for name, lo0, lo1, la0, la1 in REGION_BOXES:
        if lo0 <= lon <= lo1 and la0 <= lat <= la1:
            return name
    return "Open Ocean"


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the cleaned, spatio-temporally structured frame."""
    df = load_raw() if df is None else df.copy()

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "lon", "lat"])

    # Valid geographic range.
    df = df[(df.lon.between(-180, 180)) & (df.lat.between(-90, 90))]

    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["season"] = df.month.map(SEASONS)
    df["is_open"] = df["closed"].isna()

    df["region"] = [
        _assign_region(lo, la) for lo, la in zip(df.lon, df.lat)
    ]
    # Grid cell (lower-left corner of the containing GRID_DEG cell).
    df["cell_lon"] = (np.floor(df.lon / GRID_DEG) * GRID_DEG).astype(int)
    df["cell_lat"] = (np.floor(df.lat / GRID_DEG) * GRID_DEG).astype(int)
    df["cell"] = df.cell_lon.astype(str) + "_" + df.cell_lat.astype(str)

    df = df.reset_index(drop=True)
    return df


def save(df: pd.DataFrame) -> None:
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"[preprocess] saved -> {PROCESSED_CSV.relative_to(DATA_DIR.parent)}")


def load_clean() -> pd.DataFrame:
    """Load the cached clean frame, building it on first run."""
    if PROCESSED_CSV.exists():
        out = pd.read_csv(PROCESSED_CSV, parse_dates=["date"])
        return out
    out = clean()
    save(out)
    return out


if __name__ == "__main__":
    frame = clean()
    save(frame)
    print(f"{len(frame):,} points | {frame.year.min()}-{frame.year.max()}")
    print("\nBy region:")
    print(frame.region.value_counts())
    print("\nBy category:")
    print(frame.category_title.value_counts())
