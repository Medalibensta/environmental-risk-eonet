"""
Geographic hotspot detection with density-based spatial clustering.

DBSCAN on the haversine metric groups events that are geographically dense
without assuming a number of clusters or a particular shape — exactly what we
want for hazard hotspots. We cluster the wildfire events (the densest, most
discrete category) because storms are long tracks rather than point events.

    * metric   = haversine (inputs in radians), so `eps` is a real distance;
    * eps      = 150 km, min_samples = 30 -> "a recurring wildfire zone";
    * noise    (label -1) = isolated events, excluded from hotspots.

Outputs:
    * reports/figures/wildfire_hotspots.png   (static)
    * reports/figures/eonet_hotspots_map.html (interactive Plotly map)
    * reports/hotspots.csv                    (ranked hotspot table)

Run:
    python src/hotspots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN

from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
EARTH_KM = 6371.0088


def cluster_category(
    df: pd.DataFrame,
    category: str = "Wildfires",
    eps_km: float = 150.0,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Run haversine DBSCAN on one category's events; return labelled points."""
    events = (df[df.category_title == category]
              .sort_values("date")
              .groupby("event_id", as_index=False).first())
    coords = np.radians(events[["lat", "lon"]].to_numpy())
    db = DBSCAN(eps=eps_km / EARTH_KM, min_samples=min_samples,
                metric="haversine", algorithm="ball_tree")
    events = events.assign(cluster=db.fit_predict(coords))
    return events


def summarise(events: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters (excluding noise) by event count with a centroid."""
    hot = events[events.cluster >= 0]
    summary = (hot.groupby("cluster")
               .agg(n_events=("event_id", "count"),
                    lat=("lat", "mean"),
                    lon=("lon", "mean"),
                    top_region=("region", lambda s: s.mode().iloc[0]),
                    first_year=("year", "min"),
                    last_year=("year", "max"))
               .sort_values("n_events", ascending=False)
               .reset_index())
    return summary


def run(save: bool = True) -> pd.DataFrame:
    df = load_clean()
    events = cluster_category(df)
    summary = summarise(events)
    n_clusters = (events.cluster >= 0).sum()
    n_hot = events.cluster.nunique() - (1 if (events.cluster == -1).any() else 0)
    print(f"[hotspots] {n_hot} wildfire hotspots covering "
          f"{n_clusters:,} events (noise excluded)")
    print(summary.head(10).to_string(index=False))

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(REPORT_DIR / "hotspots.csv", index=False)

        # Static scatter — clustered points coloured, noise greyed.
        plt.figure(figsize=(13, 6.5))
        noise = events[events.cluster == -1]
        clustered = events[events.cluster >= 0]
        plt.scatter(noise.lon, noise.lat, s=3, c="lightgray", alpha=0.4,
                    label="isolés (bruit)")
        plt.scatter(clustered.lon, clustered.lat, s=6, c=clustered.cluster,
                    cmap="tab20", alpha=0.7)
        plt.scatter(summary.lon, summary.lat, s=summary.n_events / 3,
                    facecolors="none", edgecolors="red", linewidths=1.5,
                    label="centre de hotspot")
        plt.xlim(-180, 180); plt.ylim(-90, 90)
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title("Hotspots d'incendies (DBSCAN haversine, eps=150 km)")
        plt.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "wildfire_hotspots.png", dpi=130)
        plt.close()

        # Interactive Plotly map of ranked hotspots.
        fig = px.scatter_geo(
            summary, lat="lat", lon="lon", size="n_events",
            color="n_events", color_continuous_scale="YlOrRd",
            hover_name="top_region",
            hover_data={"n_events": True, "first_year": True,
                        "last_year": True, "lat": ":.1f", "lon": ":.1f"},
            projection="natural earth",
            title="Hotspots mondiaux d'incendies (EONET) — taille = nb d'événements")
        fig.write_html(FIG_DIR / "eonet_hotspots_map.html",
                       include_plotlyjs="cdn")
        print(f"[hotspots] saved figures + map + hotspots.csv to {REPORT_DIR}")

    return summary


if __name__ == "__main__":
    run()
