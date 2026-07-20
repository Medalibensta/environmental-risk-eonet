"""
Composite natural-hazard risk index per zone (brief step 6).

With no consistent severity field across categories (magnitude is populated only
for storms), the index combines three interpretable, min-max-normalised
components per 5° grid cell:

    frequency  — number of distinct events in the cell (log-scaled, heavy tail);
    recency    — share of the cell's events that occurred in the last 3 years
                 (a cell active recently is riskier than a historically active
                 but now-quiet one);
    diversity  — number of distinct hazard categories present (a cell exposed to
                 fires AND floods AND storms is more complex to manage).

    risk = 0.5·frequency + 0.3·recency + 0.2·diversity          (weights tunable)

The same recipe is also rolled up to the coarse region level for a readable
ranking. This is a *relative exposure* score, explicitly not an absolute
probability — see the README limits.

Run:
    python src/risk_index.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

RECENT_YEARS = 3
WEIGHTS = {"frequency": 0.5, "recency": 0.3, "diversity": 0.2}


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def _components(events: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cutoff = int(events.year.max()) - RECENT_YEARS + 1
    grp = events.groupby(by)
    out = grp.agg(
        n_events=("event_id", "count"),
        n_categories=("category_title", "nunique"),
    )
    recent = (events[events.year >= cutoff].groupby(by).event_id.count()
              .rename("n_recent"))
    out = out.join(recent).fillna({"n_recent": 0})
    out["recency"] = out.n_recent / out.n_events
    out["frequency"] = np.log1p(out.n_events)
    out["diversity"] = out.n_categories
    return out.reset_index()


def _score(comp: pd.DataFrame) -> pd.DataFrame:
    comp = comp.copy()
    for col in ("frequency", "recency", "diversity"):
        comp[f"{col}_n"] = _minmax(comp[col])
    comp["risk_index"] = (
        WEIGHTS["frequency"] * comp.frequency_n
        + WEIGHTS["recency"] * comp.recency_n
        + WEIGHTS["diversity"] * comp.diversity_n
    )
    return comp.sort_values("risk_index", ascending=False)


def run(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_clean()
    events = (df.sort_values("date").groupby("event_id", as_index=False).first())

    cell = _score(_components(events, ["cell_lon", "cell_lat"]))
    region = _score(_components(events, ["region"]))

    print("=== Risk index by region (top) ===")
    print(region[["region", "n_events", "n_categories", "recency",
                  "risk_index"]].round(3).to_string(index=False))

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        cell.to_csv(REPORT_DIR / "risk_index_grid.csv", index=False)
        region.to_csv(REPORT_DIR / "risk_index_region.csv", index=False)

        # Region bar chart.
        plt.figure(figsize=(9, 5.5))
        r = region.sort_values("risk_index")
        plt.barh(r.region, r.risk_index, color=plt.cm.YlOrRd(r.risk_index))
        plt.xlabel("Indice de risque composite (0-1)")
        plt.title("Exposition composite aux aléas naturels par région")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "risk_index_region.png", dpi=130)
        plt.close()

        # Interactive grid-cell risk map (cell centre).
        cell_map = cell.copy()
        cell_map["lon"] = cell_map.cell_lon + 2.5
        cell_map["lat"] = cell_map.cell_lat + 2.5
        fig = px.scatter_geo(
            cell_map, lat="lat", lon="lon", size="n_events",
            color="risk_index", color_continuous_scale="YlOrRd",
            hover_data={"n_events": True, "n_categories": True,
                        "recency": ":.2f", "risk_index": ":.3f"},
            projection="natural earth",
            title="Indice de risque composite par cellule 5° (EONET)")
        fig.write_html(FIG_DIR / "risk_index_map.html", include_plotlyjs="cdn")
        print(f"[risk] saved 2 tables + region chart + map to {REPORT_DIR}")

    return cell, region


if __name__ == "__main__":
    run()
