"""
Exploratory spatio-temporal analysis of EONET natural events.

Produces four exported figures:
    1. events per year, stacked by category   -> temporal dynamics
    2. category x region heatmap               -> where each hazard concentrates
    3. monthly seasonality of the top hazards  -> intra-year cycles
    4. distinct events per year (dedup track points to events)

All counts are computed on distinct EVENTS where the question is "how many
events" and on observation points only where density matters, to avoid
long-lived storms (many track points) dominating the picture.

Run:
    python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

# Focus reporting on the categories with enough events to model.
TOP_CATEGORIES = ["Severe Storms", "Wildfires", "Sea and Lake Ice",
                  "Volcanoes", "Floods"]


def _events_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct event (first observation), for event-level counts."""
    return (df.sort_values("date")
              .groupby("event_id", as_index=False)
              .first())


def run(save: bool = True) -> pd.DataFrame:
    df = load_clean()
    df = df[df.year.between(2012, 2026)]
    events = _events_frame(df)
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. events per year, stacked by category -------------------------
    piv = (events[events.category_title.isin(TOP_CATEGORIES)]
           .pivot_table(index="year", columns="category_title",
                        values="event_id", aggfunc="count", fill_value=0))
    ax = piv.plot(kind="bar", stacked=True, figsize=(11, 5.5), width=0.85,
                  colormap="tab10")
    ax.set(xlabel="Année", ylabel="Nombre d'événements",
           title="Événements naturels EONET par an et par catégorie")
    ax.legend(title="Catégorie", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "events_per_year.png", dpi=130)
    plt.close()

    # --- 2. category x region heatmap ------------------------------------
    ct = (events.pivot_table(index="category_title", columns="region",
                             values="event_id", aggfunc="count", fill_value=0))
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(12, 5.5))
    sns.heatmap(ct, cmap="rocket_r", annot=True, fmt="d", cbar_kws={"label": "événements"})
    plt.title("Concentration géographique par type d'aléa")
    plt.xlabel("Région"); plt.ylabel("Catégorie")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "category_region_heatmap.png", dpi=130)
    plt.close()

    # --- 3. monthly seasonality ------------------------------------------
    plt.figure(figsize=(10, 5.5))
    for cat in TOP_CATEGORIES:
        m = (events[events.category_title == cat]
             .groupby("month").event_id.count())
        m = m.reindex(range(1, 13), fill_value=0)
        if m.sum() > 0:
            plt.plot(range(1, 13), m.values / m.sum() * 100, marker="o", label=cat)
    plt.xticks(range(1, 13),
               ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    plt.xlabel("Mois"); plt.ylabel("% des événements de la catégorie")
    plt.title("Saisonnalité intra-annuelle par type d'aléa")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "seasonality.png", dpi=130)
    plt.close()

    # --- 4. distinct events per year -------------------------------------
    per_year = events.groupby("year").event_id.count()
    plt.figure(figsize=(10, 4.5))
    plt.plot(per_year.index, per_year.values, marker="o", color="#c44e52")
    plt.fill_between(per_year.index, per_year.values, alpha=0.2, color="#c44e52")
    plt.xlabel("Année"); plt.ylabel("Événements distincts")
    plt.title("Volume annuel d'événements naturels suivis (EONET)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "events_trend.png", dpi=130)
    plt.close()

    if save:
        summary = (events.groupby("category_title")
                   .agg(events=("event_id", "count"),
                        first_year=("year", "min"),
                        last_year=("year", "max"))
                   .sort_values("events", ascending=False))
        summary.to_csv(REPORT_DIR / "category_summary.csv")
        print("[eda] saved 4 figures + category_summary.csv")
        print(summary)

    return events


if __name__ == "__main__":
    run()
