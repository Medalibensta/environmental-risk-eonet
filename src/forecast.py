"""
Forecasting future event frequency with Prophet.

For the highest-volume categories we build a monthly count series of distinct
events and fit Prophet (yearly seasonality on, weekly/daily off — monthly data)
to project the next 18 months with uncertainty bands. Prophet is well suited
here because these hazards have a strong, stable annual cycle (fire season,
storm season) on top of a slow trend.

To avoid the partial-current-month artefact, the last (incomplete) month is
dropped before fitting.

Outputs:
    * reports/figures/forecast_<category>.png
    * reports/forecast_summary.csv   (next-12-months projected totals)

Run:
    python src/forecast.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

from preprocessing import load_clean

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

CATEGORIES = ["Wildfires", "Severe Storms", "Floods"]
HORIZON_MONTHS = 18


def monthly_counts(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Distinct events per calendar month for one category -> Prophet frame."""
    events = (df[df.category_title == category]
              .sort_values("date")
              .groupby("event_id", as_index=False).first())
    events["month"] = events.date.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    ts = (events.groupby("month").event_id.count()
          .rename("y").reset_index().rename(columns={"month": "ds"}))
    # Reindex to a continuous monthly axis (fill gaps with 0 events).
    full = pd.date_range(ts.ds.min(), ts.ds.max(), freq="MS")
    ts = ts.set_index("ds").reindex(full, fill_value=0).rename_axis("ds").reset_index()
    return ts.iloc[:-1]  # drop the incomplete current month


def forecast_category(ts: pd.DataFrame) -> tuple[Prophet, pd.DataFrame]:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="multiplicative",
                interval_width=0.8)
    m.fit(ts)
    future = m.make_future_dataframe(periods=HORIZON_MONTHS, freq="MS")
    fc = m.predict(future)
    return m, fc


def run(save: bool = True) -> pd.DataFrame:
    df = load_clean()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for cat in CATEGORIES:
        ts = monthly_counts(df, cat)
        if len(ts) < 24:
            print(f"[forecast] {cat}: too few months ({len(ts)}), skipped")
            continue
        print(f"[forecast] fitting Prophet for {cat} ({len(ts)} months)...")
        m, fc = forecast_category(ts)

        if save:
            fig = m.plot(fc, figsize=(10, 5))
            ax = fig.gca()
            ax.set_title(f"Prévision de fréquence mensuelle — {cat}")
            ax.set_xlabel("Date"); ax.set_ylabel("Événements / mois")
            fig.tight_layout()
            slug = cat.lower().replace(" ", "_")
            fig.savefig(FIG_DIR / f"forecast_{slug}.png", dpi=130)
            plt.close(fig)

        future_only = fc[fc.ds > ts.ds.max()].head(12)
        rows.append({
            "category": cat,
            "next12m_projected": float(future_only.yhat.clip(lower=0).sum()),
            "next12m_low": float(future_only.yhat_lower.clip(lower=0).sum()),
            "next12m_high": float(future_only.yhat_upper.clip(lower=0).sum()),
            "last12m_actual": float(ts.tail(12).y.sum()),
        })

    summary = pd.DataFrame(rows)
    if save and not summary.empty:
        summary.to_csv(REPORT_DIR / "forecast_summary.csv", index=False)
        print("\n[forecast] 12-month projections:")
        print(summary.to_string(index=False))
        print(f"[forecast] saved figures + summary to {REPORT_DIR}")
    return summary


if __name__ == "__main__":
    run()
