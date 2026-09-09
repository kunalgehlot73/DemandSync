"""
Phase 3 — builds the forecasting table (Step 7) with feature groups A-D (Step 8).

Leakage discipline (this matters more here than in Phase 4's split logic):
every lag/rolling/EWMA feature is computed from `.shift(1)` BEFORE windowing,
so a feature for date t never sees Final_Demand at date t. This is easy to
get backwards — `series.rolling(7).mean()` on the raw (unshifted) series
silently includes today's value in "yesterday's 7-day average," which is
leakage even though it looks like a legitimate rolling stat.

Run:
    python src/features/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

LAGS = [1, 2, 3, 7, 14, 28, 56]
ROLLING_WINDOWS_MEAN = [7, 14, 28]
ROLLING_WINDOWS_STD = [7, 28]
EWMA_SPAN = 7

# Columns confirmed (by reading synthetic_generator.py) to have ZERO causal
# effect on Final_Demand. Kept as static features anyway since they ARE
# known at prediction time and a real ERP system would supply them — but
# flagged here so Phase 6 feature-importance checks have a documented
# expectation: these should come out near zero.
NON_CAUSAL_STATIC_COLS = ["Lead_Time", "Unit_Cost", "Holding_Cost_Rate", "Ordering_Cost", "Stockout_Penalty"]

# Generation-artifact columns that ARE leakage (Phase 2 finding) — never features.
LEAKAGE_COLS = ["Seasonal_Factor", "Weekly_Factor", "Noise", "Base_Demand"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    mm = pd.read_csv(RAW_DIR / "material_master.csv", keep_default_na=False, na_values=[])
    demand = pd.read_csv(RAW_DIR / "synthetic_7year_demand.csv", parse_dates=["Date"])
    demand = demand.drop(columns=LEAKAGE_COLS)
    return mm, demand


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Material_ID")["Final_Demand"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # shift(1) FIRST: rolling stats for row t are computed only from t-1 backwards.
    shifted = df.groupby("Material_ID")["Final_Demand"].shift(1)
    df["_shifted"] = shifted

    for w in ROLLING_WINDOWS_MEAN:
        df[f"rolling_mean_{w}"] = df.groupby("Material_ID")["_shifted"].transform(
            lambda s: s.rolling(window=w, min_periods=w).mean()
        )
    for w in ROLLING_WINDOWS_STD:
        df[f"rolling_std_{w}"] = df.groupby("Material_ID")["_shifted"].transform(
            lambda s: s.rolling(window=w, min_periods=w).std()
        )
    df[f"ewma_{EWMA_SPAN}"] = df.groupby("Material_ID")["_shifted"].transform(
        lambda s: s.ewm(span=EWMA_SPAN, min_periods=EWMA_SPAN).mean()
    )
    df = df.drop(columns=["_shifted"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_static_features(df: pd.DataFrame, mm: pd.DataFrame) -> pd.DataFrame:
    static_cols = ["Material_ID", "Category", "Seasonality_Type", "Base_Daily_Demand"] + NON_CAUSAL_STATIC_COLS
    return df.merge(mm[static_cols], on="Material_ID", how="left")


def build() -> pd.DataFrame:
    mm, demand = load_raw()
    demand = demand.sort_values(["Material_ID", "Date"]).reset_index(drop=True)

    df = add_lag_features(demand)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_static_features(df, mm)

    df = df.rename(columns={"Final_Demand": "Demand"})
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build()

    n_before = len(df)
    # earliest rows per SKU can't have lag_56 / rolling_28 populated — expected,
    # not a bug. We keep them in the saved table (so nothing is silently lost)
    # but report how many would be dropped by the deepest feature (lag_56).
    n_incomplete = df["lag_56"].isna().sum()
    print(f"Total rows: {n_before}")
    print(f"Rows with incomplete lag_56 history (first 56 days per SKU): {n_incomplete} "
          f"({n_incomplete/n_before:.2%})")
    print(f"Columns: {list(df.columns)}")

    out_path = PROCESSED_DIR / "forecasting_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  shape={df.shape}")


if __name__ == "__main__":
    main()
