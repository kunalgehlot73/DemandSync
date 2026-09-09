"""
Phase 5 — Establish Baselines.

The three baselines the roadmap specifies are already sitting in the
feature table as byproducts of Phase 3:
  - Naive           -> lag_1            (forecast t = actual t-1)
  - Seasonal Naive   -> lag_7            (forecast t = actual t-7, same weekday)
  - Moving Average   -> rolling_mean_7   (forecast t = mean of actual t-7..t-1)

Evaluated ONLY on val (2024). Test (2025) stays untouched until Phase 10's
full model comparison -- baselines have no hyperparameters to overfit, so
touching test now wouldn't technically leak anything, but keeping the same
discipline for every model (including baselines) avoids that becoming a
habit once XGBoost/LSTM/TFT tuning starts in Phase 6+.

MASE denominator: in-sample one-step naive MAE, computed PER SKU on the
TRAIN split only (never val/test), per Hyndman & Koehler (2006). MASE is
then averaged across SKUs (macro-average) -- this matters because a
pooled/micro-average would be dominated by high-volume SKUs like MAT008.
MAE/RMSE/WAPE are reported pooled across all SKUs, since those aren't
scale-sensitive in the same way WAPE already normalizes.

Run:
    python src/evaluation/baselines.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

BASELINES = {
    "Naive": "lag_1",
    "Seasonal Naive": "lag_7",
    "Moving Average (7d)": "rolling_mean_7",
}


def load() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DIR / "forecasting_table_split.csv",
        parse_dates=["Date"],
        keep_default_na=False,
        na_values=["", "NaN", "nan"],
    )


def compute_train_naive_mae_per_sku(df: pd.DataFrame) -> pd.Series:
    train = df[df["split"] == "train"].dropna(subset=["lag_1"])
    return train.groupby("Material_ID").apply(
        lambda g: (g["Demand"] - g["lag_1"]).abs().mean()
    )


def evaluate(df: pd.DataFrame, forecast_col: str, naive_mae_per_sku: pd.Series) -> dict:
    d = df.dropna(subset=[forecast_col]).copy()
    error = d["Demand"] - d[forecast_col]
    abs_error = error.abs()

    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    wape = abs_error.sum() / d["Demand"].sum()

    d["abs_error"] = abs_error
    per_sku_mae = d.groupby("Material_ID")["abs_error"].mean()
    per_sku_mase = per_sku_mae / naive_mae_per_sku
    mase = per_sku_mase.mean()

    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MASE": mase, "n": len(d)}


def main() -> None:
    df = load()
    val = df[df["split"] == "val"]

    naive_mae_per_sku = compute_train_naive_mae_per_sku(df)

    print("Evaluating baselines on val (2024) — test set untouched.\n")
    rows = []
    for name, col in BASELINES.items():
        metrics = evaluate(val, col, naive_mae_per_sku)
        rows.append({"Model": name, **metrics})
        print(f"{name:22s}  MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
              f"WAPE={metrics['WAPE']:.4f}  MASE={metrics['MASE']:.3f}  (n={metrics['n']})")

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "baseline_metrics_val.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\nNote: MASE < 1.0 means the baseline beats the in-sample naive")
    print("forecast; MASE = 1.0 is exactly as good as train-period naive.")


if __name__ == "__main__":
    main()
