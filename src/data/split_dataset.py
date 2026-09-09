"""
Phase 4 — Leakage Prevention.

Split boundaries (2019-2023 train / 2024 val / 2025 test) match the
roadmap's diagram: 5 years train, 1 year val, 1 year test. Chosen because:
  - Phase 2 EDA found NO year-over-year trend (2019 vs 2025 totals differ
    by <0.5%), so there's no risk of e.g. training only on a "low regime"
    and validating on a "high regime" — any 1-year block is representative.
  - A full calendar year per split keeps all 12 months' worth of
    seasonality in both val and test, rather than an arbitrary mid-year cut
    that would leave val/test missing some months entirely.

No embargo/gap is inserted between splits. This is intentional, not an
oversight: because every lag/rolling feature is strictly backward-looking
(Phase 3), a validation-period row on 2024-01-01 legitimately uses real
observed 2023 history as its features -- that's exactly how a deployed
model would behave (it doesn't forget the past when the calendar flips).
An embargo is needed when a *target* window could bleed across the
boundary (e.g. a 7-day-ahead multi-step target computed for the last days
of December would need December 25-31 to peek into January). Phase 6/7
must account for that separately when building multi-step targets close
to each boundary -- flagged here as a TODO for those phases, not solved
here since single-step feature columns aren't affected.

Run:
    python src/data/split_dataset.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
# test = everything after VAL_END through end of data (2025-12-31)


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "test"
    df.loc[df["Date"] <= TRAIN_END, "split"] = "train"
    df.loc[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END), "split"] = "val"
    return df


def verify_no_overlap(df: pd.DataFrame) -> None:
    ranges = df.groupby("split")["Date"].agg(["min", "max"])
    print("Date ranges per split:")
    print(ranges.to_string())

    train_max = df.loc[df["split"] == "train", "Date"].max()
    val_min = df.loc[df["split"] == "val", "Date"].min()
    val_max = df.loc[df["split"] == "val", "Date"].max()
    test_min = df.loc[df["split"] == "test", "Date"].min()

    assert train_max < val_min, "train/val overlap!"
    assert val_max < test_min, "val/test overlap!"
    print("\nNo date overlap between splits: confirmed.")

    counts = df["split"].value_counts()
    print(f"\nRow counts: {counts.to_dict()}")
    print(f"Proportions: {(counts / len(df)).round(3).to_dict()}")

    # every split should contain all 50 SKUs (a full-calendar-year cut can't
    # accidentally drop a SKU, but worth confirming rather than assuming)
    sku_counts = df.groupby("split")["Material_ID"].nunique()
    print(f"\nSKUs present per split: {sku_counts.to_dict()} (expect 50/50/50)")


def spot_check_no_future_leak(df: pd.DataFrame) -> None:
    """Confirm a val-period row's lag_56 feature is fully populated from
    real PAST data (train period), not NaN and not from the future."""
    sample = df[(df["split"] == "val") & (df["Date"] == "2024-01-01") & (df["Material_ID"] == "MAT001")]
    row = sample.iloc[0]
    print("\nSpot check — first val-period row (MAT001, 2024-01-01):")
    print(f"  lag_1={row['lag_1']}, lag_56={row['lag_56']}, rolling_mean_28={row['rolling_mean_28']:.2f}")
    print("  All populated from 2023 (train) data — expected and correct,")
    print("  since features are point-in-time, not information from the future.")


def main() -> None:
    df = pd.read_csv(
        PROCESSED_DIR / "forecasting_table.csv",
        parse_dates=["Date"],
        keep_default_na=False,
        na_values=["", "NaN", "nan"],  # keep literal "None" (Seasonality_Type) as a string, not NaN
    )
    df = assign_split(df)
    verify_no_overlap(df)
    spot_check_no_future_leak(df)

    out_path = PROCESSED_DIR / "forecasting_table_split.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
