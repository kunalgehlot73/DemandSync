"""
Phase 6 — XGBoost model with Optuna hyperparameter search.

Run locally (this needs `pip install xgboost optuna`, not available in the
sandbox this was drafted in -- validated instead with sklearn's
HistGradientBoostingRegressor as a same-family proxy; see results/
phase6_proxy_notes.md for what that run found).

Optimization objective: minimize validation WAPE (primary metric per the
roadmap). RMSE is logged alongside every trial as a secondary check, not
folded into the objective -- Optuna needs a single scalar, and WAPE is the
metric we actually care about for the inventory-cost link in Phase 18.

IMPORTANT CAVEAT (flagged in the original roadmap critique): this uses a
SINGLE static train/val split for tuning, not rolling-origin/walk-forward
CV. With one year of val data that's a real risk of overfitting Optuna's
search to 2024's specific realized noise. Given this dataset has no
year-over-year trend (Phase 2 finding), the risk is lower than it would be
on real retail data, but it's not zero -- if results look too good, that's
the first thing to revisit.

Run:
    python src/models/train_xgboost.py
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28", "lag_56",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28", "ewma_7",
    "day_of_week", "week_of_year", "month", "quarter", "weekend",
    "Category", "Seasonality_Type", "Base_Daily_Demand",
    "Lead_Time", "Unit_Cost", "Holding_Cost_Rate", "Ordering_Cost", "Stockout_Penalty",
]
TARGET = "Demand"
N_TRIALS = 50


def load_splits():
    df = pd.read_csv(
        PROCESSED_DIR / "forecasting_table_split.csv",
        parse_dates=["Date"],
        keep_default_na=False,
        na_values=["", "NaN", "nan"],
    )
    df = df.dropna(subset=["lag_56"])  # drop warm-up rows

    for c in ["Category", "Seasonality_Type"]:
        df[c] = df[c].astype("category")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    return train, val


def wape(y_true, y_pred) -> float:
    return np.abs(y_true - y_pred).sum() / y_true.sum()


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)

    val_wape = wape(y_val.values, pred)
    val_rmse = np.sqrt(np.mean((y_val.values - pred) ** 2))
    trial.set_user_attr("rmse", val_rmse)  # logged, not optimized
    return val_wape


def main() -> None:
    train, val = load_splits()
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]

    study = optuna.create_study(direction="minimize", study_name="xgboost_demand_wape")
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val), n_trials=N_TRIALS)

    print(f"Best val WAPE: {study.best_value:.4f}")
    print(f"Best trial RMSE: {study.best_trial.user_attrs['rmse']:.3f}")
    print(f"Best params: {study.best_params}")

    # refit final model on train with best params, save
    best_model = xgb.XGBRegressor(
        **study.best_params, tree_method="hist", enable_categorical=True, random_state=42
    )
    best_model.fit(X_train, y_train)

    RESULTS_DIR.mkdir(exist_ok=True)
    best_model.save_model(RESULTS_DIR / "xgboost_model.json")
    study.trials_dataframe().to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)

    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    importances.to_csv(RESULTS_DIR / "xgboost_feature_importance.csv")
    print("\nTop 10 feature importances:")
    print(importances.head(10).round(4).to_string())

    print(f"\nSaved model, trial history, and feature importances to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
