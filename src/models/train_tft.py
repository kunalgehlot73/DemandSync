"""
Phase 8 — Temporal Fusion Transformer, via pytorch-forecasting.

Run locally:
    pip install torch pytorch-forecasting pytorch-lightning optuna

NOT run in the drafting sandbox (no network, no torch at all -- see
Phase 7's LSTM script for the same caveat). This is the heaviest model in
the roadmap; per your choice, this keeps the full Optuna search as
originally planned rather than scaling it down, but be aware: each TFT
trial is substantially more expensive than an LSTM trial (attention +
variable-selection networks on top of an LSTM encoder-decoder), so expect
this to run considerably longer than Phase 7 did on CPU. Consider running
it overnight.

Version note: pytorch-forecasting's API has had breaking changes across
versions. This targets the commonly-documented 1.x API pattern
(TimeSeriesDataSet + TemporalFusionTransformer.from_dataset). If your
installed version errors on an argument name, check
`pip show pytorch-forecasting` and the changelog for that version --
the core structure (dataset -> from_dataset -> Trainer.fit) should still
hold even if a few kwarg names moved.

Design choices, and why:

- Point forecast (loss=MAE), not quantile, to stay comparable with
  XGBoost/LSTM's point-forecast WAPE/MASE evaluation. TFT's real strength
  is native quantile output -- that's deliberately saved for Phase 9's
  probabilistic upgrade, which will swap this to QuantileLoss and reuse
  most of this script's dataset-construction logic unchanged.

- max_encoder_length=28, max_prediction_length=1: single-step-ahead,
  matching every other model's evaluation so results are comparable
  apples-to-apples. TFT's multi-horizon capability isn't needed for that
  comparison, even though it's part of what makes TFT distinctive --
  worth a footnote in the eventual writeup rather than silently ignoring.

- Static categoricals: Category, Seasonality_Type (SKU-level, constant
  over time). Time-varying known reals: day_of_week, month, weekend,
  time_idx (all known in advance for any future date). Time-varying
  UNKNOWN real: Final_Demand itself (the target, unknown at prediction time
  for the decoder step -- this is exactly the encoder/decoder split TFT is
  built around).

- Uses pytorch_forecasting's built-in optimize_hyperparameters helper
  (wraps Optuna internally) rather than hand-rolling the Optuna loop like
  Phase 6/7 did -- this is the library's documented, intended workflow for
  tuning a TFT specifically, and reimplementing it manually risks missing
  TFT-specific constraints (e.g. valid hidden_size/attention_head_size
  combinations).

Run:
    python src/models/train_tft.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

MAX_ENCODER_LENGTH = 28
MAX_PREDICTION_LENGTH = 1
N_TRIALS = 10          # heavier per-trial cost than LSTM -- kept modest even for a "run it long" budget
MAX_EPOCHS_SEARCH = 10
MAX_EPOCHS_FINAL = 30
BATCH_SIZE = 64


def load_data() -> pd.DataFrame:
    mm = pd.read_csv(ROOT / "data" / "raw" / "material_master.csv", keep_default_na=False, na_values=[])
    demand = pd.read_csv(ROOT / "data" / "raw" / "synthetic_7year_demand.csv", parse_dates=["Date"])
    df = demand.merge(mm[["Material_ID", "Category", "Seasonality_Type"]], on="Material_ID")

    df = df.sort_values(["Material_ID", "Date"]).reset_index(drop=True)
    df["Final_Demand"] = df["Final_Demand"].astype("float32")
    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days.astype(int)
    df["day_of_week"] = df["Date"].dt.dayofweek.astype(str)
    df["month"] = df["Date"].dt.month.astype(str)
    df["weekend"] = (df["Date"].dt.dayofweek >= 5).astype(str)
    df["Material_ID"] = df["Material_ID"].astype(str)
    df["Category"] = df["Category"].astype(str)
    df["Seasonality_Type"] = df["Seasonality_Type"].astype(str)
    return df


def build_datasets(df: pd.DataFrame):
    train_end_date = pd.Timestamp("2023-12-31")
    val_end_date = pd.Timestamp("2024-12-31")
    train_cutoff = int(df.loc[df["Date"] == train_end_date, "time_idx"].iloc[0])
    val_cutoff = int(df.loc[df["Date"] == val_end_date, "time_idx"].iloc[0])

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="Final_Demand",
        group_ids=["Material_ID"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["Category", "Seasonality_Type"],
        time_varying_known_categoricals=["day_of_week", "month", "weekend"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["Final_Demand"],
        target_normalizer=GroupNormalizer(groups=["Material_ID"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # validation: same series, but only generate samples whose PREDICTION
    # window falls in 2024 -- encoder is allowed to reach back into train
    # (same no-embargo reasoning as every prior phase).
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= val_cutoff],
        min_prediction_idx=train_cutoff + 1,
        stop_randomization=True,
    )

    return training, validation, train_cutoff, val_cutoff


def wape(y_true, y_pred) -> float:
    return np.abs(y_true - y_pred).sum() / y_true.sum()


def main():
    print("Loading data and building TimeSeriesDataSets (this can take a few minutes)...")
    df = load_data()
    training, validation, train_cutoff, val_cutoff = build_datasets(df)

    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

    print(f"Train samples: {len(training)}  Val samples: {len(validation)}")

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nStarting Optuna search ({N_TRIALS} trials, {MAX_EPOCHS_SEARCH} epochs each)...")
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path=str(RESULTS_DIR / "tft_tuning"),
        n_trials=N_TRIALS,
        max_epochs=MAX_EPOCHS_SEARCH,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(16, 128),
        hidden_continuous_size_range=(8, 64),
        attention_head_size_range=(1, 4),
        learning_rate_range=(1e-4, 1e-1),
        dropout_range=(0.1, 0.4),
        trainer_kwargs=dict(accelerator="cpu"),
        loss=MAE(),
    )

    print(f"\nBest val loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    # final retrain with best params, full epoch budget
    print("\nRetraining best config with full epoch budget...")
    best_params = study.best_trial.params
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=best_params["learning_rate"],
        hidden_size=best_params["hidden_size"],
        attention_head_size=best_params["attention_head_size"],
        dropout=best_params["dropout"],
        hidden_continuous_size=best_params["hidden_continuous_size"],
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=4, mode="min")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS_FINAL,
        gradient_clip_val=best_params.get("gradient_clip_val", 0.1),
        callbacks=[early_stop],
        accelerator="cpu",
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # evaluate: point predictions (median of internal quantile output when loss=MAE
    # still yields a single point estimate, since MAE loss trains a point-forecast head)
    predictions = tft.predict(val_dataloader, mode="prediction", return_y=True)
    y_pred = predictions.output.numpy().flatten()
    y_true = predictions.y[0].numpy().flatten()
    y_pred = np.clip(y_pred, 0, None)

    val_wape = wape(y_true, y_pred)
    val_mae = np.mean(np.abs(y_true - y_pred))
    val_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"\nFinal TFT val metrics:  MAE={val_mae:.3f}  RMSE={val_rmse:.3f}  WAPE={val_wape:.4f}")

    trainer.save_checkpoint(RESULTS_DIR / "tft_model.ckpt")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(RESULTS_DIR / "tft_val_predictions.csv", index=False)
    print(f"Saved model checkpoint and predictions to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
