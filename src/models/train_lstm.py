"""
Phase 7 — LSTM model with Optuna hyperparameter search.

Run locally (`pip install torch optuna` — torch isn't available in the
sandbox this was drafted in, so unlike Phase 6, there's no proxy run here;
an MLP or other non-sequential stand-in wouldn't actually test what an
LSTM does, so it would've been theater rather than a useful check).

Design choices, and why:

- GLOBAL model, not per-SKU: one LSTM trained across all 50 SKUs, with a
  learned SKU embedding concatenated onto the LSTM's final hidden state.
  This mirrors the XGBoost setup (Material_ID as a feature) rather than
  training 50 separate tiny models, which would badly underfit given how
  little history any single SKU has relative to a neural net's appetite.

- Input sequence = raw daily Demand + per-day calendar features for the
  PRIOR `SEQ_LEN` days; target = Demand on the next day. This is
  deliberately different from the XGBoost feature set (which fed
  pre-computed lag/rolling columns) -- the LSTM is meant to learn its own
  temporal representation from the raw sequence rather than be handed the
  same engineered features, which is the actual point of comparing
  architectures in Phase 28's "is added complexity justified" question.
  If the LSTM does no better than XGBoost despite learning its own
  features AND having access to a longer effective history, that's a
  meaningful data point for that comparison.

- No embargo between train/val at the sequence level either, same
  reasoning as Phase 4: a val-period target's input sequence legitimately
  reaches back into train-period history, matching real deployment.

- Optuna trial count is intentionally modest (15) compared to XGBoost's 50
  -- each trial trains a full neural net, not a single boosted-tree fit,
  so the per-trial cost is much higher. Increase N_TRIALS if you have a
  GPU and time to spare; on CPU, 15 trials will already take a while on a
  dataset this size.

Run:
    python src/models/train_lstm.py
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEQ_LEN = 28
N_TRIALS = 15
MAX_EPOCHS = 30
PATIENCE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_raw():
    mm = pd.read_csv(ROOT / "data" / "raw" / "material_master.csv", keep_default_na=False, na_values=[])
    demand = pd.read_csv(ROOT / "data" / "raw" / "synthetic_7year_demand.csv", parse_dates=["Date"])
    demand = demand.sort_values(["Material_ID", "Date"]).reset_index(drop=True)
    sku_to_idx = {sku: i for i, sku in enumerate(mm["Material_ID"])}
    return mm, demand, sku_to_idx


def build_sequences(demand: pd.DataFrame, sku_to_idx: dict, train_end: str, val_end: str):
    """Returns (X, sku_idx, y, split) arrays. X channels: [demand, day_of_week/7, month/12, weekend]."""
    demand = demand.copy()
    demand["dow"] = demand["Date"].dt.dayofweek / 7.0
    demand["mon"] = demand["Date"].dt.month / 12.0
    demand["weekend"] = (demand["Date"].dt.dayofweek >= 5).astype(float)

    X_list, sku_list, y_list, split_list = [], [], [], []

    for sku, g in demand.groupby("Material_ID"):
        g = g.reset_index(drop=True)
        vals = g[["Final_Demand", "dow", "mon", "weekend"]].to_numpy(dtype=np.float32)
        dates = g["Date"].values
        n = len(g)
        for t in range(SEQ_LEN, n):
            X_list.append(vals[t - SEQ_LEN:t])
            y_list.append(g.loc[t, "Final_Demand"])
            sku_list.append(sku_to_idx[sku])
            d = pd.Timestamp(dates[t])
            if d <= pd.Timestamp(train_end):
                split_list.append("train")
            elif d <= pd.Timestamp(val_end):
                split_list.append("val")
            else:
                split_list.append("test")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    sku_idx = np.array(sku_list, dtype=np.int64)
    split = np.array(split_list)
    return X, sku_idx, y, split


class SequenceDataset(Dataset):
    def __init__(self, X, sku_idx, y):
        self.X = torch.from_numpy(X)
        self.sku_idx = torch.from_numpy(sku_idx)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.sku_idx[i], self.y[i]


class LSTMForecaster(nn.Module):
    def __init__(self, n_skus, input_size=4, hidden_size=64, num_layers=2, dropout=0.2, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(n_skus, embed_dim)
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, sku_idx):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        emb = self.embedding(sku_idx)
        out = self.head(torch.cat([last_hidden, emb], dim=1))
        return out.squeeze(-1)


def wape(y_true, y_pred) -> float:
    return np.abs(y_true - y_pred).sum() / y_true.sum()


def train_one_model(params, X_train, sku_train, y_train, X_val, sku_val, y_val, n_skus, max_epochs=MAX_EPOCHS):
    train_ds = SequenceDataset(X_train, sku_train, y_train)
    val_ds = SequenceDataset(X_val, sku_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = LSTMForecaster(
        n_skus=n_skus, hidden_size=params["hidden_size"],
        num_layers=params["num_layers"], dropout=params["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.L1Loss()  # MAE, matches WAPE's numerator better than MSE

    best_val_wape = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, sb, yb in train_loader:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, sb, yb in val_loader:
                xb, sb = xb.to(DEVICE), sb.to(DEVICE)
                preds.append(model(xb, sb).cpu().numpy())
                actuals.append(yb.numpy())
        preds, actuals = np.concatenate(preds), np.concatenate(actuals)
        val_wape = wape(actuals, np.clip(preds, 0, None))

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val_wape


def objective(trial, X_train, sku_train, y_train, X_val, sku_val, y_val, n_skus):
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }
    _, val_wape = train_one_model(params, X_train, sku_train, y_train, X_val, sku_val, y_val, n_skus, max_epochs=15)
    return val_wape


def main():
    print(f"Device: {DEVICE}")
    mm, demand, sku_to_idx = load_raw()
    n_skus = len(sku_to_idx)

    print("Building sequences (this takes a bit)...")
    X, sku_idx, y, split = build_sequences(demand, sku_to_idx, train_end="2023-12-31", val_end="2024-12-31")
    print(f"Total sequences: {len(y)}  (train={sum(split=='train')}, val={sum(split=='val')}, test={sum(split=='test')})")

    X_train, sku_train, y_train = X[split == "train"], sku_idx[split == "train"], y[split == "train"]
    X_val, sku_val, y_val = X[split == "val"], sku_idx[split == "val"], y[split == "val"]

    study = optuna.create_study(direction="minimize", study_name="lstm_demand_wape")
    study.optimize(
        lambda t: objective(t, X_train, sku_train, y_train, X_val, sku_val, y_val, n_skus),
        n_trials=N_TRIALS,
    )

    print(f"\nBest val WAPE (during search, 15-epoch budget): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # final retrain with best params, full epoch budget
    print("\nRetraining best config with full epoch budget...")
    final_model, final_wape = train_one_model(
        study.best_params, X_train, sku_train, y_train, X_val, sku_val, y_val, n_skus, max_epochs=MAX_EPOCHS
    )
    print(f"Final val WAPE: {final_wape:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(final_model.state_dict(), RESULTS_DIR / "lstm_model.pt")
    study.trials_dataframe().to_csv(RESULTS_DIR / "lstm_optuna_trials.csv", index=False)
    print(f"Saved model and trial history to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
