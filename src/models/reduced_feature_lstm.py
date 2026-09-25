# src/models/reduced_feature_lstm.py

from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SEQ_LEN = 30
BATCH_SIZE = 256

HIDDEN_SIZE = 48
NUM_LAYERS = 2

DROPOUT = 0.30

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
PATIENCE = 12

TOP_K_VALUES = [8, 12, 16, 20, 38]


TRAIN_START = "2015-10-16"
TRAIN_END = "2023-05-30"

VAL_START = "2023-05-31"
VAL_END = "2025-01-21"

TEST_START = "2025-01-22"
TEST_END = "2026-09-10"


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",

    "log_return_1d",

    "price_ma20_ratio",
    "price_ma50_ratio",
    "price_ma200_ratio",

    "momentum_5",
    "momentum_10",
    "momentum_20",

    "volatility_10",
    "volatility_20",
    "volatility_60",

    "rsi_14",

    "macd",
    "macd_signal",
    "macd_hist",

    "volume_change",
    "volume_ratio_20",

    "relative_strength_1d",
    "relative_strength_20d",

    "beta_60",

    "vix",
    "vix_change",
    "vix_ma10",
    "vix_ma20",
    "vix_ratio_ma20",

    "sector_Consumer",
    "sector_Energy",
    "sector_Financials",
    "sector_Technology",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    required = FEATURES + [
        "Ticker",
        "Date",
        "future_return",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# DATE SPLIT
# ============================================================

def assign_split(df):

    df = df.copy()

    df["split"] = "outside"

    train_mask = (
        (df["Date"] >= pd.Timestamp(TRAIN_START))
        &
        (df["Date"] <= pd.Timestamp(TRAIN_END))
    )

    val_mask = (
        (df["Date"] >= pd.Timestamp(VAL_START))
        &
        (df["Date"] <= pd.Timestamp(VAL_END))
    )

    test_mask = (
        (df["Date"] >= pd.Timestamp(TEST_START))
        &
        (df["Date"] <= pd.Timestamp(TEST_END))
    )

    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "validation"
    df.loc[test_mask, "split"] = "test"

    return df


# ============================================================
# TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    df["cross_sectional_target"] = np.nan

    grouped = df.groupby("Date")

    mean_future = grouped["future_return"].transform("mean")
    std_future = grouped["future_return"].transform("std")

    valid = std_future > 0

    df.loc[valid, "cross_sectional_target"] = (
        (
            df.loc[valid, "future_return"]
            - mean_future[valid]
        )
        /
        std_future[valid]
    )

    return df


# ============================================================
# TRAINING IC FEATURE RANKING
# ============================================================

def calculate_feature_ic(df, feature):

    records = []

    train = df[
        df["split"] == "train"
    ][
        [
            "Date",
            feature,
            "future_return",
        ]
    ].copy()

    train = train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    train = train.dropna(
        subset=[feature, "future_return"]
    )

    for date, day in train.groupby("Date"):

        if len(day) < 3:
            continue

        x = day[feature].values
        y = day["future_return"].values

        if np.all(x == x[0]):
            continue

        if np.all(y == y[0]):
            continue

        ic, _ = spearmanr(x, y)

        if np.isfinite(ic):
            records.append(ic)

    if not records:
        return {
            "feature": feature,
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "positive_ic_pct": np.nan,
        }

    ic = np.asarray(records)

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)

    return {
        "feature": feature,
        "mean_ic": mean_ic,
        "median_ic": np.median(ic),
        "ic_std": std_ic,
        "icir": (
            mean_ic / std_ic
            if std_ic > 0
            else np.nan
        ),
        "positive_ic_pct": (
            (ic > 0).mean() * 100
        ),
    }


def rank_features(df):

    print()
    print("=" * 75)
    print("FEATURE SELECTION USING TRAIN ONLY")
    print("=" * 75)

    results = []

    for feature in FEATURES:

        result = calculate_feature_ic(
            df,
            feature,
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    # Rank by absolute IC.
    #
    # Why absolute?
    # A consistently negative feature is still useful because
    # the model can learn the inverse relationship.
    result_df["abs_mean_ic"] = (
        result_df["mean_ic"].abs()
    )

    result_df = (
        result_df
        .sort_values(
            "abs_mean_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    path = (
        OUTPUT_DIR
        / "reduced_lstm_train_feature_ranking.csv"
    )

    result_df.to_csv(
        path,
        index=False,
    )

    print(result_df.to_string(index=False))

    print()
    print(f"Saved: {path}")

    return result_df


# ============================================================
# SCALER
# ============================================================

class StandardScaler:

    def __init__(self):

        self.mean_ = None
        self.std_ = None

    def fit(self, x):

        self.mean_ = np.nanmean(
            x,
            axis=0,
        )

        self.std_ = np.nanstd(
            x,
            axis=0,
        )

        self.std_[self.std_ < 1e-8] = 1.0

        return self

    def transform(self, x):

        return (
            x - self.mean_
        ) / self.std_

    def fit_transform(self, x):

        self.fit(x)

        return self.transform(x)


# ============================================================
# BUILD SEQUENCES
# ============================================================

def build_sequences(
    df,
    feature_list,
    scaler,
):

    sequences = {
        "train_X": [],
        "train_y": [],
        "train_dates": [],
        "train_tickers": [],
        "train_returns": [],

        "val_X": [],
        "val_y": [],
        "val_dates": [],
        "val_tickers": [],
        "val_returns": [],

        "test_X": [],
        "test_y": [],
        "test_dates": [],
        "test_tickers": [],
        "test_returns": [],
    }

    # Build separately for each ticker to avoid mixing stocks
    for ticker, ticker_df in df.groupby(
        "Ticker",
        sort=False,
    ):

        ticker_df = (
            ticker_df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        x = ticker_df[feature_list].values.astype(
            np.float32
        )

        y = ticker_df[
            "cross_sectional_target"
        ].values.astype(np.float32)

        future_return = ticker_df[
            "future_return"
        ].values.astype(np.float32)

        dates = ticker_df["Date"].values

        ticker_name = ticker

        for i in range(
            SEQ_LEN - 1,
            len(ticker_df),
        ):

            target_date = dates[i]

            split = ticker_df.loc[
                i,
                "split"
            ]

            if split not in [
                "train",
                "validation",
                "test",
            ]:
                continue

            sequence = x[
                i - SEQ_LEN + 1:
                i + 1
            ]

            target = y[i]

            realized_return = future_return[i]

            if np.any(~np.isfinite(sequence)):
                continue

            if not np.isfinite(target):
                continue

            sequence = scaler.transform(
                sequence
            ).astype(np.float32)

            prefix = "val" if split == "validation" else split

            sequences[
                f"{prefix}_X"
            ].append(sequence)

            sequences[
                f"{prefix}_y"
            ].append(target)

            sequences[
                f"{prefix}_dates"
            ].append(target_date)

            sequences[
                f"{prefix}_tickers"
            ].append(ticker_name)

            sequences[
                f"{prefix}_returns"
            ].append(realized_return)

    return sequences


# ============================================================
# DATASET
# ============================================================

def to_loader(X, y, shuffle):

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


# ============================================================
# MODEL
# ============================================================

class CrossSectionalLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=48,
        num_layers=2,
        dropout=0.30,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
            if num_layers > 1
            else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_hidden = output[:, -1, :]

        return self.fc(
            last_hidden
        ).squeeze(-1)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_loader,
    val_loader,
    input_size,
):

    model = CrossSectionalLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.HuberLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            pred = model(X_batch)

            loss = criterion(
                pred,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                pred = model(X_batch)

                loss = criterion(
                    pred,
                    y_batch,
                )

                val_losses.append(
                    loss.item()
                )

        train_loss = np.mean(
            train_losses
        )

        val_loss = np.mean(
            val_losses
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.5f} | "
                f"Val {val_loss:.5f}"
            )

        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    X,
    y,
    dates,
    tickers,
    returns,
):

    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                np.asarray(
                    X,
                    dtype=np.float32,
                )
            ),
            torch.tensor(
                np.asarray(
                    y,
                    dtype=np.float32,
                )
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(DEVICE)

            pred = model(
                X_batch
            )

            predictions.extend(
                pred.cpu().numpy()
            )

    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Ticker": tickers,
            "target": y,
            "future_return": returns,
            "prediction": predictions,
        }
    )

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred_df):

    daily_ic = []

    daily_spreads = []

    daily_hit = []

    for date, day in pred_df.groupby(
        "Date"
    ):

        if len(day) < 4:
            continue

        ic, _ = spearmanr(
            day["prediction"],
            day["future_return"],
        )

        if np.isfinite(ic):
            daily_ic.append(ic)

        day = day.sort_values(
            "prediction"
        )

        long = day.tail(2)
        short = day.head(2)

        long_return = (
            long["future_return"].mean()
        )

        short_return = (
            short["future_return"].mean()
        )

        spread = (
            long_return
            - short_return
        )

        daily_spreads.append(
            spread
        )

        daily_hit.append(
            float(spread > 0)
        )

    ic = np.asarray(daily_ic)
    spreads = np.asarray(
        daily_spreads
    )

    mean_ic = (
        ic.mean()
        if len(ic)
        else np.nan
    )

    spread_mean = (
        spreads.mean()
        if len(spreads)
        else np.nan
    )

    spread_std = (
        spreads.std(ddof=1)
        if len(spreads) > 1
        else np.nan
    )

    sharpe = np.nan

    if (
        np.isfinite(spread_std)
        and spread_std > 0
    ):
        sharpe = (
            spread_mean
            / spread_std
            * np.sqrt(252)
        )

    return {
        "IC": mean_ic,
        "Median_IC": (
            np.median(ic)
            if len(ic)
            else np.nan
        ),
        "Top2_Return": (
            (
                pred_df
                .groupby("Date")
                ["prediction"]
                .apply(
                    lambda x:
                    pred_df.loc[
                        x.nlargest(2).index,
                        "future_return",
                    ].mean()
                )
                .mean()
            )
        ),
        "Bottom2_Return": (
            (
                pred_df
                .groupby("Date")
                ["prediction"]
                .apply(
                    lambda x:
                    pred_df.loc[
                        x.nsmallest(2).index,
                        "future_return",
                    ].mean()
                )
                .mean()
            )
        ),
        "Long_Short": spread_mean,
        "Hit_Rate": (
            np.mean(daily_hit)
            if daily_hit
            else np.nan
        ),
        "Sharpe": sharpe,
        "Days": len(spreads),
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    df,
    feature_list,
    label,
):

    print()
    print("=" * 75)
    print(
        f"RUNNING: {label}"
    )
    print("=" * 75)

    train_df = df[
        df["split"] == "train"
    ].copy()

    # --------------------------------------------------------
    # Fit scaler ONLY on train
    # --------------------------------------------------------

    train_values = train_df[
        feature_list
    ].values.astype(
        np.float32
    )

    scaler = StandardScaler()

    scaler.fit(
        train_values
    )

    # --------------------------------------------------------
    # Build sequences
    # --------------------------------------------------------

    seq = build_sequences(
        df,
        feature_list,
        scaler,
    )

    print(
        f"Features : {len(feature_list)}"
    )

    print(
        f"Train sequences : "
        f"{len(seq['train_X']):,}"
    )

    print(
        f"Validation      : "
        f"{len(seq['val_X']):,}"
    )

    print(
        f"Test            : "
        f"{len(seq['test_X']):,}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = to_loader(
        seq["train_X"],
        seq["train_y"],
        shuffle=True,
    )

    val_loader = to_loader(
        seq["val_X"],
        seq["val_y"],
        shuffle=False,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        train_loader,
        val_loader,
        input_size=len(feature_list),
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    val_pred = predict(
        model,
        seq["val_X"],
        seq["val_y"],
        seq["val_dates"],
        seq["val_tickers"],
        seq["val_returns"],
    )

    test_pred = predict(
        model,
        seq["test_X"],
        seq["test_y"],
        seq["test_dates"],
        seq["test_tickers"],
        seq["test_returns"],
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    val_metrics = calculate_metrics(
        val_pred
    )

    test_metrics = calculate_metrics(
        test_pred
    )

    print()
    print("VALIDATION")
    print(
        pd.Series(val_metrics)
    )

    print()
    print("TEST")
    print(
        pd.Series(test_metrics)
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    val_path = (
        OUTPUT_DIR
        / f"reduced_lstm_{label}_validation.csv"
    )

    test_path = (
        OUTPUT_DIR
        / f"reduced_lstm_{label}_test.csv"
    )

    val_pred.to_csv(
        val_path,
        index=False,
    )

    test_pred.to_csv(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR
        / f"reduced_lstm_{label}.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "features":
                feature_list,
            "scaler_mean":
                scaler.mean_,
            "scaler_std":
                scaler.std_,
        },
        model_path,
    )

    return {
        "label": label,
        "num_features": len(feature_list),

        "val_IC":
            val_metrics["IC"],

        "val_median_IC":
            val_metrics["Median_IC"],

        "val_long_short":
            val_metrics["Long_Short"],

        "val_sharpe":
            val_metrics["Sharpe"],

        "test_IC":
            test_metrics["IC"],

        "test_median_IC":
            test_metrics["Median_IC"],

        "test_long_short":
            test_metrics["Long_Short"],

        "test_sharpe":
            test_metrics["Sharpe"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    df = load_data()

    df = assign_split(df)

    df = create_target(df)

    # --------------------------------------------------------
    # Select features using TRAIN ONLY
    # --------------------------------------------------------

    ranking = rank_features(df)

    ranked_features = ranking[
        "feature"
    ].tolist()

    # --------------------------------------------------------
    # Always include the all-feature baseline
    # --------------------------------------------------------

    experiment_results = []

    for k in TOP_K_VALUES:

        feature_list = ranked_features[:k]

        label = f"top{k}"

        result = run_experiment(
            df,
            feature_list,
            label,
        )

        experiment_results.append(
            result
        )

    # --------------------------------------------------------
    # Save experiment comparison
    # --------------------------------------------------------

    results = pd.DataFrame(
        experiment_results
    )

    results = results.sort_values(
        "val_IC",
        ascending=False,
    )

    path = (
        OUTPUT_DIR
        / "reduced_lstm_comparison.csv"
    )

    results.to_csv(
        path,
        index=False,
    )

    print()
    print("=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved comparison -> {path}"
    )


if __name__ == "__main__":
    main()