# ============================================================
# Cross-Sectional 5-Day LSTM
# ============================================================
#
# Predicts each stock's RELATIVE 5-day forward return.
#
# Target:
#
#   future_5d = Close[t+5] / Close[t] - 1
#
#   cross_sectional_target =
#       (future_5d - mean(future_5d))
#       / std(future_5d)
#
# Split:
#   TRAIN      : 2015-10-16 -> 2023-05-30
#   VALIDATION : 2023-05-31 -> 2025-01-21
#   TEST       : 2025-01-22 -> 2026-09-10
#
# Model:
#   LSTM(38 -> 48, 2 layers)
#   FC 48 -> 16 -> 1
#
# Loss:
#   Huber
#
# Optimizer:
#   AdamW
#
# IMPORTANT:
#   - Scaler fitted ONLY on training data
#   - Test never used for model selection
#   - Validation predictions saved separately
# ============================================================


from pathlib import Path
import random

import numpy as np
import pandas as pd

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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {DEVICE}"
)


# ============================================================
# HYPERPARAMETERS
# ============================================================

HORIZON = 5

SEQ_LEN = 30

BATCH_SIZE = 256

HIDDEN_SIZE = 48

NUM_LAYERS = 2

DROPOUT = 0.30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 12

LR_PATIENCE = 4

LR_FACTOR = 0.5


# ============================================================
# DATE SPLITS
# ============================================================

TRAIN_START = "2015-10-16"
TRAIN_END = "2023-05-30"

VAL_START = "2023-05-31"
VAL_END = "2025-01-21"

TEST_START = "2025-01-22"
TEST_END = "2026-09-10"


# ============================================================
# FEATURES
# ============================================================

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
# CHECK CONFIG
# ============================================================

print(
    f"Number of features: {len(FEATURES)}"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 75)
    print("LOADING DATA")
    print("=" * 75)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # --------------------------------------------------------
    # Sort by ticker/date
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "Ticker",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            "Ticker",
            "Date",
            "Close",
        ]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Validate duplicates
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "Date",
            "Ticker",
        ]
    ).sum()

    if duplicates > 0:

        raise ValueError(
            f"Found {duplicates} duplicate "
            f"(Date, Ticker) rows."
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        f"Dates   : {df['Date'].nunique():,}"
    )

    print(
        f"Tickers : {df['Ticker'].nunique()}"
    )

    print(
        f"Range   : "
        f"{df['Date'].min().date()} "
        f"-> "
        f"{df['Date'].max().date()}"
    )

    print()

    print(
        "Tickers:"
    )

    print(
        sorted(
            df["Ticker"].unique()
        )
    )

    return df


# ============================================================
# CREATE 5-DAY FUTURE RETURN
# ============================================================

def create_future_return(
    df
):

    print()
    print("=" * 75)
    print(
        f"CREATING {HORIZON}-DAY FUTURE RETURN"
    )
    print("=" * 75)

    df = df.copy()

    # Future price after HORIZON trading days.
    #
    # shift(-5) means:
    #
    # current row: t
    # next rows:   t+1 ... t+5
    #
    # therefore:
    #
    # future_5d = Close[t+5] / Close[t] - 1

    df[
        f"future_{HORIZON}d"
    ] = (
        df
        .groupby("Ticker")["Close"]
        .shift(-HORIZON)
        / df["Close"]
        - 1.0
    )

    target_column = (
        f"future_{HORIZON}d"
    )

    valid = df[
        target_column
    ].notna()

    print(
        f"Valid target rows : "
        f"{valid.sum():,}"
    )

    print(
        f"Missing target    : "
        f"{(~valid).sum():,}"
    )

    return df


# ============================================================
# CREATE CROSS-SECTIONAL TARGET
# ============================================================

def create_cross_sectional_target(
    df
):

    print()
    print("=" * 75)
    print("CREATING CROSS-SECTIONAL TARGET")
    print("=" * 75)

    df = df.copy()

    future_column = (
        f"future_{HORIZON}d"
    )

    target_column = (
        "cross_sectional_target"
    )

    # --------------------------------------------------------
    # Cross-sectional mean
    # --------------------------------------------------------

    daily_mean = (
        df
        .groupby("Date")[
            future_column
        ]
        .transform("mean")
    )

    # --------------------------------------------------------
    # Cross-sectional std
    # --------------------------------------------------------

    daily_std = (
        df
        .groupby("Date")[
            future_column
        ]
        .transform("std")
    )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    df[
        target_column
    ] = (
        df[future_column]
        - daily_mean
    ) / daily_std.replace(
        0,
        np.nan,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    target = df[
        target_column
    ].dropna()

    print(
        f"Target rows : {len(target):,}"
    )

    print(
        f"Mean        : "
        f"{target.mean():.6f}"
    )

    print(
        f"Std         : "
        f"{target.std():.6f}"
    )

    print(
        f"Min         : "
        f"{target.min():.6f}"
    )

    print(
        f"Max         : "
        f"{target.max():.6f}"
    )

    return df


# ============================================================
# ADD SPLIT
# ============================================================

def add_split_column(
    df
):

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

    df.loc[
        train_mask,
        "split",
    ] = "train"

    df.loc[
        val_mask,
        "split",
    ] = "validation"

    df.loc[
        test_mask,
        "split",
    ] = "test"

    print()
    print("=" * 75)
    print("DATE SPLITS")
    print("=" * 75)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = df[
            df["split"] == split
        ]

        print(
            f"{split:12s} | "
            f"rows={len(subset):,} | "
            f"dates={subset['Date'].nunique():,}"
        )

        if not subset.empty:

            print(
                f"             "
                f"{subset['Date'].min().date()} "
                f"-> "
                f"{subset['Date'].max().date()}"
            )

    return df


# ============================================================
# SCALER
# ============================================================

class StandardScaler:

    def __init__(self):

        self.mean_ = None
        self.std_ = None

    def fit(
        self,
        x,
    ):

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        self.mean_ = np.nanmean(
            x,
            axis=0,
        )

        self.std_ = np.nanstd(
            x,
            axis=0,
        )

        # Avoid divide-by-zero.
        self.std_[
            self.std_ < 1e-8
        ] = 1.0

        return self

    def transform(
        self,
        x,
    ):

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        return (
            x - self.mean_
        ) / self.std_

    def fit_transform(
        self,
        x,
    ):

        self.fit(x)

        return self.transform(x)


# ============================================================
# BUILD SEQUENCES
# ============================================================

def build_sequences(
    df,
    scaler,
):

    print()
    print("=" * 75)
    print("BUILDING SEQUENCES")
    print("=" * 75)

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

    target_column = (
        "cross_sectional_target"
    )

    future_column = (
        f"future_{HORIZON}d"
    )

    # --------------------------------------------------------
    # Process each ticker independently.
    #
    # This is VERY important.
    #
    # We must never create a sequence containing rows from
    # different stocks.
    # --------------------------------------------------------

    for ticker, ticker_df in df.groupby(
        "Ticker",
        sort=False,
    ):

        ticker_df = (
            ticker_df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        feature_values = (
            ticker_df[
                FEATURES
            ]
            .values
            .astype(np.float32)
        )

        target_values = (
            ticker_df[
                target_column
            ]
            .values
            .astype(np.float32)
        )

        future_returns = (
            ticker_df[
                future_column
            ]
            .values
            .astype(np.float32)
        )

        dates = (
            ticker_df[
                "Date"
            ]
            .values
        )

        splits = (
            ticker_df[
                "split"
            ]
            .values
        )

        # ----------------------------------------------------
        # Need SEQ_LEN observations.
        # ----------------------------------------------------

        for i in range(
            SEQ_LEN - 1,
            len(ticker_df),
        ):

            target_date = (
                dates[i]
            )

            split = splits[i]

            if split not in [
                "train",
                "validation",
                "test",
            ]:
                continue

            # ------------------------------------------------
            # Sequence uses current day and previous
            # SEQ_LEN-1 observations.
            # ------------------------------------------------

            start = (
                i - SEQ_LEN + 1
            )

            sequence = (
                feature_values[
                    start:i + 1
                ]
            )

            target = (
                target_values[i]
            )

            realized_return = (
                future_returns[i]
            )

            # ------------------------------------------------
            # Skip invalid data
            # ------------------------------------------------

            if np.any(
                ~np.isfinite(
                    sequence
                )
            ):
                continue

            if not np.isfinite(
                target
            ):
                continue

            if not np.isfinite(
                realized_return
            ):
                continue

            # ------------------------------------------------
            # Transform using train-only scaler.
            # ------------------------------------------------

            sequence = (
                scaler
                .transform(
                    sequence
                )
                .astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # Map validation -> val
            # ------------------------------------------------

            if split == "validation":
                prefix = "val"
            else:
                prefix = split

            sequences[
                f"{prefix}_X"
            ].append(
                sequence
            )

            sequences[
                f"{prefix}_y"
            ].append(
                target
            )

            sequences[
                f"{prefix}_dates"
            ].append(
                target_date
            )

            sequences[
                f"{prefix}_tickers"
            ].append(
                ticker
            )

            sequences[
                f"{prefix}_returns"
            ].append(
                realized_return
            )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        f"Train sequences : "
        f"{len(sequences['train_X']):,}"
    )

    print(
        f"Validation      : "
        f"{len(sequences['val_X']):,}"
    )

    print(
        f"Test            : "
        f"{len(sequences['test_X']):,}"
    )

    return sequences


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(
    X,
    y,
    shuffle,
):

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

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )

    return loader


# ============================================================
# MODEL
# ============================================================

class ReturnLSTM(
    nn.Module
):

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
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size,
                16,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        output, _ = self.lstm(
            x
        )

        last_hidden = (
            output[:, -1, :]
        )

        prediction = (
            self.head(
                last_hidden
            )
        )

        return prediction.squeeze(
            -1
        )


# ============================================================
# MODEL PARAMETER COUNT
# ============================================================

def count_parameters(
    model,
):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    train_loader,
    val_loader,
):

    print()
    print("=" * 75)
    print("TRAINING")
    print("=" * 75)

    model = ReturnLSTM(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    criterion = nn.HuberLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
        )
    )

    best_val_loss = float(
        "inf"
    )

    best_state = None

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = (
                X_batch.to(DEVICE)
            )

            y_batch = (
                y_batch.to(DEVICE)
            )

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = (
                    X_batch.to(DEVICE)
                )

                y_batch = (
                    y_batch.to(DEVICE)
                )

                predictions = model(
                    X_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_losses.append(
                    loss.item()
                )

        train_loss = (
            np.mean(
                train_losses
            )
        )

        val_loss = (
            np.mean(
                val_losses
            )
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        scheduler.step(
            val_loss
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict()
                .items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # ====================================================
        # PRINT
        # ====================================================

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.6f} | "
                f"Val Loss {val_loss:.6f} | "
                f"LR {current_lr:.2e}"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "Best model state was never saved."
        )

    model.load_state_dict(
        best_state
    )

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        OUTPUT_DIR
        / "cross_sectional_5d_lstm_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print()
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Training history saved: "
        f"{history_path}"
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
    future_returns,
):

    model.eval()

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X),
            torch.tensor(y),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    predictions = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = (
                X_batch.to(DEVICE)
            )

            pred = model(
                X_batch
            )

            predictions.extend(
                pred
                .cpu()
                .numpy()
                .tolist()
            )

    result = pd.DataFrame(
        {
            "Date":
                pd.to_datetime(
                    dates
                ),

            "Ticker":
                tickers,

            "target":
                y,

            f"future_{HORIZON}d":
                future_returns,

            "prediction":
                predictions,
        }
    )

    return result


# ============================================================
# BASIC REGRESSION METRICS
# ============================================================

def regression_metrics(
    df,
):

    actual = (
        df["target"]
        .values
    )

    prediction = (
        df["prediction"]
        .values
    )

    errors = (
        prediction - actual
    )

    mse = np.mean(
        errors ** 2
    )

    rmse = np.sqrt(
        mse
    )

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            actual
            - actual.mean()
        ) ** 2
    )

    r2 = (
        1
        - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    pred_std = (
        np.std(
            prediction
        )
    )

    actual_std = (
        np.std(
            actual
        )
    )

    correlation = np.corrcoef(
        prediction,
        actual,
    )[0, 1]

    return {
        "RMSE": rmse,
        "R2": r2,
        "Correlation": correlation,
        "Prediction_Std": pred_std,
        "Actual_Std": actual_std,
    }


# ============================================================
# DAILY IC
# ============================================================

def daily_ic_metrics(
    df,
):

    daily_ics = []

    daily_top2 = []

    daily_bottom2 = []

    daily_spreads = []

    daily_hits = []

    for date, day in df.groupby(
        "Date",
        sort=True,
    ):

        if len(day) < 4:
            continue

        # ----------------------------------------------------
        # Spearman IC
        # ----------------------------------------------------

        ranked_prediction = (
            day["prediction"]
            .rank(
                method="average"
            )
        )

        ranked_return = (
            day[
                f"future_{HORIZON}d"
            ]
            .rank(
                method="average"
            )
        )

        ic = (
            ranked_prediction
            .corr(
                ranked_return
            )
        )

        if pd.notna(ic):

            daily_ics.append(
                ic
            )

        # ----------------------------------------------------
        # Top / bottom 2
        # ----------------------------------------------------

        sorted_day = (
            day
            .sort_values(
                "prediction"
            )
        )

        bottom2 = (
            sorted_day
            .head(2)
        )

        top2 = (
            sorted_day
            .tail(2)
        )

        top_return = (
            top2[
                f"future_{HORIZON}d"
            ]
            .mean()
        )

        bottom_return = (
            bottom2[
                f"future_{HORIZON}d"
            ]
            .mean()
        )

        spread = (
            top_return
            - bottom_return
        )

        daily_top2.append(
            top_return
        )

        daily_bottom2.append(
            bottom_return
        )

        daily_spreads.append(
            spread
        )

        daily_hits.append(
            float(
                spread > 0
            )
        )

    daily_ics = np.asarray(
        daily_ics,
        dtype=np.float64,
    )

    daily_spreads = np.asarray(
        daily_spreads,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # IC
    # --------------------------------------------------------

    mean_ic = (
        daily_ics.mean()
        if len(daily_ics)
        else np.nan
    )

    median_ic = (
        np.median(
            daily_ics
        )
        if len(daily_ics)
        else np.nan
    )

    positive_ic_pct = (
        np.mean(
            daily_ics > 0
        )
        if len(daily_ics)
        else np.nan
    )

    # --------------------------------------------------------
    # Long-short
    # --------------------------------------------------------

    mean_spread = (
        daily_spreads.mean()
        if len(daily_spreads)
        else np.nan
    )

    median_spread = (
        np.median(
            daily_spreads
        )
        if len(daily_spreads)
        else np.nan
    )

    spread_std = (
        daily_spreads.std(
            ddof=1
        )
        if len(daily_spreads) > 1
        else np.nan
    )

    # This is NOT the final portfolio Sharpe.
    # It is the annualized Sharpe of the daily
    # 5-day ranking spread diagnostic.

    spread_sharpe = (
        mean_spread
        / spread_std
        * np.sqrt(252)
        if (
            np.isfinite(
                spread_std
            )
            and spread_std > 0
        )
        else np.nan
    )

    hit_rate = (
        np.mean(
            daily_hits
        )
        if daily_hits
        else np.nan
    )

    return {
        "IC": mean_ic,
        "Median_IC": median_ic,
        "Positive_IC_Rate": positive_ic_pct,
        "Top2_Avg_Return": (
            np.mean(daily_top2)
            if daily_top2
            else np.nan
        ),
        "Bottom2_Avg_Return": (
            np.mean(daily_bottom2)
            if daily_bottom2
            else np.nan
        ),
        "Long_Short_Avg":
            mean_spread,
        "Long_Short_Median":
            median_spread,
        "Long_Short_Sharpe":
            spread_sharpe,
        "Long_Short_Hit_Rate":
            hit_rate,
        "Days":
            len(daily_spreads),
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    predictions,
    path,
):

    predictions.to_csv(
        path,
        index=False,
    )

    print(
        f"Predictions saved: "
        f"{path}"
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    name,
    predictions,
):

    print()
    print("=" * 75)
    print(name)
    print("=" * 75)

    reg = regression_metrics(
        predictions
    )

    ranking = daily_ic_metrics(
        predictions
    )

    print()
    print("REGRESSION METRICS")

    for key, value in reg.items():

        print(
            f"{key:22s}: "
            f"{value:.6f}"
        )

    print()
    print("CROSS-SECTIONAL RANKING")

    for key, value in ranking.items():

        if (
            "Rate" in key
            or "Hit" in key
        ):

            print(
                f"{key:22s}: "
                f"{value:.2%}"
                if np.isfinite(value)
                else
                f"{key:22s}: NaN"
            )

        else:

            print(
                f"{key:22s}: "
                f"{value:.6f}"
                if np.isfinite(value)
                else
                f"{key:22s}: NaN"
            )

    return {
        **reg,
        **ranking,
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    scaler,
):

    path = (
        OUTPUT_DIR
        / "cross_sectional_5d_lstm.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "features":
                FEATURES,

            "horizon":
                HORIZON,

            "sequence_length":
                SEQ_LEN,

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "scaler_mean":
                scaler.mean_,

            "scaler_std":
                scaler.std_,
        },
        path,
    )

    print(
        f"Model saved: {path}"
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Future 5D return
    # --------------------------------------------------------

    df = create_future_return(
        df
    )

    # --------------------------------------------------------
    # Cross-sectional target
    # --------------------------------------------------------

    df = create_cross_sectional_target(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    df = add_split_column(
        df
    )

    # ========================================================
    # FIT SCALER
    #
    # ONLY TRAIN DATA
    # ========================================================

    print()
    print("=" * 75)
    print("FITTING TRAIN-ONLY SCALER")
    print("=" * 75)

    train_df = df[
        df["split"] == "train"
    ]

    train_features = (
        train_df[
            FEATURES
        ]
        .values
        .astype(
            np.float32
        )
    )

    scaler = StandardScaler()

    scaler.fit(
        train_features
    )

    print(
        "Scaler fitted using "
        f"{len(train_df):,} "
        "training rows."
    )

    # ========================================================
    # BUILD SEQUENCES
    # ========================================================

    sequences = build_sequences(
        df,
        scaler,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not sequences["train_X"]:
        raise RuntimeError(
            "No training sequences created."
        )

    if not sequences["val_X"]:
        raise RuntimeError(
            "No validation sequences created."
        )

    if not sequences["test_X"]:
        raise RuntimeError(
            "No test sequences created."
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = create_loader(
        sequences["train_X"],
        sequences["train_y"],
        shuffle=True,
    )

    val_loader = create_loader(
        sequences["val_X"],
        sequences["val_y"],
        shuffle=False,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model = train_model(
        train_loader,
        val_loader,
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    val_predictions = predict(
        model,

        sequences["val_X"],
        sequences["val_y"],

        sequences["val_dates"],
        sequences["val_tickers"],

        sequences["val_returns"],
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_predictions = predict(
        model,

        sequences["test_X"],
        sequences["test_y"],

        sequences["test_dates"],
        sequences["test_tickers"],

        sequences["test_returns"],
    )

    # ========================================================
    # REPORTS
    # ========================================================

    val_metrics = print_report(
        "5-DAY LSTM VALIDATION",
        val_predictions,
    )

    test_metrics = print_report(
        "5-DAY LSTM TEST",
        test_predictions,
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    val_prediction_path = (
        OUTPUT_DIR
        / "cross_sectional_5d_lstm_validation_predictions.csv"
    )

    test_prediction_path = (
        OUTPUT_DIR
        / "cross_sectional_5d_lstm_predictions.csv"
    )

    save_predictions(
        val_predictions,
        val_prediction_path,
    )

    save_predictions(
        test_predictions,
        test_prediction_path,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = save_model(
        model,
        scaler,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame(
        [
            {
                "split":
                    "validation",
                **val_metrics,
            },

            {
                "split":
                    "test",
                **test_metrics,
            },
        ]
    )

    metrics_path = (
        OUTPUT_DIR
        / "cross_sectional_5d_lstm_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print("FILES CREATED")
    print("=" * 75)

    print(
        val_prediction_path
    )

    print(
        test_prediction_path
    )

    print(
        metrics_path
    )

    print(
        model_path
    )

    print(
        OUTPUT_DIR
        / "cross_sectional_5d_lstm_training_history.csv"
    )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()