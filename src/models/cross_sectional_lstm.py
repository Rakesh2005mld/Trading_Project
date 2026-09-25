import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/processed/cross_sectional_dataset.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42


# =========================================================
# SEQUENCE CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 30


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

BATCH_SIZE = 64
EPOCHS = 100

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 12

GRADIENT_CLIP = 1.0


# =========================================================
# MODEL CONFIGURATION
# =========================================================

HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.30


# =========================================================
# CROSS-SECTIONAL TARGET
# =========================================================

# future_return is converted into a daily
# cross-sectional z-score:
#
#     (stock_return - daily_mean)
#     --------------------------------
#          daily_std
#
# This makes the target relative to other stocks
# on the same date.

TARGET_COLUMN = "future_return"

MIN_STOCKS_PER_DATE = 3


# =========================================================
# DATA SPLIT
# =========================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# TOP/BOTTOM PORTFOLIO
# =========================================================

TOP_K = 2
BOTTOM_K = 2


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(
    seed=RANDOM_STATE
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# =========================================================
# DEVICE
# =========================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(
        f"\nUsing device: {device}"
    )

    return device


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

    required_columns = [
        "Date",
        "Ticker",
        TARGET_COLUMN,
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # -----------------------------------------------------
    # Sort by ticker and date.
    #
    # This is critical because sequences must never cross
    # ticker boundaries.
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            ["Ticker", "Date"]
        )
        .drop_duplicates(
            subset=["Ticker", "Date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df


# =========================================================
# DATASET VALIDATION
# =========================================================

def validate_dataset(df):

    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        f"\nDate range:\n"
        f"{df['Date'].min()} → "
        f"{df['Date'].max()}"
    )

    print(
        f"\nUnique tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        "\nTickers:"
    )

    print(
        sorted(
            df["Ticker"].unique()
        )
    )

    # -----------------------------------------------------
    # Check duplicate ticker/date observations.
    # -----------------------------------------------------

    duplicates = df.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    print(
        f"\nDuplicate ticker/date rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate ticker/date observations found."
        )

    # -----------------------------------------------------
    # Check missing values.
    # -----------------------------------------------------

    missing = df.isna().sum()

    missing_columns = (
        missing[
            missing > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if len(missing_columns) > 0:

        print(
            "\nMissing values:"
        )

        print(
            missing_columns
        )

    else:

        print(
            "\nMissing values: 0"
        )

    # -----------------------------------------------------
    # Check infinities.
    # -----------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    inf_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        f"\nInfinite values: "
        f"{inf_count}"
    )

    if inf_count > 0:

        raise ValueError(
            "Dataset contains infinite values."
        )

    # -----------------------------------------------------
    # Check chronological order per ticker.
    # -----------------------------------------------------

    for ticker, group in df.groupby(
        "Ticker"
    ):

        if not group[
            "Date"
        ].is_monotonic_increasing:

            raise ValueError(
                f"Dates are not chronological for "
                f"{ticker}."
            )

    # -----------------------------------------------------
    # Check expected target leakage.
    # -----------------------------------------------------

    leakage_keywords = [
        "future",
        "target",
        "forward",
        "next_day",
        "nextday",
    ]

    suspicious_columns = []

    for column in df.columns:

        lower = column.lower()

        if any(
            keyword in lower
            for keyword in leakage_keywords
        ):

            suspicious_columns.append(
                column
            )

    print(
        "\nPotentially future-related columns:"
    )

    print(
        suspicious_columns
    )

    allowed_target_columns = {
        TARGET_COLUMN,
        "target",
        "trading_target",
    }

    unexpected = [
        column
        for column in suspicious_columns
        if column not in allowed_target_columns
    ]

    if unexpected:

        raise ValueError(
            "Potential leakage columns found:\n"
            f"{unexpected}"
        )

    print(
        "\nDataset validation: OK"
    )


# =========================================================
# CREATE CROSS-SECTIONAL TARGET
# =========================================================

def create_cross_sectional_target(
    df
):

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL TARGET")
    print("=" * 70)

    df = df.copy()

    # -----------------------------------------------------
    # Number of stocks available on each day.
    # -----------------------------------------------------

    stock_count = (
        df
        .groupby("Date")[
            TARGET_COLUMN
        ]
        .transform("count")
    )

    df["cross_section_count"] = (
        stock_count
    )

    # -----------------------------------------------------
    # Daily cross-sectional mean.
    # -----------------------------------------------------

    daily_mean = (
        df
        .groupby("Date")[
            TARGET_COLUMN
        ]
        .transform("mean")
    )

    # -----------------------------------------------------
    # Daily cross-sectional standard deviation.
    # -----------------------------------------------------

    daily_std = (
        df
        .groupby("Date")[
            TARGET_COLUMN
        ]
        .transform("std")
    )

    # -----------------------------------------------------
    # Relative future return.
    #
    # Positive:
    # expected to outperform the cross-section.
    #
    # Negative:
    # expected to underperform the cross-section.
    # -----------------------------------------------------

    df["relative_future_return"] = (
        df[TARGET_COLUMN]
        - daily_mean
    )

    # -----------------------------------------------------
    # Cross-sectional z-score.
    # -----------------------------------------------------

    df["cross_sectional_target"] = (

        df["relative_future_return"]

        / daily_std.replace(
            0,
            np.nan
        )
    )

    # -----------------------------------------------------
    # Remove dates with too few stocks.
    # -----------------------------------------------------

    valid_dates = (
        df.groupby("Date")[
            "Ticker"
        ]
        .count()
        .loc[
            lambda x:
                x >= MIN_STOCKS_PER_DATE
        ]
        .index
    )

    df = df[
        df["Date"].isin(
            valid_dates
        )
    ].copy()

    # -----------------------------------------------------
    # Remove invalid target rows.
    # -----------------------------------------------------

    df = df[
        np.isfinite(
            df["cross_sectional_target"]
        )
    ].copy()

    df = (
        df
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Valid rows: "
        f"{len(df)}"
    )

    print(
        f"Valid dates: "
        f"{df['Date'].nunique()}"
    )

    print(
        f"Unique tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        "\nCross-sectional target statistics:"
    )

    print(
        df[
            "cross_sectional_target"
        ].describe()
    )

    print(
        "\nMean target by date "
        "(should be approximately 0):"
    )

    daily_target_mean = (
        df
        .groupby("Date")[
            "cross_sectional_target"
        ]
        .mean()
    )

    print(
        daily_target_mean.describe()
    )

    return df


# =========================================================
# FEATURE SELECTION
# =========================================================

def get_feature_columns(
    df
):

    # -----------------------------------------------------
    # Never use identifiers or targets as model inputs.
    # -----------------------------------------------------

    forbidden_columns = {

        "Date",

        "Ticker",

        "Sector",

        TARGET_COLUMN,

        "relative_future_return",

        "cross_sectional_target",

        "cross_section_count",

        "target",

        "trading_target",
    }

    # -----------------------------------------------------
    # Leakage-name check.
    # -----------------------------------------------------

    leakage_keywords = [

        "future",

        "target",

        "forward",

        "next_day",

        "nextday",
    ]

    suspicious = []

    for column in df.columns:

        if column in forbidden_columns:

            continue

        lower = column.lower()

        if any(
            keyword in lower
            for keyword in leakage_keywords
        ):

            suspicious.append(
                column
            )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    print(
        f"Unexpected suspicious "
        f"feature columns: {suspicious}"
    )

    if suspicious:

        raise ValueError(
            "Potential leakage features:\n"
            f"{suspicious}"
        )

    feature_columns = [

        column

        for column in df.columns

        if column not in forbidden_columns

    ]

    # -----------------------------------------------------
    # Features must be numeric.
    # -----------------------------------------------------

    non_numeric = [

        column

        for column in feature_columns

        if not pd.api.types.is_numeric_dtype(
            df[column]
        )

    ]

    if non_numeric:

        raise ValueError(
            "Non-numeric model features found:\n"
            f"{non_numeric}"
        )

    if len(feature_columns) == 0:

        raise ValueError(
            "No features found."
        )

    print(
        f"\nNumber of features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nFeatures:"
    )

    print(
        feature_columns
    )

    return feature_columns


# =========================================================
# DATE SPLIT
# =========================================================

def create_date_split(
    df
):

    print("\n" + "=" * 70)
    print("DATE-BASED SPLIT")
    print("=" * 70)

    unique_dates = np.array(
        sorted(
            df["Date"].unique()
        )
    )

    n_dates = len(
        unique_dates
    )

    train_end = int(
        n_dates
        * TRAIN_RATIO
    )

    validation_end = int(
        n_dates
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    if (
        train_end <= 0
        or validation_end <= train_end
        or validation_end >= n_dates
    ):

        raise ValueError(
            "Invalid date split."
        )

    train_end_date = (
        unique_dates[
            train_end - 1
        ]
    )

    validation_end_date = (
        unique_dates[
            validation_end - 1
        ]
    )

    train_start_date = (
        unique_dates[0]
    )

    validation_start_date = (
        unique_dates[train_end]
    )

    test_start_date = (
        unique_dates[
            validation_end
        ]
    )

    test_end_date = (
        unique_dates[-1]
    )

    print(
        f"\nTotal dates: "
        f"{n_dates}"
    )

    print(
        f"\nTrain:"
        f"\n{pd.Timestamp(train_start_date).date()} "
        f"→ "
        f"{pd.Timestamp(train_end_date).date()}"
    )

    print(
        f"\nValidation:"
        f"\n{pd.Timestamp(validation_start_date).date()} "
        f"→ "
        f"{pd.Timestamp(validation_end_date).date()}"
    )

    print(
        f"\nTest:"
        f"\n{pd.Timestamp(test_start_date).date()} "
        f"→ "
        f"{pd.Timestamp(test_end_date).date()}"
    )

    return {
        "train_start": train_start_date,
        "train_end": train_end_date,

        "val_start": validation_start_date,
        "val_end": validation_end_date,

        "test_start": test_start_date,
        "test_end": test_end_date,
    }


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features(
    df,
    feature_columns,
    split_dates
):

    print("\n" + "=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    working = df.copy()

    # -----------------------------------------------------
    # Convert model features to numeric.
    # -----------------------------------------------------

    for column in feature_columns:

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Replace infinities.
    # -----------------------------------------------------

    working[feature_columns] = (
        working[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # -----------------------------------------------------
    # Ensure targets are valid.
    # -----------------------------------------------------

    working = working[
        working[feature_columns]
        .notna()
        .all(axis=1)
    ].copy()

    working = working[
        working[
            "cross_sectional_target"
        ].notna()
    ].copy()

    working = working[
        working[
            TARGET_COLUMN
        ].notna()
    ].copy()

    working = (
        working
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # TRAIN MASK
    # -----------------------------------------------------

    train_mask = (

        working["Date"]
        >= split_dates["train_start"]

    ) & (

        working["Date"]
        <= split_dates["train_end"]

    )

    X_train_raw = (
        working
        .loc[
            train_mask,
            feature_columns
        ]
        .copy()
    )

    # -----------------------------------------------------
    # Scaler fitted on TRAIN ONLY.
    # -----------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_train_raw
    )

    print(
        "StandardScaler fitted on TRAIN only."
    )

    # -----------------------------------------------------
    # Transform ALL usable observations.
    #
    # This is safe because the scaler parameters were learned
    # strictly from train.
    # -----------------------------------------------------

    transformed = scaler.transform(
        working[
            feature_columns
        ]
    )

    working[
        feature_columns
    ] = transformed

    print(
        f"Scaled feature matrix: "
        f"{working[feature_columns].shape}"
    )

    return (
        working,
        scaler
    )


# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(
    df,
    feature_columns,
    split_dates,
    sequence_length
):

    print("\n" + "=" * 70)
    print("SEQUENCE GENERATION")
    print("=" * 70)

    train_sequences = []
    train_targets = []
    train_returns = []
    train_dates = []
    train_tickers = []

    val_sequences = []
    val_targets = []
    val_returns = []
    val_dates = []
    val_tickers = []

    test_sequences = []
    test_targets = []
    test_returns = []
    test_dates = []
    test_tickers = []

    # -----------------------------------------------------
    # Process each ticker independently.
    # -----------------------------------------------------

    for ticker, group in df.groupby(
        "Ticker",
        sort=True
    ):

        group = (
            group
            .sort_values("Date")
            .reset_index(drop=True)
        )

        X = (
            group[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        y = (
            group[
                "cross_sectional_target"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        raw_returns = (
            group[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        dates = (
            group["Date"]
            .to_numpy()
        )

        # -------------------------------------------------
        # Target at index i:
        #
        # future_return at date i
        #
        # Input:
        #
        # [i-sequence_length+1 ... i]
        #
        # Thus the current day's features are available to
        # predict the forward return.
        # -------------------------------------------------

        for i in range(
            sequence_length - 1,
            len(group)
        ):

            target_date = (
                dates[i]
            )

            if (
                target_date
                <= split_dates["train_end"]
            ):

                destination = "train"

            elif (
                target_date
                <= split_dates["val_end"]
            ):

                destination = "validation"

            else:

                destination = "test"

            sequence = X[
                i - sequence_length + 1:
                i + 1
            ]

            target = y[i]

            raw_return = (
                raw_returns[i]
            )

            if destination == "train":

                train_sequences.append(
                    sequence
                )

                train_targets.append(
                    target
                )

                train_returns.append(
                    raw_return
                )

                train_dates.append(
                    target_date
                )

                train_tickers.append(
                    ticker
                )

            elif destination == "validation":

                val_sequences.append(
                    sequence
                )

                val_targets.append(
                    target
                )

                val_returns.append(
                    raw_return
                )

                val_dates.append(
                    target_date
                )

                val_tickers.append(
                    ticker
                )

            else:

                test_sequences.append(
                    sequence
                )

                test_targets.append(
                    target
                )

                test_returns.append(
                    raw_return
                )

                test_dates.append(
                    target_date
                )

                test_tickers.append(
                    ticker
                )

    # -----------------------------------------------------
    # Convert to numpy arrays.
    # -----------------------------------------------------

    train_sequences = np.asarray(
        train_sequences,
        dtype=np.float32
    )

    train_targets = np.asarray(
        train_targets,
        dtype=np.float32
    )

    train_returns = np.asarray(
        train_returns,
        dtype=np.float32
    )

    val_sequences = np.asarray(
        val_sequences,
        dtype=np.float32
    )

    val_targets = np.asarray(
        val_targets,
        dtype=np.float32
    )

    val_returns = np.asarray(
        val_returns,
        dtype=np.float32
    )

    test_sequences = np.asarray(
        test_sequences,
        dtype=np.float32
    )

    test_targets = np.asarray(
        test_targets,
        dtype=np.float32
    )

    test_returns = np.asarray(
        test_returns,
        dtype=np.float32
    )

    train_dates = pd.Series(
        train_dates
    )

    val_dates = pd.Series(
        val_dates
    )

    test_dates = pd.Series(
        test_dates
    )

    train_tickers = pd.Series(
        train_tickers
    )

    val_tickers = pd.Series(
        val_tickers
    )

    test_tickers = pd.Series(
        test_tickers
    )

    print(
        f"\nSequence length: "
        f"{sequence_length}"
    )

    print(
        f"Train shape:"
        f" {train_sequences.shape}"
    )

    print(
        f"Validation shape:"
        f" {val_sequences.shape}"
    )

    print(
        f"Test shape:"
        f" {test_sequences.shape}"
    )

    print(
        "\nTrain date range:"
    )

    print(
        f"{train_dates.min().date()} "
        f"→ "
        f"{train_dates.max().date()}"
    )

    print(
        "\nValidation date range:"
    )

    print(
        f"{val_dates.min().date()} "
        f"→ "
        f"{val_dates.max().date()}"
    )

    print(
        "\nTest date range:"
    )

    print(
        f"{test_dates.min().date()} "
        f"→ "
        f"{test_dates.max().date()}"
    )

    return {

        "train_X": train_sequences,
        "train_y": train_targets,
        "train_returns": train_returns,
        "train_dates": train_dates,
        "train_tickers": train_tickers,

        "val_X": val_sequences,
        "val_y": val_targets,
        "val_returns": val_returns,
        "val_dates": val_dates,
        "val_tickers": val_tickers,

        "test_X": test_sequences,
        "test_y": test_targets,
        "test_returns": test_returns,
        "test_dates": test_dates,
        "test_tickers": test_tickers,
    }


# =========================================================
# PYTORCH DATASET
# =========================================================

class CrossSectionalDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y
    ):

        if len(X) != len(y):

            raise ValueError(
                "X and y lengths do not match."
            )

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


# =========================================================
# LSTM MODEL
# =========================================================

class CrossSectionalLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.fc1 = nn.Linear(
            hidden_size,
            16
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            16,
            1
        )

    def forward(
        self,
        x
    ):

        # -------------------------------------------------
        # x:
        #
        # [batch, sequence_length, features]
        # -------------------------------------------------

        lstm_output, _ = (
            self.lstm(x)
        )

        # Final timestep
        last_output = (
            lstm_output[:, -1, :]
        )

        x = self.fc1(
            last_output
        )

        x = self.relu(x)

        x = self.dropout(x)

        output = self.fc2(x)

        return output.squeeze(1)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        batch_size = (
            X_batch.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / total_samples
    )


# =========================================================
# VALIDATION LOSS
# =========================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = (
                X_batch.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        total_loss
        / total_samples
    )


# =========================================================
# PREDICTION
# =========================================================

def predict(
    model,
    loader,
    device
):

    model.eval()

    predictions = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(
                device
            )

            outputs = model(
                X_batch
            )

            predictions.extend(
                outputs
                .cpu()
                .numpy()
            )

    return np.asarray(
        predictions,
        dtype=np.float64
    )


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device
):

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL LSTM TRAINING")
    print("=" * 70)

    # -----------------------------------------------------
    # Huber loss for robust return prediction.
    # -----------------------------------------------------

    criterion = nn.HuberLoss(
        delta=1.0
    )

    # -----------------------------------------------------
    # AdamW.
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )
    )

    best_val_loss = float(
        "inf"
    )

    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss": val_loss,

            "learning_rate": current_lr

        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < (
            best_val_loss - 1e-6
        ):

            best_val_loss = val_loss

            best_state = {
                name:
                    parameter
                    .detach()
                    .cpu()
                    .clone()

                for name, parameter
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "Best model state was not saved."
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.8f}"
    )

    print(
        f"Epochs used: "
        f"{len(history)}"
    )

    return (
        model,
        pd.DataFrame(history)
    )


# =========================================================
# SAFE CORRELATION
# =========================================================

def safe_correlation(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    if (
        len(actual) < 2
        or np.std(actual) == 0
        or np.std(predicted) == 0
    ):

        return np.nan

    return np.corrcoef(
        actual,
        predicted
    )[0, 1]


# =========================================================
# SPEARMAN IC
# =========================================================

def calculate_daily_spearman_ic(
    predictions,
    actual_targets,
    dates
):

    evaluation = pd.DataFrame({

        "Date": dates.values,

        "prediction": predictions,

        "actual_target": actual_targets,

    })

    daily_ic = []

    for date, group in evaluation.groupby(
        "Date"
    ):

        if len(group) < 2:

            continue

        prediction_rank = (
            group["prediction"]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        actual_rank = (
            group["actual_target"]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        ic = safe_correlation(
            prediction_rank,
            actual_rank
        )

        if np.isfinite(ic):

            daily_ic.append(
                ic
            )

    daily_ic = np.asarray(
        daily_ic,
        dtype=np.float64
    )

    if len(daily_ic) == 0:

        return (
            np.nan,
            np.nan,
            daily_ic
        )

    return (
        float(np.mean(daily_ic)),
        float(
            np.median(daily_ic)
        ),
        daily_ic
    )


# =========================================================
# CROSS-SECTIONAL PORTFOLIO EVALUATION
# =========================================================

def evaluate_cross_sectional_portfolio(
    dates,
    tickers,
    predictions,
    actual_returns,
    actual_targets,
    dataset_name
):

    data = pd.DataFrame({

        "Date": dates.values,

        "Ticker": tickers.values,

        "prediction": predictions,

        "actual_target":
            actual_targets,

        "actual_return":
            actual_returns,

    })

    daily_ic_mean, daily_ic_median, daily_ic = (
        calculate_daily_spearman_ic(
            predictions,
            actual_targets,
            dates
        )
    )

    long_returns = []
    short_returns = []
    long_short_returns = []
    market_returns = []

    selected_rows = []

    # -----------------------------------------------------
    # Evaluate each date independently.
    # -----------------------------------------------------

    for date, group in data.groupby(
        "Date"
    ):

        group = group.sort_values(
            "prediction",
            ascending=False
        )

        if len(group) < (
            TOP_K + BOTTOM_K
        ):

            continue

        top = group.head(
            TOP_K
        )

        bottom = group.tail(
            BOTTOM_K
        )

        long_return = (
            top["actual_return"]
            .mean()
        )

        short_return = (
            bottom["actual_return"]
            .mean()
        )

        long_short = (
            long_return
            - short_return
        )

        market_return = (
            group["actual_return"]
            .mean()
        )

        long_returns.append(
            long_return
        )

        short_returns.append(
            short_return
        )

        long_short_returns.append(
            long_short
        )

        market_returns.append(
            market_return
        )

        top_copy = top.copy()

        top_copy[
            "portfolio_side"
        ] = "LONG"

        bottom_copy = bottom.copy()

        bottom_copy[
            "portfolio_side"
        ] = "SHORT"

        selected_rows.append(
            pd.concat(
                [
                    top_copy,
                    bottom_copy
                ]
            )
        )

    long_returns = np.asarray(
        long_returns,
        dtype=np.float64
    )

    short_returns = np.asarray(
        short_returns,
        dtype=np.float64
    )

    long_short_returns = np.asarray(
        long_short_returns,
        dtype=np.float64
    )

    market_returns = np.asarray(
        market_returns,
        dtype=np.float64
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mean_long = (
        np.mean(long_returns)
        if len(long_returns) > 0
        else np.nan
    )

    mean_short = (
        np.mean(short_returns)
        if len(short_returns) > 0
        else np.nan
    )

    mean_long_short = (
        np.mean(long_short_returns)
        if len(long_short_returns) > 0
        else np.nan
    )

    mean_market = (
        np.mean(market_returns)
        if len(market_returns) > 0
        else np.nan
    )

    long_short_std = (
        np.std(long_short_returns)
        if len(long_short_returns) > 1
        else np.nan
    )

    if (
        np.isfinite(long_short_std)
        and long_short_std > 0
    ):

        long_short_sharpe_daily = (
            mean_long_short
            / long_short_std
        )

    else:

        long_short_sharpe_daily = np.nan

    trading_days = len(
        long_short_returns
    )

    if (
        np.isfinite(
            long_short_sharpe_daily
        )
        and trading_days > 1
    ):

        annualized_sharpe = (
            long_short_sharpe_daily
            * np.sqrt(252)
        )

    else:

        annualized_sharpe = np.nan

    # -----------------------------------------------------
    # Hit ratio.
    # -----------------------------------------------------

    long_short_hit_rate = (

        np.mean(
            long_short_returns > 0
        )

        if len(long_short_returns) > 0

        else np.nan
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} "
        f"CROSS-SECTIONAL EVALUATION"
    )
    print("=" * 70)

    print(
        f"\nEvaluation dates: "
        f"{trading_days}"
    )

    print(
        f"Daily mean Spearman IC: "
        f"{daily_ic_mean}"
    )

    print(
        f"Daily median Spearman IC: "
        f"{daily_ic_median}"
    )

    print(
        f"Mean top-{TOP_K} return: "
        f"{mean_long}"
    )

    print(
        f"Mean bottom-{BOTTOM_K} return: "
        f"{mean_short}"
    )

    print(
        f"Mean long-short return: "
        f"{mean_long_short}"
    )

    print(
        f"Mean market return: "
        f"{mean_market}"
    )

    print(
        f"Long-short hit rate: "
        f"{long_short_hit_rate}"
    )

    print(
        f"Annualized long-short Sharpe: "
        f"{annualized_sharpe}"
    )

    if len(selected_rows) > 0:

        selected = pd.concat(
            selected_rows,
            ignore_index=True
        )

    else:

        selected = pd.DataFrame()

    metrics = {

        "daily_ic_mean":
            daily_ic_mean,

        "daily_ic_median":
            daily_ic_median,

        "mean_long_return":
            mean_long,

        "mean_short_return":
            mean_short,

        "mean_long_short_return":
            mean_long_short,

        "mean_market_return":
            mean_market,

        "long_short_hit_rate":
            long_short_hit_rate,

        "annualized_long_short_sharpe":
            annualized_sharpe,
    }

    return (
        metrics,
        data,
        selected
    )


# =========================================================
# REGRESSION METRICS
# =========================================================

def evaluate_regression(
    actual,
    predicted,
    dataset_name
):

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    correlation = safe_correlation(
        actual,
        predicted
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} "
        f"REGRESSION METRICS"
    )
    print("=" * 70)

    print(
        f"\nMAE         : {mae:.8f}"
    )

    print(
        f"RMSE        : {rmse:.8f}"
    )

    print(
        f"R²          : {r2:.8f}"
    )

    print(
        f"Correlation : {correlation}"
    )

    print(
        f"Prediction Std: "
        f"{np.std(predicted):.8f}"
    )

    print(
        f"Actual Std: "
        f"{np.std(actual):.8f}"
    )

    return {

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "correlation": correlation,

    }


# =========================================================
# SAVE PREDICTIONS
# =========================================================

def save_predictions(
    dates,
    tickers,
    actual_targets,
    predicted_targets,
    actual_returns
):

    output = pd.DataFrame({

        "Date":
            dates.values,

        "Ticker":
            tickers.values,

        "actual_cross_sectional_target":
            actual_targets,

        "predicted_cross_sectional_target":
            predicted_targets,

        "actual_return":
            actual_returns,

    })

    # -----------------------------------------------------
    # Rank predictions and actuals within each day.
    # -----------------------------------------------------

    output[
        "prediction_rank"
    ] = (
        output
        .groupby("Date")[
            "predicted_cross_sectional_target"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )

    output[
        "actual_rank"
    ] = (
        output
        .groupby("Date")[
            "actual_cross_sectional_target"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )

    output[
        "prediction_percentile"
    ] = (
        output
        .groupby("Date")[
            "predicted_cross_sectional_target"
        ]
        .rank(
            pct=True
        )
    )

    path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_predictions.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"\nPredictions saved to:\n"
        f"{path}"
    )

    return output


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model,
    input_size,
    feature_columns
):

    path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_model.pt"
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "input_size":
            input_size,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "sequence_length":
            SEQUENCE_LENGTH,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "loss":
            "HuberLoss",

        "optimizer":
            "AdamW",

        "feature_columns":
            feature_columns,

        "target":
            "cross_sectional_target",

        "target_definition":
            "(future_return - daily_mean) / daily_std",

    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"\nModel saved to:\n"
        f"{path}"
    )


# =========================================================
# SAVE SCALER
# =========================================================

def save_scaler(
    scaler,
    feature_columns
):

    path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_scaler.npz"
    )

    np.savez(

        path,

        mean=scaler.mean_,

        scale=scaler.scale_,

        feature_columns=np.asarray(
            feature_columns,
            dtype=str
        )

    )

    print(
        f"\nScaler saved to:\n"
        f"{path}"
    )


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    history
):

    path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_training_history.csv"
    )

    history.to_csv(
        path,
        index=False
    )

    print(
        f"\nTraining history saved to:\n"
        f"{path}"
    )


# =========================================================
# SAVE PORTFOLIO ANALYSIS
# =========================================================

def save_portfolio_analysis(
    portfolio_data,
    selected
):

    portfolio_path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_portfolio.csv"
    )

    portfolio_data.to_csv(
        portfolio_path,
        index=False
    )

    print(
        f"\nPortfolio evaluation saved to:\n"
        f"{portfolio_path}"
    )

    selected_path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_selected_stocks.csv"
    )

    selected.to_csv(
        selected_path,
        index=False
    )

    print(
        f"\nSelected stocks saved to:\n"
        f"{selected_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed()

    device = get_device()

    # =====================================================
    # 1. LOAD
    # =====================================================

    df = load_dataset()

    # =====================================================
    # 2. VALIDATE
    # =====================================================

    validate_dataset(
        df
    )

    # =====================================================
    # 3. CREATE CROSS-SECTIONAL TARGET
    # =====================================================

    df = create_cross_sectional_target(
        df
    )

    # =====================================================
    # 4. FEATURES
    # =====================================================

    feature_columns = get_feature_columns(
        df
    )

    # =====================================================
    # 5. DATE SPLIT
    # =====================================================

    split_dates = create_date_split(
        df
    )

    # =====================================================
    # 6. FEATURE PREPARATION
    # =====================================================

    (
        df,
        scaler
    ) = prepare_features(
        df,
        feature_columns,
        split_dates
    )

    # =====================================================
    # 7. SEQUENCES
    # =====================================================

    sequences = create_sequences(

        df,

        feature_columns,

        split_dates,

        SEQUENCE_LENGTH

    )

    # =====================================================
    # 8. DATASETS
    # =====================================================

    train_dataset = CrossSectionalDataset(

        sequences["train_X"],

        sequences["train_y"]

    )

    val_dataset = CrossSectionalDataset(

        sequences["val_X"],

        sequences["val_y"]

    )

    test_dataset = CrossSectionalDataset(

        sequences["test_X"],

        sequences["test_y"]

    )

    # =====================================================
    # 9. DATALOADERS
    # =====================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    # =====================================================
    # 10. MODEL
    # =====================================================

    input_size = (
        sequences["train_X"].shape[2]
    )

    model = CrossSectionalLSTM(

        input_size=input_size

    ).to(device)

    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)

    print(
        model
    )

    trainable_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_parameters:,}"
    )

    # =====================================================
    # 11. TRAIN
    # =====================================================

    (
        model,
        history
    ) = train_model(

        model,

        train_loader,

        val_loader,

        device

    )

    # =====================================================
    # 12. TRAIN PREDICTIONS
    # =====================================================

    train_predictions = predict(

        model,

        train_loader,

        device

    )

    train_regression = evaluate_regression(

        sequences["train_y"],

        train_predictions,

        "Train"

    )

    # =====================================================
    # 13. VALIDATION PREDICTIONS
    # =====================================================

    val_predictions = predict(

        model,

        val_loader,

        device

    )
    # =========================================================
    # SAVE VALIDATION PREDICTIONS
    # =========================================================

    val_output = pd.DataFrame({

        "Date":
            sequences["val_dates"].values,

        "Ticker":
            sequences["val_tickers"].values,

        "actual_cross_sectional_target":
            sequences["val_y"],

        "predicted_cross_sectional_target":
            val_predictions,

        "actual_return":
            sequences["val_returns"],

    })

    # -----------------------------------------------------
    # Rank predictions and actuals within each day.
    # -----------------------------------------------------

    val_output[
        "prediction_rank"
    ] = (
        val_output
        .groupby("Date")[
            "predicted_cross_sectional_target"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )

    val_output[
        "actual_rank"
    ] = (
        val_output
        .groupby("Date")[
            "actual_cross_sectional_target"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )

    val_output[
        "prediction_percentile"
    ] = (
        val_output
        .groupby("Date")[
            "predicted_cross_sectional_target"
        ]
        .rank(
            pct=True
        )
    )

    validation_prediction_path = os.path.join(
        MODEL_DIR,
        "cross_sectional_lstm_validation_predictions.csv"
    )

    val_output.to_csv(
        validation_prediction_path,
        index=False
    )

    print(
        f"\nValidation predictions saved to:\n"
        f"{validation_prediction_path}"
    )

    # =====================================================
    # 14. TEST PREDICTIONS
    # =====================================================

    test_predictions = predict(

        model,

        test_loader,

        device

    )

    test_regression = evaluate_regression(

        sequences["test_y"],

        test_predictions,

        "Test"

    )

    # =====================================================
    # 15. CROSS-SECTIONAL TEST EVALUATION
    # =====================================================

    (
        portfolio_metrics,
        portfolio_data,
        selected
    ) = evaluate_cross_sectional_portfolio(

        sequences["test_dates"],

        sequences["test_tickers"],

        test_predictions,

        sequences["test_returns"],

        sequences["test_y"],

        "Test"

    )

    # =====================================================
    # 16. PREDICTION SANITY CHECK
    # =====================================================

    print("\n" + "=" * 70)
    print("PREDICTION SANITY CHECK")
    print("=" * 70)

    prediction_std = np.std(
        test_predictions
    )

    actual_std = np.std(
        sequences["test_y"]
    )

    positive_rate = (
        test_predictions > 0
    ).mean()

    print(
        f"\nPrediction mean: "
        f"{np.mean(test_predictions):.8f}"
    )

    print(
        f"Prediction std: "
        f"{prediction_std:.8f}"
    )

    print(
        f"Actual target std: "
        f"{actual_std:.8f}"
    )

    print(
        f"Prediction positive rate: "
        f"{positive_rate:.2%}"
    )

    if (
        actual_std > 0
        and prediction_std / actual_std < 0.05
    ):

        print(
            "\nWARNING: Predictions are strongly "
            "collapsed toward the mean."
        )

    else:

        print(
            "\nPrediction variance sanity check: OK"
        )

    # =====================================================
    # 17. SAVE TEST PREDICTIONS
    # =====================================================

    prediction_dataframe = save_predictions(

        sequences["test_dates"],

        sequences["test_tickers"],

        sequences["test_y"],

        test_predictions,

        sequences["test_returns"]

    )

    # =====================================================
    # 18. SAVE PORTFOLIO DATA
    # =====================================================

    save_portfolio_analysis(

        portfolio_data,

        selected

    )

    # =====================================================
    # 19. SAVE HISTORY
    # =====================================================

    save_history(
        history
    )

    # =====================================================
    # 20. SAVE MODEL
    # =====================================================

    save_model(

        model,

        input_size,

        feature_columns

    )

    # =====================================================
    # 21. SAVE SCALER
    # =====================================================

    save_scaler(

        scaler,

        feature_columns

    )

    # =====================================================
    # 22. FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"\nTest RMSE "
        f"(cross-sectional target): "
        f"{test_regression['rmse']:.8f}"
    )

    print(
        f"Test R²: "
        f"{test_regression['r2']:.8f}"
    )

    print(
        f"Test correlation: "
        f"{test_regression['correlation']}"
    )

    print(
        f"\nDaily Spearman IC: "
        f"{portfolio_metrics['daily_ic_mean']}"
    )

    print(
        f"Median daily Spearman IC: "
        f"{portfolio_metrics['daily_ic_median']}"
    )

    print(
        f"\nTop-{TOP_K} mean return: "
        f"{portfolio_metrics['mean_long_return']}"
    )

    print(
        f"Bottom-{BOTTOM_K} mean return: "
        f"{portfolio_metrics['mean_short_return']}"
    )

    print(
        f"Long-short mean return: "
        f"{portfolio_metrics['mean_long_short_return']}"
    )

    print(
        f"Long-short hit rate: "
        f"{portfolio_metrics['long_short_hit_rate']}"
    )

    print(
        f"Annualized long-short Sharpe: "
        f"{portfolio_metrics['annualized_long_short_sharpe']}"
    )

    print(
        "\nFiles:"
    )

    print(
        "models/cross_sectional_lstm_model.pt"
    )

    print(
        "models/cross_sectional_lstm_scaler.npz"
    )

    print(
        "models/cross_sectional_lstm_predictions.csv"
    )

    print(
        "models/cross_sectional_lstm_portfolio.csv"
    )

    print(
        "models/cross_sectional_lstm_selected_stocks.csv"
    )

    print(
        "models/cross_sectional_lstm_training_history.csv"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()