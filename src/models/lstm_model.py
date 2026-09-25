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

DATA_PATH = "data/processed/training_dataset.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42


# =========================================================
# SEQUENCE
# =========================================================

SEQUENCE_LENGTH = 30


# =========================================================
# TRAINING
# =========================================================

BATCH_SIZE = 32
EPOCHS = 100

# Lower learning rate than previous version
LEARNING_RATE = 3e-4

# AdamW regularization
WEIGHT_DECAY = 1e-4

PATIENCE = 12

# =========================================================
# LSTM ARCHITECTURE
# =========================================================

HIDDEN_SIZE = 48
NUM_LAYERS = 2

# Stronger regularization
DROPOUT = 0.30

# =========================================================
# GRADIENT STABILITY
# =========================================================

GRADIENT_CLIP = 1.0

# =========================================================
# TARGET NORMALIZATION
# =========================================================

VOLATILITY_WINDOW = 20


# =========================================================
# DATA SPLIT
# =========================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# SETUP
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=RANDOM_STATE):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CUDA execution more deterministic where possible
    if torch.cuda.is_available():
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

    print(f"\nUsing device: {device}")

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

    # -----------------------------------------------------
    # Important:
    #
    # training_dataset.csv is AAPL time-series data.
    #
    # Remove duplicate dates defensively.
    # -----------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df


# =========================================================
# VALIDATE DATASET
# =========================================================

def validate_dataset(df):

    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print(
        f"\nDate range:\n"
        f"{df['Date'].min()} → "
        f"{df['Date'].max()}"
    )

    required_columns = [
        "Date",
        "future_return",
        "aapl_return",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    if df["Date"].isna().any():

        raise ValueError(
            "Invalid dates found."
        )

    if not df["Date"].is_monotonic_increasing:

        raise ValueError(
            "Dates are not chronological."
        )

    if df["future_return"].isna().any():

        raise ValueError(
            "future_return contains NaN."
        )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    inf_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        f"\nInfinite values: {inf_count}"
    )

    if inf_count > 0:

        raise ValueError(
            "Dataset contains infinite values."
        )

    print(
        "\nDataset validation: OK"
    )


# =========================================================
# FEATURE SELECTION
# =========================================================

def get_feature_columns(df):

    # -----------------------------------------------------
    # Never use targets or future information as inputs.
    # -----------------------------------------------------

    forbidden_columns = {
        "Date",
        "future_return",
        "target",
        "trading_target",
    }

    # -----------------------------------------------------
    # Remove raw moving-average price levels.
    #
    # Ratios are already present and are more scale stable.
    # -----------------------------------------------------

    unwanted_columns = {
        "aapl_ma20",
        "aapl_ma50",
        "aapl_ma200",

        "spy_ma20",
        "spy_ma50",
        "spy_ma200",
    }

    # -----------------------------------------------------
    # Leakage-name check
    # -----------------------------------------------------

    leakage_keywords = [
        "future",
        "target",
        "next_day",
        "nextday",
        "forward",
    ]

    suspicious_columns = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in leakage_keywords
        ):

            suspicious_columns.append(column)

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    print(
        f"Suspicious columns: "
        f"{suspicious_columns}"
    )

    expected_suspicious = {
        "future_return",
        "target",
        "trading_target",
    }

    unexpected = [
        column
        for column in suspicious_columns
        if column not in expected_suspicious
    ]

    if unexpected:

        raise ValueError(
            "Potential leakage columns found: "
            f"{unexpected}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden_columns
        and column not in unwanted_columns
    ]

    if not feature_columns:

        raise ValueError(
            "No feature columns found."
        )

    print(
        f"\nNumber of features: "
        f"{len(feature_columns)}"
    )

    return feature_columns


# =========================================================
# PREPARE TARGETS
# =========================================================

def prepare_dataframe(
    df,
    feature_columns
):

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    features = df[
        feature_columns
    ].copy()

    # -----------------------------------------------------
    # Convert features to numeric
    # -----------------------------------------------------

    for column in features.columns:

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Remove inf
    # -----------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # RAW TARGET
    # -----------------------------------------------------

    raw_target = (
        df["future_return"]
        .astype(float)
        .copy()
    )

    # -----------------------------------------------------
    # CURRENT VOLATILITY
    #
    # Rolling volatility contains only current and past
    # AAPL returns.
    #
    # It is therefore available at prediction time.
    # -----------------------------------------------------

    volatility = (
        df["aapl_return"]
        .rolling(
            VOLATILITY_WINDOW
        )
        .std()
    )

    volatility = volatility.replace(
        0,
        np.nan
    )

    # -----------------------------------------------------
    # NORMALIZED TARGET
    #
    # future_return / current volatility
    #
    # IMPORTANT:
    # future_return itself is never used as an input.
    # -----------------------------------------------------

    normalized_target = (
        raw_target / volatility
    )

    dates = df["Date"].copy()

    # -----------------------------------------------------
    # VALID ROWS
    # -----------------------------------------------------

    valid_rows = (
        features.notna().all(axis=1)
        & raw_target.notna()
        & normalized_target.notna()
        & volatility.notna()
        & dates.notna()
        & np.isfinite(normalized_target)
        & np.isfinite(volatility)
        & (volatility > 0)
    )

    features = (
        features
        .loc[valid_rows]
        .reset_index(drop=True)
    )

    raw_target = (
        raw_target
        .loc[valid_rows]
        .reset_index(drop=True)
    )

    normalized_target = (
        normalized_target
        .loc[valid_rows]
        .reset_index(drop=True)
    )

    volatility = (
        volatility
        .loc[valid_rows]
        .reset_index(drop=True)
    )

    dates = (
        dates
        .loc[valid_rows]
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------

    if len(features) != len(raw_target):

        raise ValueError(
            "Feature/target length mismatch."
        )

    print("\n" + "=" * 70)
    print("PREPARED DATA")
    print("=" * 70)

    print(
        f"Features shape      : "
        f"{features.shape}"
    )

    print(
        f"Raw target shape    : "
        f"{raw_target.shape}"
    )

    print(
        f"Normalized target   : "
        f"{normalized_target.shape}"
    )

    print(
        f"Volatility shape    : "
        f"{volatility.shape}"
    )

    print(
        f"Date shape          : "
        f"{dates.shape}"
    )

    print(
        "\nRaw future return statistics:"
    )

    print(
        raw_target.describe()
    )

    print(
        "\nNormalized target statistics:"
    )

    print(
        normalized_target.describe()
    )

    print(
        "\nCurrent volatility statistics:"
    )

    print(
        volatility.describe()
    )

    return (
        features,
        normalized_target,
        raw_target,
        volatility,
        dates
    )


# =========================================================
# TIME SPLIT
# =========================================================

def split_data(
    features,
    normalized_target,
    raw_target,
    volatility,
    dates
):

    n = len(features)

    if not np.isclose(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO,
        1.0
    ):

        raise ValueError(
            "Train/validation/test ratios must sum to 1."
        )

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    X_train = features.iloc[
        :train_end
    ].copy()

    y_train_normalized = normalized_target.iloc[
        :train_end
    ].copy()

    y_train_raw = raw_target.iloc[
        :train_end
    ].copy()

    vol_train = volatility.iloc[
        :train_end
    ].copy()

    dates_train = dates.iloc[
        :train_end
    ].copy()

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    X_val = features.iloc[
        train_end:validation_end
    ].copy()

    y_val_normalized = normalized_target.iloc[
        train_end:validation_end
    ].copy()

    y_val_raw = raw_target.iloc[
        train_end:validation_end
    ].copy()

    vol_val = volatility.iloc[
        train_end:validation_end
    ].copy()

    dates_val = dates.iloc[
        train_end:validation_end
    ].copy()

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    X_test = features.iloc[
        validation_end:
    ].copy()

    y_test_normalized = normalized_target.iloc[
        validation_end:
    ].copy()

    y_test_raw = raw_target.iloc[
        validation_end:
    ].copy()

    vol_test = volatility.iloc[
        validation_end:
    ].copy()

    dates_test = dates.iloc[
        validation_end:
    ].copy()

    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TIME SPLIT")
    print("=" * 70)

    print(
        f"Train      : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_val)}"
    )

    print(
        f"Test       : {len(X_test)}"
    )

    print("\nTrain:")

    print(
        f"{dates_train.iloc[0].date()} → "
        f"{dates_train.iloc[-1].date()}"
    )

    print("\nValidation:")

    print(
        f"{dates_val.iloc[0].date()} → "
        f"{dates_val.iloc[-1].date()}"
    )

    print("\nTest:")

    print(
        f"{dates_test.iloc[0].date()} → "
        f"{dates_test.iloc[-1].date()}"
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    if dates_train.max() >= dates_val.min():

        raise ValueError(
            "Train and validation overlap."
        )

    if dates_val.max() >= dates_test.min():

        raise ValueError(
            "Validation and test overlap."
        )

    return (
        X_train,
        y_train_normalized,
        y_train_raw,
        vol_train,
        dates_train,

        X_val,
        y_val_normalized,
        y_val_raw,
        vol_val,
        dates_val,

        X_test,
        y_test_normalized,
        y_test_raw,
        vol_test,
        dates_test
    )


# =========================================================
# SCALING
# =========================================================

def fit_scaler(X_train):

    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()

    # -----------------------------------------------------
    # CRITICAL:
    #
    # Fit ONLY on training data.
    # -----------------------------------------------------

    scaler.fit(X_train)

    print(
        "StandardScaler fitted on TRAIN only."
    )

    return scaler


def transform_features(
    scaler,
    X_train,
    X_val,
    X_test
):

    X_train_scaled = scaler.transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "Training features transformed."
    )

    print(
        "Validation features transformed."
    )

    print(
        "Test features transformed."
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled
    )


# =========================================================
# TRAINING SEQUENCES
# =========================================================

def create_training_sequences(
    X,
    normalized_target,
    raw_target,
    volatility,
    dates,
    sequence_length
):

    sequences = []
    normalized_targets = []
    raw_targets = []
    target_volatilities = []
    target_dates = []

    for i in range(
        sequence_length,
        len(X)
    ):

        # -------------------------------------------------
        # Input:
        #
        # [t-sequence_length, ..., t-1]
        #
        # Target:
        #
        # return at t
        #
        # No information from t is passed into X.
        # -------------------------------------------------

        sequence = X[
            i - sequence_length:i
        ]

        sequences.append(
            sequence
        )

        normalized_targets.append(
            normalized_target.iloc[i]
        )

        raw_targets.append(
            raw_target.iloc[i]
        )

        target_volatilities.append(
            volatility.iloc[i]
        )

        target_dates.append(
            dates.iloc[i]
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),

        np.asarray(
            normalized_targets,
            dtype=np.float32
        ),

        np.asarray(
            raw_targets,
            dtype=np.float32
        ),

        np.asarray(
            target_volatilities,
            dtype=np.float32
        ),

        pd.Series(
            target_dates
        ).reset_index(drop=True)
    )


# =========================================================
# VALIDATION / TEST CONTEXT SEQUENCES
# =========================================================

def create_context_sequences(
    X_context,
    normalized_target,
    raw_target,
    volatility,
    dates_target,
    sequence_length
):

    sequences = []
    normalized_targets = []
    raw_targets = []
    target_volatilities = []
    target_dates = []

    target_length = len(
        normalized_target
    )

    current_start = (
        len(X_context)
        - target_length
    )

    for j in range(target_length):

        current_index = (
            current_start + j
        )

        sequence_start = (
            current_index
            - sequence_length
        )

        if sequence_start < 0:

            raise ValueError(
                "Not enough historical context "
                "for sequence."
            )

        sequence = X_context[
            sequence_start:current_index
        ]

        sequences.append(
            sequence
        )

        normalized_targets.append(
            normalized_target.iloc[j]
        )

        raw_targets.append(
            raw_target.iloc[j]
        )

        target_volatilities.append(
            volatility.iloc[j]
        )

        target_dates.append(
            dates_target.iloc[j]
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),

        np.asarray(
            normalized_targets,
            dtype=np.float32
        ),

        np.asarray(
            raw_targets,
            dtype=np.float32
        ),

        np.asarray(
            target_volatilities,
            dtype=np.float32
        ),

        pd.Series(
            target_dates
        ).reset_index(drop=True)
    )


# =========================================================
# PYTORCH DATASET
# =========================================================

class FinancialDataset(Dataset):

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

class ReturnLSTM(nn.Module):

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

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

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

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

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

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

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

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# =========================================================
# PREDICT
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

            X_batch = X_batch.to(device)

            outputs = model(
                X_batch
            )

            predictions.extend(
                outputs.cpu().numpy()
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
    print("LSTM TRAINING")
    print("=" * 70)

    criterion = nn.HuberLoss(
        delta=1.0
    )

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

    best_val_loss = float("inf")

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

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if (
        np.std(actual) == 0
        or np.std(predicted) == 0
    ):

        return np.nan

    return np.corrcoef(
        actual,
        predicted
    )[0, 1]


# =========================================================
# EVALUATE RAW RETURN
# =========================================================

def evaluate_raw_return(
    actual_raw,
    predicted_normalized,
    volatility,
    dates,
    dataset_name
):

    actual_raw = np.asarray(
        actual_raw,
        dtype=np.float64
    )

    predicted_normalized = np.asarray(
        predicted_normalized,
        dtype=np.float64
    )

    volatility = np.asarray(
        volatility,
        dtype=np.float64
    )

    predicted_raw = (
        predicted_normalized
        * volatility
    )

    mae = mean_absolute_error(
        actual_raw,
        predicted_raw
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_raw,
            predicted_raw
        )
    )

    r2 = r2_score(
        actual_raw,
        predicted_raw
    )

    correlation = safe_correlation(
        actual_raw,
        predicted_raw
    )

    actual_direction = (
        actual_raw > 0
    )

    predicted_direction = (
        predicted_raw > 0
    )

    directional_accuracy = (
        actual_direction
        == predicted_direction
    ).mean()

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} RAW RETURN RESULTS"
    )
    print("=" * 70)

    print(
        f"Period: "
        f"{dates.iloc[0].date()} → "
        f"{dates.iloc[-1].date()}"
    )

    print(
        f"\nMAE                   : "
        f"{mae:.8f}"
    )

    print(
        f"RMSE                  : "
        f"{rmse:.8f}"
    )

    print(
        f"R²                    : "
        f"{r2:.8f}"
    )

    print(
        f"Prediction Correlation: "
        f"{correlation}"
    )

    print(
        f"Directional Accuracy  : "
        f"{directional_accuracy:.4f}"
    )

    print(
        f"Prediction Std        : "
        f"{np.std(predicted_raw):.8f}"
    )

    return {

        "predicted_normalized":
            predicted_normalized,

        "predicted_raw":
            predicted_raw,

        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "correlation":
            correlation,

        "directional_accuracy":
            directional_accuracy
    }


# =========================================================
# NORMALIZED TARGET EVALUATION
# =========================================================

def evaluate_normalized_target(
    actual_normalized,
    predicted_normalized,
    dates,
    dataset_name
):

    actual = np.asarray(
        actual_normalized,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted_normalized,
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
        f"{dataset_name.upper()} NORMALIZED-TARGET RESULTS"
    )
    print("=" * 70)

    print(
        f"Period: "
        f"{dates.iloc[0].date()} → "
        f"{dates.iloc[-1].date()}"
    )

    print(
        f"\nNormalized MAE        : "
        f"{mae:.8f}"
    )

    print(
        f"Normalized RMSE       : "
        f"{rmse:.8f}"
    )

    print(
        f"Normalized R²         : "
        f"{r2:.8f}"
    )

    print(
        f"Normalized Correlation: "
        f"{correlation}"
    )

    print(
        f"Prediction Std        : "
        f"{np.std(predicted):.8f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": correlation
    }


# =========================================================
# NAIVE RAW-RETURN BASELINE
# =========================================================

def evaluate_naive_baseline(
    y_train_raw,
    y_test_raw
):

    training_mean = float(
        y_train_raw.mean()
    )

    predictions = np.full(
        len(y_test_raw),
        training_mean,
        dtype=np.float64
    )

    actual = np.asarray(
        y_test_raw,
        dtype=np.float64
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    r2 = r2_score(
        actual,
        predictions
    )

    correlation = safe_correlation(
        actual,
        predictions
    )

    print("\n" + "=" * 70)
    print("NAIVE TEST BASELINE")
    print("=" * 70)

    print(
        f"Training mean return: "
        f"{training_mean:.8f}"
    )

    print(
        f"MAE        : "
        f"{mae:.8f}"
    )

    print(
        f"RMSE       : "
        f"{rmse:.8f}"
    )

    print(
        f"R²         : "
        f"{r2:.8f}"
    )

    print(
        f"Correlation: "
        f"{correlation}"
    )

    return {
        "predictions":
            predictions,

        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "correlation":
            correlation
    }


# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

def inspect_predictions(
    predicted_normalized,
    predicted_raw,
    actual_raw
):

    print("\n" + "=" * 70)
    print("TEST PREDICTION DISTRIBUTION")
    print("=" * 70)

    normalized_series = pd.Series(
        predicted_normalized
    )

    raw_series = pd.Series(
        predicted_raw
    )

    actual_series = pd.Series(
        actual_raw
    )

    print(
        "\nNormalized predictions:"
    )

    print(
        normalized_series.describe()
    )

    print(
        "\nRaw-return predictions:"
    )

    print(
        raw_series.describe()
    )

    print(
        "\nActual returns:"
    )

    print(
        actual_series.describe()
    )

    prediction_std = np.std(
        predicted_raw
    )

    actual_std = np.std(
        actual_raw
    )

    print(
        f"\nRaw prediction standard deviation: "
        f"{prediction_std:.10f}"
    )

    print(
        f"Actual return standard deviation: "
        f"{actual_std:.10f}"
    )

    print(
        f"Prediction/actual std ratio: "
        f"{prediction_std / actual_std:.4f}"
        if actual_std > 0
        else "Prediction/actual std ratio: NaN"
    )

    print(
        f"Predicted positive rate: "
        f"{(predicted_raw > 0).mean():.2%}"
    )

    print(
        f"Actual positive rate: "
        f"{(actual_raw > 0).mean():.2%}"
    )

    if actual_std > 0:

        std_ratio = (
            prediction_std
            / actual_std
        )

        if std_ratio < 0.05:

            print(
                "\nWARNING: Model predictions "
                "are strongly collapsed toward the mean."
            )

        elif std_ratio < 0.15:

            print(
                "\nWARNING: Model prediction variance "
                "is substantially smaller than actual returns."
            )


# =========================================================
# SAVE PREDICTIONS
# =========================================================

def save_predictions(
    dates,
    actual_normalized,
    predicted_normalized,
    actual_raw,
    predicted_raw,
    volatility
):

    output = pd.DataFrame({

        "Date":
            dates.values,

        "actual_normalized_return":
            actual_normalized,

        "predicted_normalized_return":
            predicted_normalized,

        "current_volatility":
            volatility,

        "actual_return":
            actual_raw,

        "predicted_return":
            predicted_raw,
    })

    output["prediction_error"] = (
        output["predicted_return"]
        - output["actual_return"]
    )

    output["absolute_error"] = (
        output["prediction_error"]
        .abs()
    )

    output["actual_direction"] = (
        output["actual_return"] > 0
    ).astype(int)

    output["predicted_direction"] = (
        output["predicted_return"] > 0
    ).astype(int)

    path = os.path.join(
        MODEL_DIR,
        "lstm_test_predictions.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"\nPredictions saved to:\n"
        f"{path}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model,
    input_size
):

    path = os.path.join(
        MODEL_DIR,
        "lstm_return_model.pt"
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

        "volatility_window":
            VOLATILITY_WINDOW,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "loss":
            "HuberLoss",

        "optimizer":
            "AdamW",
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
        "lstm_scaler.npz"
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

def save_history(history):

    path = os.path.join(
        MODEL_DIR,
        "lstm_training_history.csv"
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
# MAIN
# =========================================================

def main():

    set_seed()

    device = get_device()

    # 1. LOAD
    df = load_dataset()

    # 2. VALIDATE
    validate_dataset(df)

    # 3. FEATURES
    feature_columns = get_feature_columns(df)

    # 4. PREPARE
    (
        features,
        normalized_target,
        raw_target,
        volatility,
        dates
    ) = prepare_dataframe(
        df,
        feature_columns
    )

    # 5. TIME SPLIT
    (
        X_train,
        y_train_normalized,
        y_train_raw,
        vol_train,
        dates_train,

        X_val,
        y_val_normalized,
        y_val_raw,
        vol_val,
        dates_val,

        X_test,
        y_test_normalized,
        y_test_raw,
        vol_test,
        dates_test

    ) = split_data(
        features,
        normalized_target,
        raw_target,
        volatility,
        dates
    )

    # 6. SCALER
    scaler = fit_scaler(X_train)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled
    ) = transform_features(
        scaler,
        X_train,
        X_val,
        X_test
    )

    # 7. TRAIN SEQUENCES
    (
        train_sequences,
        train_normalized_targets,
        train_raw_targets,
        train_volatilities,
        train_dates

    ) = create_training_sequences(
        X_train_scaled,
        y_train_normalized,
        y_train_raw,
        vol_train,
        dates_train,
        SEQUENCE_LENGTH
    )

    # 8. VALIDATION CONTEXT
    train_context = X_train_scaled[-SEQUENCE_LENGTH:]

    X_val_context = np.concatenate(
        [train_context, X_val_scaled],
        axis=0
    )

    (
        val_sequences,
        val_normalized_targets,
        val_raw_targets,
        val_volatilities,
        val_dates

    ) = create_context_sequences(
        X_val_context,
        y_val_normalized,
        y_val_raw,
        vol_val,
        dates_val,
        SEQUENCE_LENGTH
    )

    # 9. TEST CONTEXT
    validation_context = X_val_scaled[-SEQUENCE_LENGTH:]

    X_test_context = np.concatenate(
        [validation_context, X_test_scaled],
        axis=0
    )

    (
        test_sequences,
        test_normalized_targets,
        test_raw_targets,
        test_volatilities,
        test_dates

    ) = create_context_sequences(
        X_test_context,
        y_test_normalized,
        y_test_raw,
        vol_test,
        dates_test,
        SEQUENCE_LENGTH
    )

    # 10. SEQUENCE INFORMATION
    print("\n" + "=" * 70)
    print("SEQUENCES")
    print("=" * 70)
    print(f"Sequence length : {SEQUENCE_LENGTH}")
    print(f"Train shape     : {train_sequences.shape}")
    print(f"Validation shape: {val_sequences.shape}")
    print(f"Test shape      : {test_sequences.shape}")

    # 11. DATASETS
    train_dataset = FinancialDataset(
        train_sequences,
        train_normalized_targets
    )

    val_dataset = FinancialDataset(
        val_sequences,
        val_normalized_targets
    )

    test_dataset = FinancialDataset(
        test_sequences,
        test_normalized_targets
    )

    # 12. DATALOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
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

    # 13. MODEL
    input_size = train_sequences.shape[2]

    model = ReturnLSTM(
        input_size=input_size
    ).to(device)

    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    print(model)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTrainable parameters: {trainable_parameters:,}")

    # 14. TRAIN
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device
    )

    # 15. NORMALIZED TRAIN EVALUATION
    train_normalized_predictions = predict(
        model,
        train_eval_loader,
        device
    )

    evaluate_normalized_target(
        train_normalized_targets,
        train_normalized_predictions,
        train_dates,
        "Train"
    )

    # 16. VALIDATION
    val_normalized_predictions = predict(
        model,
        val_loader,
        device
    )

    evaluate_normalized_target(
        val_normalized_targets,
        val_normalized_predictions,
        val_dates,
        "Validation"
    )

    # 17. TEST
    test_normalized_predictions = predict(
        model,
        test_loader,
        device
    )

    # 18. RAW RETURN TEST EVALUATION
    test_results = evaluate_raw_return(
        test_raw_targets,
        test_normalized_predictions,
        test_volatilities,
        test_dates,
        "Test"
    )

    predicted_raw_test = test_results["predicted_raw"]

    # 19. DISTRIBUTION
    inspect_predictions(
        test_normalized_predictions,
        predicted_raw_test,
        test_raw_targets
    )

    # 20. NAIVE BASELINE
    naive_results = evaluate_naive_baseline(
        y_train_raw,
        pd.Series(test_raw_targets)
    )

    # 21. LSTM VS NAIVE
    print("\n" + "=" * 70)
    print("LSTM VS NAIVE BASELINE")
    print("=" * 70)

    print(f"LSTM Test RMSE  : {test_results['rmse']:.8f}")
    print(f"Naive Test RMSE : {naive_results['rmse']:.8f}")

    if naive_results["rmse"] != 0:
        rmse_improvement = 1 - (test_results["rmse"] / naive_results["rmse"])
        print(f"\nRMSE improvement vs naive: {rmse_improvement:.2%}")

    correlation = test_results["correlation"]
    direction = test_results["directional_accuracy"]

    if np.isfinite(correlation) and correlation > 0:
        print("✓ Prediction correlation is positive.")
    else:
        print("✗ Prediction correlation is not positive.")

    if direction > 0.50:
        print("✓ Directional accuracy is above 50%.")
    else:
        print("✗ Directional accuracy is not above 50%.")

    # 22. SAVE HISTORY
    save_history(history)

    # 23. SAVE PREDICTIONS
    save_predictions(
        test_dates,
        test_normalized_targets,
        test_normalized_predictions,
        test_raw_targets,
        predicted_raw_test,
        test_volatilities
    )

    # 24. SAVE MODEL
    save_model(model, input_size)

    # 25. SAVE SCALER
    save_scaler(scaler, feature_columns)

    # 26. FINAL SUMMARY
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    rmse_imp = 1 - (test_results['rmse'] / naive_results['rmse'])

    print(f"Test RMSE           : {test_results['rmse']:.8f}")
    print(f"Naive RMSE          : {naive_results['rmse']:.8f}")
    print(f"RMSE improvement    : {rmse_imp:.2%}")
    print(f"Test R²             : {test_results['r2']:.8f}")
    print(f"Test Correlation    : {test_results['correlation']}")
    print(f"Test Direction      : {test_results['directional_accuracy']:.4f}")
    print(f"Prediction Std      : {np.std(predicted_raw_test):.8f}")


if __name__ == "__main__":
    main()