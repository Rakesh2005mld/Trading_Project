import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = (
    "data/processed/cross_sectional_dataset.csv"
)

OUTPUT_DIR = "models"

PREDICTIONS_PATH = (
    f"{OUTPUT_DIR}/walk_forward_predictions.csv"
)

FOLD_RESULTS_PATH = (
    f"{OUTPUT_DIR}/walk_forward_fold_results.csv"
)

SUMMARY_PATH = (
    f"{OUTPUT_DIR}/walk_forward_summary.csv"
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

RANDOM_STATE = 42


# =========================================================
# SEQUENCE
# =========================================================

SEQUENCE_LENGTH = 30


# =========================================================
# WALK-FORWARD WINDOWS
# =========================================================

TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3


# =========================================================
# TRAINING
# =========================================================

BATCH_SIZE = 64

EPOCHS = 40

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 6

GRADIENT_CLIP = 1.0


# =========================================================
# MODEL
# =========================================================

HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.30


# =========================================================
# TARGET
# =========================================================

TARGET_COLUMN = (
    "future_return"
)


# =========================================================
# PORTFOLIO
# =========================================================

TOP_K = 2
BOTTOM_K = 2


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# SEED
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

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"\nUsing device: {device}"
    )

    return device


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    if not os.path.exists(
        DATA_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

    required = [
        "Date",
        "Ticker",
        "future_return",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: "
                f"{column}"
            )

    df = (
        df
        .sort_values(
            [
                "Ticker",
                "Date"
            ]
        )
        .drop_duplicates(
            subset=[
                "Ticker",
                "Date"
            ]
        )
        .reset_index(drop=True)
    )

    return df


# =========================================================
# TARGET
# =========================================================

def create_target(
    df
):

    df = df.copy()

    daily_mean = (
        df
        .groupby("Date")[
            TARGET_COLUMN
        ]
        .transform("mean")
    )

    daily_std = (
        df
        .groupby("Date")[
            TARGET_COLUMN
        ]
        .transform("std")
    )

    df[
        "cross_sectional_target"
    ] = (

        df[
            TARGET_COLUMN
        ]
        - daily_mean

    ) / daily_std.replace(
        0,
        np.nan
    )

    df = df[
        np.isfinite(
            df[
                "cross_sectional_target"
            ]
        )
    ].copy()

    return (
        df
        .sort_values(
            [
                "Ticker",
                "Date"
            ]
        )
        .reset_index(drop=True)
    )


# =========================================================
# FEATURES
# =========================================================

def get_features(
    df
):

    forbidden = {

        "Date",

        "Ticker",

        "Sector",

        "future_return",

        "relative_future_return",

        "cross_sectional_target",

        "cross_section_count",

        "target",

        "trading_target",

    }

    leakage_keywords = [
        "future",
        "forward",
        "next_day",
        "nextday",
        "target",
    ]

    features = []

    for column in df.columns:

        if column in forbidden:

            continue

        lower = column.lower()

        if any(
            keyword in lower
            for keyword in leakage_keywords
        ):

            continue

        features.append(
            column
        )

    non_numeric = [

        column

        for column in features

        if not pd.api.types.is_numeric_dtype(
            df[column]
        )

    ]

    if non_numeric:

        raise ValueError(
            f"Non-numeric features:\n"
            f"{non_numeric}"
        )

    return features


# =========================================================
# DATASET
# =========================================================

class SequenceDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y
    ):

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
# MODEL
# =========================================================

class CrossSectionalLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=HIDDEN_SIZE,

            num_layers=NUM_LAYERS,

            batch_first=True,

            dropout=(
                DROPOUT
                if NUM_LAYERS > 1
                else 0.0
            )

        )

        self.fc1 = nn.Linear(
            HIDDEN_SIZE,
            16
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.fc2 = nn.Linear(
            16,
            1
        )

    def forward(
        self,
        x
    ):

        output, _ = self.lstm(
            x
        )

        last = output[
            :,
            -1,
            :
        ]

        x = self.fc1(
            last
        )

        x = self.relu(
            x
        )

        x = self.dropout(
            x
        )

        x = self.fc2(
            x
        )

        return x.squeeze(1)


# =========================================================
# CREATE WALK-FORWARD FOLDS
# =========================================================

def create_folds(
    min_date,
    max_date
):

    folds = []

    train_start = (
        pd.Timestamp(min_date)
    )

    while True:

        train_end = (
            train_start
            + pd.DateOffset(
                months=TRAIN_MONTHS
            )
            - pd.Timedelta(days=1)
        )

        val_start = (
            train_end
            + pd.Timedelta(days=1)
        )

        val_end = (
            val_start
            + pd.DateOffset(
                months=VALIDATION_MONTHS
            )
            - pd.Timedelta(days=1)
        )

        test_start = (
            val_end
            + pd.Timedelta(days=1)
        )

        test_end = (
            test_start
            + pd.DateOffset(
                months=TEST_MONTHS
            )
            - pd.Timedelta(days=1)
        )

        if test_end > max_date:

            break

        folds.append({

            "fold":
                len(folds) + 1,

            "train_start":
                train_start,

            "train_end":
                train_end,

            "val_start":
                val_start,

            "val_end":
                val_end,

            "test_start":
                test_start,

            "test_end":
                test_end,

        })

        train_start = (
            train_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


# =========================================================
# PREPARE FOLD DATA
# =========================================================

def prepare_fold(
    df,
    features,
    fold
):

    working = df.copy()

    # -----------------------------------------------------
    # Numeric conversion.
    # -----------------------------------------------------

    for column in features:

        working[column] = (
            pd.to_numeric(
                working[column],
                errors="coerce"
            )
        )

    working = (
        working
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )

    working = working[
        working[
            features
        ]
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
            [
                "Ticker",
                "Date"
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # TRAIN scaler ONLY.
    # -----------------------------------------------------

    train_mask = (

        working["Date"]
        >= fold["train_start"]

    ) & (

        working["Date"]
        <= fold["train_end"]

    )

    scaler = StandardScaler()

    scaler.fit(
        working.loc[
            train_mask,
            features
        ]
    )

    working[
        features
    ] = scaler.transform(
        working[
            features
        ]
    )

    return working


# =========================================================
# CREATE SEQUENCES FOR A FOLD
# =========================================================

def create_fold_sequences(
    df,
    features,
    fold
):

    train_X = []
    train_y = []

    val_X = []
    val_y = []

    test_X = []
    test_y = []

    test_returns = []
    test_dates = []
    test_tickers = []

    val_dates = []
    val_tickers = []
    val_returns = []

    # -----------------------------------------------------
    # Each ticker is handled independently.
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
                features
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

        returns = (
            group[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        dates = (
            group[
                "Date"
            ]
            .to_numpy()
        )

        for i in range(
            SEQUENCE_LENGTH - 1,
            len(group)
        ):

            target_date = (
                pd.Timestamp(
                    dates[i]
                )
            )

            sequence = X[
                i - SEQUENCE_LENGTH + 1:
                i + 1
            ]

            target = y[i]

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            if (
                fold["train_start"]
                <= target_date
                <= fold["train_end"]
            ):

                train_X.append(
                    sequence
                )

                train_y.append(
                    target
                )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            elif (
                fold["val_start"]
                <= target_date
                <= fold["val_end"]
            ):

                val_X.append(
                    sequence
                )

                val_y.append(
                    target
                )

                val_dates.append(
                    target_date
                )

                val_tickers.append(
                    ticker
                )

                val_returns.append(
                    returns[i]
                )

            # -------------------------------------------------
            # Test
            # -------------------------------------------------

            elif (
                fold["test_start"]
                <= target_date
                <= fold["test_end"]
            ):

                test_X.append(
                    sequence
                )

                test_y.append(
                    target
                )

                test_returns.append(
                    returns[i]
                )

                test_dates.append(
                    target_date
                )

                test_tickers.append(
                    ticker
                )

    return {

        "train_X":
            np.asarray(
                train_X,
                dtype=np.float32
            ),

        "train_y":
            np.asarray(
                train_y,
                dtype=np.float32
            ),

        "val_X":
            np.asarray(
                val_X,
                dtype=np.float32
            ),

        "val_y":
            np.asarray(
                val_y,
                dtype=np.float32
            ),

        "test_X":
            np.asarray(
                test_X,
                dtype=np.float32
            ),

        "test_y":
            np.asarray(
                test_y,
                dtype=np.float32
            ),

        "test_returns":
            np.asarray(
                test_returns,
                dtype=np.float32
            ),

        "test_dates":
            pd.Series(
                test_dates
            ),

        "test_tickers":
            pd.Series(
                test_tickers
            ),

        "val_dates":
            pd.Series(
                val_dates
            ),

        "val_tickers":
            pd.Series(
                val_tickers
            ),

        "val_returns":
            np.asarray(
                val_returns,
                dtype=np.float32
            ),

    }


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

        X_batch = (
            X_batch.to(device)
        )

        y_batch = (
            y_batch.to(device)
        )

        optimizer.zero_grad()

        prediction = model(
            X_batch
        )

        loss = criterion(
            prediction,
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

def validation_loss(
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

            X_batch = (
                X_batch.to(device)
            )

            y_batch = (
                y_batch.to(device)
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

            X_batch = (
                X_batch.to(device)
            )

            output = model(
                X_batch
            )

            predictions.extend(
                output
                .cpu()
                .numpy()
            )

    return np.asarray(
        predictions,
        dtype=np.float64
    )


# =========================================================
# TRAIN FOLD
# =========================================================

def train_fold(
    sequences,
    input_size,
    device,
    fold_number
):

    print("\n" + "=" * 70)
    print(
        f"TRAINING FOLD {fold_number}"
    )
    print("=" * 70)

    train_dataset = SequenceDataset(

        sequences["train_X"],

        sequences["train_y"]

    )

    val_dataset = SequenceDataset(

        sequences["val_X"],

        sequences["val_y"]

    )

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

    model = CrossSectionalLSTM(
        input_size
    ).to(device)

    criterion = nn.HuberLoss(
        delta=1.0
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    best_val_loss = float(
        "inf"
    )

    best_state = None

    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = (
            train_one_epoch(

                model,

                train_loader,

                optimizer,

                criterion,

                device

            )
        )

        val_loss = (
            validation_loss(

                model,

                val_loader,

                criterion,

                device

            )
        )

        print(

            f"Fold {fold_number:02d} | "

            f"Epoch {epoch:02d} | "

            f"Train={train_loss:.6f} | "

            f"Val={val_loss:.6f}"

        )

        if val_loss < (
            best_val_loss - 1e-6
        ):

            best_val_loss = val_loss

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"Fold {fold_number}: "
                f"Early stopping."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No best model state."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_val_loss
    )


# =========================================================
# DAILY SPEARMAN IC
# =========================================================

def calculate_daily_ic(
    dates,
    predictions,
    actual_targets
):

    evaluation = pd.DataFrame({

        "Date":
            pd.to_datetime(
                dates
            ),

        "prediction":
            predictions,

        "actual":
            actual_targets,

    })

    daily_ic = []

    for _, group in evaluation.groupby(
        "Date"
    ):

        if len(group) < 2:

            continue

        prediction_rank = (
            group[
                "prediction"
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        actual_rank = (
            group[
                "actual"
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        if (
            np.std(
                prediction_rank
            ) == 0
            or
            np.std(
                actual_rank
            ) == 0
        ):

            continue

        ic = np.corrcoef(

            prediction_rank,

            actual_rank

        )[0, 1]

        if np.isfinite(ic):

            daily_ic.append(
                ic
            )

    if len(daily_ic) == 0:

        return np.nan

    return float(
        np.mean(
            daily_ic
        )
    )


# =========================================================
# PORTFOLIO EVALUATION
# =========================================================

def evaluate_portfolio(
    dates,
    predictions,
    actual_returns
):

    data = pd.DataFrame({

        "Date":
            pd.to_datetime(
                dates
            ),

        "prediction":
            predictions,

        "actual_return":
            actual_returns,

    })

    long_returns = []
    short_returns = []
    long_short_returns = []

    for _, group in data.groupby(
        "Date"
    ):

        if len(group) < (
            TOP_K + BOTTOM_K
        ):

            continue

        ranked = (
            group
            .sort_values(
                "prediction",
                ascending=False
            )
        )

        longs = ranked.head(
            TOP_K
        )

        shorts = ranked.tail(
            BOTTOM_K
        )

        long_return = (
            longs[
                "actual_return"
            ]
            .mean()
        )

        short_return = (
            shorts[
                "actual_return"
            ]
            .mean()
        )

        long_short = (
            long_return
            - short_return
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

    if len(long_short_returns) == 0:

        return {

            "top_return":
                np.nan,

            "bottom_return":
                np.nan,

            "long_short_return":
                np.nan,

            "hit_rate":
                np.nan,

            "sharpe":
                np.nan,

        }

    mean_long = (
        np.mean(long_returns)
    )

    mean_short = (
        np.mean(short_returns)
    )

    mean_long_short = (
        np.mean(
            long_short_returns
        )
    )

    hit_rate = np.mean(
        long_short_returns > 0
    )

    if np.std(
        long_short_returns,
        ddof=1
    ) > 1e-12:

        sharpe = (

            mean_long_short
            / np.std(
                long_short_returns,
                ddof=1
            )

        ) * np.sqrt(252)

    else:

        sharpe = np.nan

    return {

        "top_return":
            mean_long,

        "bottom_return":
            mean_short,

        "long_short_return":
            mean_long_short,

        "hit_rate":
            hit_rate,

        "sharpe":
            sharpe,

    }


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed()

    device = get_device()

    # =====================================================
    # 1. DATA
    # =====================================================

    df = load_data()

    print("\n" + "=" * 70)
    print("WALK-FORWARD DATA")
    print("=" * 70)

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    print(
        f"Dates: "
        f"{df['Date'].nunique()}"
    )

    print(
        f"Tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        f"Range:"
        f"\n{df['Date'].min().date()} "
        f"→ "
        f"{df['Date'].max().date()}"
    )

    # =====================================================
    # 2. TARGET
    # =====================================================

    df = create_target(
        df
    )

    # =====================================================
    # 3. FEATURES
    # =====================================================

    features = get_features(
        df
    )

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    # =====================================================
    # 4. FOLDS
    # =====================================================

    folds = create_folds(

        df["Date"].min(),

        df["Date"].max()

    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD FOLDS")
    print("=" * 70)

    print(
        f"\nNumber of folds: "
        f"{len(folds)}"
    )

    for fold in folds:

        print(

            f"\nFold {fold['fold']}:"

            f"\nTrain      : "
            f"{fold['train_start'].date()} "
            f"→ "
            f"{fold['train_end'].date()}"

            f"\nValidation : "
            f"{fold['val_start'].date()} "
            f"→ "
            f"{fold['val_end'].date()}"

            f"\nTest       : "
            f"{fold['test_start'].date()} "
            f"→ "
            f"{fold['test_end'].date()}"

        )

    # =====================================================
    # 5. RUN FOLDS
    # =====================================================

    all_predictions = []

    fold_results = []

    for fold in folds:

        fold_number = fold["fold"]

        print("\n" + "=" * 70)

        print(
            f"PREPARING FOLD {fold_number}"
        )

        print("=" * 70)

        fold_df = prepare_fold(

            df,

            features,

            fold

        )

        sequences = create_fold_sequences(

            fold_df,

            features,

            fold

        )

        print(
            f"\nTrain sequences: "
            f"{len(sequences['train_X']):,}"
        )

        print(
            f"Validation sequences: "
            f"{len(sequences['val_X']):,}"
        )

        print(
            f"Test sequences: "
            f"{len(sequences['test_X']):,}"
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        model, best_val_loss = train_fold(

            sequences,

            len(features),

            device,

            fold_number

        )

        # -------------------------------------------------
        # Test predictions
        # -------------------------------------------------

        test_dataset = SequenceDataset(

            sequences["test_X"],

            sequences["test_y"]

        )

        test_loader = DataLoader(

            test_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False

        )

        test_predictions = predict(

            model,

            test_loader,

            device

        )

        # -------------------------------------------------
        # Fold IC
        # -------------------------------------------------

        fold_ic = calculate_daily_ic(

            sequences[
                "test_dates"
            ],

            test_predictions,

            sequences[
                "test_y"
            ]

        )

        # -------------------------------------------------
        # Fold portfolio
        # -------------------------------------------------

        fold_portfolio = evaluate_portfolio(

            sequences[
                "test_dates"
            ],

            test_predictions,

            sequences[
                "test_returns"
            ]

        )

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold_number} RESULTS"
        )
        print("=" * 70)

        print(
            f"\nBest validation loss: "
            f"{best_val_loss:.6f}"
        )

        print(
            f"Test IC: "
            f"{fold_ic:.5f}"
        )

        print(
            f"Top-{TOP_K} return: "
            f"{fold_portfolio['top_return']:.5%}"
        )

        print(
            f"Bottom-{BOTTOM_K} return: "
            f"{fold_portfolio['bottom_return']:.5%}"
        )

        print(
            f"Long-short return: "
            f"{fold_portfolio['long_short_return']:.5%}"
        )

        print(
            f"Hit rate: "
            f"{fold_portfolio['hit_rate']:.2%}"
        )

        print(
            f"Sharpe: "
            f"{fold_portfolio['sharpe']:.3f}"
        )

        # -------------------------------------------------
        # Save fold predictions.
        # -------------------------------------------------

        fold_prediction_df = pd.DataFrame({

            "Date":
                sequences[
                    "test_dates"
                ].values,

            "Ticker":
                sequences[
                    "test_tickers"
                ].values,

            "actual_target":
                sequences[
                    "test_y"
                ],

            "predicted_target":
                test_predictions,

            "actual_return":
                sequences[
                    "test_returns"
                ],

            "fold":
                fold_number,

        })

        fold_prediction_df[
            "prediction_rank"
        ] = (

            fold_prediction_df
            .groupby("Date")[
                "predicted_target"
            ]
            .rank(
                ascending=False,
                method="first"
            )

        )

        all_predictions.append(
            fold_prediction_df
        )

        fold_results.append({

            "fold":
                fold_number,

            "train_start":
                fold[
                    "train_start"
                ],

            "train_end":
                fold[
                    "train_end"
                ],

            "validation_start":
                fold[
                    "val_start"
                ],

            "validation_end":
                fold[
                    "val_end"
                ],

            "test_start":
                fold[
                    "test_start"
                ],

            "test_end":
                fold[
                    "test_end"
                ],

            "best_validation_loss":
                best_val_loss,

            "test_ic":
                fold_ic,

            "top_return":
                fold_portfolio[
                    "top_return"
                ],

            "bottom_return":
                fold_portfolio[
                    "bottom_return"
                ],

            "long_short_return":
                fold_portfolio[
                    "long_short_return"
                ],

            "hit_rate":
                fold_portfolio[
                    "hit_rate"
                ],

            "sharpe":
                fold_portfolio[
                    "sharpe"
                ],

        })

    # =====================================================
    # 6. COMBINE OOS PREDICTIONS
    # =====================================================

    predictions = pd.concat(

        all_predictions,

        ignore_index=True

    )

    predictions = (

        predictions
        .sort_values(
            [
                "Date",
                "Ticker"
            ]
        )
        .reset_index(drop=True)

    )

    # -----------------------------------------------------
    # Check duplicate OOS observations.
    # -----------------------------------------------------

    duplicates = (
        predictions
        .duplicated(
            subset=[
                "Date",
                "Ticker"
            ]
        )
        .sum()
    )

    if duplicates > 0:

        raise RuntimeError(
            f"Duplicate walk-forward "
            f"predictions found: "
            f"{duplicates}"
        )

    # =====================================================
    # 7. OVERALL OOS METRICS
    # =====================================================

    overall_ic = calculate_daily_ic(

        predictions["Date"],

        predictions[
            "predicted_target"
        ].to_numpy(),

        predictions[
            "actual_target"
        ].to_numpy()

    )

    overall_portfolio = (
        evaluate_portfolio(

            predictions["Date"],

            predictions[
                "predicted_target"
            ].to_numpy(),

            predictions[
                "actual_return"
            ].to_numpy()

        )
    )

    # =====================================================
    # 8. FOLD RESULTS
    # =====================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    # =====================================================
    # 9. SUMMARY STATISTICS
    # =====================================================

    mean_fold_ic = (
        fold_results_df[
            "test_ic"
        ]
        .mean()
    )

    median_fold_ic = (
        fold_results_df[
            "test_ic"
        ]
        .median()
    )

    positive_ic_folds = (
        fold_results_df[
            "test_ic"
        ]
        > 0
    ).mean()

    mean_fold_sharpe = (
        fold_results_df[
            "sharpe"
        ]
        .mean()
    )

    positive_sharpe_folds = (
        fold_results_df[
            "sharpe"
        ]
        > 0
    ).mean()

    # =====================================================
    # 10. SAVE
    # =====================================================

    predictions.to_csv(

        PREDICTIONS_PATH,

        index=False

    )

    fold_results_df.to_csv(

        FOLD_RESULTS_PATH,

        index=False

    )

    summary = pd.DataFrame({

        "total_oos_rows": [
            len(predictions)
        ],

        "oos_start": [
            predictions[
                "Date"
            ].min()
        ],

        "oos_end": [
            predictions[
                "Date"
            ].max()
        ],

        "num_folds": [
            len(folds)
        ],

        "overall_ic": [
            overall_ic
        ],

        "mean_fold_ic": [
            mean_fold_ic
        ],

        "median_fold_ic": [
            median_fold_ic
        ],

        "positive_ic_fold_fraction": [
            positive_ic_folds
        ],

        "top_k_return": [
            overall_portfolio[
                "top_return"
            ]
        ],

        "bottom_k_return": [
            overall_portfolio[
                "bottom_return"
            ]
        ],

        "long_short_return": [
            overall_portfolio[
                "long_short_return"
            ]
        ],

        "hit_rate": [
            overall_portfolio[
                "hit_rate"
            ]
        ],

        "overall_sharpe": [
            overall_portfolio[
                "sharpe"
            ]
        ],

        "mean_fold_sharpe": [
            mean_fold_sharpe
        ],

        "positive_sharpe_fold_fraction": [
            positive_sharpe_folds
        ],

    })

    summary.to_csv(

        SUMMARY_PATH,

        index=False

    )

    # =====================================================
    # 11. FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print(
        "WALK-FORWARD FINAL RESULTS"
    )
    print("=" * 70)

    print(
        f"\nOOS rows: "
        f"{len(predictions):,}"
    )

    print(
        f"OOS range:"
        f"\n{predictions['Date'].min().date()} "
        f"→ "
        f"{predictions['Date'].max().date()}"
    )

    print(
        f"\nOverall OOS IC: "
        f"{overall_ic:.5f}"
    )

    print(
        f"Mean fold IC: "
        f"{mean_fold_ic:.5f}"
    )

    print(
        f"Median fold IC: "
        f"{median_fold_ic:.5f}"
    )

    print(
        f"Positive IC folds: "
        f"{positive_ic_folds:.2%}"
    )

    print(
        f"\nTop-{TOP_K} return: "
        f"{overall_portfolio['top_return']:.5%}"
    )

    print(
        f"Bottom-{BOTTOM_K} return: "
        f"{overall_portfolio['bottom_return']:.5%}"
    )

    print(
        f"Long-short return: "
        f"{overall_portfolio['long_short_return']:.5%}"
    )

    print(
        f"Hit rate: "
        f"{overall_portfolio['hit_rate']:.2%}"
    )

    print(
        f"Overall Sharpe: "
        f"{overall_portfolio['sharpe']:.3f}"
    )

    print(
        f"Mean fold Sharpe: "
        f"{mean_fold_sharpe:.3f}"
    )

    print(
        f"Positive Sharpe folds: "
        f"{positive_sharpe_folds:.2%}"
    )

    print("\n" + "=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"\n{PREDICTIONS_PATH}"
    )

    print(
        f"{FOLD_RESULTS_PATH}"
    )

    print(
        f"{SUMMARY_PATH}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()