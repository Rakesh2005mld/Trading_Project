import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = (
    "data/processed/cross_sectional_dataset.csv"
)

MODEL_DIR = "models"

PREDICTIONS_PATH = (
    f"{MODEL_DIR}/rank_lstm_predictions.csv"
)

FOLD_RESULTS_PATH = (
    f"{MODEL_DIR}/rank_lstm_fold_results.csv"
)

SUMMARY_PATH = (
    f"{MODEL_DIR}/rank_lstm_summary.csv"
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
# RANKING LOSS
# =========================================================

# Temperature controls the softness of the pairwise
# logistic ranking loss.

RANKING_TEMPERATURE = 1.0


# =========================================================
# PORTFOLIO
# =========================================================

TOP_K = 2
BOTTOM_K = 2


# =========================================================
# DEVICE
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
# CROSS-SECTIONAL TARGET
# =========================================================

def create_target(
    df
):

    df = df.copy()

    daily_mean = (
        df
        .groupby("Date")[
            "future_return"
        ]
        .transform("mean")
    )

    daily_std = (
        df
        .groupby("Date")[
            "future_return"
        ]
        .transform("std")
    )

    df[
        "cross_sectional_target"
    ] = (

        (
            df[
                "future_return"
            ]

            - daily_mean
        )

        /

        daily_std.replace(
            0,
            np.nan
        )

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

    print("\n" + "=" * 70)
    print("FEATURES")
    print("=" * 70)

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    print(
        features
    )

    return features


# =========================================================
# CREATE WALK-FORWARD FOLDS
# =========================================================

def create_folds(
    min_date,
    max_date
):

    folds = []

    train_start = pd.Timestamp(
        min_date
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
# PREPARE FOLD
# =========================================================

def prepare_fold(
    df,
    features,
    fold
):

    working = df.copy()

    for column in features:

        working[column] = (
            pd.to_numeric(
                working[column],
                errors="coerce"
            )
        )

    working[
        features
    ] = (
        working[
            features
        ]
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
            "future_return"
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
# BUILD DATE-LEVEL SEQUENCES
# =========================================================

def create_date_groups(
    df,
    features,
    fold
):

    groups = {

        "train": [],

        "validation": [],

        "test": [],

    }

    # -----------------------------------------------------
    # Build ticker histories first.
    # -----------------------------------------------------

    ticker_histories = {}

    for ticker, group in df.groupby(
        "Ticker",
        sort=True
    ):

        ticker_histories[
            ticker
        ] = (
            group
            .sort_values("Date")
            .reset_index(drop=True)
        )

    # -----------------------------------------------------
    # For every target date, collect one sequence per
    # available ticker.
    # -----------------------------------------------------

    all_dates = sorted(
        df["Date"].unique()
    )

    for date_value in all_dates:

        target_date = pd.Timestamp(
            date_value
        )

        # Determine destination.

        if (
            fold["train_start"]
            <= target_date
            <= fold["train_end"]
        ):

            destination = "train"

        elif (
            fold["val_start"]
            <= target_date
            <= fold["val_end"]
        ):

            destination = "validation"

        elif (
            fold["test_start"]
            <= target_date
            <= fold["test_end"]
        ):

            destination = "test"

        else:

            continue

        date_sequences = []
        date_targets = []
        date_returns = []
        date_tickers = []

        for ticker, history in (
            ticker_histories.items()
        ):

            # -------------------------------------------------
            # Locate target date.
            # -------------------------------------------------

            matching = history.index[
                history["Date"]
                == target_date
            ]

            if len(matching) == 0:

                continue

            target_index = int(
                matching[0]
            )

            # -------------------------------------------------
            # Need 30 days of history for the sequence.
            # -------------------------------------------------

            if (
                target_index
                < SEQUENCE_LENGTH - 1
            ):

                continue

            start_index = (
                target_index
                - SEQUENCE_LENGTH
                + 1
            )

            sequence = (
                history
                .iloc[
                    start_index:
                    target_index + 1
                ][
                    features
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            target = float(
                history.iloc[
                    target_index
                ][
                    "cross_sectional_target"
                ]
            )

            future_return = float(
                history.iloc[
                    target_index
                ][
                    "future_return"
                ]
            )

            date_sequences.append(
                sequence
            )

            date_targets.append(
                target
            )

            date_returns.append(
                future_return
            )

            date_tickers.append(
                ticker
            )

        # -----------------------------------------------------
        # A ranking group needs enough stocks.
        # -----------------------------------------------------

        if len(date_sequences) < (
            TOP_K + BOTTOM_K
        ):

            continue

        groups[destination].append({

            "date":
                target_date,

            "X":
                np.asarray(
                    date_sequences,
                    dtype=np.float32
                ),

            "y":
                np.asarray(
                    date_targets,
                    dtype=np.float32
                ),

            "returns":
                np.asarray(
                    date_returns,
                    dtype=np.float32
                ),

            "tickers":
                date_tickers,

        })

    return groups


# =========================================================
# LSTM MODEL
# =========================================================

class RankingLSTM(
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

        score = self.fc2(
            x
        )

        return score.squeeze(-1)


# =========================================================
# PAIRWISE RANKING LOSS
# =========================================================

def pairwise_ranking_loss(
    scores,
    targets
):

    n = scores.shape[0]

    if n < 2:

        return None

    # -----------------------------------------------------
    # Pairwise score differences.
    #
    # score_diff[i,j] =
    #     score_i - score_j
    #
    # target_diff[i,j] =
    #     target_i - target_j
    # -----------------------------------------------------

    score_diff = (
        scores.unsqueeze(1)
        - scores.unsqueeze(0)
    )

    target_diff = (
        targets.unsqueeze(1)
        - targets.unsqueeze(0)
    )

    # -----------------------------------------------------
    # Keep each pair only once.
    #
    # Upper triangle:
    # i < j
    # -----------------------------------------------------

    upper = torch.triu(
        torch.ones(
            (
                n,
                n
            ),
            dtype=torch.bool,
            device=scores.device
        ),
        diagonal=1
    )

    score_diff = (
        score_diff[
            upper
        ]
    )

    target_diff = (
        target_diff[
            upper
        ]
    )

    # -----------------------------------------------------
    # Ignore ties.
    # -----------------------------------------------------

    non_ties = (
        target_diff.abs()
        > 1e-12
    )

    if not torch.any(
        non_ties
    ):

        return None

    score_diff = (
        score_diff[
            non_ties
        ]
    )

    target_diff = (
        target_diff[
            non_ties
        ]
    )

    # -----------------------------------------------------
    # Sign:
    #
    # +1 means i should rank above j.
    #
    # -1 means j should rank above i.
    # -----------------------------------------------------

    target_sign = torch.sign(
        target_diff
    )

    # -----------------------------------------------------
    # RankNet-style logistic loss:
    #
    # softplus(
    #   - sign(target_diff)
    #   * score_diff
    #   / temperature
    # )
    # -----------------------------------------------------

    loss = torch.nn.functional.softplus(

        -(

            target_sign

            *

            score_diff

            /

            RANKING_TEMPERATURE

        )

    )

    return loss.mean()


# =========================================================
# TRAIN
# =========================================================

def train_model(
    model,
    train_groups,
    validation_groups,
    device
):

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

    print("\n" + "=" * 70)
    print("RANKING LSTM TRAINING")
    print("=" * 70)

    print(
        f"\nTraining dates: "
        f"{len(train_groups)}"
    )

    print(
        f"Validation dates: "
        f"{len(validation_groups)}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_losses = []

        # -------------------------------------------------
        # Shuffle dates.
        #
        # We deliberately shuffle the DATE groups, not
        # individual stock sequences. Ranking pairs must
        # remain within the same cross-section.
        # -------------------------------------------------

        shuffled_groups = train_groups.copy()

        random.shuffle(
            shuffled_groups
        )

        for group in shuffled_groups:

            X = torch.tensor(

                group["X"],

                dtype=torch.float32,

                device=device

            )

            y = torch.tensor(

                group["y"],

                dtype=torch.float32,

                device=device

            )

            optimizer.zero_grad()

            scores = model(
                X
            )

            loss = (
                pairwise_ranking_loss(
                    scores,
                    y
                )
            )

            if loss is None:

                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                GRADIENT_CLIP

            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        # -------------------------------------------------
        # Validation.
        # -------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for group in validation_groups:

                X = torch.tensor(

                    group["X"],

                    dtype=torch.float32,

                    device=device

                )

                y = torch.tensor(

                    group["y"],

                    dtype=torch.float32,

                    device=device

                )

                scores = model(
                    X
                )

                loss = (
                    pairwise_ranking_loss(
                        scores,
                        y
                    )
                )

                if loss is not None:

                    val_losses.append(
                        loss.item()
                    )

        if len(train_losses) == 0:

            raise RuntimeError(
                "No valid training ranking groups."
            )

        if len(val_losses) == 0:

            raise RuntimeError(
                "No valid validation ranking groups."
            )

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        val_loss = float(
            np.mean(
                val_losses
            )
        )

        print(

            f"Epoch {epoch:02d} | "

            f"Train Rank Loss: "
            f"{train_loss:.6f} | "

            f"Val Rank Loss: "
            f"{val_loss:.6f}"

        )

        if val_loss < (
            best_val_loss - 1e-6
        ):

            best_val_loss = (
                val_loss
            )

            best_state = {

                name:
                    parameter
                    .detach()
                    .cpu()
                    .clone()

                for name, parameter
                in model.state_dict().items()

            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "Early stopping."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "Best ranking model state "
            "was not saved."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_val_loss
    )


# =========================================================
# PREDICT DATE GROUP
# =========================================================

def predict_group(
    model,
    group,
    device
):

    X = torch.tensor(

        group["X"],

        dtype=torch.float32,

        device=device

    )

    model.eval()

    with torch.no_grad():

        scores = model(
            X
        )

    return (
        scores
        .cpu()
        .numpy()
        .astype(np.float64)
    )


# =========================================================
# DAILY SPEARMAN IC
# =========================================================

def calculate_overall_ic(
    prediction_df
):

    daily_ic = []

    for _, group in prediction_df.groupby(
        "Date"
    ):

        if len(group) < 2:

            continue

        pred_rank = (
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
                "actual_target"
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        if (
            np.std(pred_rank) == 0
            or np.std(actual_rank) == 0
        ):

            continue

        ic = np.corrcoef(

            pred_rank,

            actual_rank

        )[0, 1]

        if np.isfinite(ic):

            daily_ic.append(
                ic
            )

    if len(daily_ic) == 0:

        return np.nan, np.nan

    return (

        float(
            np.mean(
                daily_ic
            )
        ),

        float(
            np.median(
                daily_ic
            )
        )

    )


# =========================================================
# PORTFOLIO METRICS
# =========================================================

def calculate_portfolio_metrics(
    prediction_df
):

    long_returns = []
    short_returns = []
    long_short_returns = []

    for _, group in prediction_df.groupby(
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

        ls_return = (
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
            ls_return
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

    mean_long = np.mean(
        long_returns
    )

    mean_short = np.mean(
        short_returns
    )

    mean_ls = np.mean(
        long_short_returns
    )

    hit_rate = np.mean(
        long_short_returns > 0
    )

    if (
        len(long_short_returns) > 1

        and

        np.std(
            long_short_returns,
            ddof=1
        ) > 1e-12
    ):

        sharpe = (

            mean_ls

            /

            np.std(
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
            mean_ls,

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
    # 1. LOAD
    # =====================================================

    df = load_data()

    print("\n" + "=" * 70)
    print("DATA")
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

        print(
            f"\nTrain:"
            f"\n{fold['train_start'].date()} "
            f"→ "
            f"{fold['train_end'].date()}"
        )

        print(
            f"\nValidation:"
            f"\n{fold['val_start'].date()} "
            f"→ "
            f"{fold['val_end'].date()}"
        )

        print(
            f"\nTest:"
            f"\n{fold['test_start'].date()} "
            f"→ "
            f"{fold['test_end'].date()}"
        )

        # -------------------------------------------------
        # Prepare fold
        # -------------------------------------------------

        fold_df = prepare_fold(

            df,

            features,

            fold

        )

        # -------------------------------------------------
        # Create date-level ranking groups.
        # -------------------------------------------------

        groups = create_date_groups(

            fold_df,

            features,

            fold

        )

        print(
            f"\nTrain ranking dates: "
            f"{len(groups['train'])}"
        )

        print(
            f"Validation ranking dates: "
            f"{len(groups['validation'])}"
        )

        print(
            f"Test ranking dates: "
            f"{len(groups['test'])}"
        )

        if (
            len(groups["train"]) == 0
            or len(groups["validation"]) == 0
            or len(groups["test"]) == 0
        ):

            raise RuntimeError(
                f"Insufficient ranking groups "
                f"in fold {fold_number}."
            )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = RankingLSTM(
            input_size=len(features)
        ).to(device)

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        (
            model,
            best_val_loss
        ) = train_model(

            model,

            groups["train"],

            groups["validation"],

            device

        )

        # -------------------------------------------------
        # Predict test groups.
        # -------------------------------------------------

        fold_predictions = []

        for group in groups["test"]:

            predictions = predict_group(

                model,

                group,

                device

            )

            for index in range(
                len(predictions)
            ):

                fold_predictions.append({

                    "Date":
                        group[
                            "date"
                        ],

                    "Ticker":
                        group[
                            "tickers"
                        ][index],

                    "prediction":
                        predictions[index],

                    "actual_target":
                        group[
                            "y"
                        ][index],

                    "actual_return":
                        group[
                            "returns"
                        ][index],

                    "fold":
                        fold_number,

                })

        fold_prediction_df = pd.DataFrame(
            fold_predictions
        )

        # -------------------------------------------------
        # Fold IC
        # -------------------------------------------------

        fold_ic_mean, fold_ic_median = (
            calculate_overall_ic(
                fold_prediction_df
            )
        )

        # -------------------------------------------------
        # Fold portfolio metrics
        # -------------------------------------------------

        portfolio = (
            calculate_portfolio_metrics(
                fold_prediction_df
            )
        )

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold_number} RESULTS"
        )
        print("=" * 70)

        print(
            f"\nBest validation ranking loss: "
            f"{best_val_loss:.6f}"
        )

        print(
            f"Test IC: "
            f"{fold_ic_mean:.5f}"
        )

        print(
            f"Median IC: "
            f"{fold_ic_median:.5f}"
        )

        print(
            f"Top-{TOP_K} return: "
            f"{portfolio['top_return']:.5%}"
        )

        print(
            f"Bottom-{BOTTOM_K} return: "
            f"{portfolio['bottom_return']:.5%}"
        )

        print(
            f"Long-short return: "
            f"{portfolio['long_short_return']:.5%}"
        )

        print(
            f"Hit rate: "
            f"{portfolio['hit_rate']:.2%}"
        )

        print(
            f"Sharpe: "
            f"{portfolio['sharpe']:.3f}"
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

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

            "best_validation_rank_loss":
                best_val_loss,

            "test_ic":
                fold_ic_mean,

            "median_ic":
                fold_ic_median,

            "top_return":
                portfolio[
                    "top_return"
                ],

            "bottom_return":
                portfolio[
                    "bottom_return"
                ],

            "long_short_return":
                portfolio[
                    "long_short_return"
                ],

            "hit_rate":
                portfolio[
                    "hit_rate"
                ],

            "sharpe":
                portfolio[
                    "sharpe"
                ],

        })

    # =====================================================
    # 6. COMBINE WALK-FORWARD PREDICTIONS
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
    # Duplicate check.
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
            f"predictions: {duplicates}"
        )

    # =====================================================
    # 7. OVERALL OOS METRICS
    # =====================================================

    overall_ic, overall_median_ic = (
        calculate_overall_ic(
            predictions
        )
    )

    overall_portfolio = (
        calculate_portfolio_metrics(
            predictions
        )
    )

    fold_results_df = pd.DataFrame(
        fold_results
    )

    # =====================================================
    # 8. ROBUSTNESS STATISTICS
    # =====================================================

    mean_fold_ic = (
        fold_results_df[
            "test_ic"
        ].mean()
    )

    median_fold_ic = (
        fold_results_df[
            "test_ic"
        ].median()
    )

    positive_ic_fraction = (
        fold_results_df[
            "test_ic"
        ]
        > 0
    ).mean()

    mean_fold_sharpe = (
        fold_results_df[
            "sharpe"
        ].mean()
    )

    positive_sharpe_fraction = (
        fold_results_df[
            "sharpe"
        ]
        > 0
    ).mean()

    # =====================================================
    # 9. SAVE
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

        "overall_median_ic": [
            overall_median_ic
        ],

        "mean_fold_ic": [
            mean_fold_ic
        ],

        "median_fold_ic": [
            median_fold_ic
        ],

        "positive_ic_fold_fraction": [
            positive_ic_fraction
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
            positive_sharpe_fraction
        ],

    })

    summary.to_csv(

        SUMMARY_PATH,

        index=False

    )

    # =====================================================
    # 10. FINAL RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print(
        "RANKING LSTM "
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
        f"Overall median IC: "
        f"{overall_median_ic:.5f}"
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
        f"{positive_ic_fraction:.2%}"
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
        f"{positive_sharpe_fraction:.2%}"
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