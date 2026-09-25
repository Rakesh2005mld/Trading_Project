import os
import random

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/processed/cross_sectional_dataset.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_xgboost.json"
)

PREDICTIONS_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_xgboost_predictions.csv"
)

VALIDATION_PREDICTIONS_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_xgboost_validation_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_xgboost_feature_importance.csv"
)

RANDOM_STATE = 42


# =========================================================
# DATA SPLIT
# =========================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# PORTFOLIO
# =========================================================

TOP_K = 2
BOTTOM_K = 2


# =========================================================
# XGBOOST CONFIGURATION
# =========================================================

N_ESTIMATORS = 500
MAX_DEPTH = 4
LEARNING_RATE = 0.03
SUBSAMPLE = 0.80
COLSAMPLE_BYTREE = 0.80
MIN_CHILD_WEIGHT = 5
REG_ALPHA = 0.10
REG_LAMBDA = 1.00


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

    return df


# =========================================================
# VALIDATE DATASET
# =========================================================

def validate_dataset(
    df
):

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
        f"\nDate range:"
    )

    print(
        f"{df['Date'].min()} "
        f"→ "
        f"{df['Date'].max()}"
    )

    print(
        f"\nUnique tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        sorted(
            df["Ticker"].unique()
        )
    )

    # -----------------------------------------------------
    # Duplicate check
    # -----------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "Date",
            "Ticker"
        ]
    ).sum()

    print(
        f"\nDuplicate ticker/date rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate Date/Ticker rows found."
        )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        print(
            "\nMissing values:"
        )

        print(
            missing
        )

    else:

        print(
            "\nMissing values: 0"
        )

    # -----------------------------------------------------
    # Infinite values
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
            "Infinite values found."
        )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "Date",
        "Ticker",
        "future_return",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: "
                f"{column}"
            )

    print(
        "\nDataset validation: OK"
    )


# =========================================================
# CREATE CROSS-SECTIONAL TARGET
# =========================================================

def create_target(
    df
):

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL TARGET")
    print("=" * 70)

    df = df.copy()

    # -----------------------------------------------------
    # Daily mean future return.
    # -----------------------------------------------------

    daily_mean = (
        df
        .groupby("Date")[
            "future_return"
        ]
        .transform("mean")
    )

    # -----------------------------------------------------
    # Daily standard deviation.
    # -----------------------------------------------------

    daily_std = (
        df
        .groupby("Date")[
            "future_return"
        ]
        .transform("std")
    )

    # -----------------------------------------------------
    # Relative return.
    # -----------------------------------------------------

    df[
        "relative_future_return"
    ] = (
        df["future_return"]
        - daily_mean
    )

    # -----------------------------------------------------
    # Daily standardized target.
    # -----------------------------------------------------

    df[
        "cross_sectional_target"
    ] = (

        df[
            "relative_future_return"
        ]

        / daily_std.replace(
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

    df = (
        df
        .sort_values(
            [
                "Date",
                "Ticker"
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Dates: "
        f"{df['Date'].nunique()}"
    )

    print(
        "\nTarget statistics:"
    )

    print(
        df[
            "cross_sectional_target"
        ].describe()
    )

    return df


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

    suspicious = []

    for column in df.columns:

        if column in forbidden:

            continue

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in leakage_keywords
        ):

            suspicious.append(
                column
            )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    print(
        f"Unexpected suspicious features: "
        f"{suspicious}"
    )

    if suspicious:

        raise ValueError(
            f"Potential leakage columns: "
            f"{suspicious}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden
    ]

    # -----------------------------------------------------
    # Ensure numeric
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
            f"Non-numeric features found: "
            f"{non_numeric}"
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
    print("DATE SPLIT")
    print("=" * 70)

    unique_dates = np.array(
        sorted(
            df["Date"].unique()
        )
    )

    n_dates = len(
        unique_dates
    )

    train_end_index = int(
        n_dates
        * TRAIN_RATIO
    )

    validation_end_index = int(
        n_dates
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    if (
        train_end_index <= 0
        or validation_end_index <= train_end_index
        or validation_end_index >= n_dates
    ):

        raise ValueError(
            "Invalid date split."
        )

    train_start = (
        unique_dates[0]
    )

    train_end = (
        unique_dates[
            train_end_index - 1
        ]
    )

    val_start = (
        unique_dates[
            train_end_index
        ]
    )

    val_end = (
        unique_dates[
            validation_end_index - 1
        ]
    )

    test_start = (
        unique_dates[
            validation_end_index
        ]
    )

    test_end = (
        unique_dates[-1]
    )

    print(
        f"\nTotal dates: "
        f"{n_dates}"
    )

    print(
        "\nTrain:"
    )

    print(
        f"{pd.Timestamp(train_start).date()} "
        f"→ "
        f"{pd.Timestamp(train_end).date()}"
    )

    print(
        "\nValidation:"
    )

    print(
        f"{pd.Timestamp(val_start).date()} "
        f"→ "
        f"{pd.Timestamp(val_end).date()}"
    )

    print(
        "\nTest:"
    )

    print(
        f"{pd.Timestamp(test_start).date()} "
        f"→ "
        f"{pd.Timestamp(test_end).date()}"
    )

    return {
        "train_start": train_start,
        "train_end": train_end,

        "val_start": val_start,
        "val_end": val_end,

        "test_start": test_start,
        "test_end": test_end,
    }


# =========================================================
# PREPARE MATRICES
# =========================================================

def prepare_matrices(
    df,
    feature_columns,
    split
):

    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    working = df.copy()

    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    for column in feature_columns:

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Remove invalid features
    # -----------------------------------------------------

    working = working[
        working[
            feature_columns
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
            ["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Masks
    # -----------------------------------------------------

    train_mask = (

        working["Date"]
        >= split["train_start"]

    ) & (

        working["Date"]
        <= split["train_end"]

    )

    val_mask = (

        working["Date"]
        >= split["val_start"]

    ) & (

        working["Date"]
        <= split["val_end"]

    )

    test_mask = (

        working["Date"]
        >= split["test_start"]

    ) & (

        working["Date"]
        <= split["test_end"]

    )

    # -----------------------------------------------------
    # Matrices
    # -----------------------------------------------------

    X_train = working.loc[
        train_mask,
        feature_columns
    ]

    y_train = working.loc[
        train_mask,
        "cross_sectional_target"
    ]

    X_val = working.loc[
        val_mask,
        feature_columns
    ]

    y_val = working.loc[
        val_mask,
        "cross_sectional_target"
    ]

    X_test = working.loc[
        test_mask,
        feature_columns
    ]

    y_test = working.loc[
        test_mask,
        "cross_sectional_target"
    ]

    print(
        f"\nTrain rows: "
        f"{len(X_train)}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val)}"
    )

    print(
        f"Test rows: "
        f"{len(X_test)}"
    )

    return (
        working,

        train_mask,
        val_mask,
        test_mask,

        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test
    )


# =========================================================
# MODEL
# =========================================================

def create_model():

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        min_child_weight=MIN_CHILD_WEIGHT,

        reg_alpha=REG_ALPHA,

        reg_lambda=REG_LAMBDA,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        eval_metric="rmse",

    )

    return model


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 70)
    print("XGBOOST TRAINING")
    print("=" * 70)

    print(
        "\nTraining model..."
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],

        verbose=False,
    )

    print(
        "Training completed."
    )

    return model


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
# REGRESSION EVALUATION
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
        f"REGRESSION RESULTS"
    )
    print("=" * 70)

    print(
        f"\nMAE         : "
        f"{mae:.8f}"
    )

    print(
        f"RMSE        : "
        f"{rmse:.8f}"
    )

    print(
        f"R²          : "
        f"{r2:.8f}"
    )

    print(
        f"Correlation : "
        f"{correlation}"
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
# SPEARMAN IC
# =========================================================

def calculate_daily_ic(
    dates,
    predictions,
    actual_targets
):

    evaluation = pd.DataFrame({

        "Date":
            pd.Series(
                dates
            ).reset_index(
                drop=True
            ),

        "prediction":
            predictions,

        "actual":
            actual_targets,

    })

    daily_ic = []

    for date, group in evaluation.groupby(
        "Date"
    ):

        if len(group) < 2:

            continue

        prediction_ranks = (
            group[
                "prediction"
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        actual_ranks = (
            group[
                "actual"
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        ic = safe_correlation(
            prediction_ranks,
            actual_ranks
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
        float(
            np.mean(
                daily_ic
            )
        ),

        float(
            np.median(
                daily_ic
            )
        ),

        daily_ic
    )


# =========================================================
# PORTFOLIO EVALUATION
# =========================================================

def evaluate_portfolio(
    dates,
    tickers,
    predictions,
    actual_returns,
    actual_targets,
    dataset_name
):

    data = pd.DataFrame({

        "Date":
            dates,

        "Ticker":
            tickers,

        "prediction":
            predictions,

        "actual_target":
            actual_targets,

        "actual_return":
            actual_returns,

    })

    # -----------------------------------------------------
    # Daily IC
    # -----------------------------------------------------

    ic_mean, ic_median, daily_ic = (
        calculate_daily_ic(

            data["Date"],

            data["prediction"],

            data["actual_target"]

        )
    )

    long_returns = []

    short_returns = []

    long_short_returns = []

    market_returns = []

    selected_rows = []

    # -----------------------------------------------------
    # Daily ranking
    # -----------------------------------------------------

    for date, group in data.groupby(
        "Date"
    ):

        if len(group) < (
            TOP_K
            + BOTTOM_K
        ):

            continue

        ranked = group.sort_values(
            "prediction",
            ascending=False
        )

        long_group = ranked.head(
            TOP_K
        )

        short_group = ranked.tail(
            BOTTOM_K
        )

        long_return = (
            long_group[
                "actual_return"
            ]
            .mean()
        )

        short_return = (
            short_group[
                "actual_return"
            ]
            .mean()
        )

        long_short = (
            long_return
            - short_return
        )

        market_return = (
            group[
                "actual_return"
            ]
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

        long_copy = (
            long_group.copy()
        )

        long_copy[
            "portfolio_side"
        ] = "LONG"

        short_copy = (
            short_group.copy()
        )

        short_copy[
            "portfolio_side"
        ] = "SHORT"

        selected_rows.append(
            pd.concat(
                [
                    long_copy,
                    short_copy
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
    # Mean returns
    # -----------------------------------------------------

    mean_long = (
        np.mean(long_returns)
        if len(long_returns)
        else np.nan
    )

    mean_short = (
        np.mean(short_returns)
        if len(short_returns)
        else np.nan
    )

    mean_long_short = (
        np.mean(long_short_returns)
        if len(long_short_returns)
        else np.nan
    )

    mean_market = (
        np.mean(market_returns)
        if len(market_returns)
        else np.nan
    )

    # -----------------------------------------------------
    # Hit rate
    # -----------------------------------------------------

    hit_rate = (

        np.mean(
            long_short_returns > 0
        )

        if len(long_short_returns)

        else np.nan
    )

    # -----------------------------------------------------
    # Sharpe
    # -----------------------------------------------------

    if (
        len(long_short_returns) > 1
        and np.std(
            long_short_returns
        ) > 0
    ):

        daily_sharpe = (
            mean_long_short
            / np.std(
                long_short_returns
            )
        )

        annualized_sharpe = (
            daily_sharpe
            * np.sqrt(252)
        )

    else:

        annualized_sharpe = np.nan

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} "
        f"CROSS-SECTIONAL EVALUATION"
    )
    print("=" * 70)

    print(
        f"\nEvaluation dates: "
        f"{len(long_short_returns)}"
    )

    print(
        f"Mean Spearman IC: "
        f"{ic_mean}"
    )

    print(
        f"Median Spearman IC: "
        f"{ic_median}"
    )

    print(
        f"\nTop-{TOP_K} mean return: "
        f"{mean_long}"
    )

    print(
        f"Bottom-{BOTTOM_K} mean return: "
        f"{mean_short}"
    )

    print(
        f"Long-short mean return: "
        f"{mean_long_short}"
    )

    print(
        f"Market mean return: "
        f"{mean_market}"
    )

    print(
        f"Long-short hit rate: "
        f"{hit_rate}"
    )

    print(
        f"Annualized long-short Sharpe: "
        f"{annualized_sharpe}"
    )

    if len(selected_rows):

        selected = pd.concat(
            selected_rows,
            ignore_index=True
        )

    else:

        selected = pd.DataFrame()

    return {

        "ic_mean":
            ic_mean,

        "ic_median":
            ic_median,

        "top_return":
            mean_long,

        "bottom_return":
            mean_short,

        "long_short_return":
            mean_long_short,

        "market_return":
            mean_market,

        "hit_rate":
            hit_rate,

        "annualized_sharpe":
            annualized_sharpe,

    }, data, selected


# =========================================================
# SAVE PREDICTIONS
# =========================================================

def save_predictions(
    data,
    output_path
):

    output = data.copy()

    # -----------------------------------------------------
    # Prediction rank within each day.
    # -----------------------------------------------------

    output[
        "prediction_rank"
    ] = (

        output
        .groupby("Date")[
            "prediction"
        ]
        .rank(
            ascending=False,
            method="first"
        )

    )

    # -----------------------------------------------------
    # Actual rank within each day.
    # -----------------------------------------------------

    output[
        "actual_rank"
    ] = (

        output
        .groupby("Date")[
            "actual_target"
        ]
        .rank(
            ascending=False,
            method="first"
        )

    )

    # -----------------------------------------------------
    # Prediction percentile.
    # -----------------------------------------------------

    output[
        "prediction_percentile"
    ] = (

        output
        .groupby("Date")[
            "prediction"
        ]
        .rank(
            pct=True
        )

    )

    # -----------------------------------------------------
    # Rename columns to explicit names.
    # -----------------------------------------------------

    output = output.rename(

        columns={

            "prediction":
                "predicted_cross_sectional_target",

            "actual_target":
                "actual_cross_sectional_target",

        }

    )

    output.to_csv(

        output_path,

        index=False

    )

    print(
        f"\nPredictions saved to:"
        f"\n{output_path}"
    )

    return output


# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

def save_feature_importance(
    model,
    feature_columns
):

    importance = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            model.feature_importances_,

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance.to_csv(

        FEATURE_IMPORTANCE_PATH,

        index=False

    )

    print(
        f"\nFeature importance saved to:"
        f"\n{FEATURE_IMPORTANCE_PATH}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed()

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
    # 3. TARGET
    # =====================================================

    df = create_target(
        df
    )

    # =====================================================
    # 4. FEATURES
    # =====================================================

    feature_columns = get_features(
        df
    )

    # =====================================================
    # 5. SPLIT
    # =====================================================

    split = create_date_split(
        df
    )

    # =====================================================
    # 6. PREPARE
    # =====================================================

    (
        working,

        train_mask,
        val_mask,
        test_mask,

        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test

    ) = prepare_matrices(

        df,

        feature_columns,

        split
    )

    # =====================================================
    # 7. MODEL
    # =====================================================

    model = create_model()

    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print(
        f"\nEstimators       : "
        f"{N_ESTIMATORS}"
    )

    print(
        f"Max depth        : "
        f"{MAX_DEPTH}"
    )

    print(
        f"Learning rate    : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Subsample        : "
        f"{SUBSAMPLE}"
    )

    print(
        f"Column sampling  : "
        f"{COLSAMPLE_BYTREE}"
    )

    # =====================================================
    # 8. TRAIN
    # =====================================================

    model = train_model(

        model,

        X_train,

        y_train,

        X_val,

        y_val
    )

    # =====================================================
    # 9. PREDICT
    # =====================================================

    train_predictions = (
        model.predict(
            X_train
        )
    )

    val_predictions = (
        model.predict(
            X_val
        )
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )

    # =====================================================
    # 10. REGRESSION METRICS
    # =====================================================

    train_metrics = evaluate_regression(

        y_train,

        train_predictions,

        "Train"
    )

    val_metrics = evaluate_regression(

        y_val,

        val_predictions,

        "Validation"
    )

    test_metrics = evaluate_regression(

        y_test,

        test_predictions,

        "Test"
    )

    # =====================================================
    # 11. TEST DATAFRAME
    # =====================================================

    test_rows = working.loc[
        test_mask
    ].copy()

    test_prediction_data = pd.DataFrame({

        "Date":
            test_rows[
                "Date"
            ].values,

        "Ticker":
            test_rows[
                "Ticker"
            ].values,

        "prediction":
            test_predictions,

        "actual_target":
            test_rows[
                "cross_sectional_target"
            ].values,

        "actual_return":
            test_rows[
                "future_return"
            ].values,

    })

    # =====================================================
    # 12. TEST PORTFOLIO
    # =====================================================

    (
        portfolio_metrics,

        portfolio_data,

        selected

    ) = evaluate_portfolio(

        test_prediction_data[
            "Date"
        ].reset_index(drop=True),

        test_prediction_data[
            "Ticker"
        ].reset_index(drop=True),

        test_prediction_data[
            "prediction"
        ].to_numpy(),

        test_prediction_data[
            "actual_return"
        ].to_numpy(),

        test_prediction_data[
            "actual_target"
        ].to_numpy(),

        "Test"
    )

    # =====================================================
    # 13. VALIDATION DATAFRAME
    # =====================================================

    val_rows = working.loc[
        val_mask
    ].copy()

    val_prediction_data = pd.DataFrame({

        "Date":
            val_rows[
                "Date"
            ].values,

        "Ticker":
            val_rows[
                "Ticker"
            ].values,

        "prediction":
            val_predictions,

        "actual_target":
            val_rows[
                "cross_sectional_target"
            ].values,

        "actual_return":
            val_rows[
                "future_return"
            ].values,

    })

    # =====================================================
    # 14. VALIDATION PORTFOLIO
    # =====================================================

    (
        val_portfolio_metrics,

        val_portfolio_data,

        val_selected

    ) = evaluate_portfolio(

        val_prediction_data[
            "Date"
        ].reset_index(drop=True),

        val_prediction_data[
            "Ticker"
        ].reset_index(drop=True),

        val_prediction_data[
            "prediction"
        ].to_numpy(),

        val_prediction_data[
            "actual_return"
        ].to_numpy(),

        val_prediction_data[
            "actual_target"
        ].to_numpy(),

        "Validation"
    )

    # =====================================================
    # 15. SAVE VALIDATION PREDICTIONS
    # =====================================================

    saved_validation_predictions = save_predictions(

        val_prediction_data,

        VALIDATION_PREDICTIONS_PATH

    )

    saved_validation_predictions[
        "model"
    ] = "cross_sectional_xgboost"

    saved_validation_predictions.to_csv(

        VALIDATION_PREDICTIONS_PATH,

        index=False

    )

    print(
        f"\nValidation predictions saved to:"
        f"\n{VALIDATION_PREDICTIONS_PATH}"
    )

    # =====================================================
    # 16. SAVE TEST PREDICTIONS
    # =====================================================

    saved_predictions = save_predictions(

        test_prediction_data,

        PREDICTIONS_PATH

    )

    saved_predictions[
        "model"
    ] = "cross_sectional_xgboost"

    saved_predictions.to_csv(

        PREDICTIONS_PATH,

        index=False

    )

    print(
        f"\nTest predictions saved to:"
        f"\n{PREDICTIONS_PATH}"
    )

    # =====================================================
    # 17. SAVE FEATURE IMPORTANCE
    # =====================================================

    save_feature_importance(

        model,

        feature_columns
    )

    # =====================================================
    # 18. SAVE MODEL
    # =====================================================

    model.save_model(
        MODEL_PATH
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )

    # =====================================================
    # 19. FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "\nVALIDATION"
    )

    print(
        f"RMSE: "
        f"{val_metrics['rmse']:.8f}"
    )

    print(
        f"R²: "
        f"{val_metrics['r2']:.8f}"
    )

    print(
        f"Correlation: "
        f"{val_metrics['correlation']}"
    )

    print(
        f"IC: "
        f"{val_portfolio_metrics['ic_mean']}"
    )

    print(
        f"Top-{TOP_K} return: "
        f"{val_portfolio_metrics['top_return']}"
    )

    print(
        f"Bottom-{BOTTOM_K} return: "
        f"{val_portfolio_metrics['bottom_return']}"
    )

    print(
        f"Long-short return: "
        f"{val_portfolio_metrics['long_short_return']}"
    )

    print(
        f"Hit rate: "
        f"{val_portfolio_metrics['hit_rate']}"
    )

    print(
        f"Annualized Sharpe: "
        f"{val_portfolio_metrics['annualized_sharpe']}"
    )

    print(
        "\nTEST"
    )

    print(
        f"RMSE: "
        f"{test_metrics['rmse']:.8f}"
    )

    print(
        f"R²: "
        f"{test_metrics['r2']:.8f}"
    )

    print(
        f"Correlation: "
        f"{test_metrics['correlation']}"
    )

    print(
        f"IC: "
        f"{portfolio_metrics['ic_mean']}"
    )

    print(
        f"Top-{TOP_K} return: "
        f"{portfolio_metrics['top_return']}"
    )

    print(
        f"Bottom-{BOTTOM_K} return: "
        f"{portfolio_metrics['bottom_return']}"
    )

    print(
        f"Long-short return: "
        f"{portfolio_metrics['long_short_return']}"
    )

    print(
        f"Hit rate: "
        f"{portfolio_metrics['hit_rate']}"
    )

    print(
        f"Annualized Sharpe: "
        f"{portfolio_metrics['annualized_sharpe']}"
    )

    print(
        "\nFiles:"
    )

    print(
        "models/cross_sectional_xgboost.json"
    )

    print(
        "models/cross_sectional_xgboost_validation_predictions.csv"
    )

    print(
        "models/cross_sectional_xgboost_predictions.csv"
    )

    print(
        "models/cross_sectional_xgboost_feature_importance.csv"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()