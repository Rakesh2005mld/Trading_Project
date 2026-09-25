import os
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

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

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# Winsorization limits
# These are applied using TRAINING quantiles only.
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99


# =========================================================
# SETUP
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


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
# VALIDATION
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

    print("\nDate range:")

    print(
        f"{df['Date'].min()} "
        f"→ "
        f"{df['Date'].max()}"
    )

    required_columns = [
        "Date",
        "future_return"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    if not df["Date"].is_monotonic_increasing:

        raise ValueError(
            "Dates are not chronological."
        )

    if df["future_return"].isna().any():

        raise ValueError(
            "future_return contains NaN."
        )

    # -----------------------------------------------------
    # Infinite values
    # -----------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        f"\nInfinite values: "
        f"{infinite_count}"
    )

    if infinite_count > 0:

        raise ValueError(
            "Dataset contains infinite values."
        )


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features(df):

    # -----------------------------------------------------
    # Never use these as features
    # -----------------------------------------------------

    forbidden_columns = {
        "Date",
        "future_return",
        "target",
        "trading_target",
    }

    # -----------------------------------------------------
    # Leakage check
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
        "Suspicious columns:"
    )

    print(suspicious_columns)

    expected_suspicious = {
        "future_return",
        "target",
        "trading_target"
    }

    unexpected = [
        column
        for column in suspicious_columns
        if column not in expected_suspicious
    ]

    if unexpected:

        raise ValueError(
            "Potential leakage columns: "
            f"{unexpected}"
        )

    # -----------------------------------------------------
    # Remove raw price-level moving averages
    # -----------------------------------------------------

    unwanted_columns = {
        "aapl_ma20",
        "aapl_ma50",
        "aapl_ma200",

        "spy_ma20",
        "spy_ma50",
        "spy_ma200",
    }

    feature_columns = [

        column

        for column in df.columns

        if column not in forbidden_columns

        and column not in unwanted_columns

    ]

    X = df[feature_columns].copy()

    y = df["future_return"].astype(float)

    dates = df["Date"].copy()

    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Remove infinities
    # -----------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # Remove invalid rows
    # -----------------------------------------------------

    valid_rows = (
        X.notna().all(axis=1)
        & y.notna()
        & dates.notna()
    )

    X = (
        X.loc[valid_rows]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_rows]
        .reset_index(drop=True)
    )

    dates = (
        dates.loc[valid_rows]
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Rows    : {len(X)}"
    )

    print("\nFeature list:")

    for feature in feature_columns:

        print(
            f"  - {feature}"
        )

    print("\nFuture return statistics:")

    print(
        y.describe()
    )

    return (
        X,
        y,
        dates,
        feature_columns
    )


# =========================================================
# TIME SPLIT
# =========================================================

def time_split(
    X,
    y,
    dates
):

    n = len(X)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    X_train = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    dates_train = dates.iloc[
        :train_end
    ].copy()

    X_val = X.iloc[
        train_end:validation_end
    ].copy()

    y_val = y.iloc[
        train_end:validation_end
    ].copy()

    dates_val = dates.iloc[
        train_end:validation_end
    ].copy()

    X_test = X.iloc[
        validation_end:
    ].copy()

    y_test = y.iloc[
        validation_end:
    ].copy()

    dates_test = dates.iloc[
        validation_end:
    ].copy()

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
        f"{dates_train.iloc[0].date()} "
        f"→ "
        f"{dates_train.iloc[-1].date()}"
    )

    print("\nValidation:")

    print(
        f"{dates_val.iloc[0].date()} "
        f"→ "
        f"{dates_val.iloc[-1].date()}"
    )

    print("\nTest:")

    print(
        f"{dates_test.iloc[0].date()} "
        f"→ "
        f"{dates_test.iloc[-1].date()}"
    )

    # -----------------------------------------------------
    # Safety
    # -----------------------------------------------------

    if dates_train.max() >= dates_val.min():

        raise ValueError(
            "Train/validation overlap."
        )

    if dates_val.max() >= dates_test.min():

        raise ValueError(
            "Validation/test overlap."
        )

    return (
        X_train,
        y_train,
        dates_train,

        X_val,
        y_val,
        dates_val,

        X_test,
        y_test,
        dates_test
    )


# =========================================================
# TRAIN-ONLY WINSORIZATION
# =========================================================

def calculate_return_clip(
    y_train
):

    lower_bound = y_train.quantile(
        LOWER_QUANTILE
    )

    upper_bound = y_train.quantile(
        UPPER_QUANTILE
    )

    print("\n" + "=" * 70)
    print("TRAIN TARGET ROBUSTIFICATION")
    print("=" * 70)

    print(
        f"Lower {LOWER_QUANTILE:.0%} quantile: "
        f"{lower_bound:.6f}"
    )

    print(
        f"Upper {UPPER_QUANTILE:.0%} quantile: "
        f"{upper_bound:.6f}"
    )

    return (
        lower_bound,
        upper_bound
    )


def winsorize_return(
    y,
    lower_bound,
    upper_bound
):

    return y.clip(
        lower=lower_bound,
        upper=upper_bound
    )


# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST REGRESSOR")
    print("=" * 70)

    model = XGBRegressor(

        objective="reg:pseudohubererror",

        n_estimators=1000,

        learning_rate=0.02,

        max_depth=3,

        min_child_weight=5,

        subsample=0.80,

        colsample_bytree=0.80,

        gamma=0.0,

        reg_alpha=0.1,

        reg_lambda=2.0,

        eval_metric="mae",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        early_stopping_rounds=75,
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_val,
                y_val
            )
        ],

        verbose=False
    )

    print(
        "\nTraining completed."
    )

    if hasattr(
        model,
        "best_iteration"
    ):

        print(
            f"Best iteration: "
            f"{model.best_iteration}"
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
        actual
    )

    predicted = np.asarray(
        predicted
    )

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
# EVALUATION
# =========================================================

def evaluate_model(
    model,
    X,
    y,
    dates,
    dataset_name
):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    mse = mean_squared_error(
        y,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y,
        predictions
    )

    correlation = safe_correlation(
        y,
        predictions
    )

    # -----------------------------------------------------
    # Directional accuracy
    # -----------------------------------------------------

    actual_direction = (
        y.to_numpy() > 0
    )

    predicted_direction = (
        predictions > 0
    )

    directional_accuracy = (
        actual_direction
        == predicted_direction
    ).mean()

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} RESULTS"
    )
    print("=" * 70)

    print(
        f"Period: "
        f"{dates.iloc[0].date()} "
        f"→ "
        f"{dates.iloc[-1].date()}"
    )

    print(
        f"\nMAE                  : "
        f"{mae:.8f}"
    )

    print(
        f"RMSE                 : "
        f"{rmse:.8f}"
    )

    print(
        f"R²                   : "
        f"{r2:.8f}"
    )

    if np.isnan(correlation):

        print(
            "Prediction Corr.     : NaN"
        )

    else:

        print(
            f"Prediction Corr.     : "
            f"{correlation:.8f}"
        )

    print(
        f"Directional Accuracy : "
        f"{directional_accuracy:.4f}"
    )

    return {
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "correlation": correlation,
        "directional_accuracy":
            directional_accuracy,
    }


# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

def inspect_predictions(
    predictions,
    actual
):

    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    predicted_series = pd.Series(
        predictions
    )

    actual_series = pd.Series(
        actual
    )

    print("\nPredicted returns:")

    print(
        predicted_series.describe()
    )

    print("\nActual returns:")

    print(
        actual_series.describe()
    )

    print(
        f"\nPredicted positive rate: "
        f"{(predictions > 0).mean():.2%}"
    )

    print(
        f"Actual positive rate: "
        f"{(actual > 0).mean():.2%}"
    )


# =========================================================
# QUINTILE / RANKING ANALYSIS
# =========================================================

def quintile_analysis(
    actual,
    predicted
):

    print("\n" + "=" * 70)
    print("PREDICTION QUINTILE ANALYSIS")
    print("=" * 70)

    analysis = pd.DataFrame({

        "actual_return":
            np.asarray(actual),

        "predicted_return":
            np.asarray(predicted),
    })

    # -----------------------------------------------------
    # Rank predictions into five groups
    # -----------------------------------------------------

    try:

        analysis["quintile"] = pd.qcut(
            analysis["predicted_return"],
            q=5,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Q5",
            ],
            duplicates="drop"
        )

    except ValueError:

        print(
            "Could not create quintiles "
            "because predictions have "
            "insufficient variation."
        )

        return None

    grouped = (
        analysis
        .groupby(
            "quintile",
            observed=False
        )
        .agg(
            mean_actual_return=(
                "actual_return",
                "mean"
            ),

            median_actual_return=(
                "actual_return",
                "median"
            ),

            mean_predicted_return=(
                "predicted_return",
                "mean"
            ),

            count=(
                "actual_return",
                "count"
            )
        )
    )

    print(
        "\nActual return by prediction quintile:"
    )

    print(
        grouped.to_string()
    )

    # -----------------------------------------------------
    # Long-short spread
    # -----------------------------------------------------

    if (
        "Q1" in grouped.index
        and "Q5" in grouped.index
    ):

        q5_return = (
            grouped.loc[
                "Q5",
                "mean_actual_return"
            ]
        )

        q1_return = (
            grouped.loc[
                "Q1",
                "mean_actual_return"
            ]
        )

        spread = (
            q5_return - q1_return
        )

        print(
            f"\nQ5 - Q1 spread: "
            f"{spread:.6f}"
        )

    return grouped


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def calculate_feature_importance(
    model,
    feature_columns
):

    importance = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("TOP 20 FEATURES")
    print("=" * 70)

    print(
        importance
        .head(20)
        .to_string(index=False)
    )

    return importance


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

def save_test_predictions(
    dates,
    actual,
    predicted
):

    output = pd.DataFrame({

        "Date":
            dates.values,

        "actual_return":
            actual.values,

        "predicted_return":
            predicted,
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

    output_path = os.path.join(
        MODEL_DIR,
        "xgboost_regression_test_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPredictions saved to:"
        f"\n{output_path}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):

    model_path = os.path.join(
        MODEL_DIR,
        "xgboost_return_regressor.json"
    )

    model.save_model(
        model_path
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )


# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

def save_feature_importance(
    importance
):

    path = os.path.join(
        MODEL_DIR,
        "xgboost_regression_feature_importance.csv"
    )

    importance.to_csv(
        path,
        index=False
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. Load
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # 2. Validate
    # -----------------------------------------------------

    validate_dataset(df)

    # -----------------------------------------------------
    # 3. Prepare features
    # -----------------------------------------------------

    (
        X,
        y,
        dates,
        feature_columns
    ) = prepare_features(df)

    # -----------------------------------------------------
    # 4. Split chronologically
    # -----------------------------------------------------

    (
        X_train,
        y_train,
        dates_train,

        X_val,
        y_val,
        dates_val,

        X_test,
        y_test,
        dates_test,

    ) = time_split(
        X,
        y,
        dates
    )

    # -----------------------------------------------------
    # 5. Calculate training-only clipping bounds
    # -----------------------------------------------------

    (
        lower_bound,
        upper_bound
    ) = calculate_return_clip(
        y_train
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # We calculate clipping bounds ONLY using TRAIN.
    #
    # Validation/test are never used to calculate these
    # statistics.
    # -----------------------------------------------------

    y_train_clipped = winsorize_return(
        y_train,
        lower_bound,
        upper_bound
    )

    # -----------------------------------------------------
    # 6. Train model
    # -----------------------------------------------------

    model = train_model(
        X_train,
        y_train_clipped,
        X_val,
        y_val
    )

    # -----------------------------------------------------
    # 7. Evaluate train
    # -----------------------------------------------------

    train_results = evaluate_model(
        model,
        X_train,
        y_train,
        dates_train,
        "Train"
    )

    # -----------------------------------------------------
    # 8. Evaluate validation
    # -----------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        dates_val,
        "Validation"
    )

    # -----------------------------------------------------
    # 9. Evaluate test
    # -----------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        dates_test,
        "Test"
    )

    # -----------------------------------------------------
    # 10. Prediction distribution
    # -----------------------------------------------------

    inspect_predictions(
        test_results["predictions"],
        y_test.to_numpy()
    )

    # -----------------------------------------------------
    # 11. Information coefficient
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("INFORMATION COEFFICIENT")
    print("=" * 70)

    test_ic = test_results["correlation"]

    if np.isnan(test_ic):

        print(
            "Test IC: NaN "
            "(predictions have no variation)"
        )

    else:

        print(
            f"Test IC: {test_ic:.8f}"
        )

    # -----------------------------------------------------
    # 12. Quintile analysis
    # -----------------------------------------------------

    quintile_results = quintile_analysis(
        y_test.to_numpy(),
        test_results["predictions"]
    )

    # -----------------------------------------------------
    # 13. Feature importance
    # -----------------------------------------------------

    importance = (
        calculate_feature_importance(
            model,
            feature_columns
        )
    )

    # -----------------------------------------------------
    # 14. Save model
    # -----------------------------------------------------

    save_model(
        model
    )

    # -----------------------------------------------------
    # 15. Save predictions
    # -----------------------------------------------------

    save_test_predictions(
        dates_test,
        y_test,
        test_results["predictions"]
    )

    # -----------------------------------------------------
    # 16. Save feature importance
    # -----------------------------------------------------

    save_feature_importance(
        importance
    )

    # -----------------------------------------------------
    # 17. Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Train RMSE       : "
        f"{train_results['rmse']:.8f}"
    )

    print(
        f"Validation RMSE  : "
        f"{validation_results['rmse']:.8f}"
    )

    print(
        f"Test RMSE        : "
        f"{test_results['rmse']:.8f}"
    )

    print(
        f"\nTrain R²         : "
        f"{train_results['r2']:.8f}"
    )

    print(
        f"Validation R²    : "
        f"{validation_results['r2']:.8f}"
    )

    print(
        f"Test R²          : "
        f"{test_results['r2']:.8f}"
    )

    print(
        f"\nTest Correlation  : "
        f"{test_results['correlation']}"
    )

    print(
        f"Test Directional  : "
        f"{test_results['directional_accuracy']:.4f}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()