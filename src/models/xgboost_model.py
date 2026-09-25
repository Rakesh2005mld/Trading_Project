import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/processed/training_dataset.csv"
MODEL_DIR = "models"

# Minimum meaningful next-day move.
#
# target:
#   0 -> bearish  : future_return < -THRESHOLD
#   1 -> neutral   : -THRESHOLD <= future_return <= THRESHOLD
#   2 -> bullish  : future_return > THRESHOLD
#
# We start with 0.50% as a research parameter.
# It must NOT be tuned using the final test set.
RETURN_THRESHOLD = 0.005


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# =========================================================
# PATH SETUP
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset() -> pd.DataFrame:

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

    # Sort chronologically
    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove duplicate dates
    df = (
        df
        .drop_duplicates(
            subset=["Date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df


# =========================================================
# DATASET VALIDATION
# =========================================================

def validate_dataset(df: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")

    print("\nDate range:")
    print(f"Start      : {df['Date'].min()}")
    print(f"End        : {df['Date'].max()}")

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "Date",
        "future_return",
        "target"
    ]

    missing_required = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            f"{missing_required}"
        )

    # -----------------------------------------------------
    # Date validation
    # -----------------------------------------------------

    if df["Date"].isna().any():
        raise ValueError(
            "Dataset contains invalid dates."
        )

    if not df["Date"].is_monotonic_increasing:
        raise ValueError(
            "Dates are not sorted chronologically."
        )

    # -----------------------------------------------------
    # Future return validation
    # -----------------------------------------------------

    if df["future_return"].isna().any():
        raise ValueError(
            "future_return contains NaN values."
        )

    if not pd.api.types.is_numeric_dtype(
        df["future_return"]
    ):
        raise ValueError(
            "future_return must be numeric."
        )

    # -----------------------------------------------------
    # Original binary target
    # -----------------------------------------------------

    print("\nOriginal target distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    print("\nOriginal target proportions:")

    print(
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print(
        "\nNote: The original binary target will NOT "
        "be used by this model."
    )


# =========================================================
# CREATE BETTER TRADING TARGET
# =========================================================

def create_trading_target(
    df: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:

    df = df.copy()

    future_return = df["future_return"]

    # -----------------------------------------------------
    # 0 = Bearish
    # 1 = Neutral
    # 2 = Bullish
    # -----------------------------------------------------

    df["trading_target"] = np.select(

        [
            future_return < -threshold,

            future_return > threshold,
        ],

        [
            0,
            2,
        ],

        default=1
    )

    return df


# =========================================================
# PRINT TARGET INFORMATION
# =========================================================

def inspect_trading_target(df: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("TRADING TARGET")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{RETURN_THRESHOLD * 100:.2f}%"
    )

    print(
        "\nClasses:"
    )

    print(
        "0 = Bearish"
    )

    print(
        "1 = Neutral"
    )

    print(
        "2 = Bullish"
    )

    counts = (
        df["trading_target"]
        .value_counts()
        .sort_index()
    )

    proportions = (
        df["trading_target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print("\nCounts:")

    print(counts)

    print("\nProportions:")

    print(proportions)

    print("\nHuman-readable distribution:")

    labels = {
        0: "Bearish",
        1: "Neutral",
        2: "Bullish",
    }

    for class_id in [0, 1, 2]:

        count = int(
            counts.get(class_id, 0)
        )

        proportion = float(
            proportions.get(class_id, 0)
        )

        print(
            f"{labels[class_id]:<10} "
            f"{count:>5} "
            f"({proportion:.2%})"
        )


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features(df: pd.DataFrame):

    # -----------------------------------------------------
    # Columns that must NEVER become model features
    # -----------------------------------------------------

    forbidden_columns = {
        "Date",
        "target",
        "future_return",
        "trading_target",
    }

    # -----------------------------------------------------
    # Suspicious future-looking names
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
        "Suspicious columns detected:"
    )

    print(suspicious_columns)

    # The expected suspicious columns are:
    # future_return
    # target
    # trading_target
    #
    # Anything else needs investigation.

    expected_suspicious = {
        "future_return",
        "target",
        "trading_target",
    }

    unexpected = [
        col
        for col in suspicious_columns
        if col not in expected_suspicious
    ]

    if unexpected:
        raise ValueError(
            "Potential leakage columns found: "
            f"{unexpected}"
        )

    # -----------------------------------------------------
    # Remove raw price-level moving averages
    # -----------------------------------------------------
    #
    # Relative MA ratios are retained.
    #

    unwanted_columns = {
        "aapl_ma20",
        "aapl_ma50",
        "aapl_ma200",

        "spy_ma20",
        "spy_ma50",
        "spy_ma200",
    }

    # -----------------------------------------------------
    # Select features
    # -----------------------------------------------------

    feature_columns = [

        column

        for column in df.columns

        if column not in forbidden_columns

        and column not in unwanted_columns

    ]

    if len(feature_columns) == 0:
        raise ValueError(
            "No feature columns found."
        )

    X = df[feature_columns].copy()

    y = df["trading_target"].copy()

    dates = df["Date"].copy()

    # -----------------------------------------------------
    # Convert all features to numeric
    # -----------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Replace infinities
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
        .astype(int)
        .reset_index(drop=True)
    )

    dates = (
        dates.loc[valid_rows]
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Final safety checks
    # -----------------------------------------------------

    if X.isna().any().any():
        raise ValueError(
            "NaN values remain in X."
        )

    if np.isinf(X.to_numpy()).any():
        raise ValueError(
            "Infinite values remain in X."
        )

    if not set(y.unique()).issubset(
        {0, 1, 2}
    ):
        raise ValueError(
            "trading_target contains invalid classes."
        )

    print("\n" + "=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    print(
        f"Number of features : "
        f"{len(feature_columns)}"
    )

    print(
        f"Usable rows        : "
        f"{len(X)}"
    )

    print("\nFeatures:")

    for feature in feature_columns:
        print(f"  - {feature}")

    return (
        X,
        y,
        dates,
        feature_columns
    )


# =========================================================
# TIME-BASED SPLIT
# =========================================================

def time_split(
    X,
    y,
    dates,
    train_ratio=TRAIN_RATIO,
    validation_ratio=VALIDATION_RATIO,
):

    if (
        train_ratio
        + validation_ratio
        >= 1.0
    ):
        raise ValueError(
            "Train + validation ratio "
            "must be less than 1."
        )

    n = len(X)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n * (
            train_ratio
            + validation_ratio
        )
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    X_train = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    dates_train = dates.iloc[
        :train_end
    ].copy()

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    X_val = X.iloc[
        train_end:validation_end
    ].copy()

    y_val = y.iloc[
        train_end:validation_end
    ].copy()

    dates_val = dates.iloc[
        train_end:validation_end
    ].copy()

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

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
    print("TIME-BASED SPLIT")
    print("=" * 70)

    print(
        f"Train      : {len(X_train)} rows"
    )

    print(
        f"Validation : {len(X_val)} rows"
    )

    print(
        f"Test       : {len(X_test)} rows"
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
    # Leakage safety
    # -----------------------------------------------------

    if dates_train.max() >= dates_val.min():
        raise ValueError(
            "Train/validation dates overlap."
        )

    if dates_val.max() >= dates_test.min():
        raise ValueError(
            "Validation/test dates overlap."
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
        dates_test,
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def print_split_distribution(
    y,
    split_name
):

    print(
        f"\n{split_name} target distribution:"
    )

    counts = (
        y.value_counts()
        .sort_index()
    )

    proportions = (
        y.value_counts(
            normalize=True
        )
        .sort_index()
    )

    labels = {
        0: "Bearish",
        1: "Neutral",
        2: "Bullish",
    }

    for class_id in [0, 1, 2]:

        count = int(
            counts.get(class_id, 0)
        )

        percentage = float(
            proportions.get(class_id, 0)
        )

        print(
            f"{class_id} "
            f"({labels[class_id]:<8}) : "
            f"{count:>5} "
            f"({percentage:.2%})"
        )


# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    # -----------------------------------------------------
    # Verify all classes exist in training data
    # -----------------------------------------------------

    train_classes = set(
        y_train.unique()
    )

    expected_classes = {
        0,
        1,
        2
    }

    missing_classes = (
        expected_classes
        - train_classes
    )

    if missing_classes:
        raise ValueError(
            "Training set is missing classes: "
            f"{missing_classes}"
        )

    # -----------------------------------------------------
    # Sample weighting
    # -----------------------------------------------------
    #
    # Unlike our previous binary model, the three classes
    # may be meaningfully imbalanced because the neutral
    # region can contain many observations.
    #
    # We calculate weights ONLY from training data.
    #

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    print("\nTraining class counts:")

    print(
        y_train
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=3,

        # -------------------------------------------------
        # Model complexity
        # -------------------------------------------------

        n_estimators=500,

        learning_rate=0.03,

        max_depth=2,

        min_child_weight=8,

        # -------------------------------------------------
        # Random subsampling
        # -------------------------------------------------

        subsample=0.70,

        colsample_bytree=0.70,

        # -------------------------------------------------
        # Regularization
        # -------------------------------------------------

        gamma=0.10,

        reg_alpha=0.5,

        reg_lambda=3.0,

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        eval_metric="mlogloss",

        # -------------------------------------------------
        # Reproducibility
        # -------------------------------------------------

        random_state=RANDOM_STATE,

        n_jobs=-1,

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        early_stopping_rounds=50,
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    model.fit(

        X_train,

        y_train,

        sample_weight=sample_weights,

        eval_set=[
            (
                X_val,
                y_val
            )
        ],

        verbose=False,
    )

    print(
        "\nTraining completed."
    )

    if hasattr(
        model,
        "best_iteration"
    ):

        print(
            f"Best iteration : "
            f"{model.best_iteration}"
        )

    return model


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
    dates=None,
):

    # -----------------------------------------------------
    # Probabilities
    # -----------------------------------------------------

    probabilities = (
        model
        .predict_proba(X)
    )

    # -----------------------------------------------------
    # Class prediction
    # -----------------------------------------------------

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # -----------------------------------------------------
    # Basic metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    loss = log_loss(
        y,
        probabilities,
        labels=[0, 1, 2]
    )

    # -----------------------------------------------------
    # Multi-class ROC-AUC
    # -----------------------------------------------------

    try:

        auc = roc_auc_score(
            y,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        auc = np.nan

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2]
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} RESULTS"
    )
    print("=" * 70)

    if dates is not None:

        print(
            f"Period: "
            f"{dates.iloc[0].date()} "
            f"→ "
            f"{dates.iloc[-1].date()}"
        )

    print(
        f"\nAccuracy          : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{f1:.4f}"
    )

    if np.isnan(auc):

        print(
            "Macro ROC-AUC     : N/A"
        )

    else:

        print(
            f"Macro ROC-AUC     : "
            f"{auc:.4f}"
        )

    print(
        f"Log Loss          : "
        f"{loss:.4f}"
    )

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    print(
        "\nConfusion Matrix"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Cols = Predicted"
    )

    print(
        "             Pred"
    )

    print(
        "           B   N   U"
    )

    for i, label in enumerate(
        ["Bearish", "Neutral", "Bullish"]
    ):

        print(
            f"Actual {label[0]} "
            f"{matrix[i, 0]:4d} "
            f"{matrix[i, 1]:4d} "
            f"{matrix[i, 2]:4d}"
        )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "Bearish",
                "Neutral",
                "Bullish"
            ],
            zero_division=0
        )
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "log_loss": loss,
    }


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def calculate_feature_importance(
    model,
    feature_columns,
):

    importance_df = pd.DataFrame({

        "feature": feature_columns,

        "importance": (
            model.feature_importances_
        ),
    })

    importance_df = (
        importance_df
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
        importance_df
        .head(20)
        .to_string(index=False)
    )

    return importance_df


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

def save_test_predictions(
    dates_test,
    y_test,
    test_results,
):

    probabilities = (
        test_results["probabilities"]
    )

    predictions = (
        test_results["predictions"]
    )

    output = pd.DataFrame({

        "Date":
            dates_test.values,

        "actual_class":
            y_test.values,

        "prob_bearish":
            probabilities[:, 0],

        "prob_neutral":
            probabilities[:, 1],

        "prob_bullish":
            probabilities[:, 2],

        "predicted_class":
            predictions,

    })

    # Human-readable labels
    class_to_label = {
        0: "BEARISH",
        1: "NEUTRAL",
        2: "BULLISH",
    }

    output["actual_label"] = (
        output["actual_class"]
        .map(class_to_label)
    )

    output["predicted_label"] = (
        output["predicted_class"]
        .map(class_to_label)
    )

    output_path = os.path.join(
        MODEL_DIR,
        "xgboost_test_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nTest predictions saved to:"
        f"\n{output_path}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):

    model_path = os.path.join(
        MODEL_DIR,
        "xgboost_trading_classifier.json"
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
    importance_df
):

    importance_path = os.path.join(
        MODEL_DIR,
        "xgboost_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{importance_path}"
    )


# =========================================================
# FINAL SIGNAL DISTRIBUTION
# =========================================================

def inspect_predictions(
    test_results
):

    predictions = (
        test_results["predictions"]
    )

    print("\n" + "=" * 70)
    print("TEST PREDICTION DISTRIBUTION")
    print("=" * 70)

    labels = {
        0: "Bearish",
        1: "Neutral",
        2: "Bullish",
    }

    counts = (
        pd.Series(predictions)
        .value_counts()
        .sort_index()
    )

    for class_id in [0, 1, 2]:

        count = int(
            counts.get(class_id, 0)
        )

        percentage = (
            count / len(predictions)
        )

        print(
            f"{labels[class_id]:<10}: "
            f"{count:>5} "
            f"({percentage:.2%})"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. LOAD
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # 2. VALIDATE
    # -----------------------------------------------------

    validate_dataset(df)

    # -----------------------------------------------------
    # 3. CREATE 3-CLASS TARGET
    # -----------------------------------------------------

    df = create_trading_target(
        df,
        threshold=RETURN_THRESHOLD
    )

    # -----------------------------------------------------
    # 4. INSPECT TARGET
    # -----------------------------------------------------

    inspect_trading_target(df)

    # -----------------------------------------------------
    # 5. PREPARE FEATURES
    # -----------------------------------------------------

    (
        X,
        y,
        dates,
        feature_columns,
    ) = prepare_features(df)

    # -----------------------------------------------------
    # 6. TIME SPLIT
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
    # 7. SPLIT DISTRIBUTIONS
    # -----------------------------------------------------

    print_split_distribution(
        y_train,
        "TRAIN"
    )

    print_split_distribution(
        y_val,
        "VALIDATION"
    )

    print_split_distribution(
        y_test,
        "TEST"
    )

    # -----------------------------------------------------
    # 8. TRAIN
    # -----------------------------------------------------

    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # -----------------------------------------------------
    # 9. TRAIN EVALUATION
    # -----------------------------------------------------

    train_results = evaluate_model(
        model,
        X_train,
        y_train,
        "Train",
        dates_train
    )

    # -----------------------------------------------------
    # 10. VALIDATION EVALUATION
    # -----------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation",
        dates_val
    )

    # -----------------------------------------------------
    # 11. TEST EVALUATION
    # -----------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
        dates_test
    )

    # -----------------------------------------------------
    # 12. TEST PREDICTION DISTRIBUTION
    # -----------------------------------------------------

    inspect_predictions(
        test_results
    )

    # -----------------------------------------------------
    # 13. FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance_df = (
        calculate_feature_importance(
            model,
            feature_columns
        )
    )

    # -----------------------------------------------------
    # 14. SAVE MODEL
    # -----------------------------------------------------

    save_model(
        model
    )

    # -----------------------------------------------------
    # 15. SAVE PREDICTIONS
    # -----------------------------------------------------

    save_test_predictions(
        dates_test,
        y_test,
        test_results
    )

    # -----------------------------------------------------
    # 16. SAVE FEATURE IMPORTANCE
    # -----------------------------------------------------

    save_feature_importance(
        importance_df
    )

    # -----------------------------------------------------
    # 17. FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Train ROC-AUC      : "
        f"{train_results['roc_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC : "
        f"{validation_results['roc_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC       : "
        f"{test_results['roc_auc']:.4f}"
    )

    print(
        f"\nTrain Accuracy     : "
        f"{train_results['accuracy']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_results['accuracy']:.4f}"
    )

    print(
        f"Test Accuracy      : "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"\nTest Balanced Acc. : "
        f"{test_results['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro F1      : "
        f"{test_results['f1']:.4f}"
    )

    print(
        f"Test Log Loss      : "
        f"{test_results['log_loss']:.4f}"
    )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        "1. models/"
        "xgboost_trading_classifier.json"
    )

    print(
        "2. models/"
        "xgboost_test_predictions.csv"
    )

    print(
        "3. models/"
        "xgboost_feature_importance.csv"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()