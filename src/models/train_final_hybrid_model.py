import os
import json
import warnings

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = r"C:\Users\ASUS\Desktop\quant_trading_proj\models\final_rag_lstm_dataset.csv"

MODEL_DIR = r"C:\Users\ASUS\Desktop\quant_trading_proj\models\final_hybrid"
os.makedirs(MODEL_DIR, exist_ok=True)

PREDICTIONS_PATH = os.path.join(
    MODEL_DIR,
    "final_hybrid_predictions.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgb_final_hybrid.json"
)

IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "metrics.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("FINAL HYBRID MODEL")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "ticker"]
).reset_index(drop=True)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df)}")
print(f"Date range : {df['date'].min().date()} -> {df['date'].max().date()}")


# ============================================================
# TARGET
# ============================================================

TARGET = "future_return"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if "lstm_prediction" not in df.columns:
    raise ValueError("lstm_prediction column not found.")


# ============================================================
# BASIC SANITY CHECK
# ============================================================

print("\nTarget statistics:")
print(df[TARGET].describe())

print("\nLSTM prediction statistics:")
print(df["lstm_prediction"].describe())

print("\nMissing target values:", df[TARGET].isna().sum())
print("Missing LSTM values :", df["lstm_prediction"].isna().sum())


# ============================================================
# ONE-HOT ENCODE TICKER
# ============================================================

ticker_dummies = pd.get_dummies(
    df["ticker"],
    prefix="ticker",
    dtype=int
)

df = pd.concat(
    [df, ticker_dummies],
    axis=1
)


# ============================================================
# FEATURE SELECTION
# ============================================================

# Everything below is deliberately excluded:
#
# date             -> only for chronological splitting
# ticker           -> replaced by ticker one-hot
# Sector           -> redundant with sector_* columns
# future_return    -> TARGET
# rag_available    -> constant after filtering
#

EXCLUDE = {
    "date",
    "ticker",
    "Sector",
    "future_return",
    "rag_available"
}

feature_columns = [
    col
    for col in df.columns
    if col not in EXCLUDE
]


# Make sure no accidental target leakage slipped in.
LEAKAGE_COLUMNS = {
    "future_return",
    "target",
    "future_5d"
}

bad_columns = [
    col
    for col in feature_columns
    if col in LEAKAGE_COLUMNS
]

if bad_columns:
    raise ValueError(
        f"Potential leakage columns found: {bad_columns}"
    )


# ============================================================
# NUMERIC CLEANUP
# ============================================================

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Convert booleans to integers where necessary.
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

# Force numeric.
X = X.apply(pd.to_numeric, errors="coerce")

# Check for missing values.
missing = X.isna().sum()

if missing.sum() > 0:
    print("\nMissing feature values found:")
    print(missing[missing > 0])

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

# Remove rows where target is invalid.
valid_target = y.notna() & np.isfinite(y)

X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)
df_model = df.loc[valid_target].reset_index(drop=True)


print("\nFeature count:", len(feature_columns))

print("\nFeatures:")
for i, col in enumerate(feature_columns):
    print(f"{i:3d}. {col}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

unique_dates = np.sort(
    df_model["date"].unique()
)

n_dates = len(unique_dates)

train_end_idx = int(n_dates * 0.70)
val_end_idx = int(n_dates * 0.85)

train_end_date = unique_dates[train_end_idx - 1]
val_end_date = unique_dates[val_end_idx - 1]

train_mask = df_model["date"] <= train_end_date

val_mask = (
    (df_model["date"] > train_end_date)
    &
    (df_model["date"] <= val_end_date)
)

test_mask = df_model["date"] > val_end_date


X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

df_test = df_model.loc[test_mask].copy()


print("\n" + "=" * 75)
print("CHRONOLOGICAL SPLIT")
print("=" * 75)

print(
    f"Train : {len(X_train):,} rows | "
    f"{df_model.loc[train_mask, 'date'].min().date()} -> "
    f"{df_model.loc[train_mask, 'date'].max().date()}"
)

print(
    f"Valid : {len(X_val):,} rows | "
    f"{df_model.loc[val_mask, 'date'].min().date()} -> "
    f"{df_model.loc[val_mask, 'date'].max().date()}"
)

print(
    f"Test  : {len(X_test):,} rows | "
    f"{df_model.loc[test_mask, 'date'].min().date()} -> "
    f"{df_model.loc[test_mask, 'date'].max().date()}"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=500,
    learning_rate=0.03,

    max_depth=5,
    min_child_weight=4,

    subsample=0.85,
    colsample_bytree=0.85,

    gamma=0.05,

    reg_alpha=0.10,
    reg_lambda=2.0,

    tree_method="hist",

    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL HYBRID MODEL")
print("=" * 75)

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=False
)

print("Training complete.")


# ============================================================
# PREDICTIONS
# ============================================================

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted,
    name
):
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    direction_actual = actual > 0
    direction_pred = predicted > 0

    directional_accuracy = (
        direction_actual == direction_pred
    ).mean()

    print(f"\n{name}")
    print("-" * 50)
    print(f"RMSE                 : {rmse:.8f}")
    print(f"MAE                  : {mae:.8f}")
    print(
        f"Directional Accuracy : "
        f"{directional_accuracy * 100:.2f}%"
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": float(
            directional_accuracy
        )
    }


# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

train_metrics = calculate_metrics(
    y_train,
    train_pred,
    "TRAIN"
)

val_metrics = calculate_metrics(
    y_val,
    val_pred,
    "VALIDATION"
)

test_metrics = calculate_metrics(
    y_test,
    test_pred,
    "TEST"
)


# ============================================================
# LSTM BASELINE ON TEST
# ============================================================

lstm_test = df_test["lstm_prediction"].values

lstm_metrics = calculate_metrics(
    y_test.values,
    lstm_test,
    "LSTM BASELINE - TEST"
)


# ============================================================
# SPEARMAN IC
# ============================================================

def calculate_daily_ic(
    frame,
    prediction_column,
    target_column
):
    daily_ics = []

    for _, group in frame.groupby("date"):
        if len(group) < 3:
            continue

        pred = group[prediction_column]
        actual = group[target_column]

        if pred.nunique() < 2 or actual.nunique() < 2:
            continue

        ic, _ = spearmanr(
            pred,
            actual
        )

        if np.isfinite(ic):
            daily_ics.append(ic)

    if not daily_ics:
        return np.nan

    return float(np.mean(daily_ics))


df_test["hybrid_prediction"] = test_pred

hybrid_ic = calculate_daily_ic(
    df_test,
    "hybrid_prediction",
    TARGET
)

lstm_ic = calculate_daily_ic(
    df_test,
    "lstm_prediction",
    TARGET
)

print("\n" + "=" * 75)
print("CROSS-SECTIONAL INFORMATION")
print("=" * 75)

print(f"Hybrid mean daily IC : {hybrid_ic:.6f}")
print(f"LSTM mean daily IC   : {lstm_ic:.6f}")


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

output_columns = [
    "date",
    "ticker",
    TARGET,
    "lstm_prediction",
    "hybrid_prediction"
]

df_predictions = df_test[output_columns].copy()

df_predictions["hybrid_direction"] = (
    df_predictions["hybrid_prediction"] > 0
).astype(int)

df_predictions["lstm_direction"] = (
    df_predictions["lstm_prediction"] > 0
).astype(int)

df_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    MODEL_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "dataset": {
        "rows": int(len(df)),
        "features": int(len(feature_columns)),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date())
    },

    "split": {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_end": str(pd.Timestamp(train_end_date).date()),
        "validation_end": str(pd.Timestamp(val_end_date).date())
    },

    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,

    "lstm_baseline_test": lstm_metrics,

    "cross_sectional_ic": {
        "hybrid": hybrid_ic,
        "lstm": lstm_ic
    },

    "features": feature_columns
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 75)
print("TOP 20 FEATURES")
print("=" * 75)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("FINAL HYBRID MODEL SAVED")
print("=" * 75)

print(f"Model       : {MODEL_PATH}")
print(f"Predictions : {PREDICTIONS_PATH}")
print(f"Importance  : {IMPORTANCE_PATH}")
print(f"Metrics     : {METRICS_PATH}")

print("\nDone.")