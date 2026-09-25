from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

HORIZONS = [1, 3, 5, 10, 20]

MIN_STOCKS = 5


FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",

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
]


# ============================================================
# LOAD
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    print("=" * 75)
    print("DATA")
    print("=" * 75)

    print("Rows    :", len(df))
    print("Tickers :", df["Ticker"].nunique())
    print(
        "Dates   :",
        df["Date"].nunique()
    )

    print(
        "Range   :",
        df["Date"].min().date(),
        "->",
        df["Date"].max().date(),
    )

    return df


# ============================================================
# CREATE FUTURE RETURNS
# ============================================================

def create_forward_returns(df):

    df = df.copy()

    for horizon in HORIZONS:

        df[
            f"future_{horizon}d"
        ] = (
            df.groupby("Ticker")["Close"]
            .shift(-horizon)
            / df["Close"]
            - 1.0
        )

    return df


# ============================================================
# DATE SPLIT
# ============================================================

def assign_split(df):

    df = df.copy()

    df["split"] = "outside"

    train_mask = (
        (df["Date"] >= "2015-10-16")
        &
        (df["Date"] <= "2023-05-30")
    )

    val_mask = (
        (df["Date"] >= "2023-05-31")
        &
        (df["Date"] <= "2025-01-21")
    )

    test_mask = (
        (df["Date"] >= "2025-01-22")
        &
        (df["Date"] <= "2026-09-10")
    )

    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "validation"
    df.loc[test_mask, "split"] = "test"

    return df


# ============================================================
# CROSS-SECTIONAL TARGET
# ============================================================

def add_cross_sectional_targets(df):

    df = df.copy()

    for horizon in HORIZONS:

        col = f"future_{horizon}d"

        mean_ret = (
            df.groupby("Date")[col]
            .transform("mean")
        )

        std_ret = (
            df.groupby("Date")[col]
            .transform("std")
        )

        target_col = (
            f"cross_sectional_{horizon}d"
        )

        df[target_col] = (
            df[col] - mean_ret
        ) / std_ret

    return df


# ============================================================
# DAILY IC
# ============================================================

def feature_horizon_ic(
    df,
    feature,
    horizon,
    split,
):

    target_col = f"future_{horizon}d"

    subset = df[
        df["split"] == split
    ][
        [
            "Date",
            feature,
            target_col,
        ]
    ].copy()

    subset = subset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    subset = subset.dropna()

    records = []

    for date, day in subset.groupby("Date"):

        if len(day) < MIN_STOCKS:
            continue

        x = day[feature].values
        y = day[target_col].values

        if np.all(x == x[0]):
            continue

        if np.all(y == y[0]):
            continue

        ic, _ = spearmanr(x, y)

        if np.isfinite(ic):
            records.append(ic)

    if not records:
        return None

    ic = np.asarray(records)

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)

    return {
        "feature": feature,
        "horizon": horizon,
        "split": split,
        "days": len(ic),
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


# ============================================================
# MARKET-WIDE HORIZON SUMMARY
# ============================================================

def summarize_horizon(df, horizon, split):

    target_col = f"future_{horizon}d"

    all_ic = []

    for feature in FEATURES:

        result = feature_horizon_ic(
            df,
            feature,
            horizon,
            split,
        )

        if result is not None:
            all_ic.append(result)

    result_df = pd.DataFrame(all_ic)

    return result_df


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = create_forward_returns(df)

    df = assign_split(df)

    df = add_cross_sectional_targets(df)

    all_results = []

    # ========================================================
    # ANALYZE
    # ========================================================

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print()
        print("=" * 75)
        print(
            f"{split.upper()} HORIZON ANALYSIS"
        )
        print("=" * 75)

        for horizon in HORIZONS:

            print(
                f"\nHorizon: {horizon}D"
            )

            result = summarize_horizon(
                df,
                horizon,
                split,
            )

            if result.empty:
                continue

            all_results.append(result)

            best = (
                result
                .sort_values(
                    "mean_ic",
                    ascending=False,
                )
                .head(5)
            )

            print(
                best[
                    [
                        "feature",
                        "mean_ic",
                        "median_ic",
                        "icir",
                        "positive_ic_pct",
                    ]
                ].to_string(index=False)
            )

    final = pd.concat(
        all_results,
        ignore_index=True,
    )

    output = (
        OUTPUT_DIR
        / "horizon_feature_ic.csv"
    )

    final.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 75)
    print("VALIDATION HORIZON SUMMARY")
    print("=" * 75)

    validation = final[
        final["split"] == "validation"
    ].copy()

    horizon_summary = (
        validation
        .groupby("horizon")
        .agg(
            mean_feature_ic=(
                "mean_ic",
                "mean",
            ),
            median_feature_ic=(
                "mean_ic",
                "median",
            ),
            best_feature_ic=(
                "mean_ic",
                "max",
            ),
            mean_icir=(
                "icir",
                "mean",
            ),
            best_icir=(
                "icir",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        horizon_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved -> {output}"
    )


if __name__ == "__main__":
    main()