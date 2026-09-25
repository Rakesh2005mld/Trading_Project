# src/analysis/feature_ic_analysis.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cross_sectional_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Same date split used by the current fixed-model experiments
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
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 75)
    print("LOADING DATA")
    print("=" * 75)

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df)}")
    print(f"Dates   : {df['Date'].nunique():,}")
    print(f"Tickers : {df['Ticker'].nunique()}")
    print(f"Range   : {df['Date'].min().date()} -> {df['Date'].max().date()}")

    print()

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns:\n{missing_features}"
        )

    if "future_return" not in df.columns:
        raise ValueError("future_return column not found.")

    return df


# ============================================================
# DATE SPLIT
# ============================================================

def add_split_column(df):
    df = df.copy()

    conditions = [
        (
            df["Date"] >= pd.Timestamp(TRAIN_START)
        ) & (
            df["Date"] <= pd.Timestamp(TRAIN_END)
        ),

        (
            df["Date"] >= pd.Timestamp(VAL_START)
        ) & (
            df["Date"] <= pd.Timestamp(VAL_END)
        ),

        (
            df["Date"] >= pd.Timestamp(TEST_START)
        ) & (
            df["Date"] <= pd.Timestamp(TEST_END)
        ),
    ]

    choices = ["train", "validation", "test"]

    df["split"] = np.select(
        conditions,
        choices,
        default="outside",
    )

    return df


# ============================================================
# DAILY FEATURE IC
# ============================================================

def calculate_daily_ic(df, feature):
    """
    For each date:

        Spearman(feature values, future_return)

    This is cross-sectional IC.

    Because we are comparing ranks, this is also effectively
    measuring whether the feature correctly ranks future returns.
    """

    records = []

    subset = df[["Date", "Ticker", feature, "future_return"]].copy()

    subset = subset.replace([np.inf, -np.inf], np.nan)

    subset = subset.dropna(
        subset=[feature, "future_return"]
    )

    for date, day in subset.groupby("Date", sort=True):

        # Need enough stocks to calculate meaningful correlation.
        if len(day) < 3:
            continue

        x = day[feature].values
        y = day["future_return"].values

        # Constant feature or constant return -> undefined Spearman IC
        if np.all(x == x[0]):
            continue

        if np.all(y == y[0]):
            continue

        ic, _ = spearmanr(x, y)

        if np.isfinite(ic):
            records.append(
                {
                    "Date": date,
                    "IC": ic,
                    "N": len(day),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# DAILY PORTFOLIO SPREAD
# ============================================================

def calculate_daily_spread(df, feature, k=2):
    """
    Rank stocks by the feature.

    Long:
        top K

    Short:
        bottom K

    Spread:
        mean(long future_return)
        -
        mean(short future_return)
    """

    records = []

    subset = df[
        ["Date", "Ticker", feature, "future_return"]
    ].copy()

    subset = subset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    subset = subset.dropna(
        subset=[feature, "future_return"]
    )

    for date, day in subset.groupby("Date", sort=True):

        if len(day) < 2 * k:
            continue

        day = day.sort_values(feature)

        long_group = day.tail(k)
        short_group = day.head(k)

        long_return = long_group["future_return"].mean()
        short_return = short_group["future_return"].mean()

        spread = long_return - short_return

        records.append(
            {
                "Date": date,
                "long_return": long_return,
                "short_return": short_return,
                "spread": spread,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# SUMMARY METRICS
# ============================================================

def summarize_feature(df, feature):
    daily_ic = calculate_daily_ic(df, feature)
    daily_spread = calculate_daily_spread(df, feature, k=2)

    if daily_ic.empty:
        return None

    ic = daily_ic["IC"]

    mean_ic = ic.mean()
    median_ic = ic.median()
    std_ic = ic.std(ddof=1)

    if std_ic > 0:
        icir = mean_ic / std_ic
    else:
        icir = np.nan

    positive_ic_fraction = (ic > 0).mean()

    spread_mean = daily_spread["spread"].mean()
    spread_median = daily_spread["spread"].median()

    spread_std = daily_spread["spread"].std(ddof=1)

    if spread_std > 0:
        spread_sharpe = (
            spread_mean / spread_std
        ) * np.sqrt(252)
    else:
        spread_sharpe = np.nan

    positive_spread_fraction = (
        daily_spread["spread"] > 0
    ).mean()

    return {
        "feature": feature,

        "days": len(ic),

        "mean_ic": mean_ic,
        "median_ic": median_ic,
        "ic_std": std_ic,
        "icir": icir,

        "positive_ic_pct":
            positive_ic_fraction * 100,

        "mean_top2_bottom2_spread":
            spread_mean,

        "median_top2_bottom2_spread":
            spread_median,

        "spread_sharpe":
            spread_sharpe,

        "positive_spread_pct":
            positive_spread_fraction * 100,
    }


# ============================================================
# RUN ANALYSIS FOR ONE SPLIT
# ============================================================

def analyze_split(df, split_name):

    print()
    print("=" * 75)
    print(f"{split_name.upper()} FEATURE ANALYSIS")
    print("=" * 75)

    split_df = df[df["split"] == split_name].copy()

    print(
        f"Rows   : {len(split_df):,}"
    )

    print(
        f"Dates  : {split_df['Date'].nunique():,}"
    )

    results = []

    for i, feature in enumerate(FEATURES, start=1):

        print(
            f"[{i:02d}/{len(FEATURES)}] {feature}"
        )

        summary = summarize_feature(
            split_df,
            feature,
        )

        if summary is not None:
            results.append(summary)

    results = pd.DataFrame(results)

    if results.empty:
        raise RuntimeError(
            f"No feature results generated for {split_name}."
        )

    results = results.sort_values(
        by="mean_ic",
        ascending=False,
    ).reset_index(drop=True)

    return results


# ============================================================
# IC STABILITY BY YEAR
# ============================================================

def yearly_feature_ic(df, feature):
    daily_ic = calculate_daily_ic(
        df[df["split"] != "outside"],
        feature,
    )

    if daily_ic.empty:
        return pd.DataFrame()

    daily_ic["Year"] = daily_ic["Date"].dt.year

    result = (
        daily_ic
        .groupby("Year")["IC"]
        .agg(
            mean_ic="mean",
            median_ic="median",
            std_ic="std",
            positive_ic_pct=lambda x:
                (x > 0).mean() * 100,
            days="count",
        )
        .reset_index()
    )

    return result


# ============================================================
# PERIOD STABILITY
# ============================================================

def period_feature_summary(df, feature):

    records = []

    for period in ["train", "validation", "test"]:

        period_df = df[
            df["split"] == period
        ]

        daily_ic = calculate_daily_ic(
            period_df,
            feature,
        )

        if daily_ic.empty:
            continue

        ic = daily_ic["IC"]

        records.append(
            {
                "feature": feature,
                "period": period,

                "mean_ic": ic.mean(),
                "median_ic": ic.median(),
                "ic_std": ic.std(ddof=1),

                "positive_ic_pct":
                    (ic > 0).mean() * 100,

                "days": len(ic),
            }
        )

    return records


# ============================================================
# SAVE DAILY IC
# ============================================================

def save_daily_ic(df):

    all_records = []

    for feature in FEATURES:

        print(
            f"Daily IC -> {feature}"
        )

        daily = calculate_daily_ic(
            df,
            feature,
        )

        if daily.empty:
            continue

        daily["feature"] = feature

        all_records.append(daily)

    if not all_records:
        return

    result = pd.concat(
        all_records,
        ignore_index=True,
    )

    result = result[
        [
            "Date",
            "feature",
            "IC",
            "N",
        ]
    ]

    output = OUTPUT_DIR / "feature_daily_ic.csv"

    result.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved daily IC -> {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = add_split_column(df)

    print()
    print("=" * 75)
    print("SPLIT COUNTS")
    print("=" * 75)

    print(
        df.groupby("split")["Date"]
        .nunique()
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_results = analyze_split(
        df,
        "train",
    )

    train_path = (
        OUTPUT_DIR
        / "feature_ic_train.csv"
    )

    train_results.to_csv(
        train_path,
        index=False,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_results = analyze_split(
        df,
        "validation",
    )

    val_path = (
        OUTPUT_DIR
        / "feature_ic_validation.csv"
    )

    val_results.to_csv(
        val_path,
        index=False,
    )

    # --------------------------------------------------------
    # TEST
    #
    # Diagnostic only.
    # DO NOT use this to select features.
    # --------------------------------------------------------

    test_results = analyze_split(
        df,
        "test",
    )

    test_path = (
        OUTPUT_DIR
        / "feature_ic_test.csv"
    )

    test_results.to_csv(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # PERIOD STABILITY
    # --------------------------------------------------------

    stability_records = []

    for feature in FEATURES:

        records = period_feature_summary(
            df,
            feature,
        )

        stability_records.extend(records)

    stability = pd.DataFrame(
        stability_records
    )

    stability_path = (
        OUTPUT_DIR
        / "feature_ic_period_stability.csv"
    )

    stability.to_csv(
        stability_path,
        index=False,
    )

    # --------------------------------------------------------
    # DAILY IC
    # --------------------------------------------------------

    save_daily_ic(df)

    # --------------------------------------------------------
    # PRINT TOP FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP FEATURES BY VALIDATION IC")
    print("=" * 75)

    display_cols = [
        "feature",
        "mean_ic",
        "median_ic",
        "ic_std",
        "icir",
        "positive_ic_pct",
        "spread_sharpe",
        "positive_spread_pct",
    ]

    print(
        val_results[
            display_cols
        ]
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # PRINT WORST FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("BOTTOM FEATURES BY VALIDATION IC")
    print("=" * 75)

    print(
        val_results[
            display_cols
        ]
        .tail(15)
        .sort_values(
            "mean_ic",
            ascending=True,
        )
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # TRAIN VS VALIDATION
    # --------------------------------------------------------

    comparison = (
        train_results[
            [
                "feature",
                "mean_ic",
                "median_ic",
                "icir",
                "positive_ic_pct",
                "spread_sharpe",
            ]
        ]
        .rename(
            columns={
                "mean_ic": "train_mean_ic",
                "median_ic": "train_median_ic",
                "icir": "train_icir",
                "positive_ic_pct":
                    "train_positive_ic_pct",
                "spread_sharpe":
                    "train_spread_sharpe",
            }
        )
        .merge(
            val_results[
                [
                    "feature",
                    "mean_ic",
                    "median_ic",
                    "icir",
                    "positive_ic_pct",
                    "spread_sharpe",
                ]
            ].rename(
                columns={
                    "mean_ic": "val_mean_ic",
                    "median_ic": "val_median_ic",
                    "icir": "val_icir",
                    "positive_ic_pct":
                        "val_positive_ic_pct",
                    "spread_sharpe":
                        "val_spread_sharpe",
                }
            ),
            on="feature",
        )
    )

    comparison["ic_decay"] = (
        comparison["val_mean_ic"]
        - comparison["train_mean_ic"]
    )

    comparison["absolute_ic_decay"] = (
        comparison["ic_decay"].abs()
    )

    comparison = comparison.sort_values(
        "val_mean_ic",
        ascending=False,
    )

    comparison_path = (
        OUTPUT_DIR
        / "feature_ic_train_vs_validation.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FILES CREATED")
    print("=" * 75)

    print(train_path)
    print(val_path)
    print(test_path)
    print(stability_path)
    print(comparison_path)

    print()
    print("Feature IC analysis complete.")


if __name__ == "__main__":
    main()