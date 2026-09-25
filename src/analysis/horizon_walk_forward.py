# src/analysis/horizon_walk_forward.py

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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Same walk-forward idea used in the previous LSTM experiment
TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3


HORIZONS = [
    1,
    3,
    5,
    10,
    20,
]


FEATURES = [
    "price_ma200_ratio",
    "return_60d",
    "beta_60",

    "volatility_10",
    "volatility_20",
    "volatility_60",

    "price_ma50_ratio",

    "volume_ratio_20",

    "rsi_14",

    "momentum_20",
]


MIN_STOCKS = 5


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("=" * 75)
    print("LOADING DATA")
    print("=" * 75)

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Tickers : {df['Ticker'].nunique()}"
    )

    print(
        f"Dates   : {df['Date'].nunique():,}"
    )

    print(
        f"Range   : "
        f"{df['Date'].min().date()} "
        f"-> "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# FORWARD RETURNS
# ============================================================

def create_forward_returns(
    df
):

    df = df.copy()

    for horizon in HORIZONS:

        df[
            f"future_{horizon}d"
        ] = (
            df
            .groupby("Ticker")["Close"]
            .shift(-horizon)
            / df["Close"]
            - 1.0
        )

    return df


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

def create_folds(
    dates
):

    dates = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(
                dates
            ).unique()
        )
    )

    first_date = dates.min()
    last_date = dates.max()

    folds = []

    fold_id = 1

    train_start = first_date

    while True:

        train_end = (
            train_start
            + pd.DateOffset(
                months=TRAIN_MONTHS
            )
        )

        validation_end = (
            train_end
            + pd.DateOffset(
                months=VALIDATION_MONTHS
            )
        )

        test_end = (
            validation_end
            + pd.DateOffset(
                months=TEST_MONTHS
            )
        )

        # Need complete test period
        if test_end > last_date:

            break

        folds.append(
            {
                "fold": fold_id,

                "train_start":
                    train_start,

                "train_end":
                    train_end,

                "validation_start":
                    train_end,

                "validation_end":
                    validation_end,

                "test_start":
                    validation_end,

                "test_end":
                    test_end,
            }
        )

        fold_id += 1

        train_start = (
            train_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


# ============================================================
# DAILY CROSS-SECTIONAL IC
# ============================================================

def calculate_daily_ic(
    df,
    feature,
    target_column,
):

    data = df[
        [
            "Date",
            "Ticker",
            feature,
            target_column,
        ]
    ].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            feature,
            target_column,
        ]
    )

    records = []

    for date, day in data.groupby(
        "Date",
        sort=True,
    ):

        if len(day) < MIN_STOCKS:
            continue

        x = day[feature].values
        y = day[target_column].values

        if np.all(
            x == x[0]
        ):
            continue

        if np.all(
            y == y[0]
        ):
            continue

        ic, _ = spearmanr(
            x,
            y,
        )

        if np.isfinite(ic):

            records.append(
                {
                    "Date": date,
                    "IC": ic,
                    "N": len(day),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# FOLD ANALYSIS
# ============================================================

def analyze_fold(
    df,
    fold,
):

    test_start = fold[
        "validation_end"
    ]

    test_end = fold[
        "test_end"
    ]

    mask = (
        (df["Date"] >= test_start)
        &
        (df["Date"] < test_end)
    )

    test_df = df[
        mask
    ].copy()

    results = []

    for horizon in HORIZONS:

        target = (
            f"future_{horizon}d"
        )

        for feature in FEATURES:

            daily_ic = calculate_daily_ic(
                test_df,
                feature,
                target,
            )

            if daily_ic.empty:
                continue

            ic = daily_ic[
                "IC"
            ].values

            mean_ic = np.mean(
                ic
            )

            median_ic = np.median(
                ic
            )

            std_ic = (
                np.std(
                    ic,
                    ddof=1,
                )
                if len(ic) > 1
                else np.nan
            )

            icir = (
                mean_ic / std_ic
                if (
                    np.isfinite(
                        std_ic
                    )
                    and std_ic > 0
                )
                else np.nan
            )

            results.append(
                {
                    "fold":
                        fold["fold"],

                    "test_start":
                        test_start,

                    "test_end":
                        test_end,

                    "horizon":
                        horizon,

                    "feature":
                        feature,

                    "days":
                        len(ic),

                    "mean_ic":
                        mean_ic,

                    "median_ic":
                        median_ic,

                    "ic_std":
                        std_ic,

                    "icir":
                        icir,

                    "positive_ic_pct":
                        (
                            np.mean(
                                ic > 0
                            )
                            * 100
                        ),
                }
            )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = create_forward_returns(
        df
    )

    folds = create_folds(
        df["Date"]
    )

    print()
    print("=" * 75)
    print("WALK-FORWARD FOLDS")
    print("=" * 75)

    print(
        f"Number of folds: "
        f"{len(folds)}"
    )

    all_results = []

    for fold in folds:

        print(
            f"\nFold {fold['fold']}: "
            f"{fold['test_start'].date()} "
            f"-> "
            f"{fold['test_end'].date()}"
        )

        result = analyze_fold(
            df,
            fold,
        )

        all_results.append(
            result
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_path = (
        OUTPUT_DIR
        / "horizon_walk_forward_folds.csv"
    )

    results.to_csv(
        fold_path,
        index=False,
    )

    # ========================================================
    # AGGREGATE
    # ========================================================

    summary = (
        results
        .groupby(
            [
                "horizon",
                "feature",
            ]
        )
        .agg(
            mean_ic=(
                "mean_ic",
                "mean",
            ),

            median_fold_ic=(
                "mean_ic",
                "median",
            ),

            fold_ic_std=(
                "mean_ic",
                "std",
            ),

            mean_icir=(
                "icir",
                "mean",
            ),

            positive_fold_pct=(
                "mean_ic",
                lambda x:
                (
                    x > 0
                ).mean()
                * 100,
            ),

            mean_positive_daily_pct=(
                "positive_ic_pct",
                "mean",
            ),

            folds=(
                "fold",
                "count",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # RANK
    # ========================================================

    summary[
        "abs_mean_ic"
    ] = summary[
        "mean_ic"
    ].abs()

    summary = summary.sort_values(
        [
            "horizon",
            "mean_ic",
        ],
        ascending=[
            True,
            False,
        ],
    )

    summary_path = (
        OUTPUT_DIR
        / "horizon_walk_forward_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # PRINT HORIZON SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "WALK-FORWARD HORIZON SUMMARY"
    )
    print("=" * 75)

    horizon_summary = (
        summary
        .groupby("horizon")
        .agg(
            best_mean_ic=(
                "mean_ic",
                "max",
            ),

            median_feature_ic=(
                "mean_ic",
                "median",
            ),

            best_positive_fold_pct=(
                "positive_fold_pct",
                "max",
            ),

            best_mean_icir=(
                "mean_icir",
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

    # ========================================================
    # PRINT TOP FEATURES PER HORIZON
    # ========================================================

    for horizon in HORIZONS:

        print()
        print(
            "=" * 75
        )
        print(
            f"TOP FEATURES - {horizon}D"
        )
        print(
            "=" * 75
        )

        subset = summary[
            summary["horizon"]
            == horizon
        ].copy()

        subset = subset.sort_values(
            "mean_ic",
            ascending=False,
        )

        print(
            subset[
                [
                    "feature",
                    "mean_ic",
                    "median_fold_ic",
                    "fold_ic_std",
                    "mean_icir",
                    "positive_fold_pct",
                    "mean_positive_daily_pct",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 75)
    print("FILES")
    print("=" * 75)

    print(
        fold_path
    )

    print(
        summary_path
    )

    print()
    print(
        "Walk-forward horizon analysis complete."
    )


if __name__ == "__main__":
    main()