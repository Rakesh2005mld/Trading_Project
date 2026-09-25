import os

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_DIR = "models"

LSTM_VALIDATION_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_lstm_validation_predictions.csv"
)

XGB_VALIDATION_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_xgboost_validation_predictions.csv"
)

LSTM_TEST_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_lstm_predictions.csv"
)

XGB_TEST_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_xgboost_predictions.csv"
)

ENSEMBLE_VALIDATION_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_ensemble_validation_predictions.csv"
)

ENSEMBLE_TEST_PATH = os.path.join(
    MODEL_DIR,
    "cross_sectional_ensemble_predictions.csv"
)

WEIGHT_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "ensemble_validation_weights.csv"
)

# Test LSTM weights from 0.0 to 1.0.
# XGBoost weight = 1 - LSTM weight.
WEIGHTS = np.round(
    np.arange(0.0, 1.01, 0.1),
    1
)

TOP_K = 2
BOTTOM_K = 2


# =========================================================
# LOAD PREDICTIONS
# =========================================================

def load_predictions(
    path
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"]
    )

    required_columns = {
        "Date",
        "Ticker",
        "predicted_cross_sectional_target",
        "actual_return",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"\nMissing columns in {path}:\n"
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )

    return df


# =========================================================
# DAILY CROSS-SECTIONAL Z-SCORE
# =========================================================

def add_daily_zscore(
    df,
    prediction_column,
    zscore_column
):

    df = df.copy()

    daily_mean = (
        df
        .groupby("Date")[
            prediction_column
        ]
        .transform("mean")
    )

    daily_std = (
        df
        .groupby("Date")[
            prediction_column
        ]
        .transform("std")
    )

    df[zscore_column] = np.where(

        daily_std > 1e-12,

        (
            df[prediction_column]
            - daily_mean
        )
        / daily_std,

        0.0

    )

    return df


# =========================================================
# MERGE LSTM + XGBOOST
# =========================================================

def merge_models(
    lstm_df,
    xgb_df
):

    merged = pd.merge(

        lstm_df[
            [
                "Date",
                "Ticker",
                "predicted_cross_sectional_target",
                "actual_return",
            ]
        ],

        xgb_df[
            [
                "Date",
                "Ticker",
                "predicted_cross_sectional_target",
                "actual_return",
            ]
        ],

        on=[
            "Date",
            "Ticker"
        ],

        how="inner",

        suffixes=(
            "_lstm",
            "_xgb"
        )
    )

    if merged.empty:

        raise ValueError(
            "No overlapping Date/Ticker rows "
            "between LSTM and XGBoost."
        )

    print(
        f"Matched rows : "
        f"{len(merged):,}"
    )

    # -----------------------------------------------------
    # Verify that actual returns are identical.
    #
    # Small differences can occur because the two CSVs
    # may have been written with slightly different
    # floating-point precision.
    # -----------------------------------------------------

    return_difference = (
        merged["actual_return_lstm"]
        - merged["actual_return_xgb"]
    ).abs()

    max_difference = (
        return_difference.max()
    )

    print(
        f"Max return difference: "
        f"{max_difference:.12e}"
    )

    if not np.allclose(

        merged[
            "actual_return_lstm"
        ].to_numpy(),

        merged[
            "actual_return_xgb"
        ].to_numpy(),

        rtol=1e-7,

        atol=1e-7

    ):

        raise ValueError(
            "LSTM and XGBoost actual returns "
            "do not match within tolerance."
        )

    # Use one canonical actual-return column.
    merged[
        "actual_return"
    ] = merged[
        "actual_return_lstm"
    ]

    # -----------------------------------------------------
    # Daily standardization.
    #
    # This is important because the raw prediction
    # magnitudes of LSTM and XGBoost are very different.
    # -----------------------------------------------------

    merged = add_daily_zscore(

        merged,

        "predicted_cross_sectional_target_lstm",

        "lstm_z"

    )

    merged = add_daily_zscore(

        merged,

        "predicted_cross_sectional_target_xgb",

        "xgb_z"

    )

    return merged


# =========================================================
# CALCULATE DAILY SPEARMAN IC
# =========================================================

def calculate_daily_ic(
    df,
    score_column
):

    daily_ic = []

    for date, group in df.groupby(
        "Date"
    ):

        if len(group) < 2:

            continue

        prediction_ranks = (
            group[
                score_column
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        actual_ranks = (
            group[
                "actual_return"
            ]
            .rank(
                method="average"
            )
            .to_numpy()
        )

        if (
            np.std(prediction_ranks) == 0
            or np.std(actual_ranks) == 0
        ):

            continue

        correlation = np.corrcoef(

            prediction_ranks,

            actual_ranks

        )[0, 1]

        if np.isfinite(
            correlation
        ):

            daily_ic.append(
                correlation
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
# PORTFOLIO METRICS
# =========================================================

def calculate_portfolio_metrics(
    df,
    score_column
):

    long_returns = []
    short_returns = []
    long_short_returns = []
    market_returns = []

    selected_rows = []

    for date, group in df.groupby(
        "Date"
    ):

        if len(group) < (
            TOP_K
            + BOTTOM_K
        ):

            continue

        ranked = (
            group
            .sort_values(
                score_column,
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        long_group = (
            ranked
            .head(TOP_K)
        )

        short_group = (
            ranked
            .tail(BOTTOM_K)
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

        long_short_return = (
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
            long_short_return
        )

        market_returns.append(
            market_return
        )

        long_copy = (
            long_group
            .copy()
        )

        long_copy[
            "portfolio_side"
        ] = "LONG"

        short_copy = (
            short_group
            .copy()
        )

        short_copy[
            "portfolio_side"
        ] = "SHORT"

        selected_rows.append(
            pd.concat(
                [
                    long_copy,
                    short_copy
                ],
                ignore_index=True
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
    # IC
    # -----------------------------------------------------

    ic_mean, ic_median, daily_ic = (
        calculate_daily_ic(
            df,
            score_column
        )
    )

    # -----------------------------------------------------
    # Mean returns
    # -----------------------------------------------------

    if len(long_returns) > 0:

        mean_long_return = (
            np.mean(long_returns)
        )

        mean_short_return = (
            np.mean(short_returns)
        )

        mean_long_short_return = (
            np.mean(
                long_short_returns
            )
        )

        mean_market_return = (
            np.mean(
                market_returns
            )
        )

    else:

        mean_long_return = np.nan
        mean_short_return = np.nan
        mean_long_short_return = np.nan
        mean_market_return = np.nan

    # -----------------------------------------------------
    # Hit rate
    # -----------------------------------------------------

    if len(long_short_returns) > 0:

        hit_rate = np.mean(
            long_short_returns > 0
        )

    else:

        hit_rate = np.nan

    # -----------------------------------------------------
    # Sharpe
    # -----------------------------------------------------

    if (

        len(long_short_returns) > 1

        and np.std(
            long_short_returns,
            ddof=1
        ) > 1e-12

    ):

        daily_sharpe = (

            mean_long_short_return

            / np.std(
                long_short_returns,
                ddof=1
            )

        )

        annualized_sharpe = (
            daily_sharpe
            * np.sqrt(252)
        )

    else:

        annualized_sharpe = np.nan

    # -----------------------------------------------------
    # Selected stocks
    # -----------------------------------------------------

    if len(selected_rows) > 0:

        selected = pd.concat(
            selected_rows,
            ignore_index=True
        )

    else:

        selected = pd.DataFrame()

    metrics = {

        "ic_mean":
            ic_mean,

        "ic_median":
            ic_median,

        "top_return":
            mean_long_return,

        "bottom_return":
            mean_short_return,

        "long_short_return":
            mean_long_short_return,

        "market_return":
            mean_market_return,

        "hit_rate":
            hit_rate,

        "annualized_sharpe":
            annualized_sharpe,

        "num_days":
            len(long_short_returns),

    }

    return (
        metrics,
        selected
    )


# =========================================================
# ADD ENSEMBLE SCORE
# =========================================================

def add_ensemble_score(
    df,
    lstm_weight
):

    df = df.copy()

    xgb_weight = (
        1.0
        - lstm_weight
    )

    df[
        "ensemble_score"
    ] = (

        lstm_weight
        * df["lstm_z"]

        +

        xgb_weight
        * df["xgb_z"]

    )

    return df


# =========================================================
# VALIDATION WEIGHT SEARCH
# =========================================================

def search_validation_weights(
    validation
):

    print("\n" + "=" * 70)
    print("VALIDATION WEIGHT SEARCH")
    print("=" * 70)

    results = []

    for lstm_weight in WEIGHTS:

        xgb_weight = (
            1.0
            - lstm_weight
        )

        scored = add_ensemble_score(
            validation,
            lstm_weight
        )

        metrics, _ = (
            calculate_portfolio_metrics(
                scored,
                "ensemble_score"
            )
        )

        results.append({

            "lstm_weight":
                float(lstm_weight),

            "xgb_weight":
                float(xgb_weight),

            "ic":
                metrics["ic_mean"],

            "median_ic":
                metrics["ic_median"],

            "top_k_return":
                metrics["top_return"],

            "bottom_k_return":
                metrics["bottom_return"],

            "long_short_return":
                metrics[
                    "long_short_return"
                ],

            "hit_rate":
                metrics["hit_rate"],

            "sharpe":
                metrics[
                    "annualized_sharpe"
                ],

            "num_days":
                metrics["num_days"],

        })

        print(

            f"LSTM={lstm_weight:.1f} | "

            f"XGB={xgb_weight:.1f} | "

            f"IC="
            f"{metrics['ic_mean']:.5f} | "

            f"LS="
            f"{metrics['long_short_return']:.5%} | "

            f"Hit="
            f"{metrics['hit_rate']:.2%} | "

            f"Sharpe="
            f"{metrics['annualized_sharpe']:.3f}"

        )

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Primary criterion:
    #
    # Mean daily Spearman IC
    #
    # IC directly measures whether the model ranks
    # stocks correctly, which is the objective of our
    # cross-sectional strategy.
    # -----------------------------------------------------

    valid_results = (
        results_df[
            np.isfinite(
                results_df["ic"]
            )
        ]
        .copy()
    )

    if valid_results.empty:

        raise RuntimeError(
            "No valid ensemble weights "
            "were produced."
        )

    best_index = (
        valid_results[
            "ic"
        ]
        .idxmax()
    )

    best_row = (
        valid_results.loc[
            best_index
        ]
    )

    return (
        best_row,
        results_df
    )


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    name,
    metrics
):

    print(
        f"\n{name}"
    )

    print(
        f"Mean IC           : "
        f"{metrics['ic_mean']:.5f}"
    )

    print(
        f"Median IC         : "
        f"{metrics['ic_median']:.5f}"
    )

    print(
        f"Top-{TOP_K} return       : "
        f"{metrics['top_return']:.5%}"
    )

    print(
        f"Bottom-{BOTTOM_K} return    : "
        f"{metrics['bottom_return']:.5%}"
    )

    print(
        f"Long-short return : "
        f"{metrics['long_short_return']:.5%}"
    )

    print(
        f"Market return     : "
        f"{metrics['market_return']:.5%}"
    )

    print(
        f"Hit rate          : "
        f"{metrics['hit_rate']:.2%}"
    )

    print(
        f"Annualized Sharpe : "
        f"{metrics['annualized_sharpe']:.3f}"
    )

    print(
        f"Evaluation days   : "
        f"{metrics['num_days']}"
    )


# =========================================================
# SAVE SCORED DATA
# =========================================================

def save_scored_predictions(
    df,
    score_column,
    output_path
):

    output = df.copy()

    # -----------------------------------------------------
    # Rank within each day.
    # -----------------------------------------------------

    output[
        "prediction_rank"
    ] = (

        output
        .groupby("Date")[
            score_column
        ]
        .rank(
            ascending=False,
            method="first"
        )

    )

    # -----------------------------------------------------
    # Percentile within each day.
    # -----------------------------------------------------

    output[
        "prediction_percentile"
    ] = (

        output
        .groupby("Date")[
            score_column
        ]
        .rank(
            pct=True
        )

    )

    output[
        "ensemble_score"
    ] = output[
        score_column
    ]

    columns = [

        "Date",

        "Ticker",

        "actual_return",

        "lstm_z",

        "xgb_z",

        "ensemble_score",

        "prediction_rank",

        "prediction_percentile",

    ]

    output = output[
        columns
    ].copy()

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved ensemble predictions to:"
        f"\n{output_path}"
    )

    return output


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "CROSS-SECTIONAL "
        "LSTM + XGBOOST ENSEMBLE"
    )
    print("=" * 70)

    # =====================================================
    # 1. LOAD VALIDATION PREDICTIONS
    # =====================================================

    print("\n" + "=" * 70)
    print("LOADING VALIDATION PREDICTIONS")
    print("=" * 70)

    lstm_validation = load_predictions(
        LSTM_VALIDATION_PATH
    )

    xgb_validation = load_predictions(
        XGB_VALIDATION_PATH
    )

    print(
        f"\nLSTM validation rows : "
        f"{len(lstm_validation):,}"
    )

    print(
        f"XGB validation rows  : "
        f"{len(xgb_validation):,}"
    )

    # =====================================================
    # 2. MERGE VALIDATION
    # =====================================================

    validation = merge_models(

        lstm_validation,

        xgb_validation

    )

    print(
        f"\nValidation dates:"
        f"\n{validation['Date'].min().date()} "
        f"→ "
        f"{validation['Date'].max().date()}"
    )

    print(
        f"Validation tickers: "
        f"{validation['Ticker'].nunique()}"
    )

    # =====================================================
    # 3. VALIDATION WEIGHT SEARCH
    # =====================================================

    (
        best_row,
        weight_results

    ) = search_validation_weights(
        validation
    )

    weight_results.to_csv(

        WEIGHT_RESULTS_PATH,

        index=False

    )

    best_lstm_weight = float(
        best_row["lstm_weight"]
    )

    best_xgb_weight = float(
        best_row["xgb_weight"]
    )

    # =====================================================
    # 4. BEST VALIDATION WEIGHT
    # =====================================================

    print("\n" + "=" * 70)
    print("BEST VALIDATION WEIGHT")
    print("=" * 70)

    print(
        f"\nLSTM weight : "
        f"{best_lstm_weight:.1f}"
    )

    print(
        f"XGB weight  : "
        f"{best_xgb_weight:.1f}"
    )

    print(
        f"Validation IC : "
        f"{best_row['ic']:.5f}"
    )

    print(
        f"Validation long-short : "
        f"{best_row['long_short_return']:.5%}"
    )

    print(
        f"Validation hit rate : "
        f"{best_row['hit_rate']:.2%}"
    )

    print(
        f"Validation Sharpe : "
        f"{best_row['sharpe']:.3f}"
    )

    print(
        f"\nWeight search saved to:"
        f"\n{WEIGHT_RESULTS_PATH}"
    )

    # =====================================================
    # 5. SCORE VALIDATION USING FROZEN WEIGHT
    # =====================================================

    validation_scored = (
        add_ensemble_score(
            validation,
            best_lstm_weight
        )
    )

    validation_metrics, _ = (
        calculate_portfolio_metrics(
            validation_scored,
            "ensemble_score"
        )
    )

    save_scored_predictions(

        validation_scored,

        "ensemble_score",

        ENSEMBLE_VALIDATION_PATH

    )

    # =====================================================
    # 6. LOAD TEST PREDICTIONS
    # =====================================================

    print("\n" + "=" * 70)
    print("LOADING TEST PREDICTIONS")
    print("=" * 70)

    lstm_test = load_predictions(
        LSTM_TEST_PATH
    )

    xgb_test = load_predictions(
        XGB_TEST_PATH
    )

    print(
        f"\nLSTM test rows : "
        f"{len(lstm_test):,}"
    )

    print(
        f"XGB test rows  : "
        f"{len(xgb_test):,}"
    )

    # =====================================================
    # 7. MERGE TEST
    # =====================================================

    test = merge_models(

        lstm_test,

        xgb_test

    )

    print(
        f"\nTest dates:"
        f"\n{test['Date'].min().date()} "
        f"→ "
        f"{test['Date'].max().date()}"
    )

    print(
        f"Test tickers: "
        f"{test['Ticker'].nunique()}"
    )

    # =====================================================
    # 8. APPLY FROZEN VALIDATION WEIGHT
    # =====================================================

    print("\n" + "=" * 70)
    print("APPLYING FROZEN ENSEMBLE WEIGHT")
    print("=" * 70)

    print(
        f"\nLSTM weight: "
        f"{best_lstm_weight:.1f}"
    )

    print(
        f"XGB weight: "
        f"{best_xgb_weight:.1f}"
    )

    test_scored = (
        add_ensemble_score(
            test,
            best_lstm_weight
        )
    )

    # =====================================================
    # 9. TEST MODEL COMPARISON
    # =====================================================

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    # -----------------------------------------------------
    # LSTM
    # -----------------------------------------------------

    lstm_test_metrics, _ = (
        calculate_portfolio_metrics(
            test_scored,
            "lstm_z"
        )
    )

    print_metrics(
        "LSTM",
        lstm_test_metrics
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    xgb_test_metrics, _ = (
        calculate_portfolio_metrics(
            test_scored,
            "xgb_z"
        )
    )

    print_metrics(
        "XGBoost",
        xgb_test_metrics
    )

    # -----------------------------------------------------
    # Ensemble
    # -----------------------------------------------------

    ensemble_test_metrics, _ = (
        calculate_portfolio_metrics(
            test_scored,
            "ensemble_score"
        )
    )

    print_metrics(
        "ENSEMBLE",
        ensemble_test_metrics
    )

    # =====================================================
    # 10. SAVE TEST ENSEMBLE
    # =====================================================

    save_scored_predictions(

        test_scored,

        "ensemble_score",

        ENSEMBLE_TEST_PATH

    )

    # =====================================================
    # 11. FINAL COMPARISON
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    comparison = pd.DataFrame({

        "Model": [
            "LSTM",
            "XGBoost",
            "Ensemble",
        ],

        "Mean_IC": [

            lstm_test_metrics[
                "ic_mean"
            ],

            xgb_test_metrics[
                "ic_mean"
            ],

            ensemble_test_metrics[
                "ic_mean"
            ],

        ],

        "Median_IC": [

            lstm_test_metrics[
                "ic_median"
            ],

            xgb_test_metrics[
                "ic_median"
            ],

            ensemble_test_metrics[
                "ic_median"
            ],

        ],

        "Top_2_Return": [

            lstm_test_metrics[
                "top_return"
            ],

            xgb_test_metrics[
                "top_return"
            ],

            ensemble_test_metrics[
                "top_return"
            ],

        ],

        "Bottom_2_Return": [

            lstm_test_metrics[
                "bottom_return"
            ],

            xgb_test_metrics[
                "bottom_return"
            ],

            ensemble_test_metrics[
                "bottom_return"
            ],

        ],

        "Long_Short_Return": [

            lstm_test_metrics[
                "long_short_return"
            ],

            xgb_test_metrics[
                "long_short_return"
            ],

            ensemble_test_metrics[
                "long_short_return"
            ],

        ],

        "Hit_Rate": [

            lstm_test_metrics[
                "hit_rate"
            ],

            xgb_test_metrics[
                "hit_rate"
            ],

            ensemble_test_metrics[
                "hit_rate"
            ],

        ],

        "Annualized_Sharpe": [

            lstm_test_metrics[
                "annualized_sharpe"
            ],

            xgb_test_metrics[
                "annualized_sharpe"
            ],

            ensemble_test_metrics[
                "annualized_sharpe"
            ],

        ],

    })

    print(
        "\n"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison_path = os.path.join(
        MODEL_DIR,
        "ensemble_test_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nComparison saved to:"
        f"\n{comparison_path}"
    )

    # =====================================================
    # 12. CONCLUSION
    # =====================================================

    print("\n" + "=" * 70)
    print("ENSEMBLE CONCLUSION")
    print("=" * 70)

    ensemble_ic = (
        ensemble_test_metrics[
            "ic_mean"
        ]
    )

    lstm_ic = (
        lstm_test_metrics[
            "ic_mean"
        ]
    )

    xgb_ic = (
        xgb_test_metrics[
            "ic_mean"
        ]
    )

    print(
        f"\nTest LSTM IC     : "
        f"{lstm_ic:.5f}"
    )

    print(
        f"Test XGBoost IC  : "
        f"{xgb_ic:.5f}"
    )

    print(
        f"Test Ensemble IC : "
        f"{ensemble_ic:.5f}"
    )

    if ensemble_ic > lstm_ic:

        print(
            "\nEnsemble improved "
            "over LSTM on test IC."
        )

    elif ensemble_ic < lstm_ic:

        print(
            "\nEnsemble did NOT improve "
            "over LSTM on test IC."
        )

        print(
            "LSTM may be the better "
            "standalone ranking model."
        )

    else:

        print(
            "\nEnsemble and LSTM have "
            "the same test IC."
        )

    print("\n" + "=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"\n{WEIGHT_RESULTS_PATH}"
    )

    print(
        f"{ENSEMBLE_VALIDATION_PATH}"
    )

    print(
        f"{ENSEMBLE_TEST_PATH}"
    )

    print(
        f"{comparison_path}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()