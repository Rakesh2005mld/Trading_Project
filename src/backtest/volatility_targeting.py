import os
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

LSTM_VALIDATION_PATH = (
    "models/cross_sectional_lstm_validation_predictions.csv"
)

LSTM_TEST_PATH = (
    "models/cross_sectional_lstm_predictions.csv"
)

OUTPUT_DIR = "models"

RESULTS_PATH = (
    f"{OUTPUT_DIR}/volatility_targeting_results.csv"
)

DAILY_PATH = (
    f"{OUTPUT_DIR}/volatility_targeting_daily.csv"
)

POSITION_PATH = (
    f"{OUTPUT_DIR}/volatility_targeting_positions.csv"
)


# =========================================================
# PORTFOLIO CONFIGURATION
# =========================================================

TOP_K = 2
BOTTOM_K = 2

# Gross exposure = 1.0
#
# 2 longs:
#   +0.25
#   +0.25
#
# 2 shorts:
#   -0.25
#   -0.25
#
# Net exposure = 0
# Gross exposure = 1

LONG_WEIGHT = 0.5 / TOP_K
SHORT_WEIGHT = -0.5 / BOTTOM_K


# =========================================================
# VOLATILITY TARGETING
# =========================================================

VOLATILITY_LOOKBACK = 20

TARGET_VOLATILITIES = [
    0.10,
    0.15,
    0.20,
    0.25,
]

# Exposure multiplier limits.

MIN_EXPOSURE_MULTIPLIER = 0.25
MAX_EXPOSURE_MULTIPLIER = 1.50


# =========================================================
# TRANSACTION COST
# =========================================================

TRANSACTION_COST = 0.001   # 10 bps


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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

    required_columns = [
        "Date",
        "Ticker",
        "actual_return",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column "
                f"'{column}' in {path}"
            )

    # -----------------------------------------------------
    # Detect prediction column.
    # -----------------------------------------------------

    if (
        "predicted_cross_sectional_target"
        in df.columns
    ):

        prediction_column = (
            "predicted_cross_sectional_target"
        )

    elif (
        "prediction"
        in df.columns
    ):

        prediction_column = "prediction"

    else:

        raise ValueError(
            f"No prediction column found in {path}"
        )

    df = df.rename(
        columns={
            prediction_column:
                "prediction"
        }
    )

    df = df[
        [
            "Date",
            "Ticker",
            "prediction",
            "actual_return",
        ]
    ].copy()

    # -----------------------------------------------------
    # Duplicate check.
    # -----------------------------------------------------

    duplicate_count = (
        df
        .duplicated(
            subset=[
                "Date",
                "Ticker"
            ]
        )
        .sum()
    )

    if duplicate_count > 0:

        raise ValueError(
            f"{path} contains "
            f"{duplicate_count} duplicate "
            f"Date/Ticker rows."
        )

    # -----------------------------------------------------
    # Missing values.
    # -----------------------------------------------------

    if df[
        [
            "prediction",
            "actual_return",
        ]
    ].isna().any().any():

        raise ValueError(
            f"{path} contains NaN values."
        )

    # -----------------------------------------------------
    # Infinite values.
    # -----------------------------------------------------

    numeric = df[
        [
            "prediction",
            "actual_return",
        ]
    ].to_numpy()

    if not np.isfinite(
        numeric
    ).all():

        raise ValueError(
            f"{path} contains infinite values."
        )

    # -----------------------------------------------------
    # Sort.
    # -----------------------------------------------------

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

    return df


# =========================================================
# CREATE BASE PORTFOLIO
# =========================================================

def create_base_portfolio(
    predictions
):

    rows = []

    for date, group in predictions.groupby(
        "Date",
        sort=True
    ):

        group = (
            group
            .sort_values(
                "prediction",
                ascending=False
            )
            .reset_index(drop=True)
        )

        if len(group) < (
            TOP_K + BOTTOM_K
        ):

            continue

        # -------------------------------------------------
        # Long positions.
        # -------------------------------------------------

        long_indices = (
            group
            .head(TOP_K)
            .index
        )

        # -------------------------------------------------
        # Short positions.
        # -------------------------------------------------

        short_indices = (
            group
            .tail(BOTTOM_K)
            .index
        )

        group["base_position"] = 0.0

        group.loc[
            long_indices,
            "base_position"
        ] = LONG_WEIGHT

        group.loc[
            short_indices,
            "base_position"
        ] = SHORT_WEIGHT

        selected = group[
            group["base_position"] != 0
        ].copy()

        for _, row in selected.iterrows():

            side = (
                "LONG"
                if row["base_position"] > 0
                else "SHORT"
            )

            rows.append({

                "Date":
                    row["Date"],

                "Ticker":
                    row["Ticker"],

                "prediction":
                    row["prediction"],

                "base_position":
                    row["base_position"],

                "portfolio_side":
                    side,

                "actual_return":
                    row["actual_return"],

            })

    if len(rows) == 0:

        raise RuntimeError(
            "No valid portfolio observations."
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# BASE PORTFOLIO DAILY RETURNS
# =========================================================

def calculate_base_daily_returns(
    portfolio
):

    rows = []

    for date, group in portfolio.groupby(
        "Date",
        sort=True
    ):

        portfolio_return = np.sum(
            group["base_position"]
            * group["actual_return"]
        )

        rows.append({

            "Date":
                date,

            "base_return":
                portfolio_return,

        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# REALIZED VOLATILITY
# =========================================================

def calculate_lagged_realized_volatility(
    daily_returns,
    lookback=VOLATILITY_LOOKBACK
):

    daily = daily_returns.copy()

    # -----------------------------------------------------
    # Volatility estimated using ONLY returns that were
    # already known before the current date.
    #
    # Example:
    #
    # today's exposure uses the previous 20 days.
    #
    # shift(1) is essential.
    # -----------------------------------------------------

    daily[
        "realized_volatility"
    ] = (

        daily[
            "base_return"
        ]
        .rolling(
            window=lookback,
            min_periods=lookback
        )
        .std(
            ddof=1
        )
        .shift(1)
        * np.sqrt(252)

    )

    return daily


# =========================================================
# APPLY VOLATILITY TARGETING
# =========================================================

def apply_volatility_targeting(
    daily,
    target_volatility
):

    data = daily.copy()

    # -----------------------------------------------------
    # Exposure multiplier:
    #
    # target vol
    # -----------
    # realized vol
    # -----------------------------------------------------

    multiplier = (
        target_volatility
        / data[
            "realized_volatility"
        ]
    )

    # -----------------------------------------------------
    # Initial period does not have enough history.
    #
    # Keep normal 1.0x exposure rather than using future
    # data.
    # -----------------------------------------------------

    multiplier = (
        multiplier
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(1.0)
    )

    # -----------------------------------------------------
    # Exposure limits.
    # -----------------------------------------------------

    multiplier = (
        multiplier
        .clip(
            lower=MIN_EXPOSURE_MULTIPLIER,
            upper=MAX_EXPOSURE_MULTIPLIER
        )
    )

    data[
        "exposure_multiplier"
    ] = multiplier

    # -----------------------------------------------------
    # Targeted portfolio return.
    # -----------------------------------------------------

    data[
        "gross_return"
    ] = (
        data[
            "base_return"
        ]
        * data[
            "exposure_multiplier"
        ]
    )

    return data


# =========================================================
# CALCULATE TURNOVER
# =========================================================

def calculate_turnover(
    portfolio,
    exposure_series
):

    # -----------------------------------------------------
    # Base position matrix.
    # -----------------------------------------------------

    position_matrix = (
        portfolio
        .pivot_table(
            index="Date",
            columns="Ticker",
            values="base_position",
            fill_value=0.0
        )
        .sort_index()
    )

    # -----------------------------------------------------
    # Exposure multipliers aligned by date.
    # -----------------------------------------------------

    exposure = (
        exposure_series
        .set_index("Date")[
            "exposure_multiplier"
        ]
        .reindex(
            position_matrix.index
        )
        .fillna(1.0)
    )

    # -----------------------------------------------------
    # Actual target positions after volatility scaling.
    # -----------------------------------------------------

    targeted_positions = (
        position_matrix
        .multiply(
            exposure,
            axis=0
        )
    )

    # -----------------------------------------------------
    # Previous positions.
    # -----------------------------------------------------

    previous_positions = (
        targeted_positions
        .shift(1)
        .fillna(0.0)
    )

    changes = (
        targeted_positions
        - previous_positions
    )

    # -----------------------------------------------------
    # Opening portfolio.
    # -----------------------------------------------------

    first_date = (
        targeted_positions.index[0]
    )

    changes.loc[
        first_date
    ] = (
        targeted_positions
        .loc[first_date]
    )

    turnover = (
        changes
        .abs()
        .sum(axis=1)
    )

    return pd.DataFrame({

        "Date":
            turnover.index,

        "turnover":
            turnover.values,

    })


# =========================================================
# PERFORMANCE METRICS
# =========================================================

def calculate_metrics(
    daily,
    target_volatility,
    cost_rate
):

    data = daily.copy()

    # -----------------------------------------------------
    # Transaction cost.
    # -----------------------------------------------------

    data[
        "transaction_cost"
    ] = (
        data["turnover"]
        * cost_rate
    )

    # -----------------------------------------------------
    # Net return.
    # -----------------------------------------------------

    data[
        "net_return"
    ] = (
        data["gross_return"]
        - data["transaction_cost"]
    )

    returns = (
        data["net_return"]
        .to_numpy(
            dtype=np.float64
        )
    )

    gross_returns = (
        data["gross_return"]
        .to_numpy(
            dtype=np.float64
        )
    )

    returns = returns[
        np.isfinite(
            returns
        )
    ]

    gross_returns = gross_returns[
        np.isfinite(
            gross_returns
        )
    ]

    if len(returns) < 2:

        raise RuntimeError(
            "Not enough observations."
        )

    # =====================================================
    # TOTAL RETURN
    # =====================================================

    equity = np.cumprod(
        1.0 + returns
    )

    final_value = (
        equity[-1]
    )

    total_return = (
        final_value - 1.0
    )

    # =====================================================
    # CAGR
    # =====================================================

    years = (
        len(returns)
        / 252.0
    )

    if (
        final_value > 0
        and years > 0
    ):

        cagr = (
            final_value
            ** (1.0 / years)
        ) - 1.0

    else:

        cagr = np.nan

    # =====================================================
    # VOLATILITY
    # =====================================================

    daily_volatility = np.std(
        returns,
        ddof=1
    )

    annualized_volatility = (
        daily_volatility
        * np.sqrt(252)
    )

    # =====================================================
    # SHARPE
    # =====================================================

    if daily_volatility > 1e-12:

        sharpe = (
            np.mean(returns)
            / daily_volatility
        ) * np.sqrt(252)

    else:

        sharpe = np.nan

    # =====================================================
    # SORTINO
    # =====================================================

    negative_returns = returns[
        returns < 0
    ]

    if len(negative_returns) > 0:

        downside_deviation = np.sqrt(
            np.mean(
                negative_returns ** 2
            )
        )

    else:

        downside_deviation = np.nan

    if (
        np.isfinite(
            downside_deviation
        )
        and downside_deviation > 1e-12
    ):

        sortino = (
            np.mean(returns)
            / downside_deviation
        ) * np.sqrt(252)

    else:

        sortino = np.nan

    # =====================================================
    # MAX DRAWDOWN
    # =====================================================

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_drawdown = (
        np.min(drawdown)
    )

    # =====================================================
    # HIT RATE
    # =====================================================

    hit_rate = np.mean(
        returns > 0
    )

    # =====================================================
    # TURNOVER
    # =====================================================

    average_turnover = (
        data["turnover"]
        .mean()
    )

    total_turnover = (
        data["turnover"]
        .sum()
    )

    # =====================================================
    # COST
    # =====================================================

    total_transaction_cost = (
        data[
            "transaction_cost"
        ].sum()
    )

    # =====================================================
    # EXPOSURE
    # =====================================================

    average_exposure = (
        data[
            "exposure_multiplier"
        ].mean()
    )

    median_exposure = (
        data[
            "exposure_multiplier"
        ].median()
    )

    minimum_exposure = (
        data[
            "exposure_multiplier"
        ].min()
    )

    maximum_exposure = (
        data[
            "exposure_multiplier"
        ].max()
    )

    # =====================================================
    # BEST / WORST
    # =====================================================

    best_day = np.max(
        returns
    )

    worst_day = np.min(
        returns
    )

    # =====================================================
    # GROSS RETURN
    # =====================================================

    gross_total_return = (
        np.prod(
            1.0 + gross_returns
        )
        - 1.0
    )

    return {

        "target_volatility":
            target_volatility,

        "cost_bps":
            cost_rate * 10000.0,

        "days":
            len(returns),

        "gross_total_return":
            gross_total_return,

        "net_total_return":
            total_return,

        "CAGR":
            cagr,

        "annualized_volatility":
            annualized_volatility,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "max_drawdown":
            max_drawdown,

        "hit_rate":
            hit_rate,

        "best_day":
            best_day,

        "worst_day":
            worst_day,

        "average_turnover":
            average_turnover,

        "total_turnover":
            total_turnover,

        "transaction_cost":
            total_transaction_cost,

        "average_exposure":
            average_exposure,

        "median_exposure":
            median_exposure,

        "minimum_exposure":
            minimum_exposure,

        "maximum_exposure":
            maximum_exposure,

    }, data


# =========================================================
# RUN ONE PERIOD
# =========================================================

def build_period(
    predictions,
    target_volatility
):

    portfolio = create_base_portfolio(
        predictions
    )

    base_daily = (
        calculate_base_daily_returns(
            portfolio
        )
    )

    base_daily = (
        calculate_lagged_realized_volatility(
            base_daily
        )
    )

    targeted = (
        apply_volatility_targeting(
            base_daily,
            target_volatility
        )
    )

    turnover = (
        calculate_turnover(
            portfolio,
            targeted[
                [
                    "Date",
                    "exposure_multiplier"
                ]
            ]
        )
    )

    targeted = pd.merge(

        targeted,

        turnover,

        on="Date",

        how="inner",

        validate="one_to_one"

    )

    return (
        portfolio,
        targeted
    )


# =========================================================
# VALIDATION TARGET SEARCH
# =========================================================

def select_best_target(
    validation_predictions
):

    print("\n" + "=" * 70)
    print(
        "VALIDATION VOLATILITY TARGET SEARCH"
    )
    print("=" * 70)

    results = []

    # -----------------------------------------------------
    # Build base validation portfolio once.
    # -----------------------------------------------------

    validation_portfolio = (
        create_base_portfolio(
            validation_predictions
        )
    )

    validation_base_daily = (
        calculate_base_daily_returns(
            validation_portfolio
        )
    )

    validation_base_daily = (
        calculate_lagged_realized_volatility(
            validation_base_daily
        )
    )

    for target_volatility in TARGET_VOLATILITIES:

        targeted = (
            apply_volatility_targeting(

                validation_base_daily,

                target_volatility

            )
        )

        turnover = (
            calculate_turnover(

                validation_portfolio,

                targeted[
                    [
                        "Date",
                        "exposure_multiplier"
                    ]
                ]

            )
        )

        targeted = pd.merge(

            targeted,

            turnover,

            on="Date",

            how="inner",

            validate="one_to_one"

        )

        metrics, _ = (
            calculate_metrics(

                targeted,

                target_volatility,

                TRANSACTION_COST

            )
        )

        results.append(
            metrics
        )

        print(
            f"\nTarget = "
            f"{target_volatility:.0%}"
        )

        print(
            f"CAGR       : "
            f"{metrics['CAGR']:.2%}"
        )

        print(
            f"Sharpe     : "
            f"{metrics['Sharpe']:.3f}"
        )

        print(
            f"Sortino    : "
            f"{metrics['Sortino']:.3f}"
        )

        print(
            f"Max DD     : "
            f"{metrics['max_drawdown']:.2%}"
        )

        print(
            f"Turnover   : "
            f"{metrics['average_turnover']:.2%}"
        )

        print(
            f"Avg Exposure: "
            f"{metrics['average_exposure']:.3f}x"
        )

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Select based ONLY on validation Sharpe.
    # -----------------------------------------------------

    valid = results_df[
        np.isfinite(
            results_df["Sharpe"]
        )
    ].copy()

    if valid.empty:

        raise RuntimeError(
            "No valid volatility targets."
        )

    best = (
        valid
        .sort_values(
            "Sharpe",
            ascending=False
        )
        .iloc[0]
    )

    return (
        float(
            best["target_volatility"]
        ),
        results_df
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)
    print(
        "VALIDATION-SELECTED "
        "VOLATILITY TARGETING"
    )
    print("=" * 70)

    # =====================================================
    # 1. LOAD VALIDATION
    # =====================================================

    validation_predictions = load_predictions(
        LSTM_VALIDATION_PATH
    )

    print("\n" + "=" * 70)
    print("VALIDATION DATA")
    print("=" * 70)

    print(
        f"\nRows    : "
        f"{len(validation_predictions):,}"
    )

    print(
        f"Dates   : "
        f"{validation_predictions['Date'].nunique()}"
    )

    print(
        f"Tickers : "
        f"{validation_predictions['Ticker'].nunique()}"
    )

    print(
        f"Range   : "
        f"{validation_predictions['Date'].min().date()} "
        f"→ "
        f"{validation_predictions['Date'].max().date()}"
    )

    # =====================================================
    # 2. SELECT VOLATILITY TARGET
    # =====================================================

    (
        selected_target,
        validation_results
    ) = select_best_target(
        validation_predictions
    )

    print("\n" + "=" * 70)
    print("SELECTED VOLATILITY TARGET")
    print("=" * 70)

    print(
        f"\nSelected using VALIDATION only:"
        f"\n{selected_target:.0%}"
    )

    # =====================================================
    # SAVE VALIDATION SEARCH
    # =====================================================

    validation_results_path = os.path.join(
        OUTPUT_DIR,
        "volatility_target_validation_search.csv"
    )

    validation_results.to_csv(
        validation_results_path,
        index=False
    )

    # =====================================================
    # 3. LOAD TEST
    # =====================================================

    test_predictions = load_predictions(
        LSTM_TEST_PATH
    )

    print("\n" + "=" * 70)
    print("TEST DATA")
    print("=" * 70)

    print(
        f"\nRows    : "
        f"{len(test_predictions):,}"
    )

    print(
        f"Dates   : "
        f"{test_predictions['Date'].nunique()}"
    )

    print(
        f"Tickers : "
        f"{test_predictions['Ticker'].nunique()}"
    )

    print(
        f"Range   : "
        f"{test_predictions['Date'].min().date()} "
        f"→ "
        f"{test_predictions['Date'].max().date()}"
    )

    # =====================================================
    # 4. TEST WITH FROZEN TARGET
    # =====================================================

    (
        test_portfolio,
        test_daily
    ) = build_period(

        test_predictions,

        selected_target

    )

    # -----------------------------------------------------
    # Test metrics.
    # -----------------------------------------------------

    test_metrics, test_daily = (
        calculate_metrics(

            test_daily,

            selected_target,

            TRANSACTION_COST

        )
    )

    # =====================================================
    # 5. TEST RESULT
    # =====================================================

    print("\n" + "=" * 70)
    print("TEST WITH FROZEN VOLATILITY TARGET")
    print("=" * 70)

    print(
        f"\nTarget volatility : "
        f"{selected_target:.0%}"
    )

    print(
        f"Gross total return: "
        f"{test_metrics['gross_total_return']:.2%}"
    )

    print(
        f"Net total return  : "
        f"{test_metrics['net_total_return']:.2%}"
    )

    print(
        f"CAGR              : "
        f"{test_metrics['CAGR']:.2%}"
    )

    print(
        f"Annualized vol    : "
        f"{test_metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe            : "
        f"{test_metrics['Sharpe']:.3f}"
    )

    print(
        f"Sortino           : "
        f"{test_metrics['Sortino']:.3f}"
    )

    print(
        f"Maximum drawdown  : "
        f"{test_metrics['max_drawdown']:.2%}"
    )

    print(
        f"Hit rate          : "
        f"{test_metrics['hit_rate']:.2%}"
    )

    print(
        f"Best day          : "
        f"{test_metrics['best_day']:.2%}"
    )

    print(
        f"Worst day         : "
        f"{test_metrics['worst_day']:.2%}"
    )

    print(
        f"Average turnover  : "
        f"{test_metrics['average_turnover']:.2%}"
    )

    print(
        f"Total turnover    : "
        f"{test_metrics['total_turnover']:.2f}"
    )

    print(
        f"Transaction costs : "
        f"{test_metrics['transaction_cost']:.2%}"
    )

    print(
        f"Average exposure  : "
        f"{test_metrics['average_exposure']:.3f}x"
    )

    print(
        f"Median exposure   : "
        f"{test_metrics['median_exposure']:.3f}x"
    )

    print(
        f"Minimum exposure  : "
        f"{test_metrics['minimum_exposure']:.3f}x"
    )

    print(
        f"Maximum exposure  : "
        f"{test_metrics['maximum_exposure']:.3f}x"
    )

    # =====================================================
    # 6. CREATE EQUITY CURVE
    # =====================================================

    test_daily[
        "equity_curve"
    ] = np.cumprod(
        1.0
        + test_daily[
            "net_return"
        ]
    )

    test_daily[
        "high_water_mark"
    ] = (
        test_daily[
            "equity_curve"
        ]
        .cummax()
    )

    test_daily[
        "drawdown"
    ] = (

        test_daily[
            "equity_curve"
        ]

        / test_daily[
            "high_water_mark"
        ]

        - 1.0

    )

    # =====================================================
    # 7. SAVE DAILY RESULTS
    # =====================================================

    test_daily.to_csv(
        DAILY_PATH,
        index=False
    )

    # =====================================================
    # 8. SAVE POSITIONS
    # =====================================================

    exposure_lookup = test_daily[
        [
            "Date",
            "exposure_multiplier"
        ]
    ]

    positions = test_portfolio.merge(

        exposure_lookup,

        on="Date",

        how="left",

        validate="many_to_one"

    )

    positions[
        "targeted_position"
    ] = (

        positions[
            "base_position"
        ]

        *

        positions[
            "exposure_multiplier"
        ]

    )

    positions.to_csv(
        POSITION_PATH,
        index=False
    )

    # =====================================================
    # 9. SAVE FINAL SUMMARY
    # =====================================================

    summary = pd.DataFrame([
        test_metrics
    ])

    summary.to_csv(
        RESULTS_PATH,
        index=False
    )

    # =====================================================
    # 10. FINAL
    # =====================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nValidation search:"
        f"\n{validation_results_path}"
    )

    print(
        f"\nTest daily results:"
        f"\n{DAILY_PATH}"
    )

    print(
        f"\nTest positions:"
        f"\n{POSITION_PATH}"
    )

    print(
        f"\nTest summary:"
        f"\n{RESULTS_PATH}"
    )

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(
        "\nThe volatility target was selected "
        "using validation data only."
    )

    print(
        "The selected target was then frozen "
        "before evaluating the test period."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()