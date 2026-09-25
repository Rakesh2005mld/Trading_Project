import os
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

PREDICTIONS_PATH = (
    "models/cross_sectional_lstm_predictions.csv"
)

OUTPUT_DIR = "models"

BACKTEST_RESULTS_PATH = (
    f"{OUTPUT_DIR}/realistic_backtest_daily.csv"
)

POSITION_PATH = (
    f"{OUTPUT_DIR}/realistic_backtest_positions.csv"
)

SUMMARY_PATH = (
    f"{OUTPUT_DIR}/realistic_backtest_summary.csv"
)

# ---------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------

TOP_K = 2
BOTTOM_K = 2

# Long + Short gross exposure = 1.0
# Example:
#   2 longs  -> +0.25 each
#   2 shorts -> -0.25 each
LONG_WEIGHT = 1.0 / TOP_K
SHORT_WEIGHT = -1.0 / BOTTOM_K

# ---------------------------------------------------------
# Transaction cost
#
# 0.001 = 10 basis points
# ---------------------------------------------------------

TRANSACTION_COST = 0.001

# ---------------------------------------------------------
# Additional cost sensitivity analysis
# ---------------------------------------------------------

COST_SCENARIOS = [
    0.0000,   # 0 bps
    0.0005,   # 5 bps
    0.0010,   # 10 bps
    0.0020,   # 20 bps
]


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

def load_predictions():

    if not os.path.exists(
        PREDICTIONS_PATH
    ):

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH,
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
                f"Missing required column: "
                f"{column}"
            )

    # -----------------------------------------------------
    # Find prediction column.
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
            "No prediction column found."
        )

    df = df.rename(
        columns={
            prediction_column:
                "prediction"
        }
    )

    # -----------------------------------------------------
    # Keep only required information.
    # -----------------------------------------------------

    df = df[
        [
            "Date",
            "Ticker",
            "prediction",
            "actual_return",
        ]
    ].copy()

    # -----------------------------------------------------
    # Remove duplicates.
    # -----------------------------------------------------

    duplicates = df.duplicated(
        subset=["Date", "Ticker"]
    ).sum()

    if duplicates > 0:

        raise ValueError(
            f"Found {duplicates} duplicate "
            f"Date/Ticker rows."
        )

    # -----------------------------------------------------
    # Sort.
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Check missing values.
    # -----------------------------------------------------

    if df[
        [
            "prediction",
            "actual_return",
        ]
    ].isna().any().any():

        raise ValueError(
            "Prediction data contains NaN values."
        )

    # -----------------------------------------------------
    # Check infinity.
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
            "Prediction data contains "
            "infinite values."
        )

    return df


# =========================================================
# CREATE DAILY TARGET POSITIONS
# =========================================================

def create_positions(
    predictions
):

    positions = []

    # -----------------------------------------------------
    # For every date:
    #
    # Top 2  -> LONG
    # Bottom 2 -> SHORT
    #
    # Equal weight.
    # -----------------------------------------------------

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
        # Start with zero position.
        # -------------------------------------------------

        group["position"] = 0.0

        # -------------------------------------------------
        # Long positions.
        # -------------------------------------------------

        long_indices = (
            group
            .head(TOP_K)
            .index
        )

        group.loc[
            long_indices,
            "position"
        ] = LONG_WEIGHT

        # -------------------------------------------------
        # Short positions.
        # -------------------------------------------------

        short_indices = (
            group
            .tail(BOTTOM_K)
            .index
        )

        group.loc[
            short_indices,
            "position"
        ] = SHORT_WEIGHT

        group["portfolio_side"] = "FLAT"

        group.loc[
            long_indices,
            "portfolio_side"
        ] = "LONG"

        group.loc[
            short_indices,
            "portfolio_side"
        ] = "SHORT"

        positions.append(
            group[
                [
                    "Date",
                    "Ticker",
                    "prediction",
                    "position",
                    "portfolio_side",
                    "actual_return",
                ]
            ]
        )

    if len(positions) == 0:

        raise RuntimeError(
            "No valid portfolio dates found."
        )

    positions = pd.concat(
        positions,
        ignore_index=True
    )

    return positions


# =========================================================
# ADD TURNOVER
# =========================================================

def calculate_turnover(
    positions
):

    positions = positions.copy()

    # -----------------------------------------------------
    # Turnover is based on absolute change in position.
    #
    # Example:
    #
    # yesterday +0.25
    # today     -0.25
    #
    # turnover = 0.50
    # -----------------------------------------------------

    position_matrix = (
        positions
        .pivot_table(
            index="Date",
            columns="Ticker",
            values="position",
            fill_value=0.0
        )
        .sort_index()
    )

    # -----------------------------------------------------
    # Previous day positions.
    # -----------------------------------------------------

    previous_positions = (
        position_matrix
        .shift(1)
        .fillna(0.0)
    )

    # -----------------------------------------------------
    # Absolute position change.
    # -----------------------------------------------------

    turnover = (
        position_matrix
        - previous_positions
    ).abs().sum(axis=1)

    # -----------------------------------------------------
    # First day requires opening all positions.
    # -----------------------------------------------------

    first_date = (
        position_matrix.index[0]
    )

    turnover.loc[
        first_date
    ] = (
        position_matrix
        .loc[first_date]
        .abs()
        .sum()
    )

    turnover_df = pd.DataFrame({
        "Date": turnover.index,
        "turnover": turnover.values,
    })

    return turnover_df


# =========================================================
# CALCULATE GROSS DAILY RETURN
# =========================================================

def calculate_daily_returns(
    positions
):

    daily_returns = []

    for date, group in positions.groupby(
        "Date",
        sort=True
    ):

        # -------------------------------------------------
        # Portfolio P&L:
        #
        # weight * realized return
        # -------------------------------------------------

        portfolio_return = np.sum(
            group["position"]
            * group["actual_return"]
        )

        # -------------------------------------------------
        # Gross long contribution.
        # -------------------------------------------------

        long_group = group[
            group["position"] > 0
        ]

        if len(long_group) > 0:

            long_return = np.sum(
                long_group["position"]
                * long_group["actual_return"]
            )

        else:

            long_return = 0.0

        # -------------------------------------------------
        # Short contribution.
        # -------------------------------------------------

        short_group = group[
            group["position"] < 0
        ]

        if len(short_group) > 0:

            short_return = np.sum(
                short_group["position"]
                * short_group["actual_return"]
            )

        else:

            short_return = 0.0

        # -------------------------------------------------
        # Market return.
        # -------------------------------------------------

        market_return = (
            group["actual_return"]
            .mean()
        )

        daily_returns.append({

            "Date": date,

            "gross_return":
                portfolio_return,

            "long_contribution":
                long_return,

            "short_contribution":
                short_return,

            "market_return":
                market_return,

        })

    return pd.DataFrame(
        daily_returns
    )


# =========================================================
# PERFORMANCE METRICS
# =========================================================

def calculate_metrics(
    daily_returns,
    cost_rate
):

    data = daily_returns.copy()

    # -----------------------------------------------------
    # Transaction costs.
    #
    # Cost = turnover * cost rate
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

    # -----------------------------------------------------
    # Remove non-finite values.
    # -----------------------------------------------------

    returns = returns[
        np.isfinite(returns)
    ]

    gross_returns = gross_returns[
        np.isfinite(gross_returns)
    ]

    if len(returns) < 2:

        raise RuntimeError(
            "Not enough observations "
            "for performance analysis."
        )

    n_days = len(returns)

    # =====================================================
    # TOTAL RETURN
    # =====================================================

    cumulative_curve = np.cumprod(
        1.0 + returns
    )

    final_value = (
        cumulative_curve[-1]
    )

    total_return = (
        final_value - 1.0
    )

    # =====================================================
    # CAGR
    # =====================================================

    years = (
        n_days / 252.0
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

    downside_returns = returns[
        returns < 0
    ]

    if len(downside_returns) > 0:

        downside_deviation = (
            np.sqrt(
                np.mean(
                    downside_returns ** 2
                )
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
        cumulative_curve
    )

    drawdown = (
        cumulative_curve
        / running_max
        - 1.0
    )

    max_drawdown = (
        np.min(drawdown)
    )

    # =====================================================
    # CALMAR
    # =====================================================

    if (
        np.isfinite(cagr)
        and max_drawdown < 0
    ):

        calmar = (
            cagr
            / abs(max_drawdown)
        )

    else:

        calmar = np.nan

    # =====================================================
    # HIT RATE
    # =====================================================

    hit_rate = np.mean(
        returns > 0
    )

    # =====================================================
    # AVERAGE RETURN
    # =====================================================

    mean_daily_return = (
        np.mean(returns)
    )

    median_daily_return = (
        np.median(returns)
    )

    # =====================================================
    # BEST / WORST DAY
    # =====================================================

    best_day = np.max(
        returns
    )

    worst_day = np.min(
        returns
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
        data["transaction_cost"]
        .sum()
    )

    average_transaction_cost = (
        data["transaction_cost"]
        .mean()
    )

    return {

        "cost_bps":
            cost_rate * 10000.0,

        "days":
            n_days,

        "total_return":
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

        "Calmar":
            calmar,

        "hit_rate":
            hit_rate,

        "mean_daily_return":
            mean_daily_return,

        "median_daily_return":
            median_daily_return,

        "best_day":
            best_day,

        "worst_day":
            worst_day,

        "average_turnover":
            average_turnover,

        "total_turnover":
            total_turnover,

        "total_transaction_cost":
            total_transaction_cost,

        "average_transaction_cost":
            average_transaction_cost,

        "gross_total_return":
            (
                np.prod(
                    1.0
                    + gross_returns
                )
                - 1.0
            ),

    }, data


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)
    print(
        "REALISTIC CROSS-SECTIONAL "
        "LSTM BACKTEST"
    )
    print("=" * 70)

    # =====================================================
    # 1. LOAD
    # =====================================================

    predictions = load_predictions()

    print("\n" + "=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        f"\nRows       : "
        f"{len(predictions):,}"
    )

    print(
        f"Dates      : "
        f"{predictions['Date'].nunique()}"
    )

    print(
        f"Tickers    : "
        f"{predictions['Ticker'].nunique()}"
    )

    print(
        f"Date range : "
        f"{predictions['Date'].min().date()} "
        f"→ "
        f"{predictions['Date'].max().date()}"
    )

    # =====================================================
    # 2. CREATE POSITIONS
    # =====================================================

    positions = create_positions(
        predictions
    )

    print("\n" + "=" * 70)
    print("PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    print(
        f"\nTop-K    : {TOP_K}"
    )

    print(
        f"Bottom-K : {BOTTOM_K}"
    )

    print(
        f"Long weight per stock : "
        f"{LONG_WEIGHT:.4f}"
    )

    print(
        f"Short weight per stock: "
        f"{SHORT_WEIGHT:.4f}"
    )

    print(
        f"\nPortfolio dates: "
        f"{positions['Date'].nunique()}"
    )

    # =====================================================
    # 3. TURNOVER
    # =====================================================

    turnover = calculate_turnover(
        positions
    )

    # =====================================================
    # 4. DAILY RETURNS
    # =====================================================

    daily_returns = calculate_daily_returns(
        positions
    )

    # -----------------------------------------------------
    # Merge turnover.
    # -----------------------------------------------------

    daily_returns = pd.merge(

        daily_returns,

        turnover,

        on="Date",

        how="left",

        validate="one_to_one"

    )

    if daily_returns["turnover"].isna().any():

        raise RuntimeError(
            "Missing turnover values."
        )

    # =====================================================
    # 5. BASE CASE: 10 BPS
    # =====================================================

    metrics, backtest_data = (
        calculate_metrics(

            daily_returns,

            TRANSACTION_COST

        )
    )

    print("\n" + "=" * 70)
    print(
        f"BASE CASE "
        f"({TRANSACTION_COST * 10000:.0f} BPS)"
    )
    print("=" * 70)

    print(
        f"\nGross total return       : "
        f"{metrics['gross_total_return']:.2%}"
    )

    print(
        f"Net total return         : "
        f"{metrics['total_return']:.2%}"
    )

    print(
        f"CAGR                     : "
        f"{metrics['CAGR']:.2%}"
    )

    print(
        f"Annualized volatility    : "
        f"{metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe                   : "
        f"{metrics['Sharpe']:.3f}"
    )

    print(
        f"Sortino                  : "
        f"{metrics['Sortino']:.3f}"
    )

    print(
        f"Maximum drawdown         : "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Calmar                   : "
        f"{metrics['Calmar']:.3f}"
    )

    print(
        f"Hit rate                 : "
        f"{metrics['hit_rate']:.2%}"
    )

    print(
        f"Mean daily return        : "
        f"{metrics['mean_daily_return']:.5%}"
    )

    print(
        f"Median daily return      : "
        f"{metrics['median_daily_return']:.5%}"
    )

    print(
        f"Best day                 : "
        f"{metrics['best_day']:.2%}"
    )

    print(
        f"Worst day                : "
        f"{metrics['worst_day']:.2%}"
    )

    print(
        f"Average daily turnover   : "
        f"{metrics['average_turnover']:.2%}"
    )

    print(
        f"Total turnover           : "
        f"{metrics['total_turnover']:.2f}"
    )

    print(
        f"Total transaction cost   : "
        f"{metrics['total_transaction_cost']:.2%}"
    )

    # =====================================================
    # 6. COST SENSITIVITY
    # =====================================================

    print("\n" + "=" * 70)
    print("TRANSACTION COST SENSITIVITY")
    print("=" * 70)

    scenario_results = []

    for cost in COST_SCENARIOS:

        scenario_metrics, _ = (
            calculate_metrics(
                daily_returns,
                cost
            )
        )

        scenario_results.append(
            scenario_metrics
        )

        print(
            f"\nCost: "
            f"{cost * 10000:.0f} bps"
        )

        print(
            f"Net Return : "
            f"{scenario_metrics['total_return']:.2%}"
        )

        print(
            f"CAGR       : "
            f"{scenario_metrics['CAGR']:.2%}"
        )

        print(
            f"Sharpe     : "
            f"{scenario_metrics['Sharpe']:.3f}"
        )

        print(
            f"Sortino    : "
            f"{scenario_metrics['Sortino']:.3f}"
        )

        print(
            f"Max DD     : "
            f"{scenario_metrics['max_drawdown']:.2%}"
        )

    scenario_df = pd.DataFrame(
        scenario_results
    )

    # =====================================================
    # 7. EQUITY CURVE
    # =====================================================

    backtest_data[
        "equity_curve"
    ] = np.cumprod(
        1.0
        + backtest_data[
            "net_return"
        ]
    )

    backtest_data[
        "high_water_mark"
    ] = (
        backtest_data[
            "equity_curve"
        ]
        .cummax()
    )

    backtest_data[
        "drawdown"
    ] = (

        backtest_data[
            "equity_curve"
        ]
        / backtest_data[
            "high_water_mark"
        ]

        - 1.0

    )

    # =====================================================
    # 8. SAVE DAILY BACKTEST
    # =====================================================

    backtest_data.to_csv(
        BACKTEST_RESULTS_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nDaily backtest:"
        f"\n{BACKTEST_RESULTS_PATH}"
    )

    # =====================================================
    # 9. SAVE POSITIONS
    # =====================================================

    positions.to_csv(
        POSITION_PATH,
        index=False
    )

    print(
        f"\nPositions:"
        f"\n{POSITION_PATH}"
    )

    # =====================================================
    # 10. SAVE SUMMARY
    # =====================================================

    summary_df = pd.DataFrame(
        scenario_results
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        f"\nSummary:"
        f"\n{SUMMARY_PATH}"
    )

    # =====================================================
    # 11. FINAL
    # =====================================================

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(
        "\nThe LSTM model was NOT retrained."
    )

    print(
        "The existing test predictions were used."
    )

    print(
        "Positions are formed from the prediction "
        "and returns are taken from future_return."
    )

    print(
        f"\nBase transaction cost: "
        f"{TRANSACTION_COST * 10000:.0f} bps"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()