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

RESULTS_PATH = (
    f"{OUTPUT_DIR}/portfolio_sensitivity_results.csv"
)

DAILY_PATH = (
    f"{OUTPUT_DIR}/portfolio_sensitivity_daily.csv"
)


# =========================================================
# PORTFOLIO CONFIGURATION
# =========================================================

K_VALUES = [1, 2, 3, 4]

COST_VALUES = [
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
# LOAD PREDICTIONS
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
    # Prediction column
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
            "Prediction column not found."
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
    # Validation
    # -----------------------------------------------------

    if df.duplicated(
        subset=["Date", "Ticker"]
    ).any():

        raise ValueError(
            "Duplicate Date/Ticker rows found."
        )

    if df[
        [
            "prediction",
            "actual_return",
        ]
    ].isna().any().any():

        raise ValueError(
            "NaN values found."
        )

    values = df[
        [
            "prediction",
            "actual_return",
        ]
    ].to_numpy()

    if not np.isfinite(values).all():

        raise ValueError(
            "Infinite values found."
        )

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
# BUILD DAILY PORTFOLIO
# =========================================================

def build_portfolio(
    predictions,
    k
):

    portfolio_rows = []

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

        # -------------------------------------------------
        # Need at least 2K stocks.
        # -------------------------------------------------

        if len(group) < 2 * k:

            continue

        # -------------------------------------------------
        # Equal weighting within each side.
        #
        # Gross exposure:
        #
        # Long  = +0.50
        # Short = -0.50
        #
        # Therefore:
        #
        # Net exposure = 0
        # Gross exposure = 1.0
        #
        # This is much cleaner than using +0.50 per stock
        # and -0.50 per stock.
        # -------------------------------------------------

        long_weight = (
            0.5 / k
        )

        short_weight = (
            -0.5 / k
        )

        group["position"] = 0.0

        # -------------------------------------------------
        # LONG
        # -------------------------------------------------

        long_indices = (
            group
            .head(k)
            .index
        )

        group.loc[
            long_indices,
            "position"
        ] = long_weight

        # -------------------------------------------------
        # SHORT
        # -------------------------------------------------

        short_indices = (
            group
            .tail(k)
            .index
        )

        group.loc[
            short_indices,
            "position"
        ] = short_weight

        group["portfolio_side"] = "FLAT"

        group.loc[
            long_indices,
            "portfolio_side"
        ] = "LONG"

        group.loc[
            short_indices,
            "portfolio_side"
        ] = "SHORT"

        selected = group[
            group["position"] != 0
        ].copy()

        for _, row in selected.iterrows():

            portfolio_rows.append({

                "Date":
                    row["Date"],

                "Ticker":
                    row["Ticker"],

                "prediction":
                    row["prediction"],

                "position":
                    row["position"],

                "portfolio_side":
                    row["portfolio_side"],

                "actual_return":
                    row["actual_return"],

                "K":
                    k,

            })

    if len(portfolio_rows) == 0:

        raise RuntimeError(
            f"No portfolio observations "
            f"for K={k}."
        )

    return pd.DataFrame(
        portfolio_rows
    )


# =========================================================
# CALCULATE DAILY PORTFOLIO RETURNS
# =========================================================

def calculate_daily_returns(
    portfolio
):

    daily = []

    for date, group in portfolio.groupby(
        "Date",
        sort=True
    ):

        portfolio_return = np.sum(
            group["position"]
            * group["actual_return"]
        )

        long_group = group[
            group["position"] > 0
        ]

        short_group = group[
            group["position"] < 0
        ]

        if len(long_group) > 0:

            long_contribution = np.sum(
                long_group["position"]
                * long_group["actual_return"]
            )

        else:

            long_contribution = 0.0

        if len(short_group) > 0:

            short_contribution = np.sum(
                short_group["position"]
                * short_group["actual_return"]
            )

        else:

            short_contribution = 0.0

        market_return = (
            group["actual_return"]
            .mean()
        )

        daily.append({

            "Date":
                date,

            "gross_return":
                portfolio_return,

            "long_contribution":
                long_contribution,

            "short_contribution":
                short_contribution,

            "market_return":
                market_return,

        })

    return pd.DataFrame(
        daily
    )


# =========================================================
# TURNOVER
# =========================================================

def calculate_turnover(
    portfolio
):

    matrix = (
        portfolio
        .pivot_table(
            index="Date",
            columns="Ticker",
            values="position",
            fill_value=0.0
        )
        .sort_index()
    )

    previous = (
        matrix
        .shift(1)
        .fillna(0.0)
    )

    turnover = (
        matrix
        - previous
    ).abs().sum(axis=1)

    # Opening portfolio.
    first_date = matrix.index[0]

    turnover.loc[
        first_date
    ] = (
        matrix
        .loc[first_date]
        .abs()
        .sum()
    )

    return pd.DataFrame({

        "Date":
            turnover.index,

        "turnover":
            turnover.values,

    })


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    daily,
    cost
):

    data = daily.copy()

    data[
        "transaction_cost"
    ] = (
        data["turnover"]
        * cost
    )

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
        np.isfinite(returns)
    ]

    gross_returns = gross_returns[
        np.isfinite(gross_returns)
    ]

    if len(returns) < 2:

        raise RuntimeError(
            "Not enough observations."
        )

    n_days = len(returns)

    # =====================================================
    # EQUITY
    # =====================================================

    equity = np.cumprod(
        1.0 + returns
    )

    final_value = equity[-1]

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

    daily_vol = np.std(
        returns,
        ddof=1
    )

    annualized_vol = (
        daily_vol
        * np.sqrt(252)
    )

    # =====================================================
    # SHARPE
    # =====================================================

    if daily_vol > 1e-12:

        sharpe = (
            np.mean(returns)
            / daily_vol
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

        downside_dev = np.sqrt(
            np.mean(
                negative_returns ** 2
            )
        )

    else:

        downside_dev = np.nan

    if (
        np.isfinite(downside_dev)
        and downside_dev > 1e-12
    ):

        sortino = (
            np.mean(returns)
            / downside_dev
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
        data["transaction_cost"]
        .sum()
    )

    gross_total_return = (
        np.prod(
            1.0 + gross_returns
        )
        - 1.0
    )

    return {

        "cost_bps":
            cost * 10000.0,

        "days":
            n_days,

        "gross_total_return":
            gross_total_return,

        "net_total_return":
            total_return,

        "CAGR":
            cagr,

        "annualized_volatility":
            annualized_vol,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "max_drawdown":
            max_drawdown,

        "hit_rate":
            hit_rate,

        "average_turnover":
            average_turnover,

        "total_turnover":
            total_turnover,

        "transaction_cost":
            total_transaction_cost,

    }, data


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)
    print(
        "PORTFOLIO CONSTRUCTION "
        "SENSITIVITY ANALYSIS"
    )
    print("=" * 70)

    predictions = load_predictions()

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

    all_results = []
    all_daily = []

    # =====================================================
    # LOOP OVER K
    # =====================================================

    for k in K_VALUES:

        print("\n" + "=" * 70)
        print(
            f"K = {k}"
        )
        print("=" * 70)

        portfolio = build_portfolio(
            predictions,
            k
        )

        daily_returns = (
            calculate_daily_returns(
                portfolio
            )
        )

        turnover = (
            calculate_turnover(
                portfolio
            )
        )

        daily = pd.merge(

            daily_returns,

            turnover,

            on="Date",

            how="inner",

            validate="one_to_one"

        )

        # -------------------------------------------------
        # Cost scenarios
        # -------------------------------------------------

        for cost in COST_VALUES:

            metrics, backtest = (
                calculate_metrics(
                    daily,
                    cost
                )
            )

            metrics["K"] = k

            all_results.append(
                metrics
            )

            # -------------------------------------------------
            # Store daily data for base 10bps case.
            # -------------------------------------------------

            if abs(
                cost - 0.001
            ) < 1e-12:

                daily_saved = backtest.copy()

                daily_saved[
                    "K"
                ] = k

                all_daily.append(
                    daily_saved
                )

            print(

                f"Cost={cost * 10000:.0f} bps | "

                f"CAGR="
                f"{metrics['CAGR']:.2%} | "

                f"Sharpe="
                f"{metrics['Sharpe']:.3f} | "

                f"Sortino="
                f"{metrics['Sortino']:.3f} | "

                f"MaxDD="
                f"{metrics['max_drawdown']:.2%} | "

                f"Turnover="
                f"{metrics['average_turnover']:.2%}"

            )

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results = pd.DataFrame(
        all_results
    )

    results = results[
        [
            "K",
            "cost_bps",
            "days",
            "gross_total_return",
            "net_total_return",
            "CAGR",
            "annualized_volatility",
            "Sharpe",
            "Sortino",
            "max_drawdown",
            "hit_rate",
            "average_turnover",
            "total_turnover",
            "transaction_cost",
        ]
    ]

    results = (
        results
        .sort_values(
            [
                "cost_bps",
                "K"
            ]
        )
        .reset_index(drop=True)
    )

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    # =====================================================
    # DAILY RESULTS
    # =====================================================

    if len(all_daily) > 0:

        daily_results = pd.concat(
            all_daily,
            ignore_index=True
        )

        daily_results = (
            daily_results
            .sort_values(
                [
                    "K",
                    "Date"
                ]
            )
            .reset_index(drop=True)
        )

        daily_results.to_csv(
            DAILY_PATH,
            index=False
        )

    # =====================================================
    # 10 BPS COMPARISON
    # =====================================================

    ten_bps = results[
        results["cost_bps"] == 10.0
    ].copy()

    ten_bps = (
        ten_bps
        .sort_values(
            "Sharpe",
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("10 BPS COMPARISON")
    print("=" * 70)

    print(
        ten_bps[
            [
                "K",
                "CAGR",
                "Sharpe",
                "Sortino",
                "max_drawdown",
                "hit_rate",
                "average_turnover",
                "net_total_return",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # BEST K
    # =====================================================

    best = (
        ten_bps
        .iloc[0]
    )

    print("\n" + "=" * 70)
    print("BEST PORTFOLIO CONFIGURATION")
    print("=" * 70)

    print(
        f"\nK = "
        f"{int(best['K'])}"
    )

    print(
        f"10 bps CAGR = "
        f"{best['CAGR']:.2%}"
    )

    print(
        f"10 bps Sharpe = "
        f"{best['Sharpe']:.3f}"
    )

    print(
        f"10 bps Sortino = "
        f"{best['Sortino']:.3f}"
    )

    print(
        f"10 bps Max Drawdown = "
        f"{best['max_drawdown']:.2%}"
    )

    print(
        f"10 bps Hit Rate = "
        f"{best['hit_rate']:.2%}"
    )

    print(
        f"Average Turnover = "
        f"{best['average_turnover']:.2%}"
    )

    # =====================================================
    # FILES
    # =====================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\n{RESULTS_PATH}"
    )

    print(
        f"{DAILY_PATH}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()