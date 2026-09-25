import os
import numpy as np
import pandas as pd
import yfinance as yf


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

VALIDATION_RESULTS_PATH = (
    f"{OUTPUT_DIR}/regime_filter_validation_results.csv"
)

TEST_RESULTS_PATH = (
    f"{OUTPUT_DIR}/regime_filter_test_results.csv"
)

TEST_DAILY_PATH = (
    f"{OUTPUT_DIR}/regime_filter_test_daily.csv"
)

TEST_POSITIONS_PATH = (
    f"{OUTPUT_DIR}/regime_filter_test_positions.csv"
)

REGIME_DATA_PATH = (
    f"{OUTPUT_DIR}/spy_vix_regime_data.csv"
)


# =========================================================
# PORTFOLIO
# =========================================================

TOP_K = 2
BOTTOM_K = 2

LONG_WEIGHT = 0.5 / TOP_K
SHORT_WEIGHT = -0.5 / BOTTOM_K


# =========================================================
# VOLATILITY TARGETING
# =========================================================

TARGET_VOLATILITY = 0.25

VOLATILITY_LOOKBACK = 20

MIN_EXPOSURE_MULTIPLIER = 0.25
MAX_EXPOSURE_MULTIPLIER = 1.50


# =========================================================
# TRANSACTION COST
# =========================================================

TRANSACTION_COST = 0.001     # 10 bps


# =========================================================
# SPY / VIX PARAMETERS
# =========================================================

SPY_MA_SHORT = 20
SPY_MA_LONG = 50

VIX_MA_SHORT = 20

VIX_HIGH_THRESHOLD = 25.0
VIX_EXTREME_THRESHOLD = 30.0


# =========================================================
# REGIME EXPOSURE MULTIPLIERS
# =========================================================

RISK_OFF_LEVELS = [
    0.00,
    0.25,
    0.50,
    0.75,
]


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD LSTM PREDICTIONS
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

    required = [
        "Date",
        "Ticker",
        "actual_return",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing column '{column}' "
                f"in {path}"
            )

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
            f"Prediction column not found in {path}"
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

    if df.duplicated(
        subset=[
            "Date",
            "Ticker"
        ]
    ).any():

        raise ValueError(
            f"Duplicate Date/Ticker rows in {path}"
        )

    if df.isna().any().any():

        raise ValueError(
            f"NaN values found in {path}"
        )

    numeric_values = df[
        [
            "prediction",
            "actual_return",
        ]
    ].to_numpy()

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            f"Non-finite values found in {path}"
        )

    return (
        df
        .sort_values(
            [
                "Date",
                "Ticker"
            ]
        )
        .reset_index(drop=True)
    )


# =========================================================
# DOWNLOAD SPY + VIX
# =========================================================

def download_regime_data(
    start_date,
    end_date
):

    print("\n" + "=" * 70)
    print("DOWNLOADING SPY + VIX DATA")
    print("=" * 70)

    # -----------------------------------------------------
    # Start sufficiently early so the first validation
    # date has enough history for moving averages.
    # -----------------------------------------------------

    start_timestamp = (
        pd.Timestamp(start_date)
        - pd.Timedelta(days=120)
    )

    end_timestamp = (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=1)
    )

    print(
        f"\nDownload range:"
        f"\n{start_timestamp.date()} "
        f"→ "
        f"{end_timestamp.date()}"
    )

    spy = yf.download(
        "SPY",
        start=start_timestamp.strftime("%Y-%m-%d"),
        end=end_timestamp.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )

    vix = yf.download(
        "^VIX",
        start=start_timestamp.strftime("%Y-%m-%d"),
        end=end_timestamp.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if spy.empty:

        raise RuntimeError(
            "SPY download returned no data."
        )

    if vix.empty:

        raise RuntimeError(
            "VIX download returned no data."
        )

    # -----------------------------------------------------
    # Handle possible MultiIndex columns.
    # -----------------------------------------------------

    if isinstance(
        spy.columns,
        pd.MultiIndex
    ):

        spy_close = spy[
            "Close"
        ].iloc[:, 0]

    else:

        spy_close = spy[
            "Close"
        ]

    if isinstance(
        vix.columns,
        pd.MultiIndex
    ):

        vix_close = vix[
            "Close"
        ].iloc[:, 0]

    else:

        vix_close = vix[
            "Close"
        ]

    regime = pd.DataFrame({

        "Date":
            pd.to_datetime(
                spy_close.index
            ).tz_localize(None),

        "SPY_Close":
            spy_close.to_numpy(
                dtype=np.float64
            ),

    })

    vix_frame = pd.DataFrame({

        "Date":
            pd.to_datetime(
                vix_close.index
            ).tz_localize(None),

        "VIX_Close":
            vix_close.to_numpy(
                dtype=np.float64
            ),

    })

    regime = pd.merge(
        regime,
        vix_frame,
        on="Date",
        how="inner"
    )

    regime = (
        regime
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    # =====================================================
    # MOVING AVERAGES
    # =====================================================

    regime[
        "SPY_MA20"
    ] = (
        regime["SPY_Close"]
        .rolling(
            SPY_MA_SHORT
        )
        .mean()
    )

    regime[
        "SPY_MA50"
    ] = (
        regime["SPY_Close"]
        .rolling(
            SPY_MA_LONG
        )
        .mean()
    )

    regime[
        "VIX_MA20"
    ] = (
        regime["VIX_Close"]
        .rolling(
            VIX_MA_SHORT
        )
        .mean()
    )

    # -----------------------------------------------------
    # VIX daily change.
    # -----------------------------------------------------

    regime[
        "VIX_Change"
    ] = (
        regime["VIX_Close"]
        .pct_change()
    )

    # =====================================================
    # LAG ALL REGIME INFORMATION
    # =====================================================
    #
    # Today's exposure is based on yesterday's known
    # market information.
    #
    # This is critical to avoid look-ahead.
    # =====================================================

    regime[
        "SPY_Close_lag1"
    ] = regime[
        "SPY_Close"
    ].shift(1)

    regime[
        "SPY_MA20_lag1"
    ] = regime[
        "SPY_MA20"
    ].shift(1)

    regime[
        "SPY_MA50_lag1"
    ] = regime[
        "SPY_MA50"
    ].shift(1)

    regime[
        "VIX_Close_lag1"
    ] = regime[
        "VIX_Close"
    ].shift(1)

    regime[
        "VIX_MA20_lag1"
    ] = regime[
        "VIX_MA20"
    ].shift(1)

    regime[
        "VIX_Change_lag1"
    ] = regime[
        "VIX_Change"
    ].shift(1)

    # =====================================================
    # REGIME FLAGS
    # =====================================================

    regime[
        "SPY_Below_MA20"
    ] = (

        regime[
            "SPY_Close_lag1"
        ]

        <

        regime[
            "SPY_MA20_lag1"
        ]

    )

    regime[
        "SPY_Below_MA50"
    ] = (

        regime[
            "SPY_Close_lag1"
        ]

        <

        regime[
            "SPY_MA50_lag1"
        ]

    )

    regime[
        "VIX_High"
    ] = (

        regime[
            "VIX_Close_lag1"
        ]

        >= VIX_HIGH_THRESHOLD

    )

    regime[
        "VIX_Extreme"
    ] = (

        regime[
            "VIX_Close_lag1"
        ]

        >= VIX_EXTREME_THRESHOLD

    )

    regime[
        "VIX_Rising"
    ] = (

        regime[
            "VIX_Change_lag1"
        ]

        > 0

    )

    regime[
        "VIX_Above_MA20"
    ] = (

        regime[
            "VIX_Close_lag1"
        ]

        >

        regime[
            "VIX_MA20_lag1"
        ]

    )

    regime = regime[
        [
            "Date",

            "SPY_Close",
            "SPY_MA20",
            "SPY_MA50",

            "VIX_Close",
            "VIX_MA20",
            "VIX_Change",

            "SPY_Close_lag1",
            "SPY_MA20_lag1",
            "SPY_MA50_lag1",

            "VIX_Close_lag1",
            "VIX_MA20_lag1",
            "VIX_Change_lag1",

            "SPY_Below_MA20",
            "SPY_Below_MA50",

            "VIX_High",
            "VIX_Extreme",
            "VIX_Rising",
            "VIX_Above_MA20",
        ]
    ].copy()

    regime.to_csv(
        REGIME_DATA_PATH,
        index=False
    )

    print(
        f"\nSPY/VIX rows: "
        f"{len(regime):,}"
    )

    print(
        f"Regime data saved to:"
        f"\n{REGIME_DATA_PATH}"
    )

    return regime


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

        long_indices = (
            group
            .head(TOP_K)
            .index
        )

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

            rows.append({

                "Date":
                    row["Date"],

                "Ticker":
                    row["Ticker"],

                "prediction":
                    row["prediction"],

                "base_position":
                    row["base_position"],

                "actual_return":
                    row["actual_return"],

            })

    return pd.DataFrame(
        rows
    )


# =========================================================
# BASE DAILY RETURNS
# =========================================================

def calculate_base_returns(
    portfolio
):

    rows = []

    for date, group in portfolio.groupby(
        "Date",
        sort=True
    ):

        portfolio_return = np.sum(

            group[
                "base_position"
            ]

            *

            group[
                "actual_return"
            ]

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
# LAGGED VOLATILITY
# =========================================================

def add_lagged_volatility(
    daily
):

    daily = daily.copy()

    daily[
        "realized_volatility"
    ] = (

        daily[
            "base_return"
        ]

        .rolling(
            VOLATILITY_LOOKBACK,
            min_periods=VOLATILITY_LOOKBACK
        )

        .std(
            ddof=1
        )

        .shift(1)

        * np.sqrt(252)

    )

    return daily


# =========================================================
# BASE VOLATILITY-TARGETED EXPOSURE
# =========================================================

def add_base_exposure(
    daily
):

    daily = daily.copy()

    multiplier = (

        TARGET_VOLATILITY

        /

        daily[
            "realized_volatility"
        ]

    )

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
        .clip(
            MIN_EXPOSURE_MULTIPLIER,
            MAX_EXPOSURE_MULTIPLIER
        )
    )

    daily[
        "vol_exposure"
    ] = multiplier

    return daily


# =========================================================
# REGIME RULE
# =========================================================

def regime_is_risk_off(
    row,
    rule
):

    if rule == "SPY_MA20":

        return bool(
            row[
                "SPY_Below_MA20"
            ]
        )

    if rule == "SPY_MA50":

        return bool(
            row[
                "SPY_Below_MA50"
            ]
        )

    if rule == "VIX_HIGH":

        return bool(
            row[
                "VIX_High"
            ]
        )

    if rule == "VIX_EXTREME":

        return bool(
            row[
                "VIX_Extreme"
            ]
        )

    if rule == "VIX_RISING":

        return bool(
            row[
                "VIX_Rising"
            ]
        )

    if rule == "VIX_ABOVE_MA20":

        return bool(
            row[
                "VIX_Above_MA20"
            ]
        )

    if rule == "SPY_MA20_AND_VIX_HIGH":

        return (

            bool(
                row[
                    "SPY_Below_MA20"
                ]
            )

            and

            bool(
                row[
                    "VIX_High"
                ]
            )

        )

    if rule == "SPY_MA50_AND_VIX_HIGH":

        return (

            bool(
                row[
                    "SPY_Below_MA50"
                ]
            )

            and

            bool(
                row[
                    "VIX_High"
                ]
            )

        )

    if rule == "SPY_MA20_OR_VIX_HIGH":

        return (

            bool(
                row[
                    "SPY_Below_MA20"
                ]
            )

            or

            bool(
                row[
                    "VIX_High"
                ]
            )

        )

    if rule == "SPY_MA50_OR_VIX_HIGH":

        return (

            bool(
                row[
                    "SPY_Below_MA50"
                ]
            )

            or

            bool(
                row[
                    "VIX_High"
                ]
            )

        )

    raise ValueError(
        f"Unknown regime rule: {rule}"
    )


# =========================================================
# APPLY REGIME FILTER
# =========================================================

def apply_regime_filter(
    daily,
    regime,
    rule,
    risk_off_multiplier
):

    daily = daily.copy()

    daily[
        "Date"
    ] = pd.to_datetime(
        daily["Date"]
    )

    regime = regime.copy()

    regime[
        "Date"
    ] = pd.to_datetime(
        regime["Date"]
    )

    # -----------------------------------------------------
    # Match each trading date with the latest available
    # SPY/VIX regime information.
    #
    # Regime values themselves are already lagged.
    # -----------------------------------------------------

    data = pd.merge(
        daily,
        regime,
        on="Date",
        how="left"
    )

    # -----------------------------------------------------
    # Missing regime data means we do not alter exposure.
    # -----------------------------------------------------

    risk_off = []

    for _, row in data.iterrows():

        if pd.isna(
            row["SPY_Below_MA20"]
        ):

            risk_off.append(
                False
            )

        else:

            risk_off.append(
                regime_is_risk_off(
                    row,
                    rule
                )
            )

    data[
        "risk_off"
    ] = risk_off

    data[
        "regime_multiplier"
    ] = np.where(

        data["risk_off"],

        risk_off_multiplier,

        1.0

    )

    # -----------------------------------------------------
    # Final exposure.
    # -----------------------------------------------------

    data[
        "final_exposure"
    ] = (

        data[
            "vol_exposure"
        ]

        *

        data[
            "regime_multiplier"
        ]

    )

    # -----------------------------------------------------
    # Apply maximum exposure constraint again.
    # -----------------------------------------------------

    data[
        "final_exposure"
    ] = (

        data[
            "final_exposure"
        ]

        .clip(
            0.0,
            MAX_EXPOSURE_MULTIPLIER
        )

    )

    # -----------------------------------------------------
    # Gross return after regime filter.
    # -----------------------------------------------------

    data[
        "gross_return"
    ] = (

        data[
            "base_return"
        ]

        *

        data[
            "final_exposure"
        ]

    )

    return data


# =========================================================
# CALCULATE TURNOVER
# =========================================================

def calculate_turnover(
    portfolio,
    daily
):

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

    exposure = (
        daily[
            [
                "Date",
                "final_exposure"
            ]
        ]
        .drop_duplicates(
            "Date"
        )
        .set_index("Date")
    )

    exposure = (
        exposure
        .reindex(
            position_matrix.index
        )
        .fillna(1.0)
    )

    targeted_positions = (
        position_matrix
        .multiply(
            exposure[
                "final_exposure"
            ],
            axis=0
        )
    )

    previous_positions = (
        targeted_positions
        .shift(1)
        .fillna(0.0)
    )

    changes = (
        targeted_positions
        - previous_positions
    )

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
# METRICS
# =========================================================

def calculate_metrics(
    daily,
    cost_rate
):

    data = daily.copy()

    data[
        "transaction_cost"
    ] = (
        data["turnover"]
        * cost_rate
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

    valid = np.isfinite(
        returns
    )

    returns = returns[
        valid
    ]

    gross_returns = gross_returns[
        np.isfinite(
            gross_returns
        )
    ]

    equity = np.cumprod(
        1.0 + returns
    )

    final_value = (
        equity[-1]
    )

    total_return = (
        final_value - 1.0
    )

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

    daily_vol = np.std(
        returns,
        ddof=1
    )

    annualized_vol = (
        daily_vol
        * np.sqrt(252)
    )

    if daily_vol > 1e-12:

        sharpe = (
            np.mean(returns)
            / daily_vol
        ) * np.sqrt(252)

    else:

        sharpe = np.nan

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

    hit_rate = np.mean(
        returns > 0
    )

    average_turnover = (
        data["turnover"].mean()
    )

    total_turnover = (
        data["turnover"].sum()
    )

    transaction_cost = (
        data[
            "transaction_cost"
        ].sum()
    )

    average_exposure = (
        data[
            "final_exposure"
        ].mean()
    )

    median_exposure = (
        data[
            "final_exposure"
        ].median()
    )

    risk_off_fraction = (
        data[
            "risk_off"
        ].mean()
    )

    return {

        "days":
            len(returns),

        "gross_total_return":
            (
                np.prod(
                    1.0
                    + gross_returns
                )
                - 1.0
            ),

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
            transaction_cost,

        "average_exposure":
            average_exposure,

        "median_exposure":
            median_exposure,

        "risk_off_fraction":
            risk_off_fraction,

    }, data


# =========================================================
# RUN PERIOD WITH ONE RULE
# =========================================================

def run_strategy(
    predictions,
    regime,
    rule,
    risk_off_multiplier
):

    portfolio = create_base_portfolio(
        predictions
    )

    base_daily = (
        calculate_base_returns(
            portfolio
        )
    )

    base_daily = (
        add_lagged_volatility(
            base_daily
        )
    )

    base_daily = (
        add_base_exposure(
            base_daily
        )
    )

    filtered = (
        apply_regime_filter(

            base_daily,

            regime,

            rule,

            risk_off_multiplier

        )
    )

    turnover = (
        calculate_turnover(

            portfolio,

            filtered

        )
    )

    filtered = pd.merge(

        filtered,

        turnover,

        on="Date",

        how="inner",

        validate="one_to_one"

    )

    metrics, daily = (
        calculate_metrics(

            filtered,

            TRANSACTION_COST

        )
    )

    return (
        metrics,
        daily,
        portfolio
    )


# =========================================================
# VALIDATION SEARCH
# =========================================================

def validation_search(
    validation_predictions,
    regime
):

    print("\n" + "=" * 70)
    print(
        "VALIDATION REGIME SEARCH"
    )
    print("=" * 70)

    rules = [

        "SPY_MA20",

        "SPY_MA50",

        "VIX_HIGH",

        "VIX_EXTREME",

        "VIX_RISING",

        "VIX_ABOVE_MA20",

        "SPY_MA20_AND_VIX_HIGH",

        "SPY_MA50_AND_VIX_HIGH",

        "SPY_MA20_OR_VIX_HIGH",

        "SPY_MA50_OR_VIX_HIGH",

    ]

    results = []

    # -----------------------------------------------------
    # Baseline = no regime filter.
    # -----------------------------------------------------

    baseline_metrics, _, _ = (
        run_strategy(

            validation_predictions,

            regime,

            "VIX_HIGH",

            1.0

        )
    )

    baseline_metrics[
        "rule"
    ] = "NO_FILTER"

    baseline_metrics[
        "risk_off_multiplier"
    ] = 1.0

    results.append(
        baseline_metrics
    )

    print(
        f"\nNO_FILTER | "
        f"Sharpe={baseline_metrics['Sharpe']:.3f} | "
        f"CAGR={baseline_metrics['CAGR']:.2%} | "
        f"MaxDD={baseline_metrics['max_drawdown']:.2%}"
    )

    # -----------------------------------------------------
    # Search actual filters.
    # -----------------------------------------------------

    for rule in rules:

        for risk_off_multiplier in (
            RISK_OFF_LEVELS
        ):

            metrics, _, _ = (
                run_strategy(

                    validation_predictions,

                    regime,

                    rule,

                    risk_off_multiplier

                )
            )

            metrics[
                "rule"
            ] = rule

            metrics[
                "risk_off_multiplier"
            ] = risk_off_multiplier

            results.append(
                metrics
            )

            print(

                f"{rule:28s} | "

                f"Risk-off="
                f"{risk_off_multiplier:.2f} | "

                f"Sharpe="
                f"{metrics['Sharpe']:.3f} | "

                f"CAGR="
                f"{metrics['CAGR']:.2%} | "

                f"MaxDD="
                f"{metrics['max_drawdown']:.2%}"

            )

    results_df = pd.DataFrame(
        results
    )

    valid = results_df[
        np.isfinite(
            results_df["Sharpe"]
        )
    ].copy()

    # -----------------------------------------------------
    # Select on validation Sharpe only.
    # -----------------------------------------------------

    best = (
        valid
        .sort_values(
            "Sharpe",
            ascending=False
        )
        .iloc[0]
    )

    return (
        best,
        results_df
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)
    print(
        "SPY/VIX REGIME FILTER "
        "BACKTEST"
    )
    print("=" * 70)

    # =====================================================
    # 1. LOAD VALIDATION
    # =====================================================

    validation = load_predictions(
        LSTM_VALIDATION_PATH
    )

    print("\n" + "=" * 70)
    print("VALIDATION DATA")
    print("=" * 70)

    print(
        f"\nRows: "
        f"{len(validation):,}"
    )

    print(
        f"Dates: "
        f"{validation['Date'].nunique()}"
    )

    print(
        f"Range: "
        f"{validation['Date'].min().date()} "
        f"→ "
        f"{validation['Date'].max().date()}"
    )

    # =====================================================
    # 2. LOAD TEST
    # =====================================================

    test = load_predictions(
        LSTM_TEST_PATH
    )

    print("\n" + "=" * 70)
    print("TEST DATA")
    print("=" * 70)

    print(
        f"\nRows: "
        f"{len(test):,}"
    )

    print(
        f"Dates: "
        f"{test['Date'].nunique()}"
    )

    print(
        f"Range: "
        f"{test['Date'].min().date()} "
        f"→ "
        f"{test['Date'].max().date()}"
    )

    # =====================================================
    # 3. DOWNLOAD MARKET DATA
    # =====================================================

    combined_start = min(

        validation["Date"].min(),

        test["Date"].min()

    )

    combined_end = max(

        validation["Date"].max(),

        test["Date"].max()

    )

    regime = download_regime_data(

        combined_start,

        combined_end

    )

    # =====================================================
    # 4. VALIDATION SEARCH
    # =====================================================

    (
        best,

        validation_results

    ) = validation_search(

        validation,

        regime

    )

    validation_results.to_csv(

        VALIDATION_RESULTS_PATH,

        index=False

    )

    best_rule = str(
        best["rule"]
    )

    best_multiplier = float(
        best[
            "risk_off_multiplier"
        ]
    )

    # =====================================================
    # 5. SELECTED CONFIG
    # =====================================================

    print("\n" + "=" * 70)
    print("SELECTED REGIME CONFIGURATION")
    print("=" * 70)

    print(
        f"\nRule: "
        f"{best_rule}"
    )

    print(
        f"Risk-off multiplier: "
        f"{best_multiplier:.2f}"
    )

    print(
        f"Validation Sharpe: "
        f"{best['Sharpe']:.3f}"
    )

    print(
        f"Validation CAGR: "
        f"{best['CAGR']:.2%}"
    )

    print(
        f"Validation MaxDD: "
        f"{best['max_drawdown']:.2%}"
    )

    print(
        f"Validation turnover: "
        f"{best['average_turnover']:.2%}"
    )

    # =====================================================
    # 6. FROZEN TEST EVALUATION
    # =====================================================

    (
        test_metrics,
        test_daily,
        test_portfolio

    ) = run_strategy(

        test,

        regime,

        best_rule,

        best_multiplier

    )

    # =====================================================
    # 7. TEST RESULT
    # =====================================================

    print("\n" + "=" * 70)
    print(
        "TEST WITH FROZEN REGIME RULE"
    )
    print("=" * 70)

    print(
        f"\nRule: "
        f"{best_rule}"
    )

    print(
        f"Risk-off multiplier: "
        f"{best_multiplier:.2f}"
    )

    print(
        f"\nGross total return: "
        f"{test_metrics['gross_total_return']:.2%}"
    )

    print(
        f"Net total return: "
        f"{test_metrics['net_total_return']:.2%}"
    )

    print(
        f"CAGR: "
        f"{test_metrics['CAGR']:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{test_metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe: "
        f"{test_metrics['Sharpe']:.3f}"
    )

    print(
        f"Sortino: "
        f"{test_metrics['Sortino']:.3f}"
    )

    print(
        f"Maximum drawdown: "
        f"{test_metrics['max_drawdown']:.2%}"
    )

    print(
        f"Hit rate: "
        f"{test_metrics['hit_rate']:.2%}"
    )

    print(
        f"Average turnover: "
        f"{test_metrics['average_turnover']:.2%}"
    )

    print(
        f"Total turnover: "
        f"{test_metrics['total_turnover']:.2f}"
    )

    print(
        f"Transaction costs: "
        f"{test_metrics['transaction_cost']:.2%}"
    )

    print(
        f"Average exposure: "
        f"{test_metrics['average_exposure']:.3f}x"
    )

    print(
        f"Median exposure: "
        f"{test_metrics['median_exposure']:.3f}x"
    )

    print(
        f"Risk-off fraction: "
        f"{test_metrics['risk_off_fraction']:.2%}"
    )

    # =====================================================
    # 8. EQUITY / DRAWDOWN
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

        /

        test_daily[
            "high_water_mark"
        ]

        - 1.0

    )

    # =====================================================
    # 9. SAVE TEST DAILY
    # =====================================================

    test_daily.to_csv(
        TEST_DAILY_PATH,
        index=False
    )

    # =====================================================
    # 10. SAVE TEST POSITIONS
    # =====================================================

    exposure_lookup = test_daily[
        [
            "Date",
            "final_exposure"
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
            "final_exposure"
        ]

    )

    positions.to_csv(
        TEST_POSITIONS_PATH,
        index=False
    )

    # =====================================================
    # 11. SAVE TEST SUMMARY
    # =====================================================

    summary = pd.DataFrame({

        "rule": [
            best_rule
        ],

        "risk_off_multiplier": [
            best_multiplier
        ],

        "target_volatility": [
            TARGET_VOLATILITY
        ],

        "cost_bps": [
            TRANSACTION_COST * 10000
        ],

        "gross_total_return": [
            test_metrics[
                "gross_total_return"
            ]
        ],

        "net_total_return": [
            test_metrics[
                "net_total_return"
            ]
        ],

        "CAGR": [
            test_metrics[
                "CAGR"
            ]
        ],

        "annualized_volatility": [
            test_metrics[
                "annualized_volatility"
            ]
        ],

        "Sharpe": [
            test_metrics[
                "Sharpe"
            ]
        ],

        "Sortino": [
            test_metrics[
                "Sortino"
            ]
        ],

        "max_drawdown": [
            test_metrics[
                "max_drawdown"
            ]
        ],

        "hit_rate": [
            test_metrics[
                "hit_rate"
            ]
        ],

        "average_turnover": [
            test_metrics[
                "average_turnover"
            ]
        ],

        "transaction_cost": [
            test_metrics[
                "transaction_cost"
            ]
        ],

        "average_exposure": [
            test_metrics[
                "average_exposure"
            ]
        ],

        "risk_off_fraction": [
            test_metrics[
                "risk_off_fraction"
            ]
        ],

    })

    summary.to_csv(
        TEST_RESULTS_PATH,
        index=False
    )

    # =====================================================
    # 12. FILES
    # =====================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nValidation search:"
        f"\n{VALIDATION_RESULTS_PATH}"
    )

    print(
        f"\nSPY/VIX data:"
        f"\n{REGIME_DATA_PATH}"
    )

    print(
        f"\nTest daily:"
        f"\n{TEST_DAILY_PATH}"
    )

    print(
        f"\nTest positions:"
        f"\n{TEST_POSITIONS_PATH}"
    )

    print(
        f"\nTest summary:"
        f"\n{TEST_RESULTS_PATH}"
    )

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(
        "\nThe regime rule was selected "
        "using validation data only."
    )

    print(
        "The selected rule and multiplier "
        "were frozen before test evaluation."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()