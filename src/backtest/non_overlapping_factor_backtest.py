from pathlib import Path

import numpy as np
import pandas as pd


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


# ============================================================
# STRATEGY
# ============================================================

HORIZON = 20

K = 2

TRANSACTION_COST_BPS = 10

GROSS_EXPOSURE = 1.0


# Strong factors identified in walk-forward analysis
FACTORS = [
    "price_ma200_ratio",
    "return_60d",
    "beta_60",
    "volatility_60",
    "momentum_20",
    "volatility_20",
    "volatility_10",
]


TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3

STEP_MONTHS = 3


# ============================================================
# LOAD DATA
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
        f"Dates   : {df['Date'].nunique():,}"
    )

    print(
        f"Tickers : {df['Ticker'].nunique()}"
    )

    print(
        f"Range   : "
        f"{df['Date'].min().date()} "
        f"-> "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# FOLDS
# ============================================================

def create_folds(dates):

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
# FACTOR Z SCORE
# ============================================================

def cross_sectional_zscore(
    df,
    column,
):

    mean = (
        df.groupby("Date")[column]
        .transform("mean")
    )

    std = (
        df.groupby("Date")[column]
        .transform("std")
    )

    return (
        df[column] - mean
    ) / std.replace(
        0,
        np.nan,
    )


# ============================================================
# TRAIN FACTOR WEIGHTS
# ============================================================

def calculate_factor_weights(
    train_df,
):

    records = []

    target = (
        "__future_return"
    )

    for factor in FACTORS:

        data = train_df[
            [
                "Date",
                factor,
                target,
            ]
        ].copy()

        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        data = data.dropna()

        daily_ic = []

        for date, day in data.groupby(
            "Date"
        ):

            if len(day) < 5:
                continue

            x = day[factor]

            y = day[target]

            if x.nunique() < 2:
                continue

            if y.nunique() < 2:
                continue

            ic = (
                x.rank()
                .corr(
                    y.rank()
                )
            )

            if pd.notna(ic):
                daily_ic.append(
                    ic
                )

        if len(daily_ic) == 0:

            mean_ic = 0.0
            icir = 0.0

        else:

            daily_ic = np.asarray(
                daily_ic
            )

            mean_ic = (
                daily_ic.mean()
            )

            std_ic = (
                daily_ic.std(
                    ddof=1
                )
                if len(daily_ic) > 1
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
                else 0.0
            )

        records.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "icir": icir,
            }
        )

    ic_df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Equal weight
    # --------------------------------------------------------

    equal_weights = pd.Series(
        1.0,
        index=FACTORS,
    )

    equal_weights /= (
        equal_weights.abs().sum()
    )

    # --------------------------------------------------------
    # IC weighted
    # --------------------------------------------------------

    ic_weights = (
        ic_df
        .set_index("factor")
        ["mean_ic"]
        .reindex(FACTORS)
        .fillna(0.0)
    )

    if ic_weights.abs().sum() > 0:

        ic_weights /= (
            ic_weights.abs().sum()
        )

    # --------------------------------------------------------
    # ICIR weighted
    # --------------------------------------------------------

    icir_weights = (
        ic_df
        .set_index("factor")
        ["icir"]
        .reindex(FACTORS)
        .fillna(0.0)
    )

    if icir_weights.abs().sum() > 0:

        icir_weights /= (
            icir_weights.abs().sum()
        )

    return (
        ic_df,
        {
            "equal_weight":
                equal_weights,

            "ic_weighted":
                ic_weights,

            "icir_weighted":
                icir_weights,
        },
    )


# ============================================================
# DAILY FACTOR SCORE
# ============================================================

def score_dataframe(
    df,
    weights,
):

    result = df.copy()

    result["score"] = 0.0

    for factor in FACTORS:

        z = cross_sectional_zscore(
            result,
            factor,
        )

        result["score"] += (
            weights[factor] * z
        )

    return result


# ============================================================
# BUILD NON-OVERLAPPING REBALANCE DATES
# ============================================================

def create_rebalance_dates(
    dates
):

    unique_dates = (
        pd.DatetimeIndex(
            sorted(
                pd.to_datetime(
                    dates
                ).unique()
            )
        )
    )

    return unique_dates[
        ::HORIZON
    ]


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    scored_df,
    rebalance_dates,
):

    all_dates = pd.DatetimeIndex(
        sorted(
            scored_df["Date"]
            .unique()
        )
    )

    records = []

    for start_date in rebalance_dates:

        # ----------------------------------------------------
        # Find actual date location
        # ----------------------------------------------------

        start_idx = all_dates.searchsorted(
            start_date
        )

        if start_idx >= len(all_dates):
            continue

        end_idx = (
            start_idx
            + HORIZON
        )

        if end_idx > len(all_dates):
            break

        holding_dates = all_dates[
            start_idx:end_idx
        ]

        if len(holding_dates) == 0:
            continue

        entry_date = holding_dates[0]

        exit_date = holding_dates[-1]

        entry_df = scored_df[
            scored_df["Date"]
            == entry_date
        ].copy()

        entry_df = entry_df.dropna(
            subset=[
                "score"
            ]
        )

        if len(entry_df) < 2 * K:
            continue

        entry_df = entry_df.sort_values(
            "score"
        )

        longs = entry_df.tail(K)
        shorts = entry_df.head(K)

        long_tickers = set(
            longs["Ticker"]
        )

        short_tickers = set(
            shorts["Ticker"]
        )

        # ----------------------------------------------------
        # Position weights
        # ----------------------------------------------------

        long_weight = (
            GROSS_EXPOSURE
            / (2.0 * K)
        )

        short_weight = (
            -GROSS_EXPOSURE
            / (2.0 * K)
        )

        # ----------------------------------------------------
        # Get daily returns for held stocks
        # ----------------------------------------------------

        holding_df = scored_df[
            scored_df["Date"].isin(
                holding_dates
            )
            &
            scored_df["Ticker"].isin(
                long_tickers
                | short_tickers
            )
        ].copy()

        # ----------------------------------------------------
        # Convert realized returns
        # ----------------------------------------------------

        portfolio_daily = []

        for date, day in holding_df.groupby(
            "Date"
        ):

            long_returns = day[
                day["Ticker"].isin(
                    long_tickers
                )
            ]["daily_return"]

            short_returns = day[
                day["Ticker"].isin(
                    short_tickers
                )]["daily_return"]

            if (
                len(long_returns) < K
                or len(short_returns) < K
            ):
                continue

            daily_ret = (
                long_weight
                * long_returns.sum()
                +
                short_weight
                * short_returns.sum()
            )

            portfolio_daily.append(
                {
                    "Date": date,
                    "gross_return":
                        daily_ret,
                }
            )

        if not portfolio_daily:
            continue

        holding_result = pd.DataFrame(
            portfolio_daily
        )

        # ----------------------------------------------------
        # Transaction cost
        #
        # Entry and exit:
        #
        # 4 positions x 25% =
        # 100% one-way turnover
        #
        # Entry + exit = 200%
        # ----------------------------------------------------

        turnover = (
            2.0
            * GROSS_EXPOSURE
        )

        total_cost = (
            turnover
            * TRANSACTION_COST_BPS
            / 10000.0
        )

        # Charge entry cost on first day
        # and exit cost on last day.

        n_days = len(
            holding_result
        )

        entry_cost = (
            total_cost / 2.0
        )

        exit_cost = (
            total_cost / 2.0
        )

        holding_result[
            "cost"
        ] = 0.0

        holding_result.loc[
            holding_result.index[0],
            "cost",
        ] += entry_cost

        holding_result.loc[
            holding_result.index[-1],
            "cost",
        ] += exit_cost

        holding_result[
            "net_return"
        ] = (
            holding_result[
                "gross_return"
            ]
            -
            holding_result[
                "cost"
            ]
        )

        holding_result[
            "rebalance_date"
        ] = entry_date

        holding_result[
            "exit_date"
        ] = exit_date

        records.append(
            holding_result
        )

    if not records:

        return pd.DataFrame()

    result = pd.concat(
        records,
        ignore_index=True,
    )

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    daily
):

    if daily.empty:
        return {}

    r = daily[
        "net_return"
    ].values

    gross = daily[
        "gross_return"
    ].values

    equity = (
        1.0 + r
    ).cumprod()

    # --------------------------------------------------------
    # Since returns are now actual daily returns during
    # non-overlapping holding periods, annualization is valid.
    # --------------------------------------------------------

    daily_std = (
        np.std(
            r,
            ddof=1,
        )
    )

    daily_mean = np.mean(
        r
    )

    sharpe = (
        daily_mean
        / daily_std
        * np.sqrt(252)
        if daily_std > 0
        else np.nan
    )

    downside = r[r < 0]

    downside_std = (
        np.std(
            downside,
            ddof=1,
        )
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        daily_mean
        / downside_std
        * np.sqrt(252)
        if (
            np.isfinite(
                downside_std
            )
            and downside_std > 0
        )
        else np.nan
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_dd = drawdown.min()

    years = (
        len(r)
        / 252.0
    )

    cagr = (
        equity[-1]
        ** (1.0 / years)
        - 1.0
        if years > 0
        else np.nan
    )

    total_return = (
        equity[-1] - 1.0
    )

    gross_equity = (
        1.0 + gross
    ).cumprod()

    gross_total_return = (
        gross_equity[-1] - 1.0
    )

    return {
        "days":
            len(r),

        "total_return":
            total_return,

        "gross_total_return":
            gross_total_return,

        "CAGR":
            cagr,

        "annualized_vol":
            daily_std
            * np.sqrt(252),

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "max_drawdown":
            max_dd,

        "hit_rate":
            np.mean(r > 0),

        "total_cost":
            daily["cost"].sum(),
    }


# ============================================================
# ONE FOLD
# ============================================================

def run_fold(
    df,
    fold,
):

    train_mask = (
        (df["Date"] >= fold["train_start"])
        &
        (df["Date"] < fold["train_end"])
    )

    validation_mask = (
        (df["Date"] >= fold["validation_start"])
        &
        (df["Date"] < fold["validation_end"])
    )

    test_mask = (
        (df["Date"] >= fold["test_start"])
        &
        (df["Date"] < fold["test_end"])
    )

    train_df = df[
        train_mask
    ].copy()

    validation_df = df[
        validation_mask
    ].copy()

    test_df = df[
        test_mask
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Training factor weights are based on 20-day future
    # returns from the training period only.
    # --------------------------------------------------------

    train_target = (
        train_df
        .groupby("Ticker")["Close"]
        .shift(-HORIZON)
        / train_df["Close"]
        - 1.0
    )

    train_df[
        "__future_return"
    ] = train_target

    ic_df, schemes = calculate_factor_weights(
        train_df
    )

    results = []

    for scheme_name, weights in schemes.items():

        # ----------------------------------------------------
        # SCORE validation/test
        # ----------------------------------------------------

        val_scored = score_dataframe(
            validation_df,
            weights,
        )

        test_scored = score_dataframe(
            test_df,
            weights,
        )

        # ----------------------------------------------------
        # Use every 20th trading day as rebalance date
        # ----------------------------------------------------

        val_dates = (
            pd.DatetimeIndex(
                sorted(
                    val_scored[
                        "Date"
                    ].unique()
                )
            )
        )

        test_dates = (
            pd.DatetimeIndex(
                sorted(
                    test_scored[
                        "Date"
                    ].unique()
                )
            )
        )

        val_rebalance_dates = create_rebalance_dates(
            val_dates
        )

        test_rebalance_dates = create_rebalance_dates(
            test_dates
        )

        val_daily = backtest(
            val_scored,
            val_rebalance_dates,
        )

        test_daily = backtest(
            test_scored,
            test_rebalance_dates,
        )

        val_metrics = calculate_metrics(
            val_daily
        )

        test_metrics = calculate_metrics(
            test_daily
        )

        results.append(
            {
                "fold":
                    fold["fold"],

                "scheme":
                    scheme_name,

                "validation_sharpe":
                    val_metrics.get(
                        "Sharpe",
                        np.nan,
                    ),

                "validation_cagr":
                    val_metrics.get(
                        "CAGR",
                        np.nan,
                    ),

                "validation_max_dd":
                    val_metrics.get(
                        "max_drawdown",
                        np.nan,
                    ),

                "test_sharpe":
                    test_metrics.get(
                        "Sharpe",
                        np.nan,
                    ),

                "test_cagr":
                    test_metrics.get(
                        "CAGR",
                        np.nan,
                    ),

                "test_total_return":
                    test_metrics.get(
                        "total_return",
                        np.nan,
                    ),

                "test_max_dd":
                    test_metrics.get(
                        "max_drawdown",
                        np.nan,
                    ),

                "test_hit_rate":
                    test_metrics.get(
                        "hit_rate",
                        np.nan,
                    ),

                "test_cost":
                    test_metrics.get(
                        "total_cost",
                        np.nan,
                    ),

                "test_days":
                    test_metrics.get(
                        "days",
                        np.nan,
                    ),
            }
        )

    return (
        pd.DataFrame(results),
        ic_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    # --------------------------------------------------------
    # Daily return for actual holding-period PnL
    # --------------------------------------------------------

    df[
        "daily_return"
    ] = (
        df
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    folds = create_folds(
        df["Date"]
    )

    print()
    print("=" * 75)
    print("NON-OVERLAPPING 20-DAY FACTOR BACKTEST")
    print("=" * 75)

    print(
        f"Folds     : {len(folds)}"
    )

    print(
        f"Horizon   : {HORIZON} trading days"
    )

    print(
        f"Top/Bottom: {K}"
    )

    print(
        f"Cost      : {TRANSACTION_COST_BPS} bps"
    )

    all_results = []
    all_ics = []

    for fold in folds:

        print()
        print(
            f"Fold {fold['fold']:02d} | "
            f"Test "
            f"{fold['test_start'].date()} "
            f"-> "
            f"{fold['test_end'].date()}"
        )

        fold_result, ic_df = run_fold(
            df,
            fold,
        )

        all_results.append(
            fold_result
        )

        ic_df["fold"] = fold[
            "fold"
        ]

        all_ics.append(
            ic_df
        )

        print(
            fold_result[
                [
                    "scheme",
                    "validation_sharpe",
                    "test_sharpe",
                    "test_cagr",
                    "test_max_dd",
                ]
            ].to_string(
                index=False
            )
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    ics = pd.concat(
        all_ics,
        ignore_index=True,
    )

    # ========================================================
    # SAVE
    # ========================================================

    fold_path = (
        OUTPUT_DIR
        / "non_overlapping_factor_folds.csv"
    )

    results.to_csv(
        fold_path,
        index=False,
    )

    ic_path = (
        OUTPUT_DIR
        / "non_overlapping_factor_ics.csv"
    )

    ics.to_csv(
        ic_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results
        .groupby("scheme")
        .agg(
            mean_validation_sharpe=(
                "validation_sharpe",
                "mean",
            ),

            median_validation_sharpe=(
                "validation_sharpe",
                "median",
            ),

            validation_positive_pct=(
                "validation_sharpe",
                lambda x:
                np.mean(
                    x > 0
                ) * 100,
            ),

            mean_test_sharpe=(
                "test_sharpe",
                "mean",
            ),

            median_test_sharpe=(
                "test_sharpe",
                "median",
            ),

            test_positive_pct=(
                "test_sharpe",
                lambda x:
                np.mean(
                    x > 0
                ) * 100,
            ),

            mean_test_cagr=(
                "test_cagr",
                "mean",
            ),

            mean_test_return=(
                "test_total_return",
                "mean",
            ),

            mean_test_max_dd=(
                "test_max_dd",
                "mean",
            ),

            mean_test_hit_rate=(
                "test_hit_rate",
                "mean",
            ),

            total_cost=(
                "test_cost",
                "sum",
            ),
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / "non_overlapping_factor_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 75)
    print(
        "FINAL NON-OVERLAPPING RESULTS"
    )
    print("=" * 75)

    print(
        summary.to_string(
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
        ic_path
    )

    print(
        summary_path
    )

    print()
    print(
        "Backtest complete."
    )


if __name__ == "__main__":
    main()