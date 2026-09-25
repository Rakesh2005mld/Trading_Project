# src/backtest/nested_factor_backtest.py

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

# ------------------------------------------------------------
# Walk-forward
# ------------------------------------------------------------

TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3

# ------------------------------------------------------------
# Portfolio
# ------------------------------------------------------------

HORIZON = 20

K = 2

TRANSACTION_COST_BPS = 10

GROSS_EXPOSURE = 1.0

# ============================================================
# CANDIDATE FEATURES
#
# IMPORTANT:
# None of these are pre-selected based on the global test set.
# Selection happens independently inside every fold using
# TRAIN data only.
# ============================================================

CANDIDATE_FACTORS = [

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


# Candidate portfolio constructions.
#
# All factor selection is TRAIN-only.
# Portfolio construction selection is VALIDATION-only.

STRATEGIES = [
    "best_single",
    "top3_equal",
    "top5_equal",
    "top7_equal",
    "all_equal",
    "top5_ic_weighted",
    "top5_icir_weighted",
]


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

    required = (
        CANDIDATE_FACTORS
        + [
            "Ticker",
            "Date",
            "Close",
        ]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
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
# DAILY RETURNS
# ============================================================

def add_daily_returns(df):

    df = df.copy()

    df["daily_return"] = (
        df
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    return df


# ============================================================
# WALK-FORWARD FOLDS
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
# TRAIN 20-DAY FORWARD RETURN
# ============================================================

def add_training_forward_return(
    train_df
):

    train_df = train_df.copy()

    train_df["future_return"] = (
        train_df
        .groupby("Ticker")["Close"]
        .shift(-HORIZON)
        / train_df["Close"]
        - 1.0
    )

    return train_df


# ============================================================
# TRAIN FEATURE IC
# ============================================================

def calculate_train_factor_stats(
    train_df
):

    train_df = add_training_forward_return(
        train_df
    )

    results = []

    for factor in CANDIDATE_FACTORS:

        data = train_df[
            [
                "Date",
                factor,
                "future_return",
            ]
        ].copy()

        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        data = data.dropna(
            subset=[
                factor,
                "future_return",
            ]
        )

        daily_ic = []

        for date, day in data.groupby(
            "Date"
        ):

            if len(day) < 5:
                continue

            x = day[factor]
            y = day["future_return"]

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

        if not daily_ic:

            results.append(
                {
                    "factor": factor,
                    "mean_ic": 0.0,
                    "ic_std": np.nan,
                    "icir": 0.0,
                    "positive_pct": 0.0,
                }
            )

            continue

        ic = np.asarray(
            daily_ic
        )

        mean_ic = ic.mean()

        std_ic = (
            ic.std(ddof=1)
            if len(ic) > 1
            else np.nan
        )

        icir = (
            mean_ic / std_ic
            if (
                np.isfinite(std_ic)
                and std_ic > 0
            )
            else 0.0
        )

        results.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "ic_std": std_ic,
                "icir": icir,
                "positive_pct":
                    np.mean(ic > 0) * 100,
            }
        )

    result = pd.DataFrame(
        results
    )

    # Rank by absolute predictive relationship.
    result["abs_ic"] = (
        result["mean_ic"].abs()
    )

    result = result.sort_values(
        "abs_ic",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ============================================================
# BUILD STRATEGY WEIGHTS
# ============================================================

def build_strategy_weights(
    stats,
    strategy,
):

    stats = stats.copy()

    # --------------------------------------------------------
    # Best single factor
    # --------------------------------------------------------

    if strategy == "best_single":

        row = stats.iloc[0]

        weights = {
            row["factor"]:
                np.sign(
                    row["mean_ic"]
                )
        }

        return weights

    # --------------------------------------------------------
    # Top N
    # --------------------------------------------------------

    if strategy == "top3_equal":

        selected = stats.head(3)

        weights = {}

        for _, row in selected.iterrows():

            direction = np.sign(
                row["mean_ic"]
            )

            if direction == 0:
                direction = 1.0

            weights[
                row["factor"]
            ] = direction / 3.0

        return weights

    if strategy == "top5_equal":

        selected = stats.head(5)

        weights = {}

        for _, row in selected.iterrows():

            direction = np.sign(
                row["mean_ic"]
            )

            if direction == 0:
                direction = 1.0

            weights[
                row["factor"]
            ] = direction / 5.0

        return weights

    if strategy == "top7_equal":

        selected = stats.head(7)

        weights = {}

        for _, row in selected.iterrows():

            direction = np.sign(
                row["mean_ic"]
            )

            if direction == 0:
                direction = 1.0

            weights[
                row["factor"]
            ] = direction / 7.0

        return weights

    # --------------------------------------------------------
    # All factors equal
    # --------------------------------------------------------

    if strategy == "all_equal":

        weights = {}

        n = len(stats)

        for _, row in stats.iterrows():

            direction = np.sign(
                row["mean_ic"]
            )

            if direction == 0:
                direction = 1.0

            weights[
                row["factor"]
            ] = direction / n

        return weights

    # --------------------------------------------------------
    # Top 5 IC weighted
    # --------------------------------------------------------

    if strategy == "top5_ic_weighted":

        selected = stats.head(5).copy()

        raw = []

        names = []

        for _, row in selected.iterrows():

            value = abs(
                row["mean_ic"]
            )

            if value <= 0:
                continue

            raw.append(value)
            names.append(
                row["factor"]
            )

        raw = np.asarray(
            raw
        )

        if len(raw) == 0:
            return {}

        raw = (
            raw
            / raw.sum()
        )

        weights = {}

        for factor, weight in zip(
            names,
            raw,
        ):

            row = selected[
                selected["factor"]
                == factor
            ].iloc[0]

            direction = np.sign(
                row["mean_ic"]
            )

            if direction == 0:
                direction = 1.0

            weights[factor] = (
                direction
                * weight
            )

        return weights

    # --------------------------------------------------------
    # Top 5 ICIR weighted
    # --------------------------------------------------------

    if strategy == "top5_icir_weighted":

        selected = (
            stats
            .sort_values(
                "icir",
                key=lambda x:
                    x.abs(),
                ascending=False,
            )
            .head(5)
            .copy()
        )

        raw = []

        names = []

        for _, row in selected.iterrows():

            value = abs(
                row["icir"]
            )

            if not np.isfinite(value):
                continue

            if value <= 0:
                continue

            raw.append(value)
            names.append(
                row["factor"]
            )

        raw = np.asarray(
            raw
        )

        if len(raw) == 0:
            return {}

        raw = (
            raw
            / raw.sum()
        )

        weights = {}

        for factor, weight in zip(
            names,
            raw,
        ):

            row = selected[
                selected["factor"]
                == factor
            ].iloc[0]

            direction = np.sign(
                row["mean_ic"]
            )

            if direction == 0:
                direction = 1.0

            weights[factor] = (
                direction
                * weight
            )

        return weights

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# DAILY FACTOR SCORE
# ============================================================

def score_dataframe(
    df,
    weights,
):

    result = df.copy()

    result["factor_score"] = 0.0

    for factor, weight in (
        weights.items()
    ):

        mean = (
            result
            .groupby("Date")[factor]
            .transform("mean")
        )

        std = (
            result
            .groupby("Date")[factor]
            .transform("std")
        )

        z = (
            result[factor] - mean
        ) / std.replace(
            0,
            np.nan,
        )

        result[
            "factor_score"
        ] += weight * z

    return result


# ============================================================
# PORTFOLIO FOR ONE HOLDING PERIOD
# ============================================================

def build_holding_period(
    scored_df,
    entry_date,
):

    all_dates = pd.DatetimeIndex(
        sorted(
            scored_df["Date"]
            .unique()
        )
    )

    start_idx = all_dates.searchsorted(
        entry_date
    )

    end_idx = (
        start_idx + HORIZON
    )

    if (
        start_idx >= len(all_dates)
        or end_idx > len(all_dates)
    ):
        return None

    holding_dates = all_dates[
        start_idx:end_idx
    ]

    entry_df = scored_df[
        scored_df["Date"]
        == holding_dates[0]
    ].copy()

    entry_df = entry_df.dropna(
        subset=[
            "factor_score"
        ]
    )

    if len(entry_df) < 2 * K:
        return None

    entry_df = entry_df.sort_values(
        "factor_score"
    )

    longs = entry_df.tail(K)
    shorts = entry_df.head(K)

    long_tickers = set(
        longs["Ticker"]
    )

    short_tickers = set(
        shorts["Ticker"]
    )

    long_weight = (
        GROSS_EXPOSURE
        / (2.0 * K)
    )

    short_weight = (
        -GROSS_EXPOSURE
        / (2.0 * K)
    )

    holding_df = scored_df[
        scored_df["Date"].isin(
            holding_dates
        )
    ].copy()

    records = []

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

        portfolio_return = (
            long_weight
            * long_returns.sum()
            +
            short_weight
            * short_returns.sum()
        )

        records.append(
            {
                "Date": date,
                "gross_return":
                    portfolio_return,
            }
        )

    if not records:
        return None

    result = pd.DataFrame(
        records
    )

    # Entry + exit turnover
    turnover = (
        2.0
        * GROSS_EXPOSURE
    )

    total_cost = (
        turnover
        * TRANSACTION_COST_BPS
        / 10000.0
    )

    result["cost"] = 0.0

    result.loc[
        result.index[0],
        "cost",
    ] = (
        total_cost / 2.0
    )

    result.loc[
        result.index[-1],
        "cost",
    ] = (
        total_cost / 2.0
    )

    result["net_return"] = (
        result["gross_return"]
        - result["cost"]
    )

    result["entry_date"] = (
        holding_dates[0]
    )

    result["exit_date"] = (
        holding_dates[-1]
    )

    result["longs"] = (
        ",".join(
            sorted(
                long_tickers
            )
        )
    )

    result["shorts"] = (
        ",".join(
            sorted(
                short_tickers
            )
        )
    )

    return result


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    scored_df,
):

    dates = pd.DatetimeIndex(
        sorted(
            scored_df["Date"]
            .unique()
        )
    )

    rebalance_dates = dates[
        ::HORIZON
    ]

    holdings = []

    for entry_date in rebalance_dates:

        holding = build_holding_period(
            scored_df,
            entry_date,
        )

        if holding is not None:

            holdings.append(
                holding
            )

    if not holdings:
        return pd.DataFrame()

    return pd.concat(
        holdings,
        ignore_index=True,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    daily,
):

    if daily.empty:
        return {}

    daily = (
        daily
        .sort_values("Date")
        .drop_duplicates(
            subset="Date"
        )
    )

    r = daily[
        "net_return"
    ].values

    equity = (
        1.0 + r
    ).cumprod()

    mean_r = r.mean()

    std_r = (
        r.std(ddof=1)
    )

    sharpe = (
        mean_r
        / std_r
        * np.sqrt(252)
        if std_r > 0
        else np.nan
    )

    downside = r[r < 0]

    downside_std = (
        downside.std(ddof=1)
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        mean_r
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

    running_max = (
        np.maximum.accumulate(
            equity
        )
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_dd = drawdown.min()

    total_return = (
        equity[-1] - 1.0
    )

    first_date = pd.Timestamp(
        daily["Date"].iloc[0]
    )

    last_date = pd.Timestamp(
        daily["Date"].iloc[-1]
    )

    years = max(
        (
            last_date - first_date
        ).days
        / 365.25,
        1.0 / 365.25,
    )

    cagr = (
        (1.0 + total_return)
        ** (1.0 / years)
        - 1.0
    )

    return {
        "days":
            len(daily),

        "total_return":
            total_return,

        "CAGR":
            cagr,

        "annualized_vol":
            std_r
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
# SINGLE FOLD
# ============================================================

def run_fold(
    df,
    fold,
):

    train_df = df[
        (df["Date"] >= fold["train_start"])
        &
        (df["Date"] < fold["train_end"])
    ].copy()

    validation_df = df[
        (df["Date"] >= fold["validation_start"])
        &
        (df["Date"] < fold["validation_end"])
    ].copy()

    test_df = df[
        (df["Date"] >= fold["test_start"])
        &
        (df["Date"] < fold["test_end"])
    ].copy()

    # --------------------------------------------------------
    # TRAIN-ONLY FEATURE SELECTION
    # --------------------------------------------------------

    factor_stats = (
        calculate_train_factor_stats(
            train_df
        )
    )

    strategy_rows = []

    validation_backtests = {}

    # --------------------------------------------------------
    # Evaluate each strategy on VALIDATION
    # --------------------------------------------------------

    for strategy in STRATEGIES:

        weights = (
            build_strategy_weights(
                factor_stats,
                strategy,
            )
        )

        if not weights:
            continue

        val_scored = score_dataframe(
            validation_df,
            weights,
        )

        val_daily = backtest(
            val_scored
        )

        val_metrics = calculate_metrics(
            val_daily
        )

        validation_backtests[
            strategy
        ] = val_daily

        strategy_rows.append(
            {
                "fold":
                    fold["fold"],

                "strategy":
                    strategy,

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
            }
        )

    strategy_df = pd.DataFrame(
        strategy_rows
    )

    if strategy_df.empty:
        raise RuntimeError(
            f"No valid strategies in fold "
            f"{fold['fold']}"
        )

    # --------------------------------------------------------
    # SELECT ONLY FROM VALIDATION
    # --------------------------------------------------------

    strategy_df = (
        strategy_df
        .sort_values(
            [
                "validation_sharpe",
                "validation_cagr",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    selected_strategy = (
        strategy_df.iloc[0]["strategy"]
    )

    selected_weights = (
        build_strategy_weights(
            factor_stats,
            selected_strategy,
        )
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_scored = score_dataframe(
        test_df,
        selected_weights,
    )

    test_daily = backtest(
        test_scored
    )

    test_metrics = calculate_metrics(
        test_daily
    )

    # --------------------------------------------------------
    # Factor selection
    # --------------------------------------------------------

    selected_factors = list(
        selected_weights.keys()
    )

    summary = {
        "fold":
            fold["fold"],

        "train_start":
            fold["train_start"],

        "train_end":
            fold["train_end"],

        "validation_start":
            fold["validation_start"],

        "validation_end":
            fold["validation_end"],

        "test_start":
            fold["test_start"],

        "test_end":
            fold["test_end"],

        "selected_strategy":
            selected_strategy,

        "selected_factors":
            "|".join(
                selected_factors
            ),

        "validation_sharpe":
            strategy_df.iloc[0][
                "validation_sharpe"
            ],

        "validation_cagr":
            strategy_df.iloc[0][
                "validation_cagr"
            ],

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

        "test_max_drawdown":
            test_metrics.get(
                "max_drawdown",
                np.nan,
            ),

        "test_hit_rate":
            test_metrics.get(
                "hit_rate",
                np.nan,
            ),

        "test_days":
            test_metrics.get(
                "days",
                np.nan,
            ),

        "test_total_cost":
            test_metrics.get(
                "total_cost",
                np.nan,
            ),
    }

    return (
        summary,
        strategy_df,
        factor_stats,
        test_daily,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = add_daily_returns(
        df
    )

    folds = create_folds(
        df["Date"]
    )

    print()
    print("=" * 75)
    print("NESTED WALK-FORWARD FACTOR BACKTEST")
    print("=" * 75)

    print(
        f"Folds       : {len(folds)}"
    )

    print(
        f"Horizon     : {HORIZON} days"
    )

    print(
        f"Top / Bottom: {K}"
    )

    print(
        f"Cost        : "
        f"{TRANSACTION_COST_BPS} bps"
    )

    fold_summaries = []
    strategy_history = []
    factor_history = []
    test_daily_all = []

    # ========================================================
    # RUN
    # ========================================================

    for fold in folds:

        print()
        print(
            f"Fold {fold['fold']:02d} | "
            f"Test "
            f"{fold['test_start'].date()} "
            f"-> "
            f"{fold['test_end'].date()}"
        )

        (
            summary,
            strategy_df,
            factor_stats,
            test_daily,
        ) = run_fold(
            df,
            fold,
        )

        fold_summaries.append(
            summary
        )

        strategy_df[
            "fold"
        ] = fold["fold"]

        strategy_history.append(
            strategy_df
        )

        factor_stats[
            "fold"
        ] = fold["fold"]

        factor_history.append(
            factor_stats
        )

        if not test_daily.empty:

            test_daily = (
                test_daily
                .copy()
            )

            test_daily[
                "fold"
            ] = fold["fold"]

            test_daily_all.append(
                test_daily
            )

        print(
            f"Selected strategy : "
            f"{summary['selected_strategy']}"
        )

        print(
            f"Selected factors  : "
            f"{summary['selected_factors']}"
        )

        print(
            f"Validation Sharpe : "
            f"{summary['validation_sharpe']:.3f}"
        )

        print(
            f"Test Sharpe       : "
            f"{summary['test_sharpe']:.3f}"
        )

        print(
            f"Test CAGR         : "
            f"{summary['test_cagr']:.2%}"
        )

        print(
            f"Test Max DD       : "
            f"{summary['test_max_drawdown']:.2%}"
        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    fold_results = pd.DataFrame(
        fold_summaries
    )

    strategy_results = pd.concat(
        strategy_history,
        ignore_index=True,
    )

    factor_results = pd.concat(
        factor_history,
        ignore_index=True,
    )

    full_test_daily = pd.concat(
        test_daily_all,
        ignore_index=True,
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_path = (
        OUTPUT_DIR
        / "nested_factor_fold_results.csv"
    )

    fold_results.to_csv(
        fold_path,
        index=False,
    )

    # ========================================================
    # SAVE ALL VALIDATION STRATEGIES
    # ========================================================

    strategy_path = (
        OUTPUT_DIR
        / "nested_factor_validation_strategies.csv"
    )

    strategy_results.to_csv(
        strategy_path,
        index=False,
    )

    # ========================================================
    # SAVE FACTOR HISTORY
    # ========================================================

    factor_path = (
        OUTPUT_DIR
        / "nested_factor_train_statistics.csv"
    )

    factor_results.to_csv(
        factor_path,
        index=False,
    )

    # ========================================================
    # SAVE FULL OOS DAILY RETURNS
    # ========================================================

    daily_path = (
        OUTPUT_DIR
        / "nested_factor_oos_daily.csv"
    )

    full_test_daily.to_csv(
        daily_path,
        index=False,
    )

    # ========================================================
    # OVERALL OOS METRICS
    # ========================================================

    overall = calculate_metrics(
        full_test_daily
    )

    print()
    print("=" * 75)
    print("FINAL CLEAN OOS RESULT")
    print("=" * 75)

    print(
        f"Days           : "
        f"{overall['days']}"
    )

    print(
        f"Total return   : "
        f"{overall['total_return']:.2%}"
    )

    print(
        f"CAGR           : "
        f"{overall['CAGR']:.2%}"
    )

    print(
        f"Annualized vol : "
        f"{overall['annualized_vol']:.2%}"
    )

    print(
        f"Sharpe         : "
        f"{overall['Sharpe']:.3f}"
    )

    print(
        f"Sortino        : "
        f"{overall['Sortino']:.3f}"
    )

    print(
        f"Max drawdown   : "
        f"{overall['max_drawdown']:.2%}"
    )

    print(
        f"Hit rate       : "
        f"{overall['hit_rate']:.2%}"
    )

    print(
        f"Total costs    : "
        f"{overall['total_cost']:.2%}"
    )

    # ========================================================
    # FOLD STATISTICS
    # ========================================================

    print()
    print("=" * 75)
    print("FOLD STATISTICS")
    print("=" * 75)

    print(f"Positive test folds : {np.mean(fold_results['test_sharpe'] > 0):.2%}")

    print(
        f"Mean fold Sharpe    : "
        f"{fold_results['test_sharpe'].mean():.3f}"
    )

    print(
        f"Median fold Sharpe  : "
        f"{fold_results['test_sharpe'].median():.3f}"
    )

    print(
        f"Mean fold CAGR      : "
        f"{fold_results['test_cagr'].mean():.2%}"
    )

    # ========================================================
    # STRATEGY SELECTION FREQUENCY
    # ========================================================

    print()
    print("=" * 75)
    print("SELECTED STRATEGIES")
    print("=" * 75)

    strategy_counts = (
        fold_results[
            "selected_strategy"
        ]
        .value_counts()
    )

    print(
        strategy_counts.to_string()
    )

    # ========================================================
    # FACTOR SELECTION FREQUENCY
    # ========================================================

    selected_factor_rows = []

    for _, row in fold_results.iterrows():

        if not row[
            "selected_factors"
        ]:
            continue

        factors = (
            row[
                "selected_factors"
            ]
            .split("|")
        )

        for factor in factors:

            selected_factor_rows.append(
                {
                    "fold":
                        row["fold"],

                    "factor":
                        factor,
                }
            )

    selected_factor_df = pd.DataFrame(
        selected_factor_rows
    )

    if not selected_factor_df.empty:

        counts = (
            selected_factor_df[
                "factor"
            ]
            .value_counts()
        )

        print()
        print("=" * 75)
        print("FACTOR SELECTION FREQUENCY")
        print("=" * 75)

        print(
            counts.to_string()
        )

    # ========================================================
    # SAVE FINAL SUMMARY
    # ========================================================

    final_summary = pd.DataFrame(
        [
            {
                "days":
                    overall["days"],

                "total_return":
                    overall["total_return"],

                "CAGR":
                    overall["CAGR"],

                "annualized_vol":
                    overall["annualized_vol"],

                "Sharpe":
                    overall["Sharpe"],

                "Sortino":
                    overall["Sortino"],

                "max_drawdown":
                    overall["max_drawdown"],

                "hit_rate":
                    overall["hit_rate"],

                "total_cost":
                    overall["total_cost"],

                "positive_fold_pct":
                    np.mean(
                        fold_results[
                            "test_sharpe"
                        ] > 0
                    ),

                "mean_fold_sharpe":
                    fold_results[
                        "test_sharpe"
                    ].mean(),

                "median_fold_sharpe":
                    fold_results[
                        "test_sharpe"
                    ].median(),
            }
        ]
    )

    final_path = (
        OUTPUT_DIR
        / "nested_factor_final_summary.csv"
    )

    final_summary.to_csv(
        final_path,
        index=False,
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 75)
    print("FILES")
    print("=" * 75)

    print(fold_path)
    print(strategy_path)
    print(factor_path)
    print(daily_path)
    print(final_path)

    print()
    print(
        "Nested factor backtest complete."
    )


if __name__ == "__main__":
    main()