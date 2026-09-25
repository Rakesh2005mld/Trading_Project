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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NESTED WALK-FORWARD
# ============================================================

TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3


# ============================================================
# SENSITIVITY
# ============================================================

HOLDING_PERIODS = [5, 10, 20]

K_VALUES = [1, 2, 3]

COST_BPS = [0, 5, 10, 20]


# ============================================================
# CANDIDATE FEATURES
# ============================================================

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
# STRATEGIES
# ============================================================

STRATEGIES = [
    "best_single",
    "top3_equal",
    "top5_equal",
    "top7_equal",
    "all_equal",
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

    df["daily_return"] = (
        df
        .groupby("Ticker")["Close"]
        .pct_change()
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

    train_start = first_date

    fold = 1

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
                "fold": fold,

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

        train_start += pd.DateOffset(
            months=STEP_MONTHS
        )

        fold += 1

    return folds


# ============================================================
# TRAIN FACTOR STATS
# ============================================================

def factor_stats(train_df, holding_period):

    df = train_df.copy()

    df["future_return"] = (
        df
        .groupby("Ticker")["Close"]
        .shift(-holding_period)
        / df["Close"]
        - 1.0
    )

    results = []

    for feature in FEATURES:

        temp = df[
            [
                "Date",
                feature,
                "future_return",
            ]
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        daily_ic = []

        for _, day in temp.groupby(
            "Date"
        ):

            if len(day) < 5:
                continue

            if day[feature].nunique() < 2:
                continue

            if day["future_return"].nunique() < 2:
                continue

            ic = (
                day[feature].rank()
                .corr(
                    day["future_return"].rank()
                )
            )

            if pd.notna(ic):
                daily_ic.append(ic)

        if not daily_ic:

            mean_ic = 0.0

        else:

            mean_ic = np.mean(
                daily_ic
            )

        results.append(
            {
                "feature": feature,
                "mean_ic": mean_ic,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            "mean_ic",
            key=lambda x: x.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# WEIGHTS
# ============================================================

def make_weights(
    stats,
    strategy,
):

    if strategy == "best_single":

        selected = stats.head(1)

    elif strategy == "top3_equal":

        selected = stats.head(3)

    elif strategy == "top5_equal":

        selected = stats.head(5)

    elif strategy == "top7_equal":

        selected = stats.head(7)

    elif strategy == "all_equal":

        selected = stats.copy()

    else:

        raise ValueError(
            strategy
        )

    weights = {}

    n = len(selected)

    if n == 0:
        return weights

    for _, row in selected.iterrows():

        direction = np.sign(
            row["mean_ic"]
        )

        if direction == 0:
            direction = 1.0

        weights[
            row["feature"]
        ] = direction / n

    return weights


# ============================================================
# FACTOR SCORE
# ============================================================

def add_score(
    df,
    weights,
):

    result = df.copy()

    result["score"] = 0.0

    for feature, weight in (
        weights.items()
    ):

        mean = (
            result
            .groupby("Date")[feature]
            .transform("mean")
        )

        std = (
            result
            .groupby("Date")[feature]
            .transform("std")
        )

        z = (
            result[feature] - mean
        ) / std.replace(
            0,
            np.nan,
        )

        result["score"] += (
            weight * z
        )

    return result


# ============================================================
# HOLDING PERIOD BACKTEST
# ============================================================

def backtest(
    df,
    holding_period,
    k,
    cost_bps,
):

    dates = pd.DatetimeIndex(
        sorted(
            df["Date"].unique()
        )
    )

    rebalance_dates = dates[
        ::holding_period
    ]

    all_periods = []

    for entry_date in rebalance_dates:

        idx = dates.searchsorted(
            entry_date
        )

        exit_idx = (
            idx + holding_period
        )

        if exit_idx > len(dates):
            break

        holding_dates = dates[
            idx:exit_idx
        ]

        entry = df[
            df["Date"] == entry_date
        ].copy()

        entry = entry.dropna(
            subset=["score"]
        )

        if len(entry) < 2 * k:
            continue

        entry = entry.sort_values(
            "score"
        )

        long_tickers = set(
            entry
            .tail(k)
            ["Ticker"]
        )

        short_tickers = set(
            entry
            .head(k)
            ["Ticker"]
        )

        long_weight = (
            1.0 / (2.0 * k)
        )

        short_weight = (
            -1.0 / (2.0 * k)
        )

        held = df[
            df["Date"].isin(
                holding_dates
            )
        ]

        rows = []

        for date, day in held.groupby(
            "Date"
        ):

            longs = day[
                day["Ticker"].isin(
                    long_tickers
                )
            ]["daily_return"]

            shorts = day[
                day["Ticker"].isin(
                    short_tickers
                )]["daily_return"]

            if (
                len(longs) < k
                or len(shorts) < k
            ):
                continue

            portfolio_return = (
                long_weight
                * longs.sum()
                +
                short_weight
                * shorts.sum()
            )

            rows.append(
                {
                    "Date": date,
                    "gross_return":
                        portfolio_return,
                }
            )

        if not rows:
            continue

        period = pd.DataFrame(
            rows
        )

        # Entry + exit turnover.
        #
        # Gross exposure = 1.0.
        # Entry = 100%.
        # Exit  = 100%.
        # Total = 200%.
        total_cost = (
            2.0
            * cost_bps
            / 10000.0
        )

        period["cost"] = 0.0

        period.loc[
            period.index[0],
            "cost",
        ] = (
            total_cost / 2.0
        )

        period.loc[
            period.index[-1],
            "cost",
        ] = (
            total_cost / 2.0
        )

        period["net_return"] = (
            period["gross_return"]
            - period["cost"]
        )

        all_periods.append(
            period
        )

    if not all_periods:
        return pd.DataFrame()

    result = pd.concat(
        all_periods,
        ignore_index=True,
    )

    return (
        result
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    daily
):

    if daily.empty:
        return {}

    r = daily[
        "net_return"
    ].values

    equity = (
        1.0 + r
    ).cumprod()

    mean_r = r.mean()

    std_r = r.std(
        ddof=1
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

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

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
        1 / 365.25,
    )

    cagr = (
        (1.0 + total_return)
        ** (1.0 / years)
        - 1.0
    )

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CAGR": cagr,
        "total_return": total_return,
        "max_drawdown":
            drawdown.min(),
        "hit_rate":
            np.mean(r > 0),
        "volatility":
            std_r * np.sqrt(252),
        "cost":
            daily["cost"].sum(),
    }


# ============================================================
# ONE NESTED FOLD
# ============================================================

def run_fold(
    df,
    fold,
    holding_period,
    k,
    cost_bps,
):

    train = df[
        (df["Date"] >= fold["train_start"])
        &
        (df["Date"] < fold["train_end"])
    ].copy()

    validation = df[
        (df["Date"] >= fold["validation_start"])
        &
        (df["Date"] < fold["validation_end"])
    ].copy()

    test = df[
        (df["Date"] >= fold["test_start"])
        &
        (df["Date"] < fold["test_end"])
    ].copy()

    # --------------------------------------------------------
    # Train-only factor statistics
    # --------------------------------------------------------

    stats = factor_stats(
        train,
        holding_period,
    )

    validation_results = []

    # --------------------------------------------------------
    # Select strategy on validation
    # --------------------------------------------------------

    for strategy in STRATEGIES:

        weights = make_weights(
            stats,
            strategy,
        )

        scored = add_score(
            validation,
            weights,
        )

        val_daily = backtest(
            scored,
            holding_period,
            k,
            cost_bps,
        )

        val_metrics = metrics(
            val_daily
        )

        validation_results.append(
            {
                "strategy": strategy,
                "sharpe":
                    val_metrics.get(
                        "Sharpe",
                        np.nan,
                    ),
            }
        )

    validation_results = (
        pd.DataFrame(
            validation_results
        )
        .sort_values(
            "sharpe",
            ascending=False,
        )
    )

    selected_strategy = (
        validation_results
        .iloc[0]["strategy"]
    )

    # --------------------------------------------------------
    # Freeze strategy
    # --------------------------------------------------------

    weights = make_weights(
        stats,
        selected_strategy,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    scored_test = add_score(
        test,
        weights,
    )

    test_daily = backtest(
        scored_test,
        holding_period,
        k,
        cost_bps,
    )

    test_metrics = metrics(
        test_daily
    )

    return {
        "fold":
            fold["fold"],

        "selected_strategy":
            selected_strategy,

        "selected_factors":
            "|".join(
                weights.keys()
            ),

        "validation_sharpe":
            validation_results
            .iloc[0]["sharpe"],

        "test_sharpe":
            test_metrics.get(
                "Sharpe",
                np.nan,
            ),

        "test_sortino":
            test_metrics.get(
                "Sortino",
                np.nan,
            ),

        "test_cagr":
            test_metrics.get(
                "CAGR",
                np.nan,
            ),

        "test_return":
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

        "test_volatility":
            test_metrics.get(
                "volatility",
                np.nan,
            ),

        "cost":
            test_metrics.get(
                "cost",
                np.nan,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    folds = create_folds(
        df["Date"]
    )

    print("=" * 75)
    print(
        "NESTED STRATEGY ROBUSTNESS"
    )
    print("=" * 75)

    all_results = []

    # ========================================================
    # GRID
    # ========================================================

    total = (
        len(HOLDING_PERIODS)
        * len(K_VALUES)
        * len(COST_BPS)
    )

    current = 0

    for holding_period in HOLDING_PERIODS:

        for k in K_VALUES:

            for cost_bps in COST_BPS:

                current += 1

                print()
                print(
                    f"[{current}/{total}] "
                    f"H={holding_period} "
                    f"K={k} "
                    f"Cost={cost_bps}bps"
                )

                fold_results = []

                for fold in folds:

                    result = run_fold(
                        df,
                        fold,
                        holding_period,
                        k,
                        cost_bps,
                    )

                    result[
                        "holding_period"
                    ] = holding_period

                    result[
                        "K"
                    ] = k

                    result[
                        "cost_bps"
                    ] = cost_bps

                    fold_results.append(
                        result
                    )

                fold_df = pd.DataFrame(
                    fold_results
                )

                # ------------------------------------------------
                # Aggregate
                # ------------------------------------------------

                all_results.append(
                    {
                        "holding_period":
                            holding_period,

                        "K":
                            k,

                        "cost_bps":
                            cost_bps,

                        "mean_test_sharpe":
                            fold_df[
                                "test_sharpe"
                            ].mean(),

                        "median_test_sharpe":
                            fold_df[
                                "test_sharpe"
                            ].median(),

                        "positive_fold_pct":
                            np.mean(
                                fold_df[
                                    "test_sharpe"
                                ] > 0
                            ),

                        "mean_test_cagr":
                            fold_df[
                                "test_cagr"
                            ].mean(),

                        "median_test_cagr":
                            fold_df[
                                "test_cagr"
                            ].median(),

                        "mean_test_max_dd":
                            fold_df[
                                "test_max_dd"
                            ].mean(),

                        "mean_test_hit_rate":
                            fold_df[
                                "test_hit_rate"
                            ].mean(),

                        "mean_validation_sharpe":
                            fold_df[
                                "validation_sharpe"
                            ].mean(),
                    }
                )

                print(
                    f"Mean OOS Sharpe: "
                    f"{fold_df['test_sharpe'].mean():.3f}"
                )

    summary = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "strategy_robustness_summary.csv"
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
        "FULL ROBUSTNESS RESULTS"
    )
    print("=" * 75)

    print(
        summary
        .sort_values(
            "mean_test_sharpe",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # BEST CONFIG BY COST
    # ========================================================

    for cost in COST_BPS:

        subset = summary[
            summary["cost_bps"]
            == cost
        ]

        best = subset.iloc[
            subset[
                "mean_test_sharpe"
            ].argmax()
        ]

        print()
        print(
            "=" * 75
        )
        print(
            f"BEST @ {cost} BPS"
        )
        print(
            "=" * 75
        )

        print(
            best.to_string()
        )

    print()
    print(
        f"Saved -> {summary_path}"
    )


if __name__ == "__main__":
    main()