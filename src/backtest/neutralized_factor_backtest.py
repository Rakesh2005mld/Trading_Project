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


HORIZON = 5
K = 3

TRANSACTION_COST_BPS = 10
GROSS_EXPOSURE = 1.0

TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3


FACTORS = [
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


NEUTRALIZATION_MODES = [
    "raw",
    "beta_neutral",
    "sector_neutral",
    "beta_sector_neutral",
]


STRATEGIES = [
    "best_single",
    "top3_equal",
    "top5_equal",
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
            pd.to_datetime(dates).unique()
        )
    )

    first_date = dates.min()
    last_date = dates.max()

    folds = []

    train_start = first_date
    fold_id = 1

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
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": train_end,
                "validation_end": validation_end,
                "test_start": validation_end,
                "test_end": test_end,
            }
        )

        fold_id += 1

        train_start += pd.DateOffset(
            months=STEP_MONTHS
        )

    return folds


# ============================================================
# TRAIN FACTOR STATS
# ============================================================

def factor_stats(train):

    train = train.copy()

    train["future_return"] = (
        train
        .groupby("Ticker")["Close"]
        .shift(-HORIZON)
        / train["Close"]
        - 1.0
    )

    results = []

    for factor in FACTORS:

        temp = (
            train[
                [
                    "Date",
                    factor,
                    "future_return",
                ]
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        ics = []

        for _, day in temp.groupby("Date"):

            if len(day) < 5:
                continue

            if day[factor].nunique() < 2:
                continue

            if day["future_return"].nunique() < 2:
                continue

            ic = (
                day[factor]
                .rank()
                .corr(
                    day["future_return"].rank()
                )
            )

            if pd.notna(ic):
                ics.append(ic)

        mean_ic = (
            np.mean(ics)
            if ics
            else 0.0
        )

        results.append(
            {
                "factor": factor,
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

    elif strategy == "all_equal":
        selected = stats

    else:
        raise ValueError(strategy)

    weights = {}

    if len(selected) == 0:
        return weights

    n = len(selected)

    for _, row in selected.iterrows():

        direction = np.sign(
            row["mean_ic"]
        )

        if direction == 0:
            direction = 1.0

        weights[
            row["factor"]
        ] = direction / n

    return weights


# ============================================================
# DAILY CROSS-SECTIONAL Z SCORE
# ============================================================

def cross_sectional_z(
    df,
    col,
):

    mean = (
        df
        .groupby("Date")[col]
        .transform("mean")
    )

    std = (
        df
        .groupby("Date")[col]
        .transform("std")
    )

    return (
        df[col] - mean
    ) / std.replace(
        0,
        np.nan,
    )


# ============================================================
# RAW FACTOR SCORE
# ============================================================

def raw_score(
    df,
    weights,
):

    result = df.copy()

    result["score"] = 0.0

    for factor, weight in weights.items():

        z = cross_sectional_z(
            result,
            factor,
        )

        result["score"] += (
            weight * z
        )

    return result


# ============================================================
# BETA NEUTRALIZATION
# ============================================================

def residualize_beta(
    df,
    score_column="score",
):

    result = df.copy()

    result["neutral_score"] = np.nan

    for date, day in result.groupby(
        "Date"
    ):

        mask = day[
            [
                "beta_60",
                score_column,
            ]
        ].notna().all(axis=1)

        if mask.sum() < 4:
            continue

        x = day.loc[
            mask,
            "beta_60"
        ].values

        y = day.loc[
            mask,
            score_column
        ].values

        x_mean = x.mean()
        y_mean = y.mean()

        denom = np.sum(
            (x - x_mean) ** 2
        )

        if denom <= 1e-12:

            residuals = (
                y - y_mean
            )

        else:

            beta = (
                np.sum(
                    (x - x_mean)
                    * (y - y_mean)
                )
                / denom
            )

            intercept = (
                y_mean
                - beta * x_mean
            )

            residuals = (
                y
                - (
                    intercept
                    + beta * x
                )
            )

        result.loc[
            day.index[mask],
            "neutral_score",
        ] = residuals

    return result


# ============================================================
# SECTOR NEUTRALIZATION
# ============================================================

def residualize_sector(
    df,
    score_column,
):

    result = df.copy()

    result["neutral_score"] = np.nan

    sector_columns = [
        c
        for c in [
            "sector_Consumer",
            "sector_Energy",
            "sector_Financials",
            "sector_Technology",
        ]
        if c in result.columns
    ]

    for date, day in result.groupby(
        "Date"
    ):

        available = day[
            [score_column]
            + sector_columns
        ].notna().all(axis=1)

        if available.sum() < 4:
            continue

        X = day.loc[
            available,
            sector_columns
        ].values

        y = day.loc[
            available,
            score_column
        ].values

        X = np.column_stack(
            [
                np.ones(
                    len(X)
                ),
                X,
            ]
        )

        try:

            coefficients = np.linalg.lstsq(
                X,
                y,
                rcond=None,
            )[0]

            fitted = (
                X @ coefficients
            )

            residuals = (
                y - fitted
            )

            result.loc[
                day.index[available],
                "neutral_score",
            ] = residuals

        except np.linalg.LinAlgError:

            result.loc[
                day.index[available],
                "neutral_score",
            ] = (
                y - y.mean()
            )

    return result


# ============================================================
# APPLY NEUTRALIZATION
# ============================================================

def apply_neutralization(
    df,
    mode,
):

    if mode == "raw":

        result = df.copy()

        result["final_score"] = (
            result["score"]
        )

        return result

    if mode == "beta_neutral":

        result = residualize_beta(
            df,
            "score",
        )

        result["final_score"] = (
            result["neutral_score"]
        )

        return result

    if mode == "sector_neutral":

        result = residualize_sector(
            df,
            "score",
        )

        result["final_score"] = (
            result["neutral_score"]
        )

        return result

    if mode == "beta_sector_neutral":

        result = residualize_beta(
            df,
            "score",
        )

        result = residualize_sector(
            result.rename(
                columns={
                    "neutral_score":
                        "score_beta_neutral"
                }
            ),
            "score_beta_neutral",
        )

        result["final_score"] = (
            result["neutral_score"]
        )

        return result

    raise ValueError(mode)


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    df,
):

    dates = pd.DatetimeIndex(
        sorted(
            df["Date"].unique()
        )
    )

    rebalance_dates = dates[
        ::HORIZON
    ]

    periods = []

    for entry_date in rebalance_dates:

        idx = dates.searchsorted(
            entry_date
        )

        end_idx = (
            idx + HORIZON
        )

        if end_idx > len(dates):
            break

        holding_dates = dates[
            idx:end_idx
        ]

        entry = df[
            df["Date"] == entry_date
        ].copy()

        entry = entry.dropna(
            subset=["final_score"]
        )

        if len(entry) < 2 * K:
            continue

        entry = entry.sort_values(
            "final_score"
        )

        longs = set(
            entry
            .tail(K)
            ["Ticker"]
        )

        shorts = set(
            entry
            .head(K)
            ["Ticker"]
        )

        long_weight = (
            GROSS_EXPOSURE
            / (2 * K)
        )

        short_weight = (
            -GROSS_EXPOSURE
            / (2 * K)
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

            long_returns = day[
                day["Ticker"].isin(
                    longs
                )
            ]["daily_return"]

            short_returns = day[
                day["Ticker"].isin(
                    shorts
                )]["daily_return"]

            if (
                len(long_returns) < K
                or len(short_returns) < K
            ):
                continue

            ret = (
                long_weight
                * long_returns.sum()
                +
                short_weight
                * short_returns.sum()
            )

            rows.append(
                {
                    "Date": date,
                    "gross_return": ret,
                }
            )

        if not rows:
            continue

        period = pd.DataFrame(
            rows
        )

        # Entry + exit = 200% turnover
        total_cost = (
            2.0
            * TRANSACTION_COST_BPS
            / 10000.0
        )

        period["cost"] = 0.0

        period.loc[
            period.index[0],
            "cost",
        ] = total_cost / 2

        period.loc[
            period.index[-1],
            "cost",
        ] = total_cost / 2

        period["net_return"] = (
            period["gross_return"]
            - period["cost"]
        )

        period["entry_date"] = (
            entry_date
        )

        periods.append(
            period
        )

    if not periods:
        return pd.DataFrame()

    return (
        pd.concat(
            periods,
            ignore_index=True,
        )
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    daily,
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
        downside.std(
            ddof=1
        )
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

    dd = (
        equity
        / running_max
        - 1.0
    )

    total_return = (
        equity[-1] - 1.0
    )

    first = pd.Timestamp(
        daily["Date"].iloc[0]
    )

    last = pd.Timestamp(
        daily["Date"].iloc[-1]
    )

    years = max(
        (
            last - first
        ).days
        / 365.25,
        1 / 365.25,
    )

    cagr = (
        (1 + total_return)
        ** (1 / years)
        - 1
    )

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CAGR": cagr,
        "return": total_return,
        "max_dd": dd.min(),
        "hit_rate":
            np.mean(r > 0),
    }


# ============================================================
# ONE FOLD
# ============================================================

def run_fold(
    df,
    fold,
    mode,
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

    stats = factor_stats(
        train
    )

    validation_results = []

    # --------------------------------------------------------
    # Select factor strategy using validation
    # --------------------------------------------------------

    for strategy in STRATEGIES:

        weights = make_weights(
            stats,
            strategy,
        )

        val = raw_score(
            validation,
            weights,
        )

        val = apply_neutralization(
            val,
            mode,
        )

        daily = backtest(
            val
        )

        m = metrics(
            daily
        )

        validation_results.append(
            {
                "strategy": strategy,
                "sharpe":
                    m.get(
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
        .reset_index(drop=True)
    )

    selected_strategy = (
        validation_results
        .iloc[0]["strategy"]
    )

    # --------------------------------------------------------
    # Freeze strategy and apply to TEST
    # --------------------------------------------------------

    weights = make_weights(
        stats,
        selected_strategy,
    )

    test = raw_score(
        test,
        weights,
    )

    test = apply_neutralization(
        test,
        mode,
    )

    daily = backtest(
        test
    )

    m = metrics(
        daily
    )

    return {
        "fold":
            fold["fold"],

        "mode":
            mode,

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
            m.get(
                "Sharpe",
                np.nan,
            ),

        "test_sortino":
            m.get(
                "Sortino",
                np.nan,
            ),

        "test_cagr":
            m.get(
                "CAGR",
                np.nan,
            ),

        "test_return":
            m.get(
                "return",
                np.nan,
            ),

        "test_max_dd":
            m.get(
                "max_dd",
                np.nan,
            ),

        "test_hit_rate":
            m.get(
                "hit_rate",
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
        "FACTOR NEUTRALIZATION WALK-FORWARD"
    )
    print("=" * 75)

    print(
        f"Horizon : {HORIZON} days"
    )

    print(
        f"K       : {K}"
    )

    print(
        f"Cost    : {TRANSACTION_COST_BPS} bps"
    )

    all_results = []

    for mode in NEUTRALIZATION_MODES:

        print()
        print(
            "=" * 75
        )
        print(
            f"MODE: {mode}"
        )
        print(
            "=" * 75
        )

        for fold in folds:

            result = run_fold(
                df,
                fold,
                mode,
            )

            all_results.append(
                result
            )

        mode_df = pd.DataFrame(
            [
                r
                for r in all_results
                if r["mode"] == mode
            ]
        )

        print(
            f"Mean OOS Sharpe: "
            f"{mode_df['test_sharpe'].mean():.3f}"
        )

        print(
            f"Median OOS Sharpe: "
            f"{mode_df['test_sharpe'].median():.3f}"
        )

        print(
            f"Positive folds: "
            f"{np.mean(mode_df['test_sharpe'] > 0):.2%}"
        )

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results
        .groupby("mode")
        .agg(
            mean_test_sharpe=(
                "test_sharpe",
                "mean",
            ),

            median_test_sharpe=(
                "test_sharpe",
                "median",
            ),

            positive_fold_pct=(
                "test_sharpe",
                lambda x:
                np.mean(x > 0),
            ),

            mean_test_cagr=(
                "test_cagr",
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

            mean_validation_sharpe=(
                "validation_sharpe",
                "mean",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # SAVE
    # ========================================================

    fold_path = (
        OUTPUT_DIR
        / "neutralized_factor_folds.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "neutralized_factor_summary.csv"
    )

    results.to_csv(
        fold_path,
        index=False,
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
        "FINAL NEUTRALIZATION SUMMARY"
    )
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved -> {fold_path}"
    )

    print(
        f"Saved -> {summary_path}"
    )


if __name__ == "__main__":
    main()