# src/backtest/multi_horizon_factor_strategy.py

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

# Walk-forward setup
TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3

# Strategy
HORIZON = 20
K = 2

TRANSACTION_COST_BPS = 10

GROSS_EXPOSURE = 1.0

# Factor set selected from the previous
# walk-forward horizon analysis.
FACTORS = [
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "beta_60",
    "price_ma200_ratio",
    "return_60d",
    "momentum_20",
]


# ============================================================
# DATA
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
# FORWARD RETURN
# ============================================================

def create_forward_return(df):

    df = df.copy()

    df[
        f"future_{HORIZON}d"
    ] = (
        df
        .groupby("Ticker")["Close"]
        .shift(-HORIZON)
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

        train_start = (
            train_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return folds


# ============================================================
# CROSS-SECTIONAL Z-SCORE
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
    ) / std.replace(0, np.nan)


# ============================================================
# FACTOR IC WEIGHTS
# ============================================================

def calculate_factor_ics(
    train_df
):

    results = []

    target = f"future_{HORIZON}d"

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

        daily_ics = []

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

            # Rank-based correlation
            rank_x = x.rank(
                method="average"
            )

            rank_y = y.rank(
                method="average"
            )

            corr = rank_x.corr(
                rank_y
            )

            if pd.notna(corr):
                daily_ics.append(corr)

        if daily_ics:

            daily_ics = np.asarray(
                daily_ics
            )

            mean_ic = daily_ics.mean()

            std_ic = (
                daily_ics.std(ddof=1)
                if len(daily_ics) > 1
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

        else:

            mean_ic = 0.0
            icir = 0.0
            std_ic = np.nan

        results.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "ic_std": std_ic,
                "icir": icir,
            }
        )

    result = pd.DataFrame(
        results
    )

    return result


# ============================================================
# FACTOR WEIGHT SCHEMES
# ============================================================

def build_factor_weights(
    ic_df
):

    schemes = {}

    # --------------------------------------------------------
    # Equal weight
    # --------------------------------------------------------

    equal = pd.Series(
        1.0,
        index=FACTORS,
    )

    equal = equal / equal.abs().sum()

    schemes["equal_weight"] = equal

    # --------------------------------------------------------
    # Mean IC weighting
    # --------------------------------------------------------

    ic_weights = (
        ic_df
        .set_index("factor")["mean_ic"]
        .reindex(FACTORS)
        .fillna(0.0)
    )

    if ic_weights.abs().sum() > 0:

        ic_weights = (
            ic_weights
            / ic_weights.abs().sum()
        )

    schemes["ic_weighted"] = ic_weights

    # --------------------------------------------------------
    # ICIR weighting
    # --------------------------------------------------------

    icir_weights = (
        ic_df
        .set_index("factor")["icir"]
        .reindex(FACTORS)
        .fillna(0.0)
    )

    # Preserve direction but prevent one factor from
    # dominating solely because of scale.
    if icir_weights.abs().sum() > 0:

        icir_weights = (
            icir_weights
            / icir_weights.abs().sum()
        )

    schemes["icir_weighted"] = icir_weights

    return schemes


# ============================================================
# BUILD DAILY SIGNAL
# ============================================================

def build_factor_score(
    df,
    weights,
):

    result = df.copy()

    result["factor_score"] = 0.0

    for factor in FACTORS:

        z = cross_sectional_zscore(
            result,
            factor,
        )

        weight = weights.get(
            factor,
            0.0,
        )

        result["factor_score"] += (
            weight * z
        )

    return result


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def run_portfolio(
    df
):

    target = f"future_{HORIZON}d"

    records = []

    for date, day in df.groupby(
        "Date",
        sort=True,
    ):

        day = day.dropna(
            subset=[
                "factor_score",
                target,
            ]
        ).copy()

        if len(day) < 2 * K:
            continue

        day = day.sort_values(
            "factor_score"
        )

        longs = day.tail(K)
        shorts = day.head(K)

        # Gross exposure = 1.0
        long_weight = (
            GROSS_EXPOSURE
            / 2.0
            / K
        )

        short_weight = (
            -GROSS_EXPOSURE
            / 2.0
            / K
        )

        long_return = (
            longs[target].mean()
        )

        short_return = (
            shorts[target].mean()
        )

        gross_return = (
            0.5 * long_return
            - 0.5 * short_return
        )

        # One-way turnover estimate:
        #
        # Previous portfolio isn't explicitly tracked here,
        # so this represents the portfolio's absolute weight
        # turnover for a complete rebalance.
        #
        # Total absolute position change from flat:
        # 2 * gross exposure = 2.
        #
        # We later compare this with a more realistic
        # staggered holding-period implementation.
        turnover = (
            abs(long_weight) * K
            + abs(short_weight) * K
        )

        cost = (
            turnover
            * TRANSACTION_COST_BPS
            / 10000.0
        )

        net_return = (
            gross_return - cost
        )

        records.append(
            {
                "Date": date,
                "long_return": long_return,
                "short_return": short_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
            }
        )

    return pd.DataFrame(
        records
    )


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

    n = len(r)

    mean_daily = np.mean(r)

    volatility = (
        np.std(
            r,
            ddof=1,
        )
        * np.sqrt(252)
        if n > 1
        else np.nan
    )

    sharpe = (
        mean_daily
        / np.std(
            r,
            ddof=1,
        )
        * np.sqrt(252)
        if (
            n > 1
            and np.std(
                r,
                ddof=1,
            ) > 0
        )
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
        mean_daily
        / downside_std
        * np.sqrt(252)
        if (
            np.isfinite(downside_std)
            and downside_std > 0
        )
        else np.nan
    )

    equity = (
        1.0 + r
    ).cumprod()

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / running_max
        - 1.0
    )

    max_dd = drawdown.min()

    final_value = equity[-1]

    years = n / 252.0

    cagr = (
        final_value ** (
            1.0 / years
        )
        - 1.0
        if years > 0
        else np.nan
    )

    gross_equity = (
        1.0 + gross
    ).cumprod()

    gross_total_return = (
        gross_equity[-1] - 1.0
    )

    net_total_return = (
        equity[-1] - 1.0
    )

    return {
        "days": n,
        "gross_total_return":
            gross_total_return,
        "net_total_return":
            net_total_return,
        "CAGR": cagr,
        "annualized_volatility":
            volatility,
        "Sharpe":
            sharpe,
        "Sortino":
            sortino,
        "max_drawdown":
            max_dd,
        "hit_rate":
            np.mean(r > 0),
        "average_daily_return":
            mean_daily,
        "average_turnover":
            daily["turnover"].mean(),
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

    fold_id = fold["fold"]

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
    # Learn factor directions / weights using TRAIN ONLY
    # --------------------------------------------------------

    ic_df = calculate_factor_ics(
        train_df
    )

    weight_schemes = (
        build_factor_weights(
            ic_df
        )
    )

    fold_results = []

    for scheme_name, weights in (
        weight_schemes.items()
    ):

        # ----------------------------------------------------
        # Score validation and test using weights learned
        # exclusively from train.
        # ----------------------------------------------------

        val_scored = build_factor_score(
            validation_df,
            weights,
        )

        test_scored = build_factor_score(
            test_df,
            weights,
        )

        val_daily = run_portfolio(
            val_scored
        )

        test_daily = run_portfolio(
            test_scored
        )

        val_metrics = calculate_metrics(
            val_daily
        )

        test_metrics = calculate_metrics(
            test_daily
        )

        fold_results.append(
            {
                "fold": fold_id,
                "scheme": scheme_name,

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

                "test_max_dd":
                    test_metrics.get(
                        "max_drawdown",
                        np.nan,
                    ),

                "test_total_return":
                    test_metrics.get(
                        "net_total_return",
                        np.nan,
                    ),

                "test_hit_rate":
                    test_metrics.get(
                        "hit_rate",
                        np.nan,
                    ),

                "test_turnover":
                    test_metrics.get(
                        "average_turnover",
                        np.nan,
                    ),

                "test_cost":
                    test_metrics.get(
                        "total_cost",
                        np.nan,
                    ),
            }
        )

    return (
        pd.DataFrame(fold_results),
        ic_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = create_forward_return(
        df
    )

    folds = create_folds(
        df["Date"]
    )

    print()
    print("=" * 75)
    print("WALK-FORWARD SETUP")
    print("=" * 75)

    print(
        "Folds:",
        len(folds),
    )

    print(
        "Horizon:",
        f"{HORIZON} trading days",
    )

    print(
        "Top/Bottom K:",
        K,
    )

    print(
        "Transaction cost:",
        f"{TRANSACTION_COST_BPS} bps",
    )

    all_results = []
    all_ic = []

    # ========================================================
    # RUN FOLDS
    # ========================================================

    for fold in folds:

        print()
        print(
            f"Fold {fold['fold']:02d} | "
            f"Test: "
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

        all_ic.append(
            ic_df
        )

        print(
            fold_result[
                [
                    "scheme",
                    "validation_sharpe",
                    "test_sharpe",
                    "test_cagr",
                ]
            ].to_string(
                index=False
            )
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    ic_results = pd.concat(
        all_ic,
        ignore_index=True,
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_path = (
        OUTPUT_DIR
        / "factor_strategy_walk_forward.csv"
    )

    results.to_csv(
        fold_path,
        index=False,
    )

    ic_path = (
        OUTPUT_DIR
        / "factor_strategy_fold_ics.csv"
    )

    ic_results.to_csv(
        ic_path,
        index=False,
    )

    # ========================================================
    # AGGREGATE RESULTS
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

            validation_positive_fold_pct=(
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

            test_positive_fold_pct=(
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

            mean_test_max_dd=(
                "test_max_dd",
                "mean",
            ),

            mean_test_hit_rate=(
                "test_hit_rate",
                "mean",
            ),

            mean_test_turnover=(
                "test_turnover",
                "mean",
            ),

            total_test_cost=(
                "test_cost",
                "sum",
            ),

            folds=(
                "fold",
                "count",
            ),
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / "factor_strategy_walk_forward_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "WALK-FORWARD FACTOR STRATEGY SUMMARY"
    )
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # FEATURE IC SUMMARY
    # ========================================================

    ic_summary = (
        ic_results
        .groupby("factor")
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
                np.mean(
                    x > 0
                ) * 100,
            ),
        )
        .reset_index()
        .sort_values(
            "mean_ic",
            ascending=False,
        )
    )

    ic_summary_path = (
        OUTPUT_DIR
        / "factor_strategy_ic_summary.csv"
    )

    ic_summary.to_csv(
        ic_summary_path,
        index=False,
    )

    print()
    print("=" * 75)
    print("FACTOR IC SUMMARY")
    print("=" * 75)

    print(
        ic_summary.to_string(
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

    print(
        ic_summary_path
    )

    print()
    print(
        "Factor strategy walk-forward analysis complete."
    )


if __name__ == "__main__":
    main()