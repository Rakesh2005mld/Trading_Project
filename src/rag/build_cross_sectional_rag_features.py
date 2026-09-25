from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "rag"
    / "daily_rag_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "rag"
    / "cross_sectional_rag_features.csv"
)


# ============================================================
# FEATURES
# ============================================================

RAW_FEATURES = [
    "rag_sentiment",
    "rag_confidence",
    "rag_growth_relevance",
    "rag_catalyst_relevance",
    "rag_risk_relevance",
    "rag_outlook_relevance",
]


def cross_sectional_zscore(series):
    """
    Z-score across stocks on the same date.

    If cross-sectional std is zero, return 0.
    """

    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or pd.isna(std):
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series - mean
    ) / std


def main():

    print("=" * 75)
    print("BUILDING CROSS-SECTIONAL RAG FEATURES")
    print("=" * 75)

    df = pd.read_csv(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
    )

    # --------------------------------------------------------
    # Only use rows where RAG actually exists.
    # --------------------------------------------------------

    df["rag_available"] = (
        df["rag_available"]
        .fillna(0)
        .astype(int)
    )

    available = df[
        df["rag_available"] == 1
    ].copy()

    print(
        f"Input rows       : {len(df):,}"
    )

    print(
        f"RAG rows         : {len(available):,}"
    )

    # --------------------------------------------------------
    # Cross-sectional rank
    # --------------------------------------------------------

    for feature in RAW_FEATURES:

        rank_name = (
            feature
            + "_rank"
        )

        z_name = (
            feature
            + "_z"
        )

        available[rank_name] = (
            available
            .groupby("date")[feature]
            .rank(
                method="average",
                pct=True,
            )
        )

        available[z_name] = (
            available
            .groupby("date")[feature]
            .transform(
                cross_sectional_zscore
            )
        )

    # --------------------------------------------------------
    # Age-adjusted sentiment
    #
    # Half-life = 45 trading/calendar days approximately.
    #
    # This does NOT replace the raw sentiment.
    # It gives the model an explicitly decayed version.
    # --------------------------------------------------------

    half_life = 45.0

    available[
        "rag_sentiment_decay"
    ] = (
        available[
            "rag_sentiment"
        ]
        * np.exp(
            -np.log(2)
            * available[
                "rag_age_days"
            ]
            / half_life
        )
    )

    available[
        "rag_sentiment_decay_z"
    ] = (
        available
        .groupby("date")[
            "rag_sentiment_decay"
        ]
        .transform(
            cross_sectional_zscore
        )
    )

    # --------------------------------------------------------
    # RAG availability / age
    # --------------------------------------------------------

    available[
        "rag_freshness"
    ] = 1.0 / (
        1.0
        + available[
            "rag_age_days"
        ]
    )

    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    feature_columns = [
        "date",
        "ticker",
        "filing_date",
        "rag_age_days",
        "rag_available",
        "rag_sentiment",
        "rag_confidence",
        "rag_growth_relevance",
        "rag_catalyst_relevance",
        "rag_risk_relevance",
        "rag_outlook_relevance",
        "rag_sentiment_decay",
        "rag_sentiment_decay_z",
        "rag_freshness",
    ]

    for feature in RAW_FEATURES:

        feature_columns.extend(
            [
                feature + "_rank",
                feature + "_z",
            ]
        )

    result = available[
        feature_columns
    ].copy()

    result = result.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SANITY CHECKS")
    print("=" * 75)

    print(
        f"Rows : {len(result):,}"
    )

    print(
        f"Dates: {result['date'].nunique():,}"
    )

    print(
        f"Tickers: {result['ticker'].nunique()}"
    )

    # Cross-sectional z-score means should be ~0.
    mean_abs_z = (
        result
        .groupby("date")[
            "rag_sentiment_z"
        ]
        .mean()
        .abs()
        .mean()
    )

    print(
        f"Mean absolute daily "
        f"sentiment-z mean: "
        f"{mean_abs_z:.10f}"
    )

    if (
        result["rag_sentiment_z"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "NaN values found in "
            "rag_sentiment_z."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("SAVED")
    print("=" * 75)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Columns: {len(result.columns)}"
    )

    print("\nFeature columns:")

    for column in result.columns:
        print(
            f"  {column}"
        )

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()