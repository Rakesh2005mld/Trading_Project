from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "rag"
    / "daily_rag_features.csv"
)


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print("=" * 75)
    print("RAG FEATURE DIAGNOSTICS")
    print("=" * 75)

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Dates   : {df['date'].nunique():,}"
    )

    print(
        f"Tickers : {df['ticker'].nunique()}"
    )

    # ========================================================
    # COVERAGE OVERALL
    # ========================================================

    print("\n" + "=" * 75)
    print("OVERALL COVERAGE")
    print("=" * 75)

    print(
        df["rag_available"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nCoverage: "
        f"{df['rag_available'].mean() * 100:.2f}%"
    )

    # ========================================================
    # COVERAGE BY TICKER
    # ========================================================

    print("\n" + "=" * 75)
    print("COVERAGE BY TICKER")
    print("=" * 75)

    coverage = (
        df.groupby("ticker")
        .agg(
            rows=("ticker", "size"),
            rag_rows=("rag_available", "sum"),
            coverage=("rag_available", "mean"),
        )
        .reset_index()
    )

    coverage["coverage"] *= 100

    print(
        coverage.to_string(
            index=False,
            formatters={
                "coverage": "{:.2f}%".format
            },
        )
    )

    # ========================================================
    # RAG DATE RANGE BY TICKER
    # ========================================================

    print("\n" + "=" * 75)
    print("FIRST RAG DATE BY TICKER")
    print("=" * 75)

    first_dates = (
        df[df["rag_available"] == 1]
        .groupby("ticker")["date"]
        .min()
        .reset_index(
            name="first_rag_date"
        )
    )

    print(
        first_dates.to_string(
            index=False
        )
    )

    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 75)
    print("RAG SENTIMENT DISTRIBUTION")
    print("=" * 75)

    available = df[
        df["rag_available"] == 1
    ]

    sentiment_cols = [
        "rag_sentiment",
        "rag_confidence",
        "rag_growth_relevance",
        "rag_catalyst_relevance",
        "rag_risk_relevance",
        "rag_outlook_relevance",
    ]

    print(
        available[
            sentiment_cols
        ].describe().T
    )

    # ========================================================
    # SENTIMENT BY TICKER
    # ========================================================

    print("\n" + "=" * 75)
    print("SENTIMENT BY TICKER")
    print("=" * 75)

    ticker_stats = (
        available.groupby("ticker")
        [
            [
                "rag_sentiment",
                "rag_confidence",
                "rag_growth_relevance",
                "rag_catalyst_relevance",
                "rag_risk_relevance",
                "rag_outlook_relevance",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        ticker_stats.to_string()
    )

    # ========================================================
    # INFORMATION AGE
    # ========================================================

    print("\n" + "=" * 75)
    print("RAG INFORMATION AGE")
    print("=" * 75)

    print(
        available[
            "rag_age_days"
        ].describe()
    )

    # ========================================================
    # MISSING RAG VALUES AFTER FIRST AVAILABLE DATE
    # ========================================================

    print("\n" + "=" * 75)
    print("MISSINGNESS CHECK")
    print("=" * 75)

    for ticker in sorted(
        df["ticker"].unique()
    ):

        ticker_df = df[
            df["ticker"] == ticker
        ].copy()

        first = ticker_df[
            ticker_df["rag_available"] == 1
        ]["date"].min()

        if pd.isna(first):
            continue

        post_first = ticker_df[
            ticker_df["date"] >= first
        ]

        missing_rate = (
            1
            - post_first[
                "rag_available"
            ].mean()
        )

        print(
            f"{ticker}: "
            f"first={first.date()} | "
            f"post-first missing="
            f"{missing_rate * 100:.2f}%"
        )

    # ========================================================
    # EXTREME VALUES
    # ========================================================

    print("\n" + "=" * 75)
    print("EXTREME SENTIMENT VALUES")
    print("=" * 75)

    print(
        available[
            [
                "date",
                "ticker",
                "rag_sentiment",
                "rag_confidence",
                "filing_date",
            ]
        ]
        .sort_values(
            "rag_sentiment"
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nMost positive:")

    print(
        available[
            [
                "date",
                "ticker",
                "rag_sentiment",
                "rag_confidence",
                "filing_date",
            ]
        ]
        .sort_values(
            "rag_sentiment",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()