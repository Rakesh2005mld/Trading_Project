from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MARKET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_dataset.csv"
)

LSTM_FILE = (
    PROJECT_ROOT
    / "models"
    / "cross_sectional_5d_lstm_predictions.csv"
)

RAG_FILE = (
    PROJECT_ROOT
    / "models"
    / "rag"
    / "cross_sectional_rag_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_rag_lstm_dataset.csv"
)


# ============================================================
# HELPERS
# ============================================================

def find_column(
    df,
    candidates,
    description,
):

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:

            return normalized[key]

    raise ValueError(
        f"Could not find {description}.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def standardize_keys(df):

    date_col = find_column(
        df,
        ["date", "Date", "DATE"],
        "date column",
    )

    ticker_col = find_column(
        df,
        [
            "ticker",
            "Ticker",
            "TICKER",
            "symbol",
            "Symbol",
        ],
        "ticker column",
    )

    df = df.rename(
        columns={
            date_col: "date",
            ticker_col: "ticker",
        }
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


# ============================================================
# LOAD
# ============================================================

def main():

    print("=" * 75)
    print("BUILDING FINAL RAG + LSTM DATASET")
    print("=" * 75)

    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------

    market = pd.read_csv(
        MARKET_FILE
    )

    market = standardize_keys(
        market
    )

    print(
        f"Market rows : {len(market):,}"
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm = pd.read_csv(
        LSTM_FILE
    )

    lstm = standardize_keys(
        lstm
    )

    print(
        f"\nLSTM columns:"
    )

    print(
        lstm.columns.tolist()
    )

    # --------------------------------------------------------
    # Find LSTM prediction column
    # --------------------------------------------------------

    prediction_candidates = [
        "prediction",
        "predicted",
        "predicted_return",
        "lstm_prediction",
        "lstm_pred",
        "score",
        "pred",
    ]

    lstm_prediction_col = find_column(
        lstm,
        prediction_candidates,
        "LSTM prediction column",
    )

    print(
        f"LSTM prediction column: "
        f"{lstm_prediction_col}"
    )

    # Keep only key + prediction.
    lstm_small = lstm[
        [
            "date",
            "ticker",
            lstm_prediction_col,
        ]
    ].copy()

    lstm_small = lstm_small.rename(
        columns={
            lstm_prediction_col:
                "lstm_prediction"
        }
    )

    # Prevent duplicate keys.
    lstm_small = (
        lstm_small
        .drop_duplicates(
            subset=[
                "date",
                "ticker",
            ]
        )
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag = pd.read_csv(
        RAG_FILE
    )

    rag = standardize_keys(
        rag
    )

    print(
        f"\nRAG rows : {len(rag):,}"
    )

    # RAG columns we want in the model.
    rag_feature_columns = [
        "date",
        "ticker",
        "rag_sentiment_z",
        "rag_sentiment_rank",
        "rag_sentiment_decay_z",
        "rag_confidence_z",
        "rag_growth_relevance_z",
        "rag_catalyst_relevance_z",
        "rag_risk_relevance_z",
        "rag_outlook_relevance_z",
        "rag_age_days",
        "rag_freshness",
    ]

    missing_rag = [
        c
        for c in rag_feature_columns
        if c not in rag.columns
    ]

    if missing_rag:

        raise ValueError(
            "Missing RAG columns:\n"
            + "\n".join(missing_rag)
        )

    rag_small = rag[
        rag_feature_columns
    ].copy()

    rag_small = (
        rag_small
        .drop_duplicates(
            subset=[
                "date",
                "ticker",
            ]
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print(
        "\nMerging market + LSTM..."
    )

    final = market.merge(
        lstm_small,
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        f"After LSTM merge : "
        f"{len(final):,}"
    )

    print(
        "Merging RAG..."
    )

    final = final.merge(
        rag_small,
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        f"After RAG merge : "
        f"{len(final):,}"
    )

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    final["rag_available"] = (
        final[
            "rag_sentiment_z"
        ]
        .notna()
        .astype(int)
    )

    # --------------------------------------------------------
    # Restrict to rows where LSTM exists.
    #
    # This is the actual modeling universe for the
    # LSTM + RAG experiment.
    # --------------------------------------------------------

    before = len(final)

    final = final[
        final[
            "lstm_prediction"
        ].notna()
    ].copy()

    print(
        f"LSTM-valid rows: "
        f"{len(final):,}"
    )

    print(
        f"Removed rows: "
        f"{before - len(final):,}"
    )

    # --------------------------------------------------------
    # Date/ticker order
    # --------------------------------------------------------

    final = final.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "FINAL DATASET SAVED"
    )

    print(
        "=" * 75
    )

    print(
        f"File    : {OUTPUT_FILE}"
    )

    print(
        f"Rows    : {len(final):,}"
    )

    print(
        f"Columns : {len(final.columns)}"
    )

    print(
        f"RAG coverage among LSTM rows: "
        f"{final['rag_available'].mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()