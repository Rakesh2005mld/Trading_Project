from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MARKET_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cross_sectional_dataset.csv"
)

RAG_DIR = (
    PROJECT_ROOT
    / "models"
    / "rag"
)

FILING_FEATURE_DIR = (
    RAG_DIR
    / "local_filing_features"
)

OUTPUT_FILE = (
    RAG_DIR
    / "daily_rag_features.csv"
)


# ============================================================
# RAG FEATURE COLUMNS
# ============================================================

RAG_COLUMNS = [
    "rag_sentiment",
    "rag_confidence",
    "rag_growth_relevance",
    "rag_catalyst_relevance",
    "rag_risk_relevance",
    "rag_outlook_relevance",
    "evidence_count",
]


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    print("=" * 75)
    print("LOADING MARKET DATA")
    print("=" * 75)

    if not MARKET_DATA.exists():

        raise FileNotFoundError(
            f"Market dataset not found:\n"
            f"{MARKET_DATA}"
        )

    df = pd.read_csv(
        MARKET_DATA
    )

    # Find date column robustly.
    date_column = None

    for candidate in [
        "date",
        "Date",
        "DATE",
    ]:

        if candidate in df.columns:

            date_column = candidate
            break

    if date_column is None:

        raise ValueError(
            "Could not find date column."
        )

    # --------------------------------------------------------
    # Detect ticker column
    # --------------------------------------------------------

    ticker_column = None

    ticker_candidates = [
        "ticker",
        "Ticker",
        "TICKER",
        "symbol",
        "Symbol",
        "SYMBOL",
        "stock",
        "Stock",
        "security",
        "Security",
    ]

    for candidate in ticker_candidates:

        if candidate in df.columns:
            ticker_column = candidate
            break

    # Case-insensitive fallback
    if ticker_column is None:

        normalized_columns = {
            str(column).strip().lower(): column
            for column in df.columns
        }

        for candidate in [
            "ticker",
            "symbol",
            "stock",
            "security",
        ]:

            if candidate in normalized_columns:

                ticker_column = normalized_columns[
                    candidate
                ]

                break

    if ticker_column is None:

        raise ValueError(
            "Could not detect ticker column.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Standardize names
    # --------------------------------------------------------

    df = df.rename(
        columns={
            date_column: "date",
            ticker_column: "ticker",
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

    df = df.sort_values(
        ["date","ticker"]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Dates   : "
        f"{df['date'].nunique():,}"
    )

    print(
        f"Tickers : "
        f"{df['ticker'].nunique()}"
    )

    print(
        f"Range   : "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# LOAD FILING FEATURES
# ============================================================

def load_filing_features():

    print("\n" + "=" * 75)
    print("LOADING FILING-LEVEL RAG FEATURES")
    print("=" * 75)

    files = sorted(
        FILING_FEATURE_DIR.glob(
            "*_local.json"
        )
    )

    print(
        f"Files found : {len(files)}"
    )

    if not files:

        raise FileNotFoundError(
            "No local RAG feature files found."
        )

    rows = []

    for path in files:

        try:

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            rows.append(
                {
                    "ticker":
                        str(
                            data["ticker"]
                        ).upper(),

                    "filing_date":
                        pd.to_datetime(
                            data["filing_date"]
                        ),

                    "report_date":
                        data.get(
                            "report_date",
                            None
                        ),

                    "form":
                        data.get(
                            "form",
                            None
                        ),

                    "rag_sentiment":
                        data.get(
                            "rag_sentiment",
                            np.nan
                        ),

                    "rag_confidence":
                        data.get(
                            "rag_confidence",
                            np.nan
                        ),

                    "rag_growth_relevance":
                        data.get(
                            "rag_growth_relevance",
                            np.nan
                        ),

                    "rag_catalyst_relevance":
                        data.get(
                            "rag_catalyst_relevance",
                            np.nan
                        ),

                    "rag_risk_relevance":
                        data.get(
                            "rag_risk_relevance",
                            np.nan
                        ),

                    "rag_outlook_relevance":
                        data.get(
                            "rag_outlook_relevance",
                            np.nan
                        ),

                    "evidence_count":
                        data.get(
                            "evidence_count",
                            0
                        ),
                }
            )

        except Exception as exc:

            print(
                f"WARNING: failed to read "
                f"{path.name}: {exc}"
            )

    features = pd.DataFrame(
        rows
    )

    features = features.sort_values(
        [
            "ticker",
            "filing_date",
        ]
    ).reset_index(
        drop=True
    )

    # Check duplicates.
    duplicates = features.duplicated(
        subset=[
            "ticker",
            "filing_date",
        ],
        keep=False,
    )

    if duplicates.any():

        print(
            "\nWARNING: duplicate ticker/filing_date:"
        )

        print(
            features.loc[
                duplicates,
                [
                    "ticker",
                    "filing_date",
                ]
            ]
        )

        # Keep the first occurrence.
        features = features.drop_duplicates(
            subset=[
                "ticker",
                "filing_date",
            ],
            keep="first",
        )

    print(
        f"Valid filing features : "
        f"{len(features)}"
    )

    print(
        f"Tickers                : "
        f"{features['ticker'].nunique()}"
    )

    print(
        f"Filing range            : "
        f"{features['filing_date'].min().date()} -> "
        f"{features['filing_date'].max().date()}"
    )

    return features


# ============================================================
# ALIGN FILINGS TO DAILY DATA
# ============================================================

def align_daily_features(
    market,
    filings,
):

    print("\n" + "=" * 75)
    print("ALIGNING RAG TO TRADING DATES")
    print("=" * 75)

    market = market.copy()
    filings = filings.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # merge_asof requires the 'on' column to be globally sorted.
    # Therefore sort by DATE first.
    # --------------------------------------------------------

    market = market.sort_values(
        ["date", "ticker"]
    ).reset_index(
        drop=True
    )

    filings = filings.sort_values(
        ["filing_date", "ticker"]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Strict information cutoff:
    #
    # filing_date < trading_date
    #
    # allow_exact_matches=False prevents a filing released
    # on the same calendar date from being used for that date.
    # --------------------------------------------------------

    daily = pd.merge_asof(
        market,
        filings,
        left_on="date",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )

    # --------------------------------------------------------
    # RAG availability
    # --------------------------------------------------------

    daily["rag_available"] = (
        daily["filing_date"]
        .notna()
        .astype(int)
    )

    # --------------------------------------------------------
    # Age of information
    # --------------------------------------------------------

    daily["rag_age_days"] = np.nan

    mask = (
        daily["rag_available"] == 1
    )

    daily.loc[
        mask,
        "rag_age_days"
    ] = (
        daily.loc[
            mask,
            "date"
        ]
        - daily.loc[
            mask,
            "filing_date"
        ]
    ).dt.days

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    leakage = (
        daily["filing_date"].notna()
        &
        (
            daily["filing_date"]
            >= daily["date"]
        )
    )

    if leakage.any():

        print(
            "\nLEAKAGE ROWS:"
        )

        print(
            daily.loc[
                leakage,
                [
                    "date",
                    "ticker",
                    "filing_date",
                ],
            ].head(20)
        )

        raise RuntimeError(
            "LOOK-AHEAD LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    total_rows = len(daily)

    rows_with_rag = int(
        daily["rag_available"].sum()
    )

    coverage = (
        rows_with_rag
        / total_rows
        * 100
    )

    print(
        f"Total market rows : "
        f"{total_rows:,}"
    )

    print(
        f"Rows with RAG     : "
        f"{rows_with_rag:,}"
    )

    print(
        f"RAG coverage      : "
        f"{coverage:.2f}%"
    )

    return daily

# ============================================================
# SAVE
# ============================================================

def save_daily_features(
    daily
):

    RAG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Put RAG columns near the end.
    ordered_columns = []

    for column in daily.columns:

        ordered_columns.append(
            column
        )

    daily.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("DAILY RAG FEATURES SAVED")
    print("=" * 75)

    print(
        f"File : {OUTPUT_FILE}"
    )

    print(
        f"Rows : {len(daily):,}"
    )

    print(
        f"Columns : {len(daily.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    market = load_market_data()

    filings = load_filing_features()

    daily = align_daily_features(
        market,
        filings
    )

    save_daily_features(
        daily
    )

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()