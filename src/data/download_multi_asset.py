import os
import yfinance as yf


RAW_DIR = "data/raw"

STOCK_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSLA",
    "JPM",
    "XOM",
]

MARKET_TICKERS = [
    "SPY",
    "^VIX",
]

START_DATE = "2015-01-01"


def download_ticker(ticker: str):

    print(f"\nDownloading {ticker}...")

    df = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{ticker} is missing: {missing}"
        )

    df = df[
        required_columns
    ].copy()

    df.index.name = "Date"

    path = os.path.join(
        RAW_DIR,
        f"{ticker}.csv"
    )

    df.to_csv(path)

    print(
        f"Saved {ticker}: "
        f"{len(df)} rows"
    )

    print(
        f"Period: "
        f"{df.index.min().date()} → "
        f"{df.index.max().date()}"
    )


def main():

    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )

    for ticker in (
        STOCK_TICKERS
        + MARKET_TICKERS
    ):

        try:
            download_ticker(ticker)

        except Exception as e:

            print(
                f"ERROR downloading "
                f"{ticker}: {e}"
            )


if __name__ == "__main__":
    main()