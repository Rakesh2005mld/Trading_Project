import os
import yfinance as yf


DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


def download_stock(ticker: str, start: str):

    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    if df.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    # Ensure standard columns
    df = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ]

    # Make sure index is clean
    df.index.name = "Date"

    output_path = os.path.join(
        DATA_DIR,
        f"{ticker}.csv"
    )

    df.to_csv(output_path)

    print(
        f"Saved {ticker}: "
        f"{len(df)} rows → {output_path}"
    )


if __name__ == "__main__":

    start_date = "2015-01-01"

    download_stock("AAPL", start_date)
    download_stock("SPY", start_date)
    download_stock("^VIX", start_date)