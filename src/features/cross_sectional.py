import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/cross_sectional_dataset.csv"

STOCK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "AVGO", "TSLA", "JPM", "XOM"
]

# Broad sectors used as categorical metadata/features
SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer",
    "GOOGL": "Technology",
    "META": "Technology",
    "AVGO": "Technology",
    "TSLA": "Consumer",
    "JPM": "Financials",
    "XOM": "Energy",
}


def load_price_data(ticker):
    """Load one ticker's OHLCV data."""
    path = os.path.join(RAW_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}")

    df = df[required].copy()
    df = df.sort_values("Date")
    df = df.drop_duplicates("Date")

    df.set_index("Date", inplace=True)

    return df


def add_asset_features(df, spy_returns):
    """
    Add features that use only information available on date t
    or earlier.
    """

    close = df["Close"]
    volume = df["Volume"]

    # -----------------------------
    # Returns
    # -----------------------------
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    # Log return
    df["log_return_1d"] = np.log(close / close.shift(1))

    # -----------------------------
    # Moving averages
    # -----------------------------
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    df["price_ma20_ratio"] = close / ma20
    df["price_ma50_ratio"] = close / ma50
    df["price_ma200_ratio"] = close / ma200

    # -----------------------------
    # Momentum
    # -----------------------------
    df["momentum_5"] = close / close.shift(5) - 1
    df["momentum_10"] = close / close.shift(10) - 1
    df["momentum_20"] = close / close.shift(20) - 1

    # -----------------------------
    # Volatility
    # -----------------------------
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["volatility_60"] = df["return_1d"].rolling(60).std()

    # -----------------------------
    # RSI
    # -----------------------------
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # MACD
    # -----------------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    # -----------------------------
    # Volume
    # -----------------------------
    df["volume_change"] = volume.pct_change()

    volume_ma20 = volume.rolling(20).mean()
    df["volume_ratio_20"] = volume / volume_ma20

    # -----------------------------
    # Relative strength vs SPY
    # -----------------------------
    #
    # Asset return - SPY return
    #
    df["relative_strength_1d"] = (
        df["return_1d"] - spy_returns
    )

    df["relative_strength_20d"] = (
        df["return_20d"] -
        spy_returns.rolling(20).sum()
    )

    # -----------------------------
    # Rolling beta vs SPY
    # -----------------------------
    covariance = (
        df["return_1d"]
        .rolling(60)
        .cov(spy_returns)
    )

    spy_variance = (
        spy_returns
        .rolling(60)
        .var()
    )

    df["beta_60"] = covariance / spy_variance.replace(0, np.nan)

    return df


def add_vix_features(df, vix):
    """Add market volatility regime features."""

    df["vix"] = vix
    df["vix_change"] = vix.pct_change()

    df["vix_ma10"] = vix.rolling(10).mean()
    df["vix_ma20"] = vix.rolling(20).mean()

    df["vix_ratio_ma20"] = (
        vix / df["vix_ma20"]
    )

    return df


def add_target(df):
    """
    Next-day return target.

    IMPORTANT:
    future_return[t] uses Close[t+1].
    Therefore it is a TARGET, never an input feature.
    """

    df["future_return"] = (
        df["Close"].shift(-1) / df["Close"] - 1
    )

    return df


def one_hot_sector(df):
    """One-hot encode broad sector information."""

    sector_series = df["Sector"]

    sector_dummies = pd.get_dummies(
        sector_series,
        prefix="sector",
        dtype=int
    )

    df = pd.concat(
        [df, sector_dummies],
        axis=1
    )

    return df


def main():

    print("Loading market data...")

    # ----------------------------------
    # Load SPY
    # ----------------------------------
    spy = load_price_data("SPY")

    spy_returns = spy["Close"].pct_change()

    # ----------------------------------
    # Load VIX
    # ----------------------------------
    vix = load_price_data("^VIX")

    vix_series = vix["Close"].copy()

    # ----------------------------------
    # Build each stock independently
    # ----------------------------------
    all_assets = []

    for ticker in STOCK_TICKERS:

        print(f"\nProcessing {ticker}...")

        df = load_price_data(ticker)

        # Technical + market-relative features
        df = add_asset_features(
            df,
            spy_returns
        )

        # VIX features
        df = add_vix_features(
            df,
            vix_series
        )

        # Target
        df = add_target(df)

        # Identifier
        df["Ticker"] = ticker
        df["Sector"] = SECTOR_MAP[ticker]

        # Restore Date as column
        df.reset_index(inplace=True)

        all_assets.append(df)

        print(
            f"{ticker}: "
            f"{len(df)} rows"
        )

    # ----------------------------------
    # Combine all assets
    # ----------------------------------
    data = pd.concat(
        all_assets,
        ignore_index=True
    )

    # Sort properly
    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    # ----------------------------------
    # One-hot encode sector
    # ----------------------------------
    data = one_hot_sector(data)

    # ----------------------------------
    # Remove rows with insufficient
    # historical information
    # ----------------------------------
    data = data.dropna().reset_index(drop=True)

    # ----------------------------------
    # Check infinite values
    # ----------------------------------
    numeric_cols = data.select_dtypes(
        include=[np.number]
    ).columns

    inf_count = np.isinf(
        data[numeric_cols]
    ).sum().sum()

    if inf_count > 0:
        print(
            f"Replacing {inf_count} infinite values..."
        )

        data[numeric_cols] = data[
            numeric_cols
        ].replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna().reset_index(drop=True)

    # ----------------------------------
    # Save
    # ----------------------------------
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ----------------------------------
    # Dataset summary
    # ----------------------------------
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL DATASET CREATED")
    print("=" * 60)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} "
        f"→ "
        f"{data['Date'].max().date()}"
    )

    print(
        f"Unique tickers: "
        f"{data['Ticker'].nunique()}"
    )

    print(
        "\nRows per ticker:"
    )

    print(
        data["Ticker"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget statistics:"
    )

    print(
        data["future_return"].describe()
    )

    print(
        "\nMissing values:"
    )

    print(
        data.isna().sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print(
        "\nSaved to:"
    )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()