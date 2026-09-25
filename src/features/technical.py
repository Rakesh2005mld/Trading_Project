import os
import numpy as np
import pandas as pd


DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def load_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        parse_dates=["Date"]
    )

    df = df.set_index("Date")

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    )

    df = df.sort_index()

    return df


# ---------------------------------------------------------
# RSI
# ---------------------------------------------------------

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ---------------------------------------------------------
# MACD
# ---------------------------------------------------------

def calculate_macd(series: pd.Series):

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26

    signal = macd.ewm(span=9, adjust=False).mean()

    histogram = macd - signal

    return macd, signal, histogram


# ---------------------------------------------------------
# Technical features for one asset
# ---------------------------------------------------------

def create_asset_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:

    result = pd.DataFrame(index=df.index)

    close = df["Close"]
    volume = df["Volume"]

    # 1. Daily return
    result[f"{prefix}_return"] = close.pct_change()

    # 2. Log return
    result[f"{prefix}_log_return"] = np.log(
        close / close.shift(1)
    )

    # 3. Moving averages
    result[f"{prefix}_ma20"] = close.rolling(20).mean()
    result[f"{prefix}_ma50"] = close.rolling(50).mean()
    result[f"{prefix}_ma200"] = close.rolling(200).mean()

    # 4. Price relative to moving averages
    result[f"{prefix}_price_ma20_ratio"] = (
        close / result[f"{prefix}_ma20"]
    )

    result[f"{prefix}_price_ma50_ratio"] = (
        close / result[f"{prefix}_ma50"]
    )

    result[f"{prefix}_price_ma200_ratio"] = (
        close / result[f"{prefix}_ma200"]
    )

    # 5. Momentum
    result[f"{prefix}_momentum_5"] = (
        close / close.shift(5) - 1
    )

    result[f"{prefix}_momentum_10"] = (
        close / close.shift(10) - 1
    )

    result[f"{prefix}_momentum_20"] = (
        close / close.shift(20) - 1
    )

    # 6. Volatility
    result[f"{prefix}_volatility_10"] = (
        result[f"{prefix}_return"].rolling(10).std()
    )

    result[f"{prefix}_volatility_20"] = (
        result[f"{prefix}_return"].rolling(20).std()
    )

    result[f"{prefix}_volatility_60"] = (
        result[f"{prefix}_return"].rolling(60).std()
    )

    # 7. RSI
    result[f"{prefix}_rsi"] = calculate_rsi(close)

    # 8. MACD
    macd, signal, histogram = calculate_macd(close)

    result[f"{prefix}_macd"] = macd
    result[f"{prefix}_macd_signal"] = signal
    result[f"{prefix}_macd_hist"] = histogram

    # 9. Volume change
    result[f"{prefix}_volume_change"] = volume.pct_change()

    # 10. Volume relative to 20-day average
    volume_ma20 = volume.rolling(20).mean()

    result[f"{prefix}_volume_ratio"] = (
        volume / volume_ma20
    )

    return result


# ---------------------------------------------------------
# Main dataset construction
# ---------------------------------------------------------

def build_dataset():

    aapl = load_data("AAPL")
    spy = load_data("SPY")
    vix = load_data("^VIX")

    # AAPL features
    aapl_features = create_asset_features(
        aapl,
        "aapl"
    )

    # SPY features
    spy_features = create_asset_features(
        spy,
        "spy"
    )

    # VIX
    vix_features = pd.DataFrame(index=vix.index)

    vix_close = vix["Close"]

    vix_features["vix"] = vix_close

    vix_features["vix_change"] = (
        vix_close.pct_change()
    )

    vix_features["vix_ma10"] = (
        vix_close.rolling(10).mean()
    )

    vix_features["vix_ma20"] = (
        vix_close.rolling(20).mean()
    )

    vix_features["vix_ratio_ma20"] = (
        vix_close / vix_features["vix_ma20"]
    )

    # Combine everything
    df = pd.concat(
        [
            aapl_features,
            spy_features,
            vix_features
        ],
        axis=1,
        join="inner"
    )

    # -----------------------------------------------------
    # Rolling Beta of AAPL relative to SPY
    # -----------------------------------------------------

    covariance = (
        df["aapl_return"]
        .rolling(60)
        .cov(df["spy_return"])
    )

    spy_variance = (
        df["spy_return"]
        .rolling(60)
        .var()
    )

    df["beta_spy_60"] = (
        covariance / spy_variance
    )

    # -----------------------------------------------------
    # Future return
    # -----------------------------------------------------

    df["future_return"] = (
        aapl["Close"].shift(-1) / aapl["Close"] - 1
    )

    # -----------------------------------------------------
    # Classification target
    # -----------------------------------------------------

    df["target"] = (
        df["future_return"] > 0
    ).astype(int)

    # Remove rows created by rolling windows
    df = df.dropna()

    # Remove final row because future return is unavailable
    df = df.iloc[:-1]

    output_path = os.path.join(
        PROCESSED_DIR,
        "training_dataset.csv"
    )

    df.to_csv(output_path)

    print("Dataset created successfully.")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    build_dataset()