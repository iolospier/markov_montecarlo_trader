import yfinance as yf
import pandas as pd
import os


def init_data(
    ticker: str = "SPY", start: str = "2018-01-01", save_path: str = "data/spy.csv"
):
    """
    Download market data from Yahoo Finance and prepare it for the model.
    """
    # Create data directory if missing
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data retrieved for {ticker}")

    # Calculate daily returns and volatility
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=20).std()
    df.dropna(inplace=True)

    # Save to CSV
    df.to_csv(save_path)
    print(f"Data initialised and saved to {save_path}")

    return df


df = init_data("SPY", "2018-01-01", "../data/spy.csv")
