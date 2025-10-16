import os
import pandas as pd
import yfinance as yf
from src.utils.logger import get_logger


def init_data(
    ticker: str = "SPY",
    start: str = "2018-01-01",
    save_path: str = "data/spy.csv",
    logger=None,
):
    """
    Download market data from Yahoo Finance and prepare it for the model.
    """
    logger = logger or get_logger(".")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"Downloading {ticker} data from Yahoo Finance...")

    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        raise

    if df is None or df.empty:
        logger.error(f"No data retrieved for {ticker}")
        raise ValueError(f"No data retrieved for {ticker}")

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=20).std()
    df.dropna(inplace=True)

    df.to_csv(save_path)
    logger.info(f"Data initialised and saved to {save_path}")

    return df
