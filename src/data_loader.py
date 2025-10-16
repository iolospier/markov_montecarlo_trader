import os
import pandas as pd
import yfinance as yf
from src.utils.logger import get_logger


def init_data(
    ticker: str | None = None,
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    save_dir: str = "data",
    logger=None,
):
    """
    Download or load market data for one or multiple tickers.
    Returns a dictionary {ticker: DataFrame}.
    """
    logger = logger or get_logger(".")
    os.makedirs(save_dir, exist_ok=True)

    # Normalise input (support both 'ticker' and 'tickers')
    tickers = tickers or ([ticker] if ticker else [])
    datasets = {}

    for t in tickers:
        file_path = os.path.join(save_dir, f"{t}.csv")

        # If data already exists locally, load it
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached {t} from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to read cached {t}: {e}. Redownloading...")
                result = _download_and_process(t, start, file_path, logger)
                if isinstance(result, dict):
                    datasets.update(result)
                    continue
                df = result
        else:
            result = _download_and_process(t, start, file_path, logger)
            if isinstance(result, dict):  # Multi-ticker DataFrame split
                datasets.update(result)
                continue
            df = result

        datasets[t] = df

    return datasets


def _download_and_process(ticker, start: str, save_path: str, logger):
    """Download and preprocess data for a single or multiple tickers."""
    logger.info(f"Downloading {ticker} data from Yahoo Finance...")

    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        raise

    if df is None or df.empty:
        logger.error(f"No data retrieved for {ticker}")
        raise ValueError(f"No data retrieved for {ticker}")

    #  Handle multi-index (multiple tickers returned)
    if isinstance(df.columns, pd.MultiIndex):
        logger.info(
            f"Multi-ticker DataFrame detected for {ticker}. Splitting columns..."
        )
        datasets = {}
        for t in df.columns.levels[1]:
            sub = df.xs(t, axis=1, level=1)
            sub["Return"] = sub["Close"].pct_change()
            sub["Volatility"] = sub["Return"].rolling(window=20).std()
            sub.dropna(inplace=True)
            sub_path = os.path.join(os.path.dirname(save_path), f"{t}.csv")
            sub.to_csv(sub_path)
            logger.info(f"Saved {t} data to {sub_path}")
            datasets[t] = sub
        return datasets  # return dict of DataFrames

    #  Normal singleticker download
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=20).std()
    df.dropna(inplace=True)

    df.to_csv(save_path)
    logger.info(f"Data initialised and saved to {save_path}")
    return df
