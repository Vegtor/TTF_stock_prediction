import pandas as pd
import csv
import yfinance as yf
from datetime import datetime, timedelta
from libs.data_edit import fill_missing
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter


def check_empty_target(df, target) -> pd.DataFrame:
    """
    Removes rows from the DataFrame where any target-related column has missing (NaN) values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Substring to identify target-related columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with NaNs dropped in target columns.
    """
    target_columns = [col for col in df.columns if target in col]
    df_cleaned = df.dropna(subset=target_columns)
    return df_cleaned

def get_fin_data_mult(name_tickers, date_start, date_end) -> pd.DataFrame:
    """
    Fetches historical financial data for multiple tickers from Yahoo Finance and merges them into one DataFrame.

    Args:
        name_tickers (list[str]): List of ticker symbols.
        date_start (str): Start date in 'YYYY-MM-DD' format.
        date_end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Combined DataFrame with financial data for all tickers.
    """
    ticker_sets = yf.Tickers(name_tickers)

    whole_data = pd.DataFrame()
    for i in range(len(name_tickers)):
        hist_price = ticker_sets.tickers[name_tickers[i]].history(start=date_start, end=date_end, interval='1d',
                                                                  auto_adjust=False, actions=False)
        hist_price = hist_price.rename(columns={"Open": name_tickers[i] + "_Open", "High": name_tickers[i] + "_High",
                                                "Low": name_tickers[i] + "_Low", "Close": name_tickers[i] + "_Close",
                                                "Adj Close": name_tickers[i] + "_Adj_Close",
                                                "Volume": name_tickers[i] + "_Volume"})

        hist_price.index = hist_price.index.date
        hist_price.index.name = "Date"
        if whole_data.empty:
            whole_data = hist_price
        else:
            whole_data = whole_data.merge(hist_price, left_index=True, right_index=True, how='outer')
    return whole_data

def get_fin_data_indiv(name_ticker, date_start, date_end) -> pd.DataFrame:
    """
    Fetches historical financial data for a single ticker from Yahoo Finance.

    Args:
        name_ticker (str): Ticker symbol.
        date_start (str): Start date in 'YYYY-MM-DD' format.
        date_end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical data for the ticker.
    """
    ticker_sets = yf.Ticker(name_ticker)

    hist_price = ticker_sets.history(start=date_start, end=date_end, interval='1d',
                                                                  auto_adjust=False, actions=False)
    hist_price = hist_price.rename(columns={"Open": name_ticker + "_Open", "High": name_ticker + "_High",
                                                "Low": name_ticker + "_Low", "Close": name_ticker + "_Close",
                                                "Adj Close": name_ticker + "_Adj_Close",
                                                "Volume": name_ticker + "_Volume"})

    hist_price.index = hist_price.index.date
    hist_price.index.name = "Date"
    whole_data = hist_price
    return whole_data