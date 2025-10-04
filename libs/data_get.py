import csv
from datetime import timedelta, datetime
import pandas as pd

from libs.data_edit import fill_missing, filter_by_dates
from libs.finance import get_fin_data_mult, check_empty_target, get_fin_data_indiv
from libs.finance_indicators_fft import tech_indicators, fin_fourier
from libs.trends import get_trends_3_days, get_trends


def reading_tickers(file_tickers):
    """
    Reading financial market tickers from csv file (together with filter of them).

    Parameters:
        file_tickers (str): File path containing csv with tickers and filtered version of columns used (optional)

    Returns:
        list: List of names of tickers from file.
        list List of selected features of tickers from file.
    """
    file = open(file_tickers, "r")
    reader = csv.reader(file)
    name_tickers = next(reader)
    used_indicators = next(reader)
    file.close()
    return name_tickers, used_indicators

def obtain_whole_data(start, end, file_tickers, target, filter_cols=False, trends_list=None, sentiment_list=None, pct_change=False) -> pd.DataFrame:
    """
        Get whole dataset based on given parameters.

        Parameters:
            start(str): Start date.
            end(str): End date.
            file_tickers(str): Path to file containing list of stock and other market tickers.
            target(str): Name of targeted stock.
            filter_cols(bool): Bool parameter for using filtering of columns, financial data.
                Defaults to False.
            trends_list(list): List of trends data to add.
                Defaults to None.
            sentiment_list(str): List of sentiment data to add.
                Defaults to None.
        Returns:
            pd.DataFrame: Dataframe containing certain data.
    """
    date_start = datetime.strptime(start, "%Y-%m-%d")
    date_end = datetime.strptime(end, "%Y-%m-%d")
    name_tickers, used_indicators = reading_tickers(file_tickers)

    whole_data = get_fin_data_mult(name_tickers, date_start, date_end)
    whole_data = check_empty_target(whole_data, target)
    if filter_cols and used_indicators:
        whole_data = whole_data[used_indicators]
    whole_data = fill_missing(whole_data)

    target_prices = get_fin_data_indiv(target, date_start - timedelta(days=42), date_end)
    target_prices = target_prices.iloc[:, :-2]

    if pct_change:
        whole_data = whole_data.pct_change()
        whole_data = whole_data.iloc[2:]
        target_prices = target_prices.iloc[2:]

    # Tech indicators adding
    mavg = tech_indicators(target_prices)
    mavg.index = target_prices.index
    whole_data = whole_data.merge(mavg, left_index=True, right_index=True, how='left')


    # Trend adding, based on parameter
    if trends_list is not None:
        for file in trends_list:
            trends = pd.read_csv(file)
            if trends.columns[0] == 0 or trends.columns[0] == 'Unnamed: 0':
                trends = trends.drop(trends.columns[0], axis=1)
            trends.rename(columns={'Den': 'date'}, inplace=True)
            trends['date'] = pd.to_datetime(trends['date'])
            trends.drop_duplicates(subset=['date'], keep='first', inplace=True)
            trends.set_index('date', inplace=True)
            whole_data = whole_data.merge(trends, left_index=True, right_index=True, how='left')

    # Fourier parameters (adding parameter for it?)
    fft = fin_fourier(target_prices.iloc[:, 3])
    fft.index = target_prices.index
    whole_data = whole_data.merge(fft, left_index=True, right_index=True, how='left')

    # sentiment adding from file
    if sentiment_list is not None:
        for file in sentiment_list:
            temp = pd.read_csv(file)
            if temp.index.name != 'date':
                if 'date' in temp.columns:
                    temp.set_index('date', inplace=True)
            temp.index = pd.to_datetime(temp.index)
            whole_data = whole_data.merge(temp, left_index=True, right_index=True, how='left')
    whole_data.fillna(0, inplace=True)

    return whole_data

