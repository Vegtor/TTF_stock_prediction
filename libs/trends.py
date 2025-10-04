import datetime
import pandas as pd
from pytrends.request import TrendReq


def get_trends_3_days(keywords) -> pd.DataFrame:
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=6)

    trends_data_whole = get_trends(keywords, start_date, end_date)
    return trends_data_whole

def get_trends(keywords, start_date, end_date) -> pd.DataFrame:
    """
        Get Google Trends of certain keywords using pytrends library.

        Parameters:
            keywords (str): List of keywords.
            start_date (str): Start date in string format.
            start_date (str): End date in string format.
        Returns:
            pd.DataFrame: Dataframe containing trends in columns.
    """
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    pytrends = TrendReq()

    pytrends.build_payload(keywords, cat=0, timeframe=f'{start_date_str} {end_date_str}')
    trends_data_whole = pytrends.interest_over_time()
    trends_data_whole = trends_data_whole.iloc[:,:-1]
    return trends_data_whole
