import feedparser
import requests
from datetime import datetime, timedelta
import pandas as pd

def fetch_google_news_rss(keyword, start_date, end_date, language="en-US", location="US"):
    """
    Fetches Google News RSS feed articles for a keyword within a date range.

    Args:
        keyword (str): The search term.
        start_date (datetime): Start date for news articles.
        end_date (datetime): End date for news articles.
        language (str): Language setting for news (default: 'en-US').
        location (str): Region setting for news (default: 'US').

    Returns:
        list: List of dictionaries containing news articles with columns 'title', 'link', 'source' and 'published'
    """

    base_url = "https://news.google.com/rss/search"
    query = f"{base_url}?q={keyword}%20after:{start_date.strftime('%Y-%m-%d')}%20before:{end_date.strftime('%Y-%m-%d')}&hl={language}&gl={location}&ceid={location}:{language}"

    try:
        response = requests.get(query)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        news_by_date = list()
        for entry in feed.entries:
            pub_date = datetime(*entry.published_parsed[:3])
            news_by_date.append({'title': entry.title,
                                 'link': entry.link,
                                 'source': entry.source.title,
                                 'published': pub_date})
        #news_by_date = pd.DataFrame(news_by_date)
        return news_by_date
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return {}

def get_news_ltw(keyword, start_date, end_date, window=5, language="en-US", location="US"):
    """
    Fetch news articles from Google News RSS in time windows over a date range.

    Queries Google News RSS for a specified keyword in windows of a given number of days
    between start_date and end_date. If the number of articles returned in a window is
    75 or more, the window size is decreased by one day to avoid large responses. The
    collected news entries from all windows are combined and returned as a pandas DataFrame.

    Args:
        keyword (str): Search term for Google News.
        start_date (datetime.datetime): Start date for news search.
        end_date (datetime.datetime): End date for news search.
        window (int, optional): Number of days per search window.
            Defaults to 5.
        language (str, optional): Language code (e.g., 'en-US').
            Defaults to 'en-US'.
        location (str, optional): Location code (e.g., 'US').
            Defaults to 'US'.

    Returns:
        pandas.DataFrame: DataFrame containing news articles with fields such as
                          'title', 'link', 'source', and 'published' as datetime.
    """
    news_by_date = list()
    temp_date_end = start_date + timedelta(days=window)
    temp_date_start = start_date
    while temp_date_end <= end_date:
        temp_news = fetch_google_news_rss(keyword, temp_date_start, temp_date_end, language, location)
        if len(temp_news) >= 75 and temp_date_end != temp_date_start+timedelta(days=1):
            temp_date_end = temp_date_end - timedelta(days=1)
        else:
            print(f"{temp_date_start} - {temp_date_end}")
            news_by_date.extend(temp_news)
            temp_date_start = temp_date_end
            temp_date_end = temp_date_end + timedelta(days=window)
    news_by_date = pd.DataFrame(news_by_date)
    return news_by_date

def filtered_news(df_input, filtering_keywords):
    def is_related(title):
        title_lower = str(title).lower()
        return any(keyword in title_lower for keyword in filtering_keywords)

    filtered_df = df_input[df_input["title"].apply(is_related)]
    return filtered_df