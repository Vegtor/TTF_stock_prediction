import json
import zstandard as zstd
from collections import defaultdict
from datetime import datetime
import pandas as pd


def get_submissions_by_day(file_path: str, start_date=None, end_date=None):
    """
        Get submissions from zst file in certain time window.

        Parameters:
            file_path (str): Path to zst file.
            start_date (str): Start date in string format.
                Defaults to None.
            start_date (str): End date in string format.
                Defaults to None.
        Returns:
            pd.DataFrame: Dataframe containing certain data from submissions with columns
            'date', 'title', 'selftext', 'date', 'score' and 'num_comments'.
    """
    start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
    end = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None

    daily_submissions = []

    with open(file_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buffer = b''
            while True:
                chunk = reader.read(65536)
                # 2**16 bits
                if not chunk:
                    break
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        data = json.loads(line.decode('utf-8'))

                        # Safely get and convert timestamp
                        timestamp = data.get('created_utc')
                        if not isinstance(timestamp, (int, float)):
                            continue

                        date = datetime.utcfromtimestamp(timestamp).date()

                        if (start and date < start) or (end and date > end):
                            continue

                        submission = {
                            'date': date,
                            'title': data.get('title'),
                            'selftext': data.get('selftext'),
                            'score': data.get('score'),
                            'num_comments': data.get('num_comments'),
                        }
                        daily_submissions.append(submission)

                    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
                        continue  # Skip malformed lines
    df_submisions = pd.DataFrame(daily_submissions)
    return df_submisions

def filter_submissions(df: pd.DataFrame, keywords: list(str)):
    """
        Filters the DataFrame of submissions by checking if any of the keywords appear
        in the title or selftext fields (case-insensitive).

        Parameters:
            df (pd.DataFrame): DataFrame with submissions data (date, title, selftext, date, score, num_comments).
            keywords (list of str): List of keywords to filter for.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only matching submissions.
    """
    if not keywords:
        return df

    keyword_pattern = '|'.join([rf'\b{k}\b' for k in keywords])
    mask = (
            df['title'].fillna('').str.contains(keyword_pattern, case=False, regex=True) |
            df['selftext'].fillna('').str.contains(keyword_pattern, case=False, regex=True)
    )

    return df[mask].reset_index(drop=True)

def get_top_posts_per_day(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
        Filters top posts per day based on set parameter.

        Parameters:
            data (pd.DataFrame): DataFrame with submissions data (date, title, selftext, date, score, num_comments).
            top_n (int): Number of top posts to return.
                Defaults to 20

        Returns:
            pd.DataFrame: Filtered DataFrame containing only matching submissions.
    """
    df['date'] = pd.to_datetime(data['date'])

    def rank_day(group):
        group['score_norm'] = (group['score'] - group['score'].min()) / (group['score'].max() - group['score'].min() + 1e-6)
        group['comments_norm'] = (group['num_comments'] - group['num_comments'].min()) / (group['num_comments'].max() - group['num_comments'].min() + 1e-6)
        group['rank_score'] = group['score_norm'] * 0.6 + group['comments_norm'] * 0.4
        return group.nlargest(top_n, 'rank_score')

    top_posts = df.groupby(df['date'].dt.date).apply(rank_day).reset_index(drop=True)
    return top_posts[['date', 'title', 'selftext', 'score', 'num_comments', 'rank_score']]



