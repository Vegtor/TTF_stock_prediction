import pandas as pd
import numpy as np


def are_same_month(date_str, month_str):
    """
    Check if the given date string belongs to the specified month.

    Args:
        date_str (str): Date string in a format parseable by pandas.to_datetime.
        month_str (str): Month string in 'YYYY-MM' format.

    Returns:
        bool: True if date_str is in the same month and year as month_str, False otherwise.
    """
    date = pd.to_datetime(date_str)
    month = pd.Period(month_str, freq='M')
    return date.year == month.year and date.month == month.month


def join_together_trends_plain(file_path):
    """
    Join multiple Google Trends CSV files by stacking their rows without scaling.

    Args:
        file_path (str): Path prefix to the files (expects files named 'multiTimeline (0).csv', ...).

    Returns:
        pd.DataFrame: Combined DataFrame with all rows concatenated.
    """
    file_path = file_path + "/multiTimeline ("
    data = pd.read_csv(file_path + str(0) + ").csv")
    data = data.astype({data.columns[1]: 'float'})
    for i in range(1, 10):
        temp = pd.read_csv(file_path + str(i) + ").csv")
        temp = temp.astype({data.columns[1]: 'float'})
        data = pd.concat([data, temp.iloc[1:]], ignore_index=True)
    return data


def join_together_trends_ratio(file_path):
    """
    Join multiple Google Trends CSV files by stacking rows and scaling subsequent files
    so that trends match smoothly based on overlapping values.

    Args:
        file_path (str): Path prefix to the files (expects files named 'multiTimeline (0).csv', ...).

    Returns:
        pd.DataFrame: Combined and scaled DataFrame with all rows concatenated.
    """
    file_path = file_path + "/multiTimeline ("
    data = pd.read_csv(file_path + str(0) + ").csv")
    data = data.astype({data.columns[1]: 'float'})
    for i in range(1, 10):
        temp = pd.read_csv(file_path + str(i) + ").csv")
        temp = temp.astype({data.columns[1]: 'float'})
        data_last_value = data.iloc[-1, 1]
        temp_first_value = temp.iloc[0, 1].astype(float)

        if data_last_value == 0 and temp_first_value == 0:
            ratio = 1.0
        elif data_last_value == 0:
            ratio = temp_first_value
        elif temp_first_value == 0:
            ratio = data_last_value
        else:
            ratio = data_last_value / temp_first_value

        temp.iloc[:, 1] = temp.iloc[:, 1] * ratio
        data = pd.concat([data, temp.iloc[1:]], ignore_index=True)
    return data


def normalize_trends(data, file_path):
    """
    Normalize Google Trends data based on monthly averages provided in a separate CSV.

    Args:
        data (pd.DataFrame): DataFrame containing trends data with dates in the first column and values in the second.
        file_path (str): Path prefix where 'monthly.csv' file with monthly averages is located.

    Returns:
        pd.DataFrame: Normalized trends data with values adjusted based on monthly average ratios.
    """
    temp_return = data.copy()
    monthly_avg = pd.read_csv(file_path + "/monthly.csv")

    data['M'] = pd.to_datetime(data.iloc[:, 0]).dt.to_period('M')
    monthly_data_avg = data.groupby('M')[data.columns[1]].mean()
    test = monthly_avg[monthly_avg.columns[1]].astype(float)
    monthly_ratio = monthly_avg[monthly_avg.columns[1]].astype(float).reset_index(drop=True) / monthly_data_avg.reset_index(drop=True)

    i = 0
    j = 0
    while i < data.shape[0] - 1:
        if are_same_month(data.iloc[i, 0], monthly_avg.iloc[j, 0]):
            temp_return.iloc[i, 1] = data.iloc[i, 1] * monthly_ratio.iloc[j]
            i = i + 1
        else:
            j = j + 1
    data = data.drop(columns=['M'])
    return temp_return


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing (NaN) values in the DataFrame by averaging the previous and next non-NaN values.
    If only one neighbor exists, use that neighbor's value. If none exist, fill with 0.

    Args:
        df (pd.DataFrame): Input DataFrame with possible NaN values.

    Returns:
        pd.DataFrame: DataFrame with NaNs filled based on neighboring values.
    """
    result = df.copy()
    for i in range(0, result.shape[1]):

        for j in range(0, result.shape[0]):
            if np.isnan(result.iloc[j, i]):
                if j > 0 and not np.isnan(result.iloc[j - 1, i]):
                    prev_value = result.iloc[j - 1, i]
                else:
                    prev_value = np.nan

                if j < len(result) - 1:
                    next_value = next(x for x in result.iloc[j + 1:, i] if not np.isnan(x))
                    #next_value = result.iloc[j + 1, i]
                else:
                    next_value = np.nan

                if (not np.isnan(prev_value) and not np.isnan(next_value)) and (prev_value != 0 and next_value != 0):
                    result.iloc[j, i] = (prev_value + next_value) / 2
                elif np.isnan(prev_value) and not np.isnan(next_value):
                    result.iloc[j, i] = next_value
                elif not np.isnan(prev_value) and np.isnan(next_value):
                    result.iloc[j, i] = prev_value
                else:
                    result.iloc[j, i] = 0
    return result


def filter_by_dates(dates, data):
    result = data[data.index.intersection(dates)]
    return result
