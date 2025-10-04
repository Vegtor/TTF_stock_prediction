from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from math import ceil
import numpy as np
import os
from datetime import datetime


def tokenize_process(text, tokenizer, model):
    """
    Tokenize input text and get softmax probabilities from the model.

    Args:
        text (str or list[str]): Input text or list of texts to tokenize and process.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        model (transformers.PreTrainedModel): Pretrained sequence classification model.

    Returns:
        torch.Tensor: Softmax probabilities tensor for each class.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    return probs

def process_text(text, tokenizer, max_length = 500):
    """
    Split text into chunks if it exceeds max token length.

    Args:
        text (str): Input text string.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        max_length (int, optional): Maximum token length per chunk.
            Defaults to 500.

    Returns:
        list: List of text chunks each within max token length.
    """
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']

    if len(tokens) <= max_length:
        return [text]
    else:
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            chunk_text = tokenizer.decode(chunk)
            chunks.append(chunk_text)
        return chunks

def PNN_sentiment_long_texts(list_txt: list[str], model_name ="yiyanghkust/finbert-tone"):
    """
    Compute positive, negative and neutral sentiment scores for a list of potentially long texts by chunking and averaging.

    Args:
        list_txt (list[str]): List of input texts.
        model_name (str, optional): HuggingFace model name.
            Defaults to "yiyanghkust/finbert-tone".

    Returns:
        pandas.DataFrame: DataFrame containing averaged sentiment probabilities
                          with columns 'positive', 'neutral' and 'negative'.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    output_scores = []
    for i in range(0, len(list_txt)):
        temp_text = process_text(list_txt[i], tokenizer)
        if len(temp_text) > 1:
            whole_out = []
            for text in temp_text:
                probs = tokenize_process(text, tokenizer, model)
                whole_out.append(probs.tolist())
            whole_out = np.array(whole_out)
            output_scores.append(whole_out.mean(axis=0).squeeze().tolist())
        else:
            temp_text = temp_text[0]
            probs = tokenize_process(temp_text, tokenizer, model)
            output_scores.append(probs.squeeze().tolist())
    labels = ['positive', 'neutral', 'negative']
    return pd.DataFrame(output_scores, columns=labels)

def PNN_sentiment(text, model_name ="yiyanghkust/finbert-tone"):
    """
    Compute positive, negative and neutral sentiment scores for a single text or list of short texts.

    Args:
        text (str or list[str]): Input text(s) for sentiment analysis.
        model_name (str, optional): HuggingFace model name.
            Defaults to "yiyanghkust/finbert-tone".

   Returns:
        pandas.DataFrame: DataFrame containing averaged sentiment probabilities
                          with columns 'positive', 'neutral' and 'negative'.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    probs = tokenize_process(text, tokenizer, model)

    labels = ['positive', 'neutral', 'negative']
    return pd.DataFrame(probs.tolist(), columns=labels)

def batch_3_sentiment(text: list[str] , batch=5, model_name="yiyanghkust/finbert-tone"):
    """
    Process texts in batches and return combined sentiment DataFrame.

    Args:
        text (list[str]): List of input texts.
        batch (int, optional): Batch size for processing.
            Defaults to 5.
        model_name (str, optional): HuggingFace model name.
            Defaults to "yiyanghkust/finbert-tone".

    Returns:
        pandas.DataFrame: Concatenated sentiment results for all batches.
    """
    num_batches = ceil(len(text) / batch) # python list slicing works out-of-bound
    batch_results = []

    for i in range(num_batches):
        batch_texts = text[(i * batch):((i + 1) * batch)]
        batch_df = PNN_sentiment(batch_texts, model_name)
        batch_results += [batch_df]  # batch_results.extend([batch_df])

    result = pd.concat(batch_results, ignore_index=True)
    return result

def check_strings(list_txt: list[str]):
    """
    Check if all items in a list are strings and print index of any non-string items.

    Args:
        list_txt (list[str]): List of items to check.
    """
    i = 0
    for item in list_txt:
        if not isinstance(item, str):
            print("Item is not string on index" + str(i))
        i = i + 1

def calculate_score(file_path: str):
    """
    Calculate sentiment scores from a CSV file containing sentiment probabilities and metadata.

    Supports two formats:
    - Single sentiment set (columns with 'positive', 'neutral', 'negative')
    - Reddit style with separate title and selftext scores, engagement metrics.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame or None: DataFrame with date-indexed sentiment scores.
    """
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')

    data = pd.read_csv(file_path)
    list_cols = data.columns.values.tolist()
    pnn_number = 0
    for col in list_cols:
        if "positive" in col:
            pnn_number = pnn_number + 1

    if 0 < pnn_number < 2:
        name = parts[0]
        polarity_score = (data['positive'] - data['negative'])
        confidence_score = 1 - data['neutral']
        sentiment_score = polarity_score * confidence_score
        result = pd.concat([data['date'], sentiment_score, data['source']], axis=1)
        result.columns = ['date', name + '_sentiment', 'source']
        return result
    elif pnn_number == 2:
        sp = 1.6 # sentiment parameter
        cp = 0.6 # comment parameter
        rsp = 0.9 # reddit score parameter
        esp = 0.333 # engagement parameter
        name = parts[0] + "_" + parts[2]

        max_comments = data['num_comments'].max()
        min_comments = data['num_comments'].min()
        normalized_comments = (data['num_comments'] - min_comments) / (max_comments + min_comments + 1e-6)

        max_score = data['score'].max()
        min_score = data['score'].min()
        normalized_score = (data['score'] - min_score) / (max_score - min_score)

        engagement_score = np.log1p(cp * normalized_comments + rsp * normalized_score)

        polarity_score_title = data['title_positive'] - data['title_negative']
        confidence_score_title = 1 - data['title_neutral']
        title_score = sp * polarity_score_title * confidence_score_title + esp * engagement_score

        polarity_score_selftext = data['selftext_positive'] - data['selftext_negative']
        confidence_score_selftext = 1 - data['selftext_neutral']
        selftext_score = sp * polarity_score_selftext * confidence_score_selftext + esp * engagement_score

        result = pd.concat([data['date'],  title_score, selftext_score], axis=1)
        result = result.groupby('date').mean().reset_index()
        result.columns = ['date', name + '_title_score', name + '_selftexts_score']
        return result
    else:
        return None

def score_in_between(path, start_date=None, end_date=None):
    """
    Calculate sentiment scores in a date range from a CSV file.

    Args:
        path (str): Path to sentiment score CSV file.
        start_date (str or None, optional): Start date in 'YYYY-MM-DD' format.
            Defaults to None.
        end_date (str or None, optional): End date in 'YYYY-MM-DD' format.
            Defaults to None.

    Returns:
        pandas.DataFrame: Filtered DataFrame indexed by date.
    """
    if start_date is None:
        start = datetime.strptime("2000-01-01", '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
    elif end_date is None:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.now()
    else:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

    temp = calculate_score(path)
    temp = temp[(pd.to_datetime(temp['date']) >= start) & (pd.to_datetime(temp['date']) <= end)]
    temp.set_index('date', inplace=True)
    return temp

def combine_scores_reddit(files, weights):
    """
    Combine multiple Reddit sentiment score files with given weights.

    Args:
        files (list[str]): List of file paths to Reddit sentiment CSVs.
        weights (list[float]): Corresponding weights for each file.

    Returns:
        pandas.DataFrame: DataFrame with weighted average title and selftext scores by date.
    """
    weighted_title_scores = []
    weighted_selftext_scores = []

    for i, file in enumerate(files):
        df = pd.read_csv(file)

        wt = weights[i]
        cols = df.columns.values.tolist()

        weighted_title = df[cols[1]] * wt
        weighted_selftext = df[cols[2]] * wt

        weighted_title_scores.append(pd.DataFrame({'date': df['date'], 'weighted_title': weighted_title}))
        weighted_selftext_scores.append(pd.DataFrame({'date': df['date'], 'weighted_selftext': weighted_selftext}))

    combined_title = pd.concat(weighted_title_scores)
    sum_weighted_title = combined_title.groupby('date')['weighted_title'].sum()

    combined_selftext = pd.concat(weighted_selftext_scores)
    sum_weighted_selftext = combined_selftext.groupby('date')['weighted_selftext'].sum()

    all_dates = pd.concat([df[['date']] for df in weighted_title_scores]).drop_duplicates().sort_values('date')

    sum_weights_title = sum(weights)
    sum_weights_selftext = sum(weights)

    result = pd.DataFrame()
    result['date'] = all_dates['date'].values
    result['weighted_avg_title'] = result['date'].map(sum_weighted_title).fillna(0) / sum_weights_title
    result['weighted_avg_selftext'] = result['date'].map(sum_weighted_selftext).fillna(0) / sum_weights_selftext

    return result

def news_weight_aggregation(files, weights_file, split_by_weight=False):
    """
    Aggregate news sentiment scores with weights from a weights file.

    Args:
        files (list[str]): List of sentiment CSV files.
        weights_file (str): Path to CSV containing source weights.
        split_by_weight (bool, optional): Whether to split results by weight categories.
            Defaults to False.

    Returns:
        pandas.DataFrame: Aggregated weighted sentiment scores. If split_by_weight is True,
                          returns separate columns for each weight category.
    """
    weights_df = pd.read_csv(weights_file)
    weights_dict = dict(zip(weights_df['Source'], weights_df['Weight']))
    all_data = []

    for file in files:
        df = pd.read_csv(file)
        sentiment_col = df.columns[1]
        mapped_weights = df['source'].map(weights_dict)

        if split_by_weight:
            df['weight_category'] = pd.cut(
                mapped_weights,
                bins=[-float('inf'), 0.62, 0.87, float('inf')],
                labels=['low', 'medium', 'high']
            )
            df['weight_category'] = df['weight_category'].fillna('low')
            all_data.append(df[['date', sentiment_col, 'weight_category']])
        else:
            df['weight'] = pd.cut(
                mapped_weights,
                bins=[-float('inf'), 0.62, 0.87, float('inf')],
                labels=[0.3, 0.55, 1.0]
            ).astype(float)
            df['weight'] = df['weight'].fillna(0.2)
            df['weighted_sentiment'] = df[sentiment_col] * df['weight']
            all_data.append(df[['date', 'weighted_sentiment', 'weight']])

    combined_df = pd.concat(all_data)

    if split_by_weight:
        result = combined_df.groupby(['date', 'weight_category'])[combined_df.columns[1]].mean().unstack(level=-1)
        result.columns = [f"{col}_avg_sentiment" for col in result.columns]
        result = result.fillna(0)
        result = result.reset_index()
        return result
    else:
        grouped = combined_df.groupby('date').agg(
            total_weighted_score=('weighted_sentiment', 'sum'),
            total_weight=('weight', 'sum')
        ).reset_index()
        grouped['weighted_average_sentiment'] = grouped['total_weighted_score'] / grouped['total_weight']
        return grouped[['date', 'weighted_average_sentiment']]

