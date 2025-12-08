"""
This script preprocesses the GoEmotions Dataset for sentiment analysis
"""

import pandas as pd
from pathlib import Path

def read_data(filename):
    """
    This function loads the GoEmotions.csv as a dataframe

    Input: Filename

    Output: GoEmotions Dataframe
    """

    # Load dataset as csv
    root = Path(__file__).parent.parent
    path = root/"data"/filename
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['id', 'text'], keep='first')

    return df

def clean_data(df):
    """
    This function takes the raw GoEmotions dataframe and cleans it to make it fit for sentiment analysis

    Input: Raw GoEmotions Dataframe

    Output: Clean Dataframe
    """

    # Identify emotion categories
    positive_emotions = ['admiration', 'amusement',  'approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love',  'optimism', 'pride', 'relief']
    negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'nervousness', 'remorse', 'sadness', 'grief', 'fear']
    neutral_emotions = ['neutral']
    ambigous_emotions = ['surprise', 'realization', 'curiosity', 'confusion']

    # Consolidate positive, negative, and neutral emotions to reduce 27 emotions to 3
    df_clean = df.copy()
    df_clean['positive'] = df_clean[positive_emotions].sum(axis=1)
    df_clean['negative'] = df_clean[negative_emotions].sum(axis=1)
    df_clean['neutral'] = df_clean[neutral_emotions].sum(axis=1)

    # Filter out ambigous records
    df_clean['sentiment_aggr'] = df_clean[['positive','negative','neutral']].sum(axis=1)
    df_clean = df_clean.loc[df_clean['sentiment_aggr'] > 0]

    # Drop unused features
    drop_feature_list = positive_emotions + negative_emotions + ambigous_emotions + ['sentiment_aggr', 'example_very_unclear']
    df_clean = df_clean.drop(columns=drop_feature_list)
    df_clean['sentiment'] = df_clean[['positive', 'negative', 'neutral']].idxmax(axis=1)
    df_clean = df_clean.drop(columns=['positive', 'negative', 'neutral'])

    return df_clean

if __name__ == "__main__":

    # Read and clean the data
    filename = "go_emotions_dataset.csv"
    df_raw = read_data(filename)
    df_clean = clean_data(df_raw)

    # Export csv
    output_path = Path(__file__).parent.parent / "data" / "go_emotions_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print("File exported")





