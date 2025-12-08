"""
This script predicts reddit comment sentiment using VADER
"""

import pandas as pd
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.preprocessing import read_data, clean_data

def sent_analysis_vader(df):
    """
    This function uses VADER for sentiment prediction

    Input: Cleaned GoEmotion Dataframe

    Output: Updated Dataframe with VADER predictions
    """

    # Create analyzer object
    analyzer = SentimentIntensityAnalyzer()
    
    # Determine sentiment using sentiment score thresholds
    def get_sentiment(x):
        score = analyzer.polarity_scores(x)['compound']
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    # Make sentiment prediction for every reddit comment
    df['sentiment_score_vader'] = df['text'].apply(get_sentiment)

    return df


if __name__ == "__main__":

    # Read and clean the data
    filename = "go_emotions_dataset.csv"
    df_raw = read_data(filename)
    df_clean = clean_data(df_raw)
    df_clean = df_clean.drop(columns=['sentiment'])

    # Make sentiment predictions using VADER
    df_vader_pred = sent_analysis_vader(df_clean)

    # Export csv
    output_path = Path(__file__).parent.parent / "data" / "go_emotions_vader_pred.csv"
    df_vader_pred.to_csv(output_path, index=False)
    print("File exported")



