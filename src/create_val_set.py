"""
This module creates the validation set for benchmarking the sentiment classifiers. 
"""

import pandas as pd
from pathlib import Path

def create_validation_set(subreddit, batch_size):
    """
    This function creates the validation set using the 2019 Reddit Comments dataset.

    Input: 
        subreddit: Subreddit name
        batch_size: Size for validation set

    Output: Validation set
    """
    
    root = Path(__file__).parent.parent
    path = root/"data"

    partnumberlist = ['part1','part2','part3','part4','part5']
    df_list = []

    # Combine all parts of the 2019 Reddit data
    for partnumber in partnumberlist:
        df = pd.read_csv(path/f'kaggle_RC_2019-05_{partnumber}.csv', index_col=None, header=0)
        df_list.append(df)

    # Concat all parts into one df
    reddit_df = pd.concat(df_list, axis=0, ignore_index=True)
    reddit_df = reddit_df.drop_duplicates().dropna()
    reddit_df = reddit_df[['subreddit','body']]

    # Filter for specific subreddit and limit dataset size
    filter_df = reddit_df[reddit_df['subreddit']==subreddit]
    filter_df = filter_df.head(batch_size)

    return filter_df

if __name__ == "__main__":

    # Retrieve data
    subreddit = "nba"
    batch_size = 300
    filter_df = create_validation_set(subreddit, batch_size)

    # Exports results as csv
    output_path = Path(__file__).parent.parent / "data" / "nba_subreddit_sentanalysis_raw.csv"
    filter_df.to_csv(output_path, index=False)
    print("File exported")