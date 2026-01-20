import requests
import json
import os
import pandas as pd

def get_reddit_post_data(url, headers, cache_file="reddit_cache.json"):
    """
    Fetches JSON from Reddit. Checks if a local cache exists first 
    to avoid unnecessary network requests during testing.

    Input: 
        - url: subreddit post url
        - headers: header for request

    Output: reddit post data in JSON format
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    full_cache_path = os.path.join(data_dir, cache_file)

    # Load from cache if the cache file exists
    if os.path.exists(full_cache_path):
        print(f"--- Loading data from local cache: {full_cache_path} ---")
        with open(full_cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Adding limit read many comments in one go
    # Currently limiting to 200 comments
    api_url = f"{url.rstrip('/')}.json?limit=200"
    
    # Make API call
    print(f"--- Requesting data from Reddit ---")
    response = requests.get(api_url, headers=headers)
    
    # If successful, save to local cache
    if response.status_code == 200:
        data = response.json()
        # Save to local cache for future use
        with open(full_cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return data
    elif response.status_code == 429:
        print("Error: Rate limited (429). Wait a few minutes and try again.")
        return None
    else:
        print(f"Error: Could not fetch data. Status code: {response.status_code}")
        return None

def extract_all_comments(comment_data, all_comments=None):
    """
    Recursively walks through the 'replies' tree to extract every single 
    available comment body and author.
    """
    if all_comments is None:
        all_comments = []

    for child in comment_data:
        # 't1' is the Reddit kind code for a Comment
        if child.get('kind') == 't1':
            data = child.get('data', {})
            all_comments.append({
                'author': data.get('author'),
                'body': data.get('body')
            })

            # Check for nested replies
            replies = data.get('replies')
            if replies and isinstance(replies, dict):
                extract_all_comments(replies['data']['children'], all_comments)
                
    return all_comments


if __name__ == "__main__":
    POST_URL = "https://www.reddit.com/r/cars/comments/1pnnh4o/ford_officially_discontinues_f150_lightning/"
    MY_HEADERS = {'User-Agent': 'pc:comment_extractor:v1.0'}

    raw_data = get_reddit_post_data(POST_URL, MY_HEADERS)

    if raw_data and len(raw_data) > 1:
        comment_root = raw_data[1]['data']['children']
        results = extract_all_comments(comment_root)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Clean up: remove any rows where the body is [removed] or [deleted]
        # (Optional but helpful for data analysis)
        df = df[~df['body'].isin(['[removed]', '[deleted]'])]
        
        # Save to CSV
        df.to_csv("r_cars_reddit_comments.csv", index=False, encoding='utf-8-sig')
        print(f"Saved {len(df)} cleaned comments to reddit_comments.csv")