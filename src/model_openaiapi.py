"""
This script uses the OpenAI API to make sentiment predictions
"""

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path
import re
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def sent_analysis_openai(df, prompt, batch_size = 10):
    """
    This function uses the OpenAI API to make sentiment predictions

    Input:
        df: Input dataframe with reddit text
        prompt: User prompt
        batch_size: User defined batch size for API calls

    Output: Updated dataframe with sentiment predictions
    """

    chatgpt_results = []

    # Use regex to identify (comment, sentiment) tuple in output
    # tuple_pattern code created using ChatGPT
    tuple_pattern = re.compile(
        r"\((.*?),\s*(positive|negative|neutral)\)",
        re.IGNORECASE | re.DOTALL
    )

    # For each records in the dataframe, make sentiment predictions 
    # Batch size used to limit # of records passed at one time
    for i in range(0, len(df), batch_size):

        # Grab comment subset and join with ||| delimitor
        batch = df["text"].iloc[i:i + batch_size].tolist()
        joined_texts = "|||".join(batch)

        user_prompt = prompt + f"INPUT:\n{joined_texts}\n"

        # Try API
        try:

            # Use GPT 5.1 for Inference
            # Use Temp=0 for deterministic results
            resp = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            # Extract results and save as list of tuples, where each element is (comment, sentiment)
            ans = resp.choices[0].message.content.strip()
            res_list = tuple_pattern.findall(ans)

            # Pad missing records
            if len(res_list) != len(batch):
                while len(res_list) < len(batch):
                    res_list.append((None, None))

            # Append list of results
            chatgpt_results.extend(res_list)

        except Exception:
            chatgpt_results.extend([(None, None)] * len(batch))

     # Create output df
    res_df = pd.DataFrame(chatgpt_results, columns=["input_text", "sentiment_pred_openai"])

    # Merge predictions with original df on text
    df = df.merge(res_df, left_on="text", right_on="input_text", how="left")
    return df


if __name__ == "__main__":

    # Running this script standalone will make sentiment predictions on the first 1000 records of the GoEmotions test set
    test_set_path = Path(__file__).parent.parent / "data" / "go_emotions_bert_pred.csv"
    df_clean = pd.read_csv(test_set_path)
    df_clean = df_clean[["text"]]
    df_clean = df_clean.iloc[0:1000]

    # Try zero shot approach by default
    prompt = (
            "You are a sentiment analyzer.\n"
            "Classify each text below as positive, negative, or neutral.\n"
            "Each input text is separated by '|||'.\n"
            "Return exactly one tuple for EACH input text, in the SAME ORDER.\n"
            "Each tuple must have this exact format: (original_text, sentiment_label)\n"
            "sentiment_label must be one of: positive, negative, neutral.\n"
            "Tuples must be separated by '|||'.\n"
            "Do NOT add or remove any text, punctuation, spaces, or line breaks.\n"
    )

    # Make sentiment prediction
    df_openai_pred = sent_analysis_openai(df_clean, prompt, 20)

    # Export results
    output_path = Path(__file__).parent.parent / "data" / "go_emotions_openai_pred.csv"
    df_openai_pred.to_csv(output_path, index=False)

    print("File exported")