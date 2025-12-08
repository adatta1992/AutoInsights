"""
This script launches the streamlit dashboard
"""

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path
import spacy

def get_query_data(query):
    """
    This function extracts key names from user query to filter dashboard

    Input: User query

    Output: List of proper nouns extracted from the query
    """
    
    if not query:
        return None

    # Parse query using the Spacy model to extract tokens and other query features
    doc = spacy_mdl(query)

    # Find proper nouns
    pnoun_list = [token.text for token in doc if token.pos_ == "PROPN"]

    # Return string of proper nouns if found
    if pnoun_list:
        pn_string = "|".join(pnoun_list)
        return pn_string
    else:
        return None

@st.cache_resource
def load_spacy_model():
    # Load the Spacy small english model to assist with nl queries
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_data(path):
    # Read csv as dataframe
    df = pd.read_csv(path)
    return df[['comment', 'comment sentiment']].dropna()

# Set page config and title
st.set_page_config(page_title="NBA Sentiment Dashboard", layout="wide")
st.title("🏀 r/NBA Reddit Sentiment Analysis")
st.write("### 2025 Univerity of Michigan MADS Capstone Project")

# Load the Spacy small english model to assist with nl queries
spacy_mdl = load_spacy_model()
path = Path(__file__).parent.parent / "data" / "nba_subreddit_sentanalysis_final.csv"

# Read csv as dataframe
df = load_data(path)

# Add search bar to enable nl queries
query = st.text_input(
    "How do people feel about your favourite players? (e.g., 'What is the sentiment regarding Steph Curry?')",
    placeholder="Let's find out! Type here..."
)

# Get query data from user input
q_data = get_query_data(query)
filtered_df = df.copy()

# Filter dataframe using user query data
if q_data:


    if query: 
        # Filter dataframe using user defined features
        filtered_df = filtered_df[filtered_df["comment"].str.contains(q_data, case=False, na=False)]

        # Feedback to user
        info_q_data = q_data.replace("|", ", ")
        st.info(f"Filtering comments using these keywords **'{info_q_data}'**")
    else:
        # Feedback to user
        st.warning("Could not identify a specific subject (Name or Noun). Showing all results.")

else:
    # Feedback to user
    st.markdown("All data displayed")

st.divider()

st.subheader("KPIs")

# Display KPIs in two columns
col1, col2, col3 = st.columns([1, 1, 1])

# Add filter widget for user sentiment input
with col3:

    filter_selection = st.multiselect(
        label="Filter by Sentiment",
        options=filtered_df['comment sentiment'].unique(),
        default=filtered_df['comment sentiment'].unique()
    )

# If the user clears the selection, this prevents the dataframe from becoming empty
if filter_selection:
    filtered_df = filtered_df[filtered_df['comment sentiment'].isin(filter_selection)]

# KPI for total number of comments
with col2:

    total_comments = len(filtered_df['comment'])
    st.metric(label="Total Comments", value=f"{total_comments:,}")

# KPI for most dominant sentiment
with col1:
    top_sentiment = filtered_df['comment sentiment'].mode()[0]
    st.metric(label="Dominant Sentiment", value=top_sentiment.title())

st.divider()

# Create two tabs: one for Visualizations and one for Raw Data
tab1, tab2 = st.tabs(["Visualizations", "Raw Data"])

with tab1:

    # Visualize sentiment data
    st.subheader("Sentiment Analysis Breakdown")

    # Aggregation
    sentiment_counts = filtered_df["comment sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    sentiment_counts['percentage'] = sentiment_counts['count'] / sentiment_counts['count'].sum()

    # Bar Chart
    st.write("**Sentiment Analysis Results**")

    bar_chart = (
        alt.Chart(sentiment_counts)
        .mark_bar()
        .encode(
            x=alt.X("sentiment", sort="-y", axis=alt.Axis(title="Sentiment Category")),
            y=alt.Y("count", axis=alt.Axis(title="Number of Comments")),
            color="sentiment",
            tooltip=["sentiment", "count", alt.Tooltip("percentage", format=".1%")]
        )
        .properties(
            height=350,
            width=800
        )
    )

    st.altair_chart(bar_chart)

    st.divider()

    # Display word cloud
    st.subheader("What's Everyone Saying?")

    # Combine text for input to WordCloud class
    wc_input = " ".join(filtered_df['comment'].astype(str))

    # Add filter data in stopwords
    if q_data:
        sw_list = q_data.split('|')
        custom_stopwords = STOPWORDS.union(set(sw_list))

    else:
        custom_stopwords = STOPWORDS

    # Create WordCloud object. This will tokenize and remove stop words
    wc = WordCloud(
        width=1200,
        height=400,
        background_color='white',
        colormap='viridis',
        stopwords=custom_stopwords,
        max_words=20
    ).generate(wc_input)


    # Render Word Cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

with tab2:

    st.subheader("Raw Data Records")

    # Display the dataframe in the second tab
    st.dataframe(filtered_df, use_container_width=True)