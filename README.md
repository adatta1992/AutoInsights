# AutoInsight
### University of Michigan MADS Capstone

Term Year: 2025
Team Members: Aabir Datta <aabir@umich.edu>

### 📊 Benchmarking Sentiment Analysis for Reddit Data
This project aimed to benchmark various sentiment analysis approaches, summarizing their performance and limitations to identify the most effective method for analyzing Reddit data. The insights generated from this investigation can help enable more informed strategic decisions on whether investing in a comprehensive and extensive social media monitoring project, as well as commercial use of social media data, is a value added proposition. The scripts and methodology provided in this repository are modular and adaptable, allowing researchers to easily apply and benchmark these sentiment approaches on their own custom Reddit datasets for domain-specific analyses.

### 💻 Interactive Sentiment Results Dashboard
The results of the sentiment analysis performed on the r/nba subreddit data can be found in the provided streamlit dashboard below.
Click here to view: 

### ✨ Key Findings
    * Top Performer: The GPT 5.1 model achieved the highest F1-score of 0.76 on the validation set, demonstrating robustness in its ability to handle Reddit slang and noise.
    * Best bang-for-buck: The fine-tuned BERT model achieved a highly competitive F1-score of 0.68 on the test set with significantly lower inference cost to the GPT model.
    * Quick Insights: The VADER approach is the best model for quick, high-level sentiment insights at the cost of worse sentiment classification performance.

### 📚 Data Source & Pre-processing
Data Source Details:
    1. GoEmotions dataset: Sourced from Kaggle and used for fine-tuning the BERT model
    2. 2019 Reddit Comments dataset: Sourced from Kaggle and used to create the validation set from comments sourced from the r/nba subreddit.

### 💻 Repository Structure
[AutoInsights]/
├── data/
│   ├── go_emotions_bert_pred.csv          # BERT predictions on test set
│   └── go_emotions_clean.csv    # Cleaned GoEmotions dataset
│   └── go_emotions_dataset.csv    # GoEmotions dataset
│   └── kaggle_RC_2019-05_part1.csv    # Subset of 2019 reddit comments data
│   └── kaggle_RC_2019-05_part2.csv    # Subset of 2019 reddit comments data
│   └── kaggle_RC_2019-05_part3.csv    # Subset of 2019 reddit comments data
│   └── kaggle_RC_2019-05_part4.csv    # Subset of 2019 reddit comments data
│   └── kaggle_RC_2019-05_part5.csv    # Subset of 2019 reddit comments data
│   └── nba_subreddit_sentanalysis_final.csv    # Sentiment analysis results on r/nba validation set
│   └── nba_subreddit_sentanalysis.csv    # Validation subset of r/nba data with human sentiment annotations
├── notebooks/
│   ├── models_benchmarking.ipynb          # Benchmark sentiment classifiers
├── src/
│   ├── create_val_set.py   # Creates validation set
│   └── model_bert.py       # Sentiment classification using BERT
│   └── model_openaiapi.py       # Sentiment classification using GPT 5.1
│   └── model_vader.py       # Sentiment classification using VADER
│   └── preprocessing.py       # Preprocess GoEmotions dataset
│   └── streamlit-dashboard.py       # Launch dashboard
└── README.md

### 🚀 Setup and Installation
All dependencies can be found in the requirements.txt file

### ▶️ Execution
To replicate this analysis for your custom reddit datasets, do the following:
    1. Use the create_validation_set funciton in create_validation_set.py to create a dataset for your subreddit of interest using the 2019 reddit data.
    2. Use the sent_analysis_vader function in model_vader.py to make predictions using VADER
    3. Run the model_bert.py script to create and save a fine-tuned BERT model trained on the GoEmotions dataset. Load this saved model to make sentiment predictions using BERT.
    4. Set your OpenAI API key in the .env file. Use the sent_analysis_openai function in model_openaiapi.py to make sentiment predictions using the OpenAI API/GPT 5.1