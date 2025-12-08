"""
This script makes sentiment predictions on the GoEmotions dataset using a fine-tuned BERT model 
"""

import numpy as np
from pathlib import Path
from src.preprocessing import read_data, clean_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

def split_data(df):
    """
    This function splits the clean dataframe to train/test sets

    Input: Clean dataframe

    Output:
        train_dataset: Training set as Hugging Face dataset for BERT
        test_dataset: Test set as Hugging Face dataset for BERT
        test_df: Test set as pandas dataframe to join with predictions
    """

    # Perform train/test split
    # 80% Train and 20% Test. Stratify by labels to handle class imbalance
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
    
    # Convert to HuggingFace dataset to be compatible with BERT 
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    return train_dataset, test_dataset, test_df

def SetTrainingArgs():
    """
    This function creates a TrainingArguements object to set Trainer inputs

    Output: Training Arguments Class Instance
    """

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100
    ) 

    return training_args

def evaluate_bert_model(pred):
    """
    This function computes weighted F1 score and loads it as the best model for test predictions

    Input: EvalPrediction Object with the following attributes: predictions, true labels ids

    Output: F1 score
    """

    # Compute F1 score
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(pred.label_ids, preds, average='weighted')
    
    return {'f1': f1}



def train_bert_model(model, training_args, train_set, test_set, tokenizer):
    """
    This function creates a HuggingFace Trainer object to use for BERT classification

    Input:
        model
        training_args: Training settings
        train_set
        test_set
        tokenizer

    Output: Trainer class
    """

    # Create trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        compute_metrics=evaluate_bert_model
    )

    return trainer


def sent_analysis_bert(df):
    """
    This function performs sentiment analysis using a fine-tuned BERT model

    Input: Clean GoEmotions dataframe

    Output: Updated dataframe with sentiment predictions
    """

    # Encode sentiment labels for BertForSequenceClassification
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['labels'] = df['sentiment'].map(label_mapping)

    # Mapping variable to decode encoded labels
    id_to_label = {v: k for k, v in label_mapping.items()}

    # Split the dataframe to train/test sets and return the test_df to join with predictions
    train_set, test_set, test_df = split_data(df)

    # Load tokenizer using the bert-base-uncased model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Use tokenizer to tokenize the reddit comments
    def tokenize_function(data):
        return tokenizer(data['text'], truncation=True, padding='max_length', max_length=128)

    # Tokenize reddit comments in batch
    train_set_tokenized = train_set.map(tokenize_function, batched=True)
    test_set_tokenized = test_set.map(tokenize_function, batched=True)

    # Drop features that aren't needed for BERT classification
    train_set_tokenized = train_set_tokenized.remove_columns(["text", "sentiment"])
    test_set_tokenized = test_set_tokenized.remove_columns(["text", "sentiment"])

    # Use BertForSequenceClassification with 3 labels (0,1,2) for (negative,neutral,positive)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    training_args = SetTrainingArgs()

    # Create trainer instance and train model
    trainer = train_bert_model(model, training_args, train_set_tokenized, test_set_tokenized, tokenizer)
    trainer.train()
    print('Training started...')

    # Save trained model
    trainer.save_model('models/bert_sentiment_model')
    tokenizer.save_pretrained('models/bert_sentiment_model')

    # Compute sentiment scores on the test set for each label using the best performing model
    print('get test set predictions...')
    predictions_testset = trainer.predict(test_set_tokenized)

    # Set sentiment as the label with the max sentiment score
    predicted_labels = np.argmax(predictions_testset.predictions, axis=1)
    test_df['predicted_label_id'] = predicted_labels
    test_df['predicted_sentiment'] = test_df['predicted_label_id'].map(id_to_label)

    return test_df

if __name__ == "__main__":

   # Read and clean the data
    filename = "go_emotions_dataset.csv"
    df_raw = read_data(filename)
    df_clean = clean_data(df_raw)
    df_clean = df_clean.drop(columns=['id'])

    # Make sentiment prediction
    df_bert_pred = sent_analysis_bert(df_clean)

    # Export predictions as csv
    output_path = Path(__file__).parent.parent / "data" / "go_emotions_bert_pred.csv"
    df_bert_pred.to_csv(output_path, index=False)
    print("File exported")
