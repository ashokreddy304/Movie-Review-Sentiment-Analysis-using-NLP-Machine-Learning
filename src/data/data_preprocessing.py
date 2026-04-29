import pandas as pd
import os
import re
import string
from typing import Optional

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging


def preprocess_text(text: str,stop_words: set,lemmatizer: WordNetLemmatizer) -> str:
    """
    Clean and normalize a single text sample.
    """
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)    # Remove URLs
    text = re.sub(r'\d+', '', text)    # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)     # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words] # Remove stopwords and apply lemmatization

    return " ".join(words)

def preprocess_dataframe(df: pd.DataFrame,col: str = 'review') -> pd.DataFrame:
    """
    Apply preprocessing to an entire dataframe column.
    """

    try:
        logging.info("Starting text preprocessing")

        # Drop NaN values before processing
        df = df.dropna(subset=[col])

        # Initialize NLP tools
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # Apply preprocessing
        df[col] = df[col].apply(lambda x: preprocess_text(x,stop_words,lemmatizer))

        logging.info("Text preprocessing completed successfully")

        return df

    except KeyError as e:
        logging.error("Column not found in dataframe: %s", e)
        raise

    except Exception as e:
        logging.error("Unexpected error during preprocessing: %s", e)
        raise


def save_processed_data(train_df: pd.DataFrame,test_df: pd.DataFrame,output_dir: str = "./data/interim") -> None:
    """
    Save processed train and test datasets.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train_processed.csv")
        test_path = os.path.join(output_dir, "test_processed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info("Processed train data saved to %s", train_path)
        logging.info("Processed test data saved to %s", test_path)

    except Exception as e:
        logging.error("Error while saving processed data: %s", e)
        raise


def load_data(train_path: str,test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets.
    """

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Train and test datasets loaded successfully")

        return train_df, test_df

    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise

    except pd.errors.ParserError as e:
        logging.error("CSV parsing error: %s", e)
        raise

    except Exception as e:
        logging.error("Unexpected error while loading data: %s", e)
        raise


def main():

    try:

        # Load raw datasets
        train_data, test_data = load_data(train_path="./data/raw/train.csv",test_path="./data/raw/test.csv")

        # Preprocess datasets
        train_processed_data = preprocess_dataframe(train_data,col='review')

        test_processed_data = preprocess_dataframe(test_data,col='review')

        # Save processed datasets
        save_processed_data(train_df=train_processed_data,test_df=test_processed_data)

        logging.info("Data preprocessing pipeline completed successfully")

    except Exception as e:
        logging.error("Failed to complete preprocessing pipeline: %s",e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()