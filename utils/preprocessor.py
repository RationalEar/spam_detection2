import hashlib
import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

from utils.data_loader import download_datasets
from utils.email_parser import parse_email

# Helper: Redact PII and hash emails
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+')
CURRENCY_PATTERN = re.compile(r'\$\d+(?:\.\d+)?')
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
PHONE_PATTERN = re.compile(r'\b\d{10,}\b')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s<>]')
WHITESPACE_PATTERN = re.compile(r'\s+')


def redact_and_hash_email(text):
    def hash_email(match):
        return hashlib.sha256(match.group(0).encode()).hexdigest()
    text = EMAIL_PATTERN.sub(hash_email, text)
    return text


def preprocess_text(raw_email):
    if not isinstance(raw_email, str):
        return ""
    
    # Remove email headers/footers
    raw_email = re.sub(r"^.*?(From:|Subject:|To:).*?\n", "", raw_email, flags=re.DOTALL | re.IGNORECASE)
    raw_email = re.sub(r"\n\S+:\s.*?\n", "\n", raw_email)  # Remove remaining headers
    
    # Normalization
    raw_email = raw_email.lower()

    # Strip HTML
    soup = BeautifulSoup(raw_email, 'html.parser')
    text = soup.get_text()

    text = URL_PATTERN.sub('<URL>', text)  # Replace URLs
    text = redact_and_hash_email(text)  # Replace emails
    text = PHONE_PATTERN.sub("<PHONE>", text)  # Replace phone numbers
    text = CURRENCY_PATTERN.sub('<CURRENCY>', text) # Replace currencies
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)  # Replace special chars with space
    text = WHITESPACE_PATTERN.sub(" ", text).strip()  # Collapse whitespace
    return text


def create_dataset():
    data = []
    
    # Process each dataset with proper labeling
    for dataset, label in [("easy_ham", 0), ("easy_ham_2", 0),
                           ("hard_ham", 0), ("spam", 1), ("spam_2", 1)]:
        dir_path = f"data/raw/{dataset}"
        if not os.path.exists(dir_path):
            continue
        
        for filename in os.listdir(dir_path):
            if filename.startswith(".") or "cmds" in filename:
                continue
            
            file_path = os.path.join(dir_path, filename)
            parsed = parse_email(file_path)
            if parsed:
                raw_email = f"{parsed['subject']} {parsed['body']}"
                data.append({
                    "text": preprocess_text(raw_email),
                    "label": label,
                    "source": dataset  # Track origin
                })
    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)
    return df


def prepare_data():
    if not os.path.exists("data/processed/train.pkl"):
        download_datasets()
        df = create_dataset()
        
        # Verify counts match original description
        print(f"\nDataset counts:")
        print(df["source"].value_counts())
        
        # Stratified split (preserve class balance)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )
        print(f"Spam ratio: {df['label'].mean():.2%}")
        
        # Save data
        os.makedirs("data/processed", exist_ok=True)
        train_df.to_csv("data/processed/train.csv", index=False)
        test_df.to_csv("data/processed/test.csv", index=False)
        
        with open("data/processed/train.pkl", "wb") as f:
            pickle.dump(train_df, f)
        with open("data/processed/test.pkl", "wb") as f:
            pickle.dump(test_df, f)
    else:
        print("Data already prepared. Loading from disk...")
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    
    print("\nData preparation complete!")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")


def load_glove_embeddings(glove_path, word2idx, embedding_dim=300):
    """
    Loads GloVe embeddings and creates an embedding matrix for the given vocabulary.
    Args:
        glove_path (str): Path to the GloVe .txt file.
        word2idx (dict): Mapping from word to index in your vocabulary.
        embedding_dim (int): Dimension of the embeddings (default 300).
    Returns:
        torch.Tensor: Embedding matrix of shape (vocab_size, embedding_dim)
    """
    import numpy as np
    import torch

    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for word, idx in word2idx.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix)

