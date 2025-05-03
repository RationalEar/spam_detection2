import os
import pickle
import tarfile
import urllib.request
import pandas as pd
import hashlib
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATASETS = [
    ("easy_ham", "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"),
    ("easy_ham_2", "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2"),
    ("hard_ham", "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2"),
    ("spam", "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"),
    ("spam_2", "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2")
]


def download_datasets():
    """
    Download and extract the datasets from the given URLs.
    """
    for dir_name, url in DATASETS:
        print(f"Downloading {dir_name}...")
        file_path = f"data/raw/{dir_name}.tar.bz2"
        # download if not already downloaded
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"{dir_name} already downloaded.")
        
        # Extract with folder structure preservation
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall("data/raw")

