import random

import numpy as np
import torch


def build_vocab(texts, min_freq=2):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word2idx = {word: idx+2 for idx, word in enumerate(sorted(vocab))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    return word2idx

    
