import random

import numpy as np
import torch


def build_vocab(texts, min_freq=2):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word2idx = {word: idx + 2 for idx, word in enumerate(sorted(vocab))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def encode(text, word2idx, max_len=200):
    tokens = text.split()
    idxs = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(idxs) < max_len:
        idxs += [word2idx['<PAD>']] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs


def cnn_tokenizer(text, word2idx, max_len, idx2word=None):
    idxs = [word2idx.get(token, word2idx['<UNK>']) for token in text.split()]
    if len(idxs) < max_len:
        idxs += [word2idx['<PAD>']] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]

    # Capture idx2word in the scope of the decode function
    _idx2word = idx2word  # Create a local reference

    def decode(idxs):
        return ' '.join([_idx2word.get(idx, '<UNK>') for idx in idxs])

    return idxs, decode


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
