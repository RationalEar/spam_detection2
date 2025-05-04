import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from models.bert import SpamBERT
from models.bilstm import BiLSTMSpam
from models.cnn import SpamCNN
from utils.functions import build_vocab

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(text, word2idx, max_len=200):
    tokens = text.split()
    idxs = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(idxs) < max_len:
        idxs += [word2idx['<PAD>']] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs


def train_model(model_type, train_df, test_df, embedding_dim=300, pretrained_embeddings=None,
                model_save_path='', max_len=200, evaluate=False):
    
    word2idx = build_vocab(train_df['text'])
    
    X_train = torch.tensor([encode(t, word2idx, max_len) for t in train_df['text']])
    y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)
    X_test = torch.tensor([encode(t, word2idx, max_len) for t in test_df['text']])
    y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)
    
    # Choose model: 'cnn', 'bilstm', or 'bert'
    if model_type == 'cnn':
        model = SpamCNN(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                        pretrained_embeddings=pretrained_embeddings)
        train_inputs, train_labels = X_train, y_train
        test_inputs, test_labels = X_test, y_test
    elif model_type == 'bilstm':
        model = BiLSTMSpam(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                           pretrained_embeddings=pretrained_embeddings)
        train_inputs, train_labels = X_train, y_train
        test_inputs, test_labels = X_test, y_test
    elif model_type == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenize with BERT tokenizer
        
        def bert_encode(texts, tokenizer, max_len=200):  # Tokenize and encode sequences
            return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_len,
                             return_tensors='pt')
        
        train_encodings = bert_encode(train_df['text'], tokenizer, max_len)
        test_encodings = bert_encode(test_df['text'], tokenizer, max_len)
        model = SpamBERT()
        train_inputs, train_labels = train_encodings, y_train
        test_inputs, test_labels = test_encodings, y_test
    else:
        raise ValueError('Invalid model_type')
    
    # Set model-specific training parameters
    if model_type == 'cnn':
        epochs = 50
        learning_rate = 1e-3
    elif model_type == 'bilstm':
        epochs = 40
        learning_rate = 8e-4
    elif model_type == 'bert':
        epochs = 10
        learning_rate = 2e-5
    else:
        raise ValueError('Invalid model_type')
    
    # Move model to GPU if available
    model = model.cuda() if torch.cuda.is_available() else model
    
    batch_size = 32
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    if model_type in ['cnn', 'bilstm']:
        train_dataset = TensorDataset(train_inputs, train_labels)
        test_dataset = TensorDataset(test_inputs, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:  # BERT
        train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
        test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == 'bert':
                input_ids, attention_mask, labels = [b.cuda() if torch.cuda.is_available() else b for b in batch]
                outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                inputs, labels = [b.cuda() if torch.cuda.is_available() else b for b in batch]
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            # Check tensor dimensions before squeezing
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")
    
    # Save model to Google Drive
    model_save_file = os.path.join(model_save_path, f'spam_{model_type}.pt')
    model.save(model_save_file)
    print(f"Model saved to {model_save_file}")
    
    if evaluate:
        if model_type == 'bert':
            evaluate_model(model, model_type, test_loader=test_loader)
        else:
            evaluate_model(model, model_type, X_test=test_inputs, y_test=test_labels)
    
    return model


def evaluate_model(model, model_type, X_test=None, y_test=None, test_loader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if model_type == 'bert':
        y_pred, y_true = [], []
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = model(**inputs)
            logits = outputs.logits
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_true.extend(batch[2].cpu().numpy())
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        
    elif model_type in ['cnn', 'bilstm']:
        X_test = X_test.to(device)
        with torch.no_grad():
            outputs = model(X_test)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # Check tensor dimensions before squeezing
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            predictions = (outputs > 0.5).cpu().numpy()
        
        print(classification_report(y_test.numpy(), predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test.numpy(), predictions))

