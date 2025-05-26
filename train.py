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


def train_bilstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, max_norm=1.0, 
             adversarial_training=True, epsilon=0.1):
    """
    Training loop for BiLSTM model with gradient clipping and adversarial training
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            inputs, labels = [b.to(device) for b in batch]
            batch_size = inputs.size(0)
            
            # Regular forward pass
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            # Adversarial training
            if adversarial_training:
                # Generate adversarial examples
                adv_embeddings = model.generate_adversarial_example(inputs, labels, epsilon=epsilon)
                # Forward pass with adversarial examples
                adv_outputs, _ = model(adv_embeddings)
                adv_loss = criterion(adv_outputs, labels)
                # Combine losses
                loss = 0.5 * (loss + adv_loss)
            
            loss.backward()
            # Clip gradients
            model.clip_gradients(max_norm)
            optimizer.step()
            
            total_train_loss += loss.item() * batch_size
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += batch_size
        
        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = [b.to(device) for b in batch]
                batch_size = inputs.size(0)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item() * batch_size
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += batch_size
        
        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val
        
        # Update training history
        model.update_training_history(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save('best_model.pt')
    
    return model


def train_model(model_type, train_df, test_df, embedding_dim=300, pretrained_embeddings=None,
                model_save_path='', max_len=200, evaluate=False):
    
    word2idx, idx2word = build_vocab(train_df['text'])
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    batch_size = 32
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    if model_type == 'bilstm':
        # Create data loaders for BiLSTM
        train_dataset = TensorDataset(train_inputs, train_labels)
        val_dataset = TensorDataset(test_inputs, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train BiLSTM with specialized training loop
        model = train_bilstm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=epochs,
            device=device,
            max_norm=1.0,
            adversarial_training=True,
            epsilon=0.1
        )
    else:  # CNN or BERT
        if model_type == 'bert':
            train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
            test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
        else:  # CNN
            train_dataset = TensorDataset(train_inputs, train_labels)
            test_dataset = TensorDataset(test_inputs, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Standard training loop for CNN and BERT
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                if model_type == 'bert':
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                else:  # CNN
                    inputs, labels = [b.to(device) for b in batch]
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
    
    # Save model to specified path
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
                
            # Fix: Handle tuple output from SpamBERT model
            if isinstance(outputs, tuple):
                probs = outputs[0]  # First element contains the probabilities
            else:
                probs = outputs
                
            predictions = (probs > 0.5).long().cpu().numpy()
            y_pred.extend(predictions)
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
