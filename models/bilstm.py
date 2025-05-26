import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch_size, seq_len, hidden_dim*2)
        attn_weights = self.attn(lstm_output).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim*2)
        return context, attn_weights


class BiLSTMSpam(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings, hidden_dim=128, num_layers=2, num_classes=1, dropout=0.5):
        super(BiLSTMSpam, self).__init__()
        self.config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'dropout': dropout
        }
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            # Regular forward pass with input indices
            x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # If x is already an embedding tensor, use it directly
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        context, attn_weights = self.attention(lstm_out, mask)
        x = self.dropout(F.relu(self.fc1(context)))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x, attn_weights

    def generate_adversarial_example(self, x, y, epsilon=0.1, num_steps=10):
        """
        Generate adversarial examples using the Fast Gradient Sign Method (FGSM)
        Args:
            x: Input tensor (batch_size, seq_len)
            y: Target labels
            epsilon: Maximum perturbation size
            num_steps: Number of optimization steps
        Returns:
            Adversarial examples as token indices
        """
        self.train()  # Enable gradients
        
        # Ensure inputs are on the correct device
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device)
        
        # Get initial embeddings
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Create adversarial embeddings starting from original embeddings
        emb_adv = embeddings.clone().detach().requires_grad_(True)
        criterion = nn.BCELoss()
        
        # PGD attack
        alpha = epsilon / num_steps  # Step size
        
        for step in range(num_steps):
            # Forward pass through LSTM layers with current adversarial embeddings
            batch_size = emb_adv.size(0)
            
            # Pack the embeddings to handle variable lengths
            lengths = (x != 0).sum(dim=1)  # Get actual sequence lengths
            packed_embeddings = pack_padded_sequence(emb_adv, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
            # Pass through LSTM
            lstm_out, _ = self.lstm(packed_embeddings)
            
            # Unpack the sequence
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            
            # Apply attention
            if hasattr(self, 'attention'):
                attn_mask = (x != 0).float()  # Create attention mask
                context, _ = self.attention(lstm_out, attn_mask)
            else:
                # If no attention, use last hidden state
                context = lstm_out[:, -1, :]
            
            # Final classification
            x_fc1 = self.dropout(F.relu(self.fc1(context)))
            outputs = torch.sigmoid(self.fc2(x_fc1))
            
            # Compute loss (maximize loss to create adversarial example)
            loss = -criterion(outputs.squeeze(), y)  # Negative because we want to maximize loss
            
            # Compute gradients
            loss.backward()
            
            # Update adversarial embeddings
            with torch.no_grad():
                # Get gradient sign
                grad_sign = emb_adv.grad.sign()
                
                # Update embeddings
                emb_adv.data = emb_adv.data + alpha * grad_sign
                
                # Project back to epsilon ball around original embeddings
                delta = emb_adv.data - embeddings
                delta = torch.clamp(delta, -epsilon, epsilon)
                emb_adv.data = embeddings + delta
                
                # Reset gradients for next step
                if step < num_steps - 1:
                    emb_adv.grad.zero_()
        
        # Convert perturbed embeddings back to token indices
        with torch.no_grad():
            # Get embedding matrix
            emb_matrix = self.embedding.weight  # (vocab_size, embedding_dim)
            
            # Initialize output tensor
            batch_size, seq_len, emb_dim = emb_adv.size()
            x_adv = torch.zeros_like(x)
            
            # Compute cosine similarity between perturbed embeddings and embedding matrix
            emb_adv_flat = emb_adv.view(-1, emb_dim)
            emb_matrix_norm = F.normalize(emb_matrix, p=2, dim=1)
            emb_adv_norm = F.normalize(emb_adv_flat, p=2, dim=1)
            
            # Compute similarities and get closest tokens
            similarities = torch.mm(emb_adv_norm, emb_matrix_norm.t())
            closest_tokens = similarities.argmax(dim=1)
            x_adv = closest_tokens.view(batch_size, seq_len)
            
            # Preserve padding tokens from original input
            pad_mask = (x == 0)
            x_adv = torch.where(pad_mask, x, x_adv)
        
        self.eval()  # Reset to evaluation mode
        return x_adv

    def clip_gradients(self, max_norm=1.0):
        """
        Clips gradients of the model parameters.
        Args:
            max_norm: Maximum norm for gradient clipping
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def update_training_history(self, train_loss, val_loss, train_acc, val_acc):
        """
        Updates the training history with new metrics
        """
        self.training_history['train_loss'].append(float(train_loss))
        self.training_history['val_loss'].append(float(val_loss))
        self.training_history['train_acc'].append(float(train_acc))
        self.training_history['val_acc'].append(float(val_acc))

    def save(self, path):
        """
        Save model state, configuration, and training history
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        
        # Save config and history separately for easy access
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'training_history': self.training_history
            }, f, indent=4)

    def load(self, path):
        """
        Load model state, configuration, and training history
        """
        checkpoint = torch.load(path)
        
        # Load configuration if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        
        self.eval()
