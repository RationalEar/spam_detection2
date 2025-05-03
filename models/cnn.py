import torch
import torch.nn as nn
import torch.nn.functional as F


class SpamCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings, num_classes=1, dropout=0.5):
        super(SpamCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)

        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x).squeeze(-1)  # (batch_size, 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x  # shape: (batch_size, 1)


    def grad_cam(self, x):
        # Placeholder for Grad-CAM explainability integration
        # To be implemented: generate class activation maps for input x
        pass


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
