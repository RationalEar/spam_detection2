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


    def grad_cam(self, x, target_class=None):
        """
        Compute Grad-CAM for the input batch x.
        Args:
            x: input tensor (batch_size, seq_len)
            target_class: index of the class to compute Grad-CAM for (default: predicted class)
        Returns:
            cam: class activation map (batch_size, seq_len)
        """
        self.eval()
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        # Register hooks on the last conv layer
        handle_fwd = self.conv3.register_forward_hook(forward_hook)
        handle_bwd = self.conv3.register_backward_hook(backward_hook)

        x_emb = self.embedding(x)
        x_perm = x_emb.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x_perm))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        pooled = self.global_max_pool(x3).squeeze(-1)
        x_fc1 = self.dropout(F.relu(self.fc1(pooled)))
        logits = self.fc2(x_fc1)
        probs = torch.sigmoid(logits)

        if target_class is None:
            target_class = (probs > 0.5).long()
        # For binary, target_class is shape (batch, 1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target_class, 1.0)
        self.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get activations and gradients
        act = activations[0]  # (batch, channels, seq_len)
        grad = gradients[0]   # (batch, channels, seq_len)
        weights = grad.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        cam = (weights * act).sum(dim=1)  # (batch, seq_len)
        cam = F.relu(cam)
        # Normalize each CAM to [0, 1]
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Remove hooks
        handle_fwd.remove()
        handle_bwd.remove()
        return cam


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
