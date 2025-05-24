import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc


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
        device = next(self.parameters()).device
        x = x.to(device)
        # Ensure x requires gradients for backward
        # x.requires_grad_()
        
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
        
        # Ensure target_class is on the same device
        target_class = target_class.to(device)
        
        # For binary, target_class is shape (batch, 1)
        one_hot = torch.zeros_like(probs, device=device)
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


    def predict(self, x):
        """
        Make predictions on input data
        Args:
            x: input tensor (batch_size, seq_len)
        Returns:
            predictions: tensor of predictions (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.squeeze(1)


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, map_location=None):
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()


    def compute_metrics(self, y_true, y_pred, y_prob=None, fp_cost=0.3, fn_cost=0.7):
        """
        Compute comprehensive evaluation metrics including AUC-ROC, FPR, FNR, and cost-sensitive evaluation.
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities (for AUC-ROC)
            fp_cost: Cost weight for false positives
            fn_cost: Cost weight for false negatives
        Returns:
            dict: Dictionary containing all metrics
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()

        # Calculate confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate basic metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Calculate cost-sensitive error
        weighted_error = (fp_cost * fp + fn_cost * fn) / len(y_true)

        # Calculate AUC-ROC if probabilities are provided
        auc_roc = None
        if y_prob is not None:
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
            auc_roc = auc(fpr_curve, tpr_curve)

        return {
            'fpr': fpr,
            'fnr': fnr,
            'weighted_error': weighted_error,
            'auc_roc': auc_roc,
            'confusion_matrix': {
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
        }


    def compute_explanation_metrics(self, x, cam_maps, num_perturbations=10):
        """
        Compute explainability metrics including faithfulness and stability.
        Args:
            x: Input tensor (batch_size, seq_len)
            cam_maps: Grad-CAM activation maps (batch_size, seq_len)
            num_perturbations: Number of perturbations for stability testing
        Returns:
            dict: Dictionary containing explainability metrics
        """
        self.eval()
        device = next(self.parameters()).device
        batch_size = x.size(0)
        
        # Compute AUC-Del (Area Under Deletion curve)
        def compute_auc_del(x_single, cam_single):
            # Sort indices by importance
            _, indices = torch.sort(cam_single, descending=True)
            deletions = []
            x_perturbed = x_single.clone()
            
            # Progressively delete most important tokens
            for i in range(len(indices)):
                x_perturbed[indices[i]] = 0  # Zero out token
                with torch.no_grad():
                    pred = self(x_perturbed.unsqueeze(0))
                deletions.append(pred.item())
            
            # Calculate AUC
            auc_del = np.trapz(deletions) / len(indices)
            return auc_del

        # Compute AUC-Ins (Area Under Insertion curve)
        def compute_auc_ins(x_single, cam_single):
            # Sort indices by importance
            _, indices = torch.sort(cam_single, descending=True)
            insertions = []
            x_perturbed = torch.zeros_like(x_single)
            
            # Progressively insert most important tokens
            for i in range(len(indices)):
                x_perturbed[indices[i]] = x_single[indices[i]]
                with torch.no_grad():
                    pred = self(x_perturbed.unsqueeze(0))
                insertions.append(pred.item())
            
            # Calculate AUC
            auc_ins = np.trapz(insertions) / len(indices)
            return auc_ins

        # Compute Jaccard Stability
        def compute_stability(x_single, cam_single, k=5):
            # Generate perturbations
            perturbations = []
            for _ in range(num_perturbations):
                # Add small random noise to input embeddings
                noise = torch.randn_like(x_single) * 0.1
                x_perturbed = x_single + noise
                
                # Get new explanation
                with torch.no_grad():
                    cam_perturbed = self.grad_cam(x_perturbed.unsqueeze(0))[0]
                
                # Get top-k indices
                _, top_k_orig = torch.topk(cam_single, k)
                _, top_k_pert = torch.topk(cam_perturbed, k)
                
                # Calculate Jaccard similarity
                intersection = len(set(top_k_orig.tolist()) & set(top_k_pert.tolist()))
                union = len(set(top_k_orig.tolist()) | set(top_k_pert.tolist()))
                jaccard = intersection / union
                perturbations.append(jaccard)
            
            return np.mean(perturbations)

        # Calculate metrics for each sample in batch
        metrics = {
            'auc_del': [],
            'auc_ins': [],
            'stability': [],
            'ecs': []  # Explanation Consistency Score
        }

        for i in range(batch_size):
            auc_del = compute_auc_del(x[i], cam_maps[i])
            auc_ins = compute_auc_ins(x[i], cam_maps[i])
            stability = compute_stability(x[i], cam_maps[i])
            
            # Calculate ECS (Explanation Consistency Score)
            # ECS = 0.4*Faithfulness + 0.3*Stability + 0.2*Plausibility + 0.1*Simplicity
            # Here we use a simplified version without plausibility (which requires human evaluation)
            faithfulness = (auc_ins - auc_del + 1) / 2  # Normalize to [0,1]
            simplicity = 1 - (torch.count_nonzero(cam_maps[i]) / cam_maps[i].numel())
            ecs = 0.5 * faithfulness + 0.4 * stability + 0.1 * simplicity
            
            metrics['auc_del'].append(auc_del)
            metrics['auc_ins'].append(auc_ins)
            metrics['stability'].append(stability)
            metrics['ecs'].append(ecs)

        # Average metrics across batch
        return {k: np.mean(v) for k, v in metrics.items()}

    def generate_adversarial_example(self, x, y, epsilon=0.1, num_steps=10):
        """
        Generate adversarial examples using the Fast Gradient Sign Method (FGSM)
        Args:
            x: Input tensor (batch_size, seq_len)
            y: Target labels
            epsilon: Maximum perturbation size
            num_steps: Number of optimization steps
        Returns:
            Adversarial examples
        """
        self.train()  # Enable gradients
        x_adv = x.clone().detach().requires_grad_(True)
        criterion = nn.BCELoss()
        
        for _ in range(num_steps):
            outputs = self(x_adv)
            loss = criterion(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # Update adversarial example
            x_adv = x_adv + epsilon * grad.sign()
            # Project back to valid token space (this is a simplified version)
            x_adv = torch.clamp(x_adv, 0, self.embedding.num_embeddings - 1)
            x_adv = x_adv.detach().requires_grad_(True)
        
        self.eval()
        return x_adv.detach()

    def measure_adversarial_robustness(self, x, y, epsilon_range=[0.01, 0.05, 0.1]):
        """
        Measure model robustness against adversarial attacks
        Args:
            x: Input tensor
            y: True labels
            epsilon_range: List of perturbation sizes to test
        Returns:
            dict: Dictionary containing robustness metrics
        """
        metrics = {
            'clean_accuracy': None,
            'adversarial_accuracy': {},
            'explanation_shift': {}
        }
        
        # Get clean predictions and explanations
        with torch.no_grad():
            clean_preds = self(x)
            clean_cam = self.grad_cam(x)
        clean_acc = ((clean_preds > 0.5).float() == y).float().mean().item()
        metrics['clean_accuracy'] = clean_acc
        
        # Test different perturbation sizes
        for epsilon in epsilon_range:
            # Generate adversarial examples
            x_adv = self.generate_adversarial_example(x, y, epsilon=epsilon)
            
            # Get adversarial predictions and explanations
            with torch.no_grad():
                adv_preds = self(x_adv)
                adv_cam = self.grad_cam(x_adv)
            
            # Calculate adversarial accuracy
            adv_acc = ((adv_preds > 0.5).float() == y).float().mean().item()
            metrics['adversarial_accuracy'][epsilon] = adv_acc
            
            # Calculate explanation shift
            cos_sim = F.cosine_similarity(clean_cam.view(x.size(0), -1), 
                                        adv_cam.view(x.size(0), -1), dim=1)
            avg_shift = 1 - cos_sim.mean().item()  # Convert similarity to distance
            metrics['explanation_shift'][epsilon] = avg_shift
        
        return metrics

    def evaluate_adversarial_examples(self, x, y):
        """
        Comprehensive evaluation of model behavior under adversarial attack
        Args:
            x: Input tensor
            y: True labels
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Standard metrics on clean data
        clean_metrics = self.compute_metrics(y, (self(x) > 0.5).float(), self(x))
        
        # Generate adversarial examples
        x_adv = self.generate_adversarial_example(x, y)
        
        # Metrics on adversarial examples
        adv_metrics = self.compute_metrics(y, (self(x_adv) > 0.5).float(), self(x_adv))
        
        # Get explanations for both clean and adversarial
        clean_cam = self.grad_cam(x)
        adv_cam = self.grad_cam(x_adv)
        
        # Compute explanation metrics for both
        clean_exp_metrics = self.compute_explanation_metrics(x, clean_cam)
        adv_exp_metrics = self.compute_explanation_metrics(x_adv, adv_cam)
        
        return {
            'clean': {
                'performance': clean_metrics,
                'explanations': clean_exp_metrics
            },
            'adversarial': {
                'performance': adv_metrics,
                'explanations': adv_exp_metrics
            }
        }
