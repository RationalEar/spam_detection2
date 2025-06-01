import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_attack_success_rate_cnn(original_preds_prob, adversarial_preds_prob, threshold=0.5):
    """
    Calculates the Attack Success Rate (ASR).
    Args:
        original_preds_prob: PyTorch tensor or NumPy array of original prediction probabilities (batch_size,).
        adversarial_preds_prob: PyTorch tensor or NumPy array of adversarial prediction probabilities (batch_size,).
        threshold: Classification threshold (default 0.5).
    Returns:
        float: Attack Success Rate.
    """
    if isinstance(original_preds_prob, torch.Tensor):
        original_preds_prob = original_preds_prob.detach().cpu().numpy()
    if isinstance(adversarial_preds_prob, torch.Tensor):
        adversarial_preds_prob = adversarial_preds_prob.detach().cpu().numpy()

    original_binary_preds = (original_preds_prob > threshold).astype(int)
    adversarial_binary_preds = (adversarial_preds_prob > threshold).astype(int)
    
    successful_attacks = np.sum(original_binary_preds != adversarial_binary_preds)
    asr = successful_attacks / len(original_binary_preds) if len(original_binary_preds) > 0 else 0.0
    return asr

def calculate_explanation_shift_cnn(original_cams, adversarial_cams):
    """
    Calculates the Explanation Shift using Cosine Similarity between CAMs.
    Args:
        original_cams: PyTorch tensor or NumPy array of original CAMs (batch_size, seq_len).
        adversarial_cams: PyTorch tensor or NumPy array of adversarial CAMs (batch_size, seq_len).
    Returns:
        float: Average cosine distance (1 - cosine_similarity).
    """
    if original_cams.shape[0] == 0 or adversarial_cams.shape[0] == 0:
        return np.nan
        
    if isinstance(original_cams, torch.Tensor):
        original_cams = original_cams.detach().cpu().numpy()
    if isinstance(adversarial_cams, torch.Tensor):
        adversarial_cams = adversarial_cams.detach().cpu().numpy()

    if original_cams.shape != adversarial_cams.shape:
        # Attempt to truncate/pad if lengths are slightly different (e.g. due to minor tokenization changes for adv example)
        # This is less likely for CNNs with fixed max_len inputs but good for robustness
        min_len = min(original_cams.shape[1], adversarial_cams.shape[1])
        original_cams = original_cams[:, :min_len]
        adversarial_cams = adversarial_cams[:, :min_len]
        if original_cams.shape[1] != adversarial_cams.shape[1] or original_cams.shape[1] == 0:
            print(f"Warning: CAM shapes still mismatch after attempting to align: {original_cams.shape} vs {adversarial_cams.shape}. Returning NaN for explanation shift.")
            return np.nan

    similarities = []
    for i in range(original_cams.shape[0]):
        sim = cosine_similarity(original_cams[i].reshape(1, -1), adversarial_cams[i].reshape(1, -1))[0, 0]
        similarities.append(sim)
    
    avg_cosine_similarity = np.nanmean(similarities) if similarities else np.nan
    return 1.0 - avg_cosine_similarity # Return distance

def calculate_top_k_retention_cnn(original_cams, adversarial_cams, k=5):
    """
    Calculates Top-k Retention for CAMs.
    Args:
        original_cams: PyTorch tensor or NumPy array of original CAMs (batch_size, seq_len).
        adversarial_cams: PyTorch tensor or NumPy array of adversarial CAMs (batch_size, seq_len).
        k: Number of top features to consider.
    Returns:
        float: Average Top-k Retention score.
    """
    if original_cams.shape[0] == 0 or adversarial_cams.shape[0] == 0:
        return np.nan

    if isinstance(original_cams, torch.Tensor):
        original_cams = original_cams.detach().cpu().numpy()
    if isinstance(adversarial_cams, torch.Tensor):
        adversarial_cams = adversarial_cams.detach().cpu().numpy()

    if original_cams.shape != adversarial_cams.shape:
        min_len = min(original_cams.shape[1], adversarial_cams.shape[1])
        original_cams = original_cams[:, :min_len]
        adversarial_cams = adversarial_cams[:, :min_len]
        if original_cams.shape[1] != adversarial_cams.shape[1] or original_cams.shape[1] == 0:
            print(f"Warning: CAM shapes still mismatch for top-k retention: {original_cams.shape} vs {adversarial_cams.shape}. Returning NaN.")
            return np.nan

    retention_scores = []
    num_features = original_cams.shape[1]
    actual_k = min(k, num_features)

    if actual_k == 0: return np.nan # Cannot compute retention for k=0 if num_features > 0, or 1.0 if num_features also 0

    for i in range(original_cams.shape[0]):
        orig_top_k_indices = np.argsort(original_cams[i])[-actual_k:]
        adv_top_k_indices = np.argsort(adversarial_cams[i])[-actual_k:]
        
        intersection = len(set(orig_top_k_indices).intersection(set(adv_top_k_indices)))
        retention = intersection / actual_k
        retention_scores.append(retention)
        
    return np.nanmean(retention_scores) if retention_scores else np.nan 