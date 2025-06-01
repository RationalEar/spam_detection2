import torch
import numpy as np
from sklearn.metrics import auc
from scipy.stats import spearmanr


def area_under_deletion_curve(model, input_tensor, attributions, baseline_output):
    """
    Calculates the Area Under the Deletion Curve (AUDC) for faithfulness.
    Args:
        model: The model to evaluate.
        input_tensor: The input tensor (batch_size, seq_len).
        attributions: Attribution scores for each token (batch_size, seq_len).
        baseline_output: The model's output on the original input.
    Returns:
        AUDC score.
    """
    num_features = input_tensor.size(1)
    sorted_indices = torch.argsort(attributions, descending=True, dim=1)
    
    deletion_scores = []
    current_input = input_tensor.clone()
    
    for i in range(num_features):
        # Create a mask for the current feature to delete
        mask = torch.ones_like(current_input)
        # Set the i-th most important feature (across the batch) to 0 (or a PAD token)
        # This needs to be done carefully if PAD token is not 0 or if dealing with embeddings directly
        # For simplicity, assuming 0 is a PAD token or a neutral value after embedding.
        # A more robust way might be to replace with a specific PAD token ID.
        for batch_idx in range(input_tensor.size(0)):
            mask[batch_idx, sorted_indices[batch_idx, i]] = 0 # Mask out the token
        
        # Apply mask. If input is token IDs, this means setting token ID to 0 (PAD)
        # If input is embeddings, this means zeroing out the embedding vector.
        # Assuming input_tensor here are token IDs and 0 is PAD.
        masked_input = current_input * mask 
        
        with torch.no_grad():
            output, _ = model(masked_input.long()) # Ensure input is long for embedding layer
        
        # Assuming binary classification and sigmoid output
        # We measure the drop from the baseline
        score_drop = baseline_output - output 
        deletion_scores.append(score_drop.mean().item()) # Average over batch if batch_size > 1
        
        current_input = masked_input # Update current input for next iteration

    # Normalize by the number of features
    return np.trapz(deletion_scores, dx=1.0/num_features)


def area_under_insertion_curve(model, input_tensor, attributions, baseline_value=0.0):
    """
    Calculates the Area Under the Insertion Curve (AUIC) for faithfulness.
    Args:
        model: The model to evaluate.
        input_tensor: The input tensor (batch_size, seq_len).
        attributions: Attribution scores for each token (batch_size, seq_len).
        baseline_value: The model's output on a baseline input (e.g., all PAD tokens).
                        Often, this is assumed to be close to 0 or 0.5 for balanced classes.
    Returns:
        AUIC score.
    """
    num_features = input_tensor.size(1)
    sorted_indices = torch.argsort(attributions, descending=True, dim=1) # Most important first
    
    insertion_scores = []
    # Start with a baseline input (e.g., all PAD tokens, assuming 0 is PAD ID)
    current_input = torch.zeros_like(input_tensor) 
    
    for i in range(num_features):
        # Add the i-th most important feature
        for batch_idx in range(input_tensor.size(0)):
            token_to_insert_idx = sorted_indices[batch_idx, i]
            original_token_value = input_tensor[batch_idx, token_to_insert_idx]
            current_input[batch_idx, token_to_insert_idx] = original_token_value
            
        with torch.no_grad():
            output, _ = model(current_input.long()) # Ensure input is long
            
        # Measure the increase from the baseline_value
        score_increase = output - baseline_value 
        insertion_scores.append(score_increase.mean().item()) # Average over batch

    # Normalize by the number of features
    return np.trapz(insertion_scores, dx=1.0/num_features)


def comprehensiveness_score(model, input_tensor, attributions, baseline_output, k=5):
    """
    Calculates the Comprehensiveness score.
    This is the change in prediction when the top-k most important features are removed.
    Args:
        model: The model to evaluate.
        input_tensor: The input tensor (batch_size, seq_len).
        attributions: Attribution scores for each token (batch_size, seq_len).
        baseline_output: The model's output on the original input.
        k: The number of top features to remove.
    Returns:
        Comprehensiveness score.
    """
    sorted_indices = torch.argsort(attributions, descending=True, dim=1)
    top_k_indices = sorted_indices[:, :k]
    
    masked_input = input_tensor.clone()
    for batch_idx in range(input_tensor.size(0)):
        masked_input[batch_idx, top_k_indices[batch_idx]] = 0 # Assuming 0 is PAD/neutral
        
    with torch.no_grad():
        output_after_removal, _ = model(masked_input.long())
        
    # Comprehensiveness is the difference in output probability
    # Higher is better if it means removing important features significantly changes output towards the opposite class
    # Or simply, the magnitude of change.
    # Here, we define it as baseline_output - output_after_removal. 
    # A large positive value means removing features REDUCED the original prediction score.
    comprehensiveness = (baseline_output - output_after_removal).mean().item()
    return comprehensiveness


def jaccard_stability(attributions1, attributions2, k=5):
    """
    Calculates Jaccard Stability of top-k features between two explanations.
    Args:
        attributions1: Attribution scores for the first input (batch_size, seq_len).
        attributions2: Attribution scores for the second (similar) input (batch_size, seq_len).
        k: The number of top features to consider.
    Returns:
        Jaccard stability score (average over batch).
    """
    if attributions1.shape != attributions2.shape:
        raise ValueError("Attribution shapes must match.")

    top_k_indices1 = torch.argsort(attributions1, descending=True, dim=1)[:, :k]
    top_k_indices2 = torch.argsort(attributions2, descending=True, dim=1)[:, :k]
    
    jaccard_scores = []
    for i in range(attributions1.size(0)):
        set1 = set(top_k_indices1[i].tolist())
        set2 = set(top_k_indices2[i].tolist())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            jaccard_scores.append(1.0) # Both sets are empty, perfect match
        else:
            jaccard_scores.append(intersection / union)
            
    return np.mean(jaccard_scores)


def rank_correlation(attributions1, attributions2):
    """
    Calculates Spearman's rank correlation between two sets of attributions.
    Args:
        attributions1: Attribution scores for the first input (batch_size, seq_len).
        attributions2: Attribution scores for the second (e.g., perturbed) input (batch_size, seq_len).
    Returns:
        Average Spearman's rank correlation coefficient over the batch.
    """
    if attributions1.shape != attributions2.shape:
        raise ValueError("Attribution shapes must match.")
    
    correlations = []
    for i in range(attributions1.size(0)):
        # Flatten or ensure 1D for spearmanr
        attr1_flat = attributions1[i].flatten().cpu().numpy()
        attr2_flat = attributions2[i].flatten().cpu().numpy()
        
        # Remove NaNs or Infs if any, though ideally attributions should be clean
        valid_mask = np.isfinite(attr1_flat) & np.isfinite(attr2_flat)
        attr1_flat = attr1_flat[valid_mask]
        attr2_flat = attr2_flat[valid_mask]

        if len(attr1_flat) < 2 or len(attr2_flat) < 2:
            # Not enough data points to compute correlation for this sample
            correlations.append(np.nan) # Or 0, or skip
            continue
            
        correlation, _ = spearmanr(attr1_flat, attr2_flat)
        correlations.append(correlation)
        
    return np.nanmean(correlations) # Use nanmean to ignore NaNs if any sample had issues 