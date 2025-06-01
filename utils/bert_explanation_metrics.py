import torch
import numpy as np
from scipy.stats import spearmanr
# No sklearn.metrics.auc needed if using np.trapz directly for AUDC/AUIC

def get_token_level_attributions_from_ig(ig_attributions):
    """Converts Integrated Gradients from (batch, seq, emb_dim) to (batch, seq) by summing embeddings."""
    if ig_attributions.ndim == 3:
        return ig_attributions.sum(dim=2)
    return ig_attributions # If already token-level

def get_token_level_attributions_from_attention(attention_map):
    """Converts an attention map (batch, seq, seq) to token importance (batch, seq) by averaging received attention."""
    if attention_map.ndim == 3: # (batch, seq_len_src, seq_len_tgt)
        # Importance of a token is how much other tokens attend to it (or mean attention it receives)
        return attention_map.mean(dim=1) # Average attention received by each target token from all source tokens
    elif attention_map.ndim == 2: # (seq_len_src, seq_len_tgt) - add batch dim
        return attention_map.mean(dim=0).unsqueeze(0)
    return attention_map

def area_under_deletion_curve_bert(model, tokenizer, input_ids, attention_mask, token_level_attributions, baseline_output):
    """
    Calculates AUDC for BERT models.
    token_level_attributions: (batch_size, seq_len)
    baseline_output: (batch_size,)
    """
    device = input_ids.device
    pad_token_id = tokenizer.pad_token_id
    batch_size, num_features = input_ids.size()
    
    # Ensure attributions match seq_len of input_ids, considering [CLS] and [SEP]
    # Attributions should correspond to the tokens in input_ids
    if token_level_attributions.shape[1] != num_features:
        # This might happen if attributions did not cover CLS/SEP or were for different tokenization
        # For now, we assume they match. Add padding/truncation logic for attributions if necessary.
        print(f"Warning: AUDC attribution length ({token_level_attributions.shape[1]}) != input_ids length ({num_features}). Truncating/padding attributions.")
        if token_level_attributions.shape[1] > num_features:
            token_level_attributions = token_level_attributions[:, :num_features]
        else:
            padding = torch.zeros(batch_size, num_features - token_level_attributions.shape[1], device=device)
            token_level_attributions = torch.cat([token_level_attributions, padding], dim=1)

    sorted_indices = torch.argsort(token_level_attributions, descending=True, dim=1)
    
    deletion_scores_all_batches = []

    for b_idx in range(batch_size):
        current_input_ids = input_ids[b_idx].clone().unsqueeze(0) # (1, seq_len)
        current_attention_mask = attention_mask[b_idx].clone().unsqueeze(0) # (1, seq_len)
        original_baseline_output = baseline_output[b_idx] if baseline_output.ndim > 0 else baseline_output

        batch_deletion_scores = []
        for i in range(num_features):
            token_idx_to_mask = sorted_indices[b_idx, i].item()
            
            # Mask token by replacing with PAD and updating attention mask
            masked_input_ids_step = current_input_ids.clone()
            masked_input_ids_step[0, token_idx_to_mask] = pad_token_id
            
            masked_attention_mask_step = current_attention_mask.clone()
            masked_attention_mask_step[0, token_idx_to_mask] = 0 # Mask out in attention
            
            with torch.no_grad():
                output, _ = model(input_ids=masked_input_ids_step, attention_mask=masked_attention_mask_step)
            
            score_drop = original_baseline_output - output.squeeze()
            batch_deletion_scores.append(score_drop.item())
            
            current_input_ids = masked_input_ids_step
            current_attention_mask = masked_attention_mask_step
        deletion_scores_all_batches.append(np.trapz(batch_deletion_scores, dx=1.0/num_features) if num_features > 0 else 0.0)
    
    return np.mean(deletion_scores_all_batches)

def area_under_insertion_curve_bert(model, tokenizer, input_ids, attention_mask, token_level_attributions, baseline_value_initial):
    """
    Calculates AUIC for BERT models.
    token_level_attributions: (batch_size, seq_len)
    baseline_value_initial: scalar, model output on all-PAD input.
    """
    device = input_ids.device
    pad_token_id = tokenizer.pad_token_id
    batch_size, num_features = input_ids.size()

    if token_level_attributions.shape[1] != num_features:
        print(f"Warning: AUIC attribution length ({token_level_attributions.shape[1]}) != input_ids length ({num_features}). Truncating/padding attributions.")
        if token_level_attributions.shape[1] > num_features:
            token_level_attributions = token_level_attributions[:, :num_features]
        else:
            padding = torch.zeros(batch_size, num_features - token_level_attributions.shape[1], device=device)
            token_level_attributions = torch.cat([token_level_attributions, padding], dim=1)

    sorted_indices = torch.argsort(token_level_attributions, descending=True, dim=1)
    
    insertion_scores_all_batches = []

    for b_idx in range(batch_size):
        # Start with all PAD tokens
        current_input_ids = torch.full_like(input_ids[b_idx], pad_token_id).unsqueeze(0)
        current_attention_mask = torch.zeros_like(attention_mask[b_idx]).unsqueeze(0)
        
        batch_insertion_scores = []
        for i in range(num_features):
            token_idx_to_insert = sorted_indices[b_idx, i].item()
            original_token_value = input_ids[b_idx, token_idx_to_insert]
            
            current_input_ids[0, token_idx_to_insert] = original_token_value
            current_attention_mask[0, token_idx_to_insert] = 1 # Unmask in attention
            
            with torch.no_grad():
                output, _ = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
            
            score_increase = output.squeeze() - baseline_value_initial 
            batch_insertion_scores.append(score_increase.item())
        
        insertion_scores_all_batches.append(np.trapz(batch_insertion_scores, dx=1.0/num_features) if num_features > 0 else 0.0)

    return np.mean(insertion_scores_all_batches)

def comprehensiveness_score_bert(model, tokenizer, input_ids, attention_mask, token_level_attributions, baseline_output, k=5):
    """
    Calculates Comprehensiveness for BERT.
    baseline_output: (batch_size,)
    """
    device = input_ids.device
    pad_token_id = tokenizer.pad_token_id
    batch_size, num_features = input_ids.size()

    if token_level_attributions.shape[1] != num_features:
        print(f"Warning: Comp attribution length ({token_level_attributions.shape[1]}) != input_ids length ({num_features}). Truncating/padding attributions.")
        if token_level_attributions.shape[1] > num_features:
            token_level_attributions = token_level_attributions[:, :num_features]
        else:
            padding = torch.zeros(batch_size, num_features - token_level_attributions.shape[1], device=device)
            token_level_attributions = torch.cat([token_level_attributions, padding], dim=1)
            
    actual_k = min(k, num_features)
    if actual_k == 0: return 0.0

    sorted_indices = torch.argsort(token_level_attributions, descending=True, dim=1)
    top_k_indices = sorted_indices[:, :actual_k]
    
    all_comprehensiveness_scores = []

    for b_idx in range(batch_size):
        masked_input_ids = input_ids[b_idx].clone().unsqueeze(0)
        masked_attention_mask = attention_mask[b_idx].clone().unsqueeze(0)
        original_baseline_output = baseline_output[b_idx] if baseline_output.ndim > 0 else baseline_output

        for token_idx_to_mask in top_k_indices[b_idx]:
            masked_input_ids[0, token_idx_to_mask] = pad_token_id
            masked_attention_mask[0, token_idx_to_mask] = 0
            
        with torch.no_grad():
            output_after_removal, _ = model(input_ids=masked_input_ids, attention_mask=masked_attention_mask)
            
        comprehensiveness = original_baseline_output - output_after_removal.squeeze()
        all_comprehensiveness_scores.append(comprehensiveness.item())

    return np.mean(all_comprehensiveness_scores)


def jaccard_stability_bert(attributions1, attributions2, k=5):
    """Calculates Jaccard Stability (can be generic if attributions are token-level)."""
    if attributions1.shape != attributions2.shape:
        # Try to align lengths if they differ slightly due to tokenization of perturbed text
        len1, len2 = attributions1.shape[1], attributions2.shape[1]
        if len1 > len2:
            padding = torch.zeros(attributions1.shape[0], len1 - len2, device=attributions2.device)
            attributions2 = torch.cat([attributions2, padding], dim=1)
        elif len2 > len1:
            padding = torch.zeros(attributions1.shape[0], len2 - len1, device=attributions1.device)
            attributions1 = torch.cat([attributions1, padding], dim=1)
        # If still not matching after simple padding, raise error or return NaN
        if attributions1.shape[1] != attributions2.shape[1]:
             print(f"Warning: Jaccard stability attribution shapes {attributions1.shape} and {attributions2.shape} still mismatch after padding. Returning NaN.")
             return np.nan

    num_features = attributions1.size(1)
    actual_k = min(k, num_features)
    if actual_k == 0: return 1.0 # Or np.nan if preferred for empty sets

    top_k_indices1 = torch.argsort(attributions1, descending=True, dim=1)[:, :actual_k]
    top_k_indices2 = torch.argsort(attributions2, descending=True, dim=1)[:, :actual_k]
    
    jaccard_scores = []
    for i in range(attributions1.size(0)):
        set1 = set(top_k_indices1[i].tolist())
        set2 = set(top_k_indices2[i].tolist())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            jaccard_scores.append(1.0) 
        else:
            jaccard_scores.append(intersection / union)
            
    return np.mean(jaccard_scores)


def rank_correlation_bert(attributions1, attributions2):
    """Calculates Spearman's Rank Correlation (can be generic)."""
    if attributions1.shape != attributions2.shape:
        len1, len2 = attributions1.shape[1], attributions2.shape[1]
        if len1 > len2:
            padding = torch.zeros(attributions1.shape[0], len1 - len2, device=attributions2.device)
            attributions2 = torch.cat([attributions2, padding], dim=1)
        elif len2 > len1:
            padding = torch.zeros(attributions1.shape[0], len2 - len1, device=attributions1.device)
            attributions1 = torch.cat([attributions1, padding], dim=1)
        if attributions1.shape[1] != attributions2.shape[1]:
            print(f"Warning: Rank correlation attribution shapes {attributions1.shape} and {attributions2.shape} still mismatch after padding. Returning NaN.")
            return np.nan
    
    correlations = []
    for i in range(attributions1.size(0)):
        attr1_flat = attributions1[i].flatten().cpu().numpy()
        attr2_flat = attributions2[i].flatten().cpu().numpy()
        
        valid_mask = np.isfinite(attr1_flat) & np.isfinite(attr2_flat)
        attr1_flat_valid = attr1_flat[valid_mask]
        attr2_flat_valid = attr2_flat[valid_mask]

        if len(attr1_flat_valid) < 2 or len(attr2_flat_valid) < 2:
            correlations.append(np.nan)
            continue
            
        correlation, _ = spearmanr(attr1_flat_valid, attr2_flat_valid)
        correlations.append(correlation)
        
    return np.nanmean(correlations) 