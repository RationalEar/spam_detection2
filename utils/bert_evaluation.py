import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
from utils.bert_explanation_metrics import (
    get_token_level_attributions_from_ig,
    get_token_level_attributions_from_attention,
    area_under_deletion_curve_bert,
    area_under_insertion_curve_bert,
    comprehensiveness_score_bert,
    jaccard_stability_bert,
    rank_correlation_bert
)


def compute_metrics(predictions, labels):
    """Compute all classification metrics"""
    preds = (predictions > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auc_roc': roc_auc_score(labels, predictions),
        'spam_catch_rate': recall_score(labels, preds),  # Same as recall for spam class
        'ham_preservation_rate': 1 - (preds[labels == 0].sum() / (labels == 0).sum())  # 1 - FPR
    }

    return metrics


def evaluate_model(model, test_df, tokenizer, batch_size=32, device=None):
    """Evaluate model on test data"""
    all_predictions = []
    all_labels = test_df['label'].values

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), batch_size)):
            batch_texts = test_df['text'].iloc[i:i + batch_size].tolist()
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            probs, _ = model(**inputs)
            all_predictions.extend(probs.cpu().numpy())

    return np.array(all_predictions), all_labels


def analyze_explanations(model, text, tokenizer, device=None, chunk_size=128):
    """Generate and analyze model explanations with memory optimization
    Args:
        model: Trained SpamBERT model
        text: Input text to analyze
        tokenizer: HuggingFace tokenizer
        device: Compute device
        chunk_size: Size of chunks for memory-efficient processing
    Returns:
        tuple: (explanation_data, prediction_probability)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids_orig = inputs['input_ids'].to(device)
        attention_mask_orig = inputs['attention_mask'].to(device)
        
        # Get predictions and attention maps with FP16
        with torch.cuda.amp.autocast():
            probs, attention_data = model(input_ids=input_ids_orig, attention_mask=attention_mask_orig, return_attentions=True)
        
        # Clear cache before computing integrated gradients
        torch.cuda.empty_cache()
        
        # Get integrated gradients with chunking
        attributions, delta = model.compute_integrated_gradients(
            **inputs, 
            chunk_size=chunk_size
        )
        
        # Compute explanation metrics
        metrics = model.get_explanation_metrics(attributions, delta)
        
        # Combine all explanation data
        explanation_data = {
            'attributions': attributions,
            'metrics': metrics,
            **attention_data
        }
        
        # --- Start: New Explanation Quality Metrics Calculation ---
        print("\nCalculating advanced explanation quality metrics...")
        current_metrics = explanation_data.get('metrics', {})

        # 1. Faithfulness Metrics (AUDC, AUIC, Comprehensiveness) using Integrated Gradients
        token_ig_attributions = get_token_level_attributions_from_ig(attributions) # (batch, seq_len)
        baseline_output_for_faithfulness = probs.detach() # (batch_size,) or scalar if batch_size=1
        
        # Ensure baseline_output is correctly shaped (batch_size,)
        if baseline_output_for_faithfulness.ndim == 0: # scalar
            baseline_output_for_faithfulness = baseline_output_for_faithfulness.unsqueeze(0)
        if token_ig_attributions.shape[0] != baseline_output_for_faithfulness.shape[0]:
             # This can happen if IG attributions were for a single sample but probs is not squeezed.
             # Or if batching is handled differently. For now, assume batch_size is consistent or 1.
             # If IG is (1, seq, dim) and probs is scalar, unsqueeze probs.
             # This section assumes attributions and probs are for the same batch size (typically 1 for analyze_explanations)
             pass # Add more robust batch handling if needed here

        #   a. AUIC Baseline (output on all PADs)
        pad_token_id = tokenizer.pad_token_id
        # Create all-PAD input with same seq length as original input_ids_orig
        seq_len = input_ids_orig.size(1)
        all_pad_input_ids = torch.full_like(input_ids_orig, pad_token_id)
        all_pad_attention_mask = torch.zeros_like(attention_mask_orig)
        # For AUIC, typically only the first token (often [CLS]) is unmasked for the all-PAD input, 
        # or the model should handle all_pad_attention_mask correctly.
        # Let's assume model can handle it, or for simplicity, that a zero-mask means effectively no input.
        # A more sophisticated AUIC might build up from a truly empty/zeroed embedding.
        with torch.no_grad(), torch.cuda.amp.autocast():
            auic_ref_output, _ = model(input_ids=all_pad_input_ids, attention_mask=all_pad_attention_mask)
        auic_baseline_val = auic_ref_output.item() # scalar

        current_metrics['audc_ig'] = area_under_deletion_curve_bert(
            model, tokenizer, input_ids_orig, attention_mask_orig, 
            token_ig_attributions, baseline_output_for_faithfulness
        )
        current_metrics['auic_ig'] = area_under_insertion_curve_bert(
            model, tokenizer, input_ids_orig, attention_mask_orig, 
            token_ig_attributions, auic_baseline_val
        )
        k_comp = min(5, token_ig_attributions.size(1) // 2 if token_ig_attributions.size(1) > 1 else 1)
        current_metrics['comprehensiveness_ig'] = comprehensiveness_score_bert(
            model, tokenizer, input_ids_orig, attention_mask_orig, 
            token_ig_attributions, baseline_output_for_faithfulness, k=k_comp
        )
        print(f"  AUDC (IG): {current_metrics['audc_ig']:.4f}")
        print(f"  AUIC (IG): {current_metrics['auic_ig']:.4f}")
        print(f"  Comprehensiveness (IG, k={k_comp}): {current_metrics['comprehensiveness_ig']:.4f}")

        # 2. Stability Metrics (Jaccard, Rank Correlation) using Attention Weights
        #    Use attention from the last target layer (e.g., layer_12 if available)
        #    The SpamBERT model stores target_layers = [6, 9, 12]
        last_target_layer_idx = model.target_layers[-1] if model.target_layers else 12 # Default to 12
        attention_key = f'layer_{last_target_layer_idx}'
        if attention_key in attention_data:
            original_attention_map = attention_data[attention_key] # (batch, seq, seq)
            # Convert to token-level attributions (batch, seq)
            token_attention_orig = get_token_level_attributions_from_attention(original_attention_map)

            #    a. Create perturbed input
            perturbed_text = text.replace('a', '@').replace('e', '3').replace('i', '1').replace('o', '0') # Simple perturbation
            inputs_perturbed = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=512)
            input_ids_pert = inputs_perturbed['input_ids'].to(device)
            attention_mask_pert = inputs_perturbed['attention_mask'].to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                _, attention_data_perturbed = model(input_ids=input_ids_pert, attention_mask=attention_mask_pert, return_attentions=True)
            
            if attention_data_perturbed and attention_key in attention_data_perturbed:
                perturbed_attention_map = attention_data_perturbed[attention_key]
                token_attention_pert = get_token_level_attributions_from_attention(perturbed_attention_map)

                k_jaccard = min(5, token_attention_orig.size(1) // 2 if token_attention_orig.size(1) > 1 else 1)
                current_metrics['jaccard_stability_attn'] = jaccard_stability_bert(
                    token_attention_orig, token_attention_pert, k=k_jaccard
                )
                current_metrics['rank_correlation_attn'] = rank_correlation_bert(
                    token_attention_orig, token_attention_pert
                )
                print(f"  Jaccard Stability (Attention L{last_target_layer_idx}, k={k_jaccard}): {current_metrics['jaccard_stability_attn']:.4f}")
                print(f"  Rank Correlation (Attention L{last_target_layer_idx}): {current_metrics['rank_correlation_attn']:.4f}")
            else:
                print(f"Warning: Could not get attention data for perturbed input (layer {last_target_layer_idx}). Stability metrics skipped.")
        else:
            print(f"Warning: Attention data for layer {last_target_layer_idx} not found. Stability metrics skipped.")
        
        explanation_data['metrics'] = current_metrics
        # --- End: New Explanation Quality Metrics Calculation ---
        
        return explanation_data, probs.item()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: Out of memory error occurred. Try reducing chunk_size or input length.")
            # Try to recover CUDA memory
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        raise e


def visualize_attention(text, attention_weights, tokenizer, max_tokens=30):
    """
    Visualize attention weights for a given text
    Args:
        text: Input text
        attention_weights: Attention weights tensor
        tokenizer: BERT tokenizer
        max_tokens: Maximum number of tokens to display in visualization
    """
    # Tokenize text with special tokens
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    
    # Get tokens including special tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    # Find most attended tokens
    token_importance = attention_weights.mean(axis=0)  # Average attention received
    top_indices = token_importance.argsort()[-max_tokens:][::-1]
    
    # Filter tokens and attention weights to show only most important ones
    filtered_tokens = [tokens[i] for i in top_indices]
    filtered_weights = attention_weights[top_indices][:, top_indices]
    
    # Create figure with appropriate size and larger font
    plt.figure(figsize=(15, 12))
    
    # Plot heatmap with adjusted font sizes
    sns.heatmap(
        filtered_weights,
        xticklabels=filtered_tokens,
        yticklabels=filtered_tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        square=True  # Make cells square
    )
    
    # Customize plot
    plt.title('Token Attention Weights (Top Most Attended Tokens)', pad=20, fontsize=14)
    plt.xlabel('Target Tokens', labelpad=10, fontsize=12)
    plt.ylabel('Source Tokens', labelpad=10, fontsize=12)
    
    # Rotate labels and adjust font size
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    
    # Print top token interactions
    attention_matrix = filtered_weights.copy()
    np.fill_diagonal(attention_matrix, 0)  # Exclude self-attention
    # Ensure there are enough elements to sort for top 5 if matrix is small
    num_elements_to_sort = min(5, attention_matrix.size // 2 if attention_matrix.size > 1 else 0)
    if num_elements_to_sort > 0:
        flat_indices = np.unravel_index(
            np.argsort(attention_matrix.ravel())[-num_elements_to_sort:], # Get up to 5
            attention_matrix.shape
        )
        
        print("\nTop token interactions (up to 5):")
        for src_idx, tgt_idx in zip(flat_indices[0], flat_indices[1]):
            src_token = filtered_tokens[src_idx]
            tgt_token = filtered_tokens[tgt_idx]
            score = attention_matrix[src_idx, tgt_idx]
            print(f"'{src_token}' → '{tgt_token}': {score:.4f}")
    else:
        print("\nNot enough token interactions to display top interactions.")
    
    # Print overall token importance
    print("\nTop 10 most attended tokens:")
    for idx in top_indices[:10]:
        print(f"'{tokens[idx]}': {token_importance[idx]:.4f}")


def evaluate_adversarial_robustness(model, test_df, tokenizer, device=None, n_samples=100, k_for_retention=5):
    """Evaluate model's robustness against adversarial attacks, including explanation stability.
    Args:
        model: Trained SpamBERT model.
        test_df: DataFrame with 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer.
        device: Compute device.
        n_samples: Number of samples to evaluate from test_df.
        k_for_retention: K for Top-K retention metric.
    Returns:
        dict: Dictionary of robustness metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Sample emails for testing
    if n_samples > len(test_df):
        n_samples = len(test_df)
        print(f"Warning: n_samples > len(test_df). Using all {n_samples} samples.")
    test_samples_df = test_df.sample(n_samples, random_state=42)
    sample_texts = test_samples_df['text'].tolist()
    
    original_predictions_probs = []
    original_attentions_list = [] # To store token-level attention for original inputs
    perturbed_predictions_probs = []
    perturbed_attentions_list = [] # To store token-level attention for perturbed inputs
    
    print(f"Evaluating adversarial robustness on {n_samples} samples...")

    last_target_layer_idx = model.target_layers[-1] if model.target_layers else 12
    attention_key = f'layer_{last_target_layer_idx}'

    for text in tqdm(sample_texts, desc="Processing samples for robustness"):
        # Original input processing
        inputs_orig = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids_orig = inputs_orig['input_ids'].to(device)
        attention_mask_orig = inputs_orig['attention_mask'].to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            prob_orig, att_data_orig = model(input_ids=input_ids_orig, attention_mask=attention_mask_orig, return_attentions=True)
        original_predictions_probs.append(prob_orig.item())
        if att_data_orig and attention_key in att_data_orig:
            token_att_orig = get_token_level_attributions_from_attention(att_data_orig[attention_key])
            original_attentions_list.append(token_att_orig.squeeze().cpu().numpy()) # Squeeze to (seq_len,)
        else:
            original_attentions_list.append(np.array([])) # Placeholder if no attention
            print(f"Warning: No attention data for original: {text[:30]}...")

        # Perturbed input processing
        # Stronger character-level perturbation
        perturbed_text = list(text)
        char_map = {
            'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 'l': '!', 't': '7',
            'g': '9', 'b': '8', 'c': '(', 'f': '#', 'h': '^', 'm': ';', 'n': '>',
            'p': '?', 'r': ']', 'u': '*', 'w': '{', 'y': '`',
            'A': '@', 'E': '3', 'I': '1', 'O': '0', 'S': '$', 'L': '!', 'T': '7',
            'G': '9', 'B': '8', 'C': '(', 'F': '#', 'H': '^', 'M': ';', 'N': '>',
            'P': '?', 'R': ']', 'U': '*', 'W': '{', 'Y': '`'
        }
        # Increase perturbation density: attempt to change up to 20% of characters
        num_chars_to_change = int(len(perturbed_text) * 0.20)
        if num_chars_to_change == 0 and len(perturbed_text) > 0: # Ensure at least one change if possible
            num_chars_to_change = 1

        indices_to_change = np.random.choice(len(perturbed_text), size=num_chars_to_change, replace=False)
        
        for idx in indices_to_change:
            char_to_replace = perturbed_text[idx]
            if char_to_replace in char_map:
                perturbed_text[idx] = char_map[char_to_replace]
            elif char_to_replace.lower() in char_map: # Handle cases missed by direct map (e.g. if map is only lower)
                 perturbed_text[idx] = char_map[char_to_replace.lower()]

        perturbed_text = "".join(perturbed_text)
        
        # Print original and perturbed text for a few samples to verify
        if len(original_predictions_probs) < 3: # Print for first 3 samples
            print(f"\nOriginal text snippet: {text[:100]}...")
            print(f"Perturbed text snippet: {perturbed_text[:100]}...")

        inputs_pert = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids_pert = inputs_pert['input_ids'].to(device)
        attention_mask_pert = inputs_pert['attention_mask'].to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            prob_pert, att_data_pert = model(input_ids=input_ids_pert, attention_mask=attention_mask_pert, return_attentions=True)
        perturbed_predictions_probs.append(prob_pert.item())
        if att_data_pert and attention_key in att_data_pert:
            token_att_pert = get_token_level_attributions_from_attention(att_data_pert[attention_key])
            perturbed_attentions_list.append(token_att_pert.squeeze().cpu().numpy()) # Squeeze to (seq_len,)
        else:
            perturbed_attentions_list.append(np.array([]))
            print(f"Warning: No attention data for perturbed: {perturbed_text[:30]}...")

    original_preds_binary = (np.array(original_predictions_probs) > 0.5).astype(int)
    perturbed_preds_binary = (np.array(perturbed_predictions_probs) > 0.5).astype(int)

    # 1. Attack Success Rate (ASR)
    attack_success_rate = np.mean(original_preds_binary != perturbed_preds_binary)

    # 2. Explanation Shift (Cosine Similarity)
    explanation_shifts = []
    for orig_attn, pert_attn in zip(original_attentions_list, perturbed_attentions_list):
        if orig_attn.size == 0 or pert_attn.size == 0 or orig_attn.shape != pert_attn.shape:
            explanation_shifts.append(np.nan) # Cannot compare if shapes differ or empty
            continue
        # Ensure 2D for cosine_similarity
        similarity = cosine_similarity(orig_attn.reshape(1, -1), pert_attn.reshape(1, -1))[0, 0]
        explanation_shifts.append(similarity)
    avg_explanation_shift_cosine_sim = np.nanmean(explanation_shifts)

    # 3. Top-k Retention
    top_k_retentions = []
    actual_k = k_for_retention
    for orig_attn, pert_attn in zip(original_attentions_list, perturbed_attentions_list):
        if orig_attn.size == 0 or pert_attn.size == 0 or orig_attn.shape != pert_attn.shape:
            top_k_retentions.append(np.nan)
            continue
        
        # Ensure k is not larger than sequence length
        current_seq_len = orig_attn.shape[0]
        if current_seq_len == 0: # Should not happen if size > 0 check passed, but defensive
             top_k_retentions.append(np.nan)
             continue
        
        current_k = min(actual_k, current_seq_len)
        if current_k == 0: # If sequence length is 0 or k is 0
            top_k_retentions.append(1.0 if current_seq_len == 0 else np.nan) # Perfect retention if no features, else NaN
            continue

        orig_top_k_indices = np.argsort(orig_attn)[-current_k:]
        pert_top_k_indices = np.argsort(pert_attn)[-current_k:]
        
        retention = len(set(orig_top_k_indices).intersection(set(pert_top_k_indices))) / current_k
        top_k_retentions.append(retention)
    avg_top_k_retention = np.nanmean(top_k_retentions)

    # Prediction stability and decision stability (from original implementation)
    prediction_diff_stability = 1 - np.mean(np.abs(np.array(original_predictions_probs) - np.array(perturbed_predictions_probs)))
    decision_stability = np.mean(original_preds_binary == perturbed_preds_binary)

    metrics = {
        'attack_success_rate': attack_success_rate,
        'avg_explanation_shift_cosine_sim': avg_explanation_shift_cosine_sim,
        'avg_top_k_retention': avg_top_k_retention,
        'prediction_output_stability': prediction_diff_stability, # Renamed for clarity from 'prediction_stability'
        'decision_flip_stability': decision_stability # Renamed for clarity from 'decision_stability'
    }
    print("\nAdversarial Robustness Metrics Calculated:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")
    return metrics
