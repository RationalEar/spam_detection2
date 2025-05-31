import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.notebook import tqdm


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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions and attention maps with FP16
        with torch.cuda.amp.autocast():
            probs, attention_data = model(**inputs, return_attentions=True)
        
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
    flat_indices = np.unravel_index(
        np.argsort(attention_matrix.ravel())[-5:],
        attention_matrix.shape
    )
    
    print("\nTop token interactions:")
    for src_idx, tgt_idx in zip(flat_indices[0], flat_indices[1]):
        src_token = filtered_tokens[src_idx]
        tgt_token = filtered_tokens[tgt_idx]
        score = attention_matrix[src_idx, tgt_idx]
        print(f"'{src_token}' → '{tgt_token}': {score:.4f}")
    
    # Print overall token importance
    print("\nTop 10 most attended tokens:")
    for idx in top_indices[:10]:
        print(f"'{tokens[idx]}': {token_importance[idx]:.4f}")


def evaluate_adversarial_robustness(model, test_df, tokenizer, n_samples=100):
    """Evaluate model's robustness against adversarial attacks"""
    # Sample emails for testing
    test_samples = test_df.sample(n_samples, random_state=42)

    # Get original predictions
    original_preds, _ = evaluate_model(model, test_samples, tokenizer)

    # Create perturbed versions (simple character-level perturbations)
    perturbed_texts = []
    for text in test_samples['text']:
        # Simple character substitution (e.g., 'a' -> '@')
        perturbed = text.replace('a', '@').replace('i', '1').replace('o', '0')
        perturbed_texts.append(perturbed)

    # Get predictions on perturbed texts
    perturbed_df = test_samples.copy()
    perturbed_df['text'] = perturbed_texts
    perturbed_preds, _ = evaluate_model(model, perturbed_df, tokenizer)

    # Calculate robustness metrics
    prediction_stability = np.mean(np.abs(original_preds - perturbed_preds))
    decision_stability = np.mean((original_preds > 0.5) == (perturbed_preds > 0.5))

    return {
        'prediction_stability': 1 - prediction_stability,  # Higher is better
        'decision_stability': decision_stability  # Higher is better
    }
