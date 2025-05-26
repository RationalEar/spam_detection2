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


def analyze_explanations(model, text, tokenizer, device=None):
    """Generate and analyze model explanations"""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions and attention maps
    probs, attention_data = model(**inputs, return_attentions=True)

    # Get integrated gradients
    attributions, delta = model.compute_integrated_gradients(**inputs)

    # Compute explanation metrics
    explanation_data = {
        'attributions': attributions,
        **attention_data
    }

    return explanation_data, probs


def visualize_attention(text, attention_weights, tokenizer):
    """Visualize attention weights for a given text"""
    tokens = tokenizer.tokenize(text)

    plt.figure(figsize=(15, 5))
    sns.heatmap(attention_weights[:, :len(tokens)],
                xticklabels=tokens,
                yticklabels=False,
                cmap='viridis')
    plt.title('Attention Weights Visualization')
    plt.xlabel('Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


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
