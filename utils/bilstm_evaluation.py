import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader

from utils.explainability import explain_bilstm_with_lime, explain_bilstm_with_shap
from utils.functions import encode, get_pad_token_id
from utils.explanation_metrics import (
    area_under_deletion_curve, area_under_insertion_curve,
    comprehensiveness_score, jaccard_stability, rank_correlation
)


def evaluate_bilstm_performance(model, test_df, word2idx, max_len, device):
    """
    Comprehensive evaluation of BiLSTM model performance
    Args:
        model: Trained BiLSTM model
        test_df: Test dataset DataFrame
        word2idx: Word to index mapping
        max_len: Maximum sequence length
        device: Device to run model on
    """
    # Prepare test data
    X_test = torch.tensor([encode(t, word2idx, max_len) for t in test_df['text']])
    y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Collect predictions
    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 1. Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")

    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def analyze_model_explanations(model, test_df, word2idx, max_len, device, num_samples=5):
    """
    Analyze model predictions using LIME and SHAP
    Args:
        model: Trained BiLSTM model
        test_df: Test dataset DataFrame
        word2idx: Word to index mapping
        max_len: Maximum sequence length
        device: Device to run model on
        num_samples: Number of samples to analyze
    """
    # Select random samples
    sample_indices = np.random.choice(len(test_df), num_samples, replace=False)
    sample_texts = test_df['text'].iloc[sample_indices].tolist()
    
    # Get LIME explanations
    lime_explanations = explain_bilstm_with_lime(
        model, 
        lambda x, max_len=max_len: encode(x, word2idx, max_len),
        max_len,
        device,
        sample_texts
    )
    
    # Get SHAP explanations
    shap_values, tokenized_texts = explain_bilstm_with_shap(
        model,
        lambda x, max_len=max_len: encode(x, word2idx, max_len),
        sample_texts,
        max_len,
        device
    )
    
    # Visualize explanations
    for idx, (text, lime_exp) in enumerate(zip(sample_texts, lime_explanations)):
        print(f"\nSample {idx + 1}:")
        print("Text:", text[:200] + "..." if len(text) > 200 else text)
        
        # Get model prediction for this sample
        input_tensor = torch.tensor([encode(text, word2idx, max_len)]).to(device)
        with torch.no_grad():
            pred, _ = model(input_tensor)
            pred = pred.item()
        print(f"Model prediction: {'Spam' if pred > 0.5 else 'Ham'} (confidence: {pred:.3f})")
        
        plt.figure(figsize=(15, 6))
        
        # LIME visualization
        plt.subplot(1, 2, 1)
        # Get the explanation data from LIME
        exp_list = lime_exp.as_list()
        words, scores = zip(*exp_list)
        y_pos = np.arange(len(words))
        
        # Create horizontal bar chart for LIME
        colors = ['red' if s < 0 else 'blue' for s in scores]
        plt.barh(y_pos, scores, color=colors)
        plt.yticks(y_pos, words)
        plt.xlabel('Impact on prediction')
        plt.title('LIME Explanation\nBlue = Spam indicator, Red = Ham indicator')
        
        # SHAP visualization
        plt.subplot(1, 2, 2)
        
        # Get the words from the original text
        words = text.split()[:max_len]
        if len(words) < max_len:
            words += ['<PAD>'] * (max_len - len(words))
            
        # Get SHAP values for this sample
        if isinstance(shap_values, list):
            shap_values_sample = shap_values[0][idx]  # For binary classification, use class 1 (spam)
        else:
            shap_values_sample = shap_values[idx]
            
        # Convert SHAP values to numpy array if needed
        if isinstance(shap_values_sample, torch.Tensor):
            shap_values_sample = shap_values_sample.cpu().numpy()
        
        # Ensure we have the right shape
        if len(shap_values_sample.shape) > 1:
            shap_values_sample = shap_values_sample.flatten()
            
        # Create word-value pairs and sort by absolute magnitude
        word_values = list(zip(words, shap_values_sample))
        word_values.sort(key=lambda x: abs(float(x[1])), reverse=True)
        
        # Take top 20 words
        top_k = min(20, len(word_values))
        top_words = [pair[0] for pair in word_values[:top_k]]
        top_values = np.array([float(pair[1]) for pair in word_values[:top_k]])
        
        # Create horizontal bar chart for SHAP
        y_pos = np.arange(top_k)
        colors = ['blue' if v < 0 else 'red' for v in top_values]
        plt.barh(y_pos, top_values, color=colors)
        plt.yticks(y_pos, top_words)
        plt.xlabel('SHAP value')
        plt.title('Top 20 Most Important Words (SHAP)\nBlue = Spam indicator, Red = Ham indicator')
        
        plt.tight_layout()
        plt.show()
        
        # Print top 5 most influential words from both methods
        print("\nTop 5 most influential words (LIME):")
        for word, score in exp_list[:5]:
            impact = "negative" if score < 0 else "positive"
            print(f"- {word}: {impact} impact (LIME score: {score:.3f})")
            
        print("\nTop 5 most influential words (SHAP):")
        top_5_shap = word_values[:5]
        for word, value in top_5_shap:
            value = float(value)  # Convert to float for printing
            impact = "negative" if value < 0 else "positive"
            print(f"- {word}: {impact} impact (SHAP value: {value:.3f})")

    # --- Start: New code for Explanation Quality Metrics ---
    print("\n\nCalculating Explanation Quality Metrics...")
    metrics_results = {}

    # Prepare inputs for metrics
    encoded_samples = torch.tensor([encode(text, word2idx, max_len) for text in sample_texts], dtype=torch.long).to(device)
    # True labels for adversarial generation (if needed for stability metrics later)
    # We need the actual labels for the selected samples.
    sample_actual_labels = torch.tensor(test_df['label'].iloc[sample_indices].values, dtype=torch.float32).to(device)


    # SHAP values are already computed as `shap_values` in this function.
    # Ensure `shap_values` is in the correct format (e.g., tensor, correct dimensions)
    if isinstance(shap_values, tuple): # If explain_bilstm_with_shap returns (values, tokens)
        shap_attributions_raw = shap_values[0]
    else:
        shap_attributions_raw = shap_values

    if isinstance(shap_attributions_raw, list): # Common for SHAP multi-output from some libraries/explainers
                                                # For binary case, often shap_values[0] is for class 0, shap_values[1] for class 1
                                                # Or it might be a list of arrays, one per sample
        # Assuming for binary classification, we are interested in attributions for the positive class (spam)
        # If shap_values is a list of arrays (one per sample), stack them.
        # If shap_values is [class0_attr, class1_attr], take class1_attr.
        # This part needs to be robust based on exact output of `explain_bilstm_with_shap`
        # For now, assuming shap_values[0] contains the relevant attributions for the samples if it's a list of arrays per class
        # Or if it's a list of arrays (one per sample), convert to tensor
        if len(shap_attributions_raw) == num_samples and isinstance(shap_attributions_raw[0], (np.ndarray, torch.Tensor)):
             all_shap_attributions = torch.stack([torch.tensor(s, device=device) for s in shap_attributions_raw]).float()
        elif len(shap_attributions_raw) > 0 and isinstance(shap_attributions_raw[0], (np.ndarray, torch.Tensor)) and shap_attributions_raw[0].ndim >= 2 : # e.g. list of [num_samples, seq_len]
            all_shap_attributions = torch.tensor(shap_attributions_raw[0], device=device).float() # Defaulting to first set, adjust if needed
        else: # Fallback or error
            print("Warning: SHAP values format not fully determined for list type. Using empty tensor.")
            all_shap_attributions = torch.empty(0, max_len, device=device)

    elif isinstance(shap_attributions_raw, (np.ndarray, torch.Tensor)):
        all_shap_attributions = torch.tensor(shap_attributions_raw, device=device).float()
    else:
        print("Warning: SHAP values are not in an expected list/array/tensor format. Using empty tensor.")
        all_shap_attributions = torch.empty(0, max_len, device=device)
    
    if all_shap_attributions.ndim > 2: # e.g. (num_samples, seq_len, 1) or (num_samples, seq_len, num_classes_for_shap)
        if all_shap_attributions.shape[-1] == 1: # Squeeze if last dim is 1
            all_shap_attributions = all_shap_attributions.squeeze(-1)
        # If SHAP returns for multiple classes, you might need to select one.
        # E.g., if binary: model outputs one value, SHAP might give for "class 0" and "class 1"
        # Typically, for a single output neuron (sigmoid), SHAP values directly correspond.
        # If model.fc2.out_features > 1 and shap returns for each, select appropriately.
        # For BiLSTMSpam, fc2 has num_classes=1, so SHAP should give one set of values per input.

    # Ensure correct shape: (num_samples, seq_len)
    use_attention_for_faithfulness = False
    if all_shap_attributions.nelement() == 0 or all_shap_attributions.shape[0] != len(sample_texts) or all_shap_attributions.shape[1] != max_len:
        print(f"Warning: SHAP attributions shape mismatch or empty ({all_shap_attributions.shape}). Expected ({len(sample_texts)}, {max_len}).")
        print("Using Attention Weights for AUDC, AUIC, Comprehensiveness as fallback.")
        use_attention_for_faithfulness = True
    
    # Get baseline model outputs and original attention weights
    with torch.no_grad():
        baseline_model_outputs, original_attention_weights = model(encoded_samples)
    baseline_model_outputs = baseline_model_outputs.squeeze() 
    original_attention_weights = original_attention_weights.detach() 

    if use_attention_for_faithfulness:
        primary_attributions = original_attention_weights
    else:
        primary_attributions = all_shap_attributions
        print("Using SHAP Attributions for AUDC, AUIC, Comprehensiveness.")


    pad_token_id = get_pad_token_id(word2idx)
    empty_input_for_auic = torch.full((encoded_samples.size(0), max_len), pad_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        auic_ref_output, _ = model(empty_input_for_auic)
    auic_baseline_val = auic_ref_output.squeeze()

    metrics_results['audc'] = area_under_deletion_curve(model, encoded_samples, primary_attributions, baseline_model_outputs)
    print(f"  Average AUDC (Faithfulness): {metrics_results['audc']:.4f} (using {'Attention' if use_attention_for_faithfulness else 'SHAP'})")

    metrics_results['auic'] = area_under_insertion_curve(model, encoded_samples, primary_attributions, baseline_value=auic_baseline_val)
    print(f"  Average AUIC (Faithfulness): {metrics_results['auic']:.4f} (using {'Attention' if use_attention_for_faithfulness else 'SHAP'})")

    k_comprehensiveness = min(5, max_len // 2 if max_len > 1 else 1)
    metrics_results['comprehensiveness'] = comprehensiveness_score(model, encoded_samples, primary_attributions, baseline_model_outputs, k=k_comprehensiveness)
    print(f"  Average Comprehensiveness (k={k_comprehensiveness}): {metrics_results['comprehensiveness']:.4f} (using {'Attention' if use_attention_for_faithfulness else 'SHAP'})")

    print("\nUsing Model's Attention Weights for Stability Metrics (Jaccard & Rank Correlation).")
    # Generate perturbed inputs (e.g., using a very small adversarial perturbation)
    # Ensure labels are correct for generating adversarial examples for stability
    perturbed_inputs = model.generate_adversarial_example(encoded_samples.clone(), sample_actual_labels, epsilon=0.005, num_steps=5)
    with torch.no_grad():
        _, perturbed_attention_weights = model(perturbed_inputs)
    perturbed_attention_weights = perturbed_attention_weights.detach()

    k_jaccard = min(5, max_len // 2 if max_len > 1 else 1)
    metrics_results['jaccard_stability_attn'] = jaccard_stability(original_attention_weights, perturbed_attention_weights, k=k_jaccard)
    print(f"  Jaccard Stability (Attention, k={k_jaccard}): {metrics_results['jaccard_stability_attn']:.4f}")

    metrics_results['rank_correlation_attn'] = rank_correlation(original_attention_weights, perturbed_attention_weights)
    print(f"  Rank Correlation (Attention): {metrics_results['rank_correlation_attn']:.4f}")
    
    print("--- End: Explanation Quality Metrics ---")
    return metrics_results # Return the calculated metrics


def analyze_adversarial_robustness(model, test_df, word2idx, max_len, device, num_samples=5, epsilon_range=[0.01, 0.05, 0.1], k_for_retention=5):
    """
    Analyze model robustness against adversarial examples
    Args:
        model: Trained BiLSTM model
        test_df: Test dataset DataFrame
        word2idx: Word to index mapping
        max_len: Maximum sequence length
        device: Device to run model on
        num_samples: Number of samples to analyze
        epsilon_range: List of perturbation sizes to test
        k_for_retention: Number of top tokens to retain for Top-k Retention
    """
    # Select random samples
    sample_indices = np.random.choice(len(test_df), num_samples, replace=False)
    sample_texts = test_df['text'].iloc[sample_indices].tolist()
    sample_labels = test_df['label'].iloc[sample_indices].values
    
    print("\nAnalyzing adversarial robustness on", num_samples, "samples")
    
    # Prepare inputs
    inputs = torch.tensor([encode(t, word2idx, max_len) for t in sample_texts]).to(device)
    labels = torch.tensor(sample_labels, dtype=torch.float32).to(device)
    
    # Get original predictions and attention weights
    model.eval()
    with torch.no_grad():
        original_outputs, original_attention_weights = model(inputs)
        original_preds = (original_outputs > 0.5).float()
        print("\nOriginal predictions:", original_preds.cpu().numpy())
        print("True labels:", labels.cpu().numpy())
        print("Original confidence scores:", original_outputs.cpu().numpy())
    
    # Store original predictions for each sample
    original_pred_list = original_preds.cpu().numpy()
    
    # Test different epsilon values
    results_summary = []
    for epsilon in epsilon_range:
        print(f"\nTesting epsilon = {epsilon}")
        # Generate adversarial examples
        adv_inputs = model.generate_adversarial_example(inputs.clone(), labels.clone(), epsilon=epsilon)
        
        print("Number of tokens changed:", (adv_inputs != inputs).sum().item())
        
        # Get predictions and attention weights on adversarial examples
        with torch.no_grad():
            adv_outputs, adv_attention_weights = model(adv_inputs)
            adv_preds = (adv_outputs > 0.5).float()
            print("Adversarial predictions:", adv_preds.cpu().numpy())
            print("Adversarial confidence scores:", adv_outputs.cpu().numpy())
        
        # Calculate attack success rate
        attack_success_rate = (adv_preds != original_preds).float().mean().item()
        print(f"Attack success rate: {attack_success_rate:.4f}")
        
        # 2. Explanation Shift (Cosine Similarity)
        explanation_shifts = []
        for i in range(num_samples):
            orig_attn = original_attention_weights[i].flatten().cpu().numpy().reshape(1, -1)
            adv_attn = adv_attention_weights[i].flatten().cpu().numpy().reshape(1, -1)
            # Ensure no NaN values from attention before similarity calculation
            if np.isnan(orig_attn).any() or np.isnan(adv_attn).any():
                similarity = np.nan # Or handle as 0 or skip
            else:
                similarity = cosine_similarity(orig_attn, adv_attn)[0, 0]
            explanation_shifts.append(similarity)
        avg_explanation_shift = np.nanmean(explanation_shifts)
        print(f"Average Explanation Shift (Cosine Similarity): {avg_explanation_shift:.4f}")

        # 3. Top-k Retention
        top_k_retentions = []
        # Ensure k_for_retention is not larger than sequence length
        actual_k = min(k_for_retention, original_attention_weights.size(1))
        if actual_k == 0: # Avoid issues if max_len is 0 or 1
             avg_top_k_retention = np.nan
        else:
            for i in range(num_samples):
                orig_top_k = torch.topk(original_attention_weights[i], k=actual_k).indices.cpu().tolist()
                adv_top_k = torch.topk(adv_attention_weights[i], k=actual_k).indices.cpu().tolist()
                
                retention = len(set(orig_top_k).intersection(set(adv_top_k))) / actual_k
                top_k_retentions.append(retention)
            avg_top_k_retention = np.mean(top_k_retentions) if top_k_retentions else np.nan
        print(f"Average Top-{actual_k} Retention: {avg_top_k_retention:.4f}")

        # Detailed changes
        changes = (adv_preds != original_preds).cpu().numpy()
        if changes.any():
            print("\nSuccessful attacks:")
            for idx in np.where(changes)[0]:
                print(f"Sample {idx}:")
                print(f"Original prediction: {original_outputs[idx].item():.4f}")
                print(f"Adversarial prediction: {adv_outputs[idx].item():.4f}")
        
        results_summary.append({
            'epsilon': epsilon,
            'attack_success_rate': attack_success_rate,
            'avg_explanation_shift_cosine_sim': avg_explanation_shift,
            'avg_top_k_retention': avg_top_k_retention,
            'original_preds': original_pred_list,
            'adversarial_preds': adv_preds.cpu().numpy()
        })
    
    # Plot results (ASR vs Epsilon is already there)
    epsilons = [r['epsilon'] for r in results_summary]
    success_rates = [r['attack_success_rate'] for r in results_summary]
    explanation_shift_scores = [r['avg_explanation_shift_cosine_sim'] for r in results_summary]
    top_k_retention_scores = [r['avg_top_k_retention'] for r in results_summary]
    
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, success_rates, marker='o')
    plt.xlabel('Epsilon (Perturbation Size)')
    plt.ylabel('Attack Success Rate')
    plt.title('Adversarial Attack Success Rate vs. Perturbation Size')
    plt.grid(True)
    plt.show()

    # Plot Explanation Shift vs Epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, explanation_shift_scores, marker='o', color='green')
    plt.xlabel('Epsilon (Perturbation Size)')
    plt.ylabel('Avg. Explanation Shift (Cosine Similarity)')
    plt.title('Explanation Shift vs. Perturbation Size')
    plt.grid(True)
    plt.show()

    # Plot Top-k Retention vs Epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, top_k_retention_scores, marker='o', color='purple')
    plt.xlabel('Epsilon (Perturbation Size)')
    plt.ylabel(f'Avg. Top-{actual_k} Retention')
    plt.title(f'Top-{actual_k} Retention vs. Perturbation Size')
    plt.grid(True)
    plt.show()
    
    # Print detailed results for each sample
    print("\nDetailed Results:")
    for idx, (text, orig_pred, label) in enumerate(zip(sample_texts, original_pred_list, labels.cpu().numpy())):
        print(f"\nSample {idx + 1}:")
        print("Text:", text[:200] + "..." if len(text) > 200 else text)
        print(f"True label: {'Spam' if label == 1 else 'Ham'}")
        print(f"Original prediction: {'Spam' if orig_pred == 1 else 'Ham'}")
        print("\nAdversarial Results:")
        for r in results_summary:
            adv_pred = r['adversarial_preds'][idx]
            if adv_pred != orig_pred:
                print(f"- At ε={r['epsilon']}: Prediction flipped to {'Spam' if adv_pred == 1 else 'Ham'}")
    
    return results_summary # analyze_adversarial_robustness now returns the summary with new metrics
