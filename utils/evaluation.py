import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from torch.utils.data import TensorDataset, DataLoader

from utils.explainability import explain_bilstm_with_lime, explain_bilstm_with_shap
from utils.functions import encode


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
        colors = ['red' if v < 0 else 'blue' for v in top_values]
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


def analyze_adversarial_robustness(model, test_df, word2idx, max_len, device, num_samples=5, epsilon_range=[0.01, 0.05, 0.1]):
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
    """
    # Select random samples
    sample_indices = np.random.choice(len(test_df), num_samples, replace=False)
    sample_texts = test_df['text'].iloc[sample_indices].tolist()
    sample_labels = test_df['label'].iloc[sample_indices].values
    
    print("\nAnalyzing adversarial robustness on", num_samples, "samples")
    
    # Prepare inputs
    inputs = torch.tensor([encode(t, word2idx, max_len) for t in sample_texts]).to(device)
    labels = torch.tensor(sample_labels, dtype=torch.float32).to(device)
    
    # Get original predictions
    model.eval()
    with torch.no_grad():
        original_outputs, _ = model(inputs)
        original_preds = (original_outputs > 0.5).float()
        print("\nOriginal predictions:", original_preds.cpu().numpy())
        print("True labels:", labels.cpu().numpy())
        print("Original confidence scores:", original_outputs.cpu().numpy())
    
    # Store original predictions for each sample
    original_pred_list = original_preds.cpu().numpy()
    
    # Test different epsilon values
    results = []
    for epsilon in epsilon_range:
        print(f"\nTesting epsilon = {epsilon}")
        # Generate adversarial examples
        adv_inputs = model.generate_adversarial_example(inputs, labels, epsilon=epsilon)
        
        print("Number of tokens changed:", (adv_inputs != inputs).sum().item())
        
        # Get predictions on adversarial examples
        with torch.no_grad():
            adv_outputs, _ = model(adv_inputs)
            adv_preds = (adv_outputs > 0.5).float()
            print("Adversarial predictions:", adv_preds.cpu().numpy())
            print("Adversarial confidence scores:", adv_outputs.cpu().numpy())
        
        # Calculate success rate (percentage of predictions that changed)
        success_rate = (adv_preds != original_preds).float().mean().item()
        print(f"Attack success rate: {success_rate:.4f}")
        
        # Detailed changes
        changes = (adv_preds != original_preds).cpu().numpy()
        if changes.any():
            print("\nSuccessful attacks:")
            for idx in np.where(changes)[0]:
                print(f"Sample {idx}:")
                print(f"Original prediction: {original_outputs[idx].item():.4f}")
                print(f"Adversarial prediction: {adv_outputs[idx].item():.4f}")
        
        results.append({
            'epsilon': epsilon,
            'success_rate': success_rate,
            'original_preds': original_pred_list,
            'adversarial_preds': adv_preds.cpu().numpy()
        })
    
    # Plot results
    epsilons = [r['epsilon'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, success_rates, marker='o')
    plt.xlabel('Epsilon (Perturbation Size)')
    plt.ylabel('Attack Success Rate')
    plt.title('Adversarial Attack Success Rate vs. Perturbation Size')
    plt.grid(True)
    plt.show()
    
    # Print detailed results
    print("\nDetailed Results:")
    for idx, (text, orig_pred, label) in enumerate(zip(sample_texts, original_pred_list, labels.cpu().numpy())):
        print(f"\nSample {idx + 1}:")
        print("Text:", text[:200] + "..." if len(text) > 200 else text)
        print(f"True label: {'Spam' if label == 1 else 'Ham'}")
        print(f"Original prediction: {'Spam' if orig_pred == 1 else 'Ham'}")
        print("\nAdversarial Results:")
        for r in results:
            adv_pred = r['adversarial_preds'][idx]
            if adv_pred != orig_pred:
                print(f"- At ε={r['epsilon']}: Prediction flipped to {'Spam' if adv_pred == 1 else 'Ham'}")
    
    return results
