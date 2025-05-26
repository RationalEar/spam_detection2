import torch
import torch.nn.functional as F
import shap
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer


def explain_spam_cnn(model, tokenizer, texts, word2idx, max_len, device, num_samples=100, rng=None):
    """
    Generate SHAP values for a batch of texts using a CNN model.
    Uses KernelExplainer for integer input compatibility.
    Args:
        model: Trained SpamCNN model
        tokenizer: Function to convert text to input ids
        texts: List of raw email texts
        word2idx (dict): Your word to index mapping.
        max_len (int): The maximum sequence length used during training.
        device: torch.device
        num_samples: Number of background samples for SHAP
        rng: Optional NumPy random number generator
    Returns:
        shap_values, tokenized_texts
    """
    model.eval()
    # Use more background samples for better results
    background_size = min(num_samples, len(texts))
    background_texts = texts[:background_size]
    background_inputs = np.array([tokenizer(t, max_len) for t in background_texts])

    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(x)
            # For binary classification, return as a single column
            return out.cpu().numpy()

    # Create the explainer with the masker
    explainer = shap.KernelExplainer(predict_fn, background_inputs, random_state=rng)

    # Get test inputs
    test_inputs = np.array([tokenizer(t, max_len) for t in texts])
    test_tensor = torch.tensor(test_inputs, dtype=torch.long).to(device)

    # Generate SHAP values with more samples for stability
    shap_values = explainer.shap_values(test_inputs, nsamples=100)

    return shap_values, test_inputs


def explain_cnn_with_shap(model, tokenizer, texts, device, max_len=128, num_samples=100, rng=None):
    """
    Generate SHAP values for a batch of texts using a CNN model.
    Uses KernelExplainer for integer input compatibility.
    Args:
        model: Trained SpamCNN model
        tokenizer: Function to convert text to input ids
        texts: List of raw email texts
        device: torch.device
        max_len: Max sequence length
        num_samples: Number of background samples for SHAP
        rng: Optional NumPy random number generator
    Returns:
        shap_values, input_ids
    """
    model.eval()
    # Use more background samples for better results
    background_size = min(num_samples, len(texts))
    background_texts = texts[:background_size]
    background_inputs = np.array([tokenizer(t, max_len) for t in background_texts])

    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(x)
            # For binary classification, return as a single column
            return out.cpu().numpy()

    # Create the explainer with the masker
    explainer = shap.KernelExplainer(predict_fn, background_inputs, random_state=rng)

    # Get test inputs
    test_inputs = np.array([tokenizer(t, max_len) for t in texts])
    test_tensor = torch.tensor(test_inputs, dtype=torch.long).to(device)

    # Generate SHAP values with more samples for stability
    shap_values = explainer.shap_values(test_inputs, nsamples=100)

    return shap_values, test_tensor


def explain_bilstm_with_shap(model, tokenizer, texts, max_len, device, num_samples=100, rng=None):
    """
    Generate SHAP values for a batch of texts using a BiLSTM model.
    Uses KernelExplainer for integer input compatibility.
    Args:
        model: Trained BiLSTMSpam model
        tokenizer: Function to convert text to input ids
        texts: List of raw email texts
        max_len (int): The maximum sequence length used during training.
        device: torch.device
        num_samples: Number of background samples for SHAP
        rng: Optional NumPy random number generator
    Returns:
        shap_values, tokenized_texts
    """
    model.eval()
    background_size = min(num_samples, len(texts))
    background_texts = texts[:background_size]
    background_inputs = np.array([tokenizer(t, max_len) for t in background_texts])

    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            probs, _ = model(x)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            return probs.cpu().numpy().reshape(-1, 1)

    explainer = shap.KernelExplainer(predict_fn, background_inputs, random_state=rng)
    test_inputs = np.array([tokenizer(t, max_len) for t in texts])
    shap_values = explainer.shap_values(test_inputs, nsamples=100)
    return shap_values, test_inputs


def explain_spam_cnn_with_lime(model, tokenizer, max_len, device, texts, num_samples=500):
    """
    Generate LIME explanations for a batch of texts using a CNN model.
    Args:
        model: Trained SpamCNN model
        tokenizer: Function to convert text to input ids
        max_len: Max sequence length
        device: torch.device
        texts: List of raw texts
        num_samples: Number of samples for LIME
    Returns:
        lime_explanations: List of LIME explanation objects
    """
    class_names = ['ham', 'spam']
    model.eval()

    def predict_proba(texts):
        # Tokenize and pad each text
        batch = [tokenizer(t, max_len) for t in texts]
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        with torch.no_grad():
            probs = model(batch).cpu().numpy()
        # Return probability for both classes (ham, spam)
        probs = np.concatenate([1 - probs, probs], axis=1)
        return probs

    explainer = LimeTextExplainer(class_names=class_names)
    lime_explanations = []
    for text in texts:
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=num_samples
        )
        lime_explanations.append(exp)
    return lime_explanations


def explain_bilstm_with_lime(model, tokenizer, max_len, device, texts, num_samples=500):
    """
    Generate LIME explanations for a batch of texts using a BiLSTM model.
    Args:
        model: Trained BiLSTMSpam model
        tokenizer: Function to convert text to input ids
        max_len: Max sequence length
        device: torch.device
        texts: List of raw texts
        num_samples: Number of samples for LIME
    Returns:
        lime_explanations: List of LIME explanation objects
    """
    class_names = ['ham', 'spam']
    model.eval()

    def predict_proba(texts):
        # Tokenize and pad each text
        batch = [tokenizer(t, max_len) for t in texts]
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        with torch.no_grad():
            probs, _ = model(batch)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            probs = probs.cpu().numpy().reshape(-1, 1)
        # Return probability for both classes (ham, spam)
        probs = np.concatenate([1 - probs, probs], axis=1)
        return probs

    explainer = LimeTextExplainer(class_names=class_names)
    lime_explanations = []
    for text in texts:
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=num_samples
        )
        lime_explanations.append(exp)
    return lime_explanations


def explain_bert_with_lime(model, tokenizer, max_len, device, texts, num_samples=500):
    """
    Generate LIME explanations for a batch of texts using a BERT model.
    Args:
        model: Trained SpamBERT model
        tokenizer: HuggingFace tokenizer (with encode_plus)
        max_len: Max sequence length
        device: torch.device
        texts: List of raw texts
        num_samples: Number of samples for LIME
    Returns:
        lime_explanations: List of LIME explanation objects
    """
    class_names = ['ham', 'spam']
    model.eval()

    def predict_proba(texts):
        # Tokenize and pad each text using HuggingFace tokenizer
        encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            probs, _ = model(input_ids, attention_mask=attention_mask)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            probs = probs.cpu().numpy().reshape(-1, 1)
        # Return probability for both classes (ham, spam)
        probs = np.concatenate([1 - probs, probs], axis=1)
        return probs

    explainer = LimeTextExplainer(class_names=class_names)
    lime_explanations = []
    for text in texts:
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=num_samples
        )
        lime_explanations.append(exp)
    return lime_explanations


def explain_bert_with_shap(model, tokenizer, texts, max_len, device, num_samples=2, nsamples=10, rng=None):
    """
    Generate SHAP values for a batch of texts using a BERT model.
    Uses KernelExplainer for integer input compatibility.
    Always runs on CPU to avoid CUDA OOM errors.
    Args:
        model: Trained SpamBERT model
        tokenizer: HuggingFace tokenizer (with encode_plus)
        texts: List of raw email texts
        max_len (int): The maximum sequence length used during training.
        device: torch.device (ignored, always runs on CPU)
        num_samples: Number of background samples for SHAP (keep small, e.g. 2)
        nsamples: Number of SHAP samples (keep small, e.g. 10)
        rng: Optional NumPy random number generator
    Returns:
        shap_values, tokenized_inputs
    """
    model_cpu = model.cpu()
    model_cpu.eval()
    background_size = min(num_samples, len(texts))
    background_texts = texts[:background_size]
    background_encodings = tokenizer(
        background_texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    background_inputs = background_encodings['input_ids'].numpy()

    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.long)
        attention_mask = (x_tensor != tokenizer.pad_token_id).long()
        with torch.no_grad():
            probs, _ = model_cpu(x_tensor, attention_mask=attention_mask)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            return probs.cpu().numpy().reshape(-1, 1)

    explainer = shap.KernelExplainer(predict_fn, background_inputs, random_state=rng)
    test_encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    test_inputs = test_encodings['input_ids'].numpy()
    shap_values = explainer.shap_values(test_inputs, nsamples=nsamples)
    return shap_values, test_inputs


def visualize_shap_for_sample(shap_values, original_text, class_names=['ham', 'spam'], prediction_threshold=0.5):
    """
    Visualizes SHAP values for a single text sample.

    Args:
        shap_values (shap.Explanation or np.array): SHAP explanation object or numpy array for a single sample.
        original_text (str): The original text of the sample.
        class_names (list of str): Names of the classes.
        prediction_threshold (float): Threshold for classifying as the positive class.
    """
    # For binary classification, SHAP returns one set of values (for the positive class)
    if isinstance(shap_values, np.ndarray):
        predicted_class_index = 1 if shap_values.mean() > prediction_threshold else 0
        shap_values_data = original_text.split()[:len(shap_values)]
        shap_values_values = shap_values
    else:
        predicted_class_index = 1 if shap_values.values.mean() > prediction_threshold else 0
        shap_values_data = shap_values.data
        shap_values_values = shap_values.values

    predicted_class_name = class_names[predicted_class_index]

    print(f"Original Text: '{original_text}'")
    print(f"Predicted Class: {predicted_class_name}")

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.RdBu  # Red for positive, Blue for negative influence

    # Normalize SHAP values for better color mapping
    max_abs_val = np.max(np.abs(shap_values_values))
    norm = plt.Normalize(-max_abs_val, max_abs_val)

    y_positions = np.arange(len(shap_values_data))
    bar_height = 1.0

    for i, word in enumerate(shap_values_data):
        # skip word if shap value is 0
        if shap_values_values[i] == 0:
            continue
        color = cmap(norm(shap_values_values[i]))

        if len(word) > 20:
            word = word[:20] + '...'

        ax.barh(y_positions[i], shap_values_values[i], height=bar_height, color=color, align='center')
        ax.text(0, y_positions[i], word, va='center', ha='left', fontsize=10)

    ax.set_yticks([])  # Remove default y-axis ticks
    ax.set_xlabel("SHAP Value (Influence on 'spam' prediction)")
    ax.set_ylabel("Words")  # Add a y-axis label
    title = original_text[:40] + '...' if len(original_text) > 40 else original_text
    ax.set_title(f"SHAP Explanation for: '{title}'")
    ax.invert_yaxis()  # To display the first word at the top
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="SHAP Value")
    plt.tight_layout()
    plt.show()


def visualize_lime_explanation(lime_exp, original_text, class_names=['ham', 'spam']):
    """
    Visualize a LIME explanation for a single text sample.
    Args:
        lime_exp: LIME explanation object
        original_text: The original text
        class_names: List of class names
    """
    print(f"Original Text: '{original_text}'")
    pred_class = np.argmax(lime_exp.predict_proba)
    print(f"Predicted Class: {class_names[pred_class]}")
    lime_exp.show_in_notebook(text=original_text)
