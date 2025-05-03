import torch
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np

# SHAP explainability for CNN model

def explain_cnn_with_shap(model, tokenizer, texts, device, max_len=128, num_samples=100):
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
    Returns:
        shap_values, input_ids
    """
    model.eval()
    background_texts = texts[:num_samples]
    background_inputs = np.array([tokenizer(t, max_len) for t in background_texts])

    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(x)
            return out.cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, background_inputs)
    test_inputs = np.array([tokenizer(t, max_len) for t in texts])
    shap_values = explainer.shap_values(test_inputs, nsamples=100)
    return shap_values, torch.tensor(test_inputs, dtype=torch.long).to(device)


# LIME explainability for CNN model

def explain_cnn_with_lime(model, tokenizer, class_names, device, max_len=128):
    """
    Returns a function to generate LIME explanations for a single text.
    Args:
        model: Trained SpamCNN model
        tokenizer: Function to convert text to input ids
        class_names: List of class names (e.g., ['ham', 'spam'])
        device: torch.device
        max_len: Max sequence length
    Returns:
        explain_fn: function(text) -> lime explanation
    """
    model.eval()
    explainer = LimeTextExplainer(class_names=class_names)
    
    def predict_proba(texts):
        inputs = [tokenizer(t, max_len) for t in texts]
        inputs = torch.tensor(inputs).to(device)
        with torch.no_grad():
            probs = model(inputs).cpu().numpy()
        # Model outputs probability of spam; return [P(ham), P(spam)]
        probs = np.stack([1 - probs, probs], axis=1)
        return probs.astype(float)  # Ensure float dtype

    def explain_fn(text):
        return explainer.explain_instance(text, predict_proba, num_features=10)

    return explain_fn

