import torch
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np

# SHAP explainability for CNN model

def explain_cnn_with_shap(model, tokenizer, texts, device, max_len=128, num_samples=100):
    """
    Generate SHAP values for a batch of texts using a CNN model.
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
    background_inputs = [tokenizer(t, max_len) for t in background_texts]
    background_inputs = torch.tensor(background_inputs).to(device)

    def predict_fn(x):
        x = torch.tensor(x).to(device)
        with torch.no_grad():
            return model(x).cpu().numpy()

    explainer = shap.DeepExplainer(model, background_inputs)
    test_inputs = [tokenizer(t, max_len) for t in texts]
    test_inputs = torch.tensor(test_inputs).to(device)
    shap_values = explainer.shap_values(test_inputs)
    return shap_values, test_inputs


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
        return probs

    def explain_fn(text):
        return explainer.explain_instance(text, predict_proba, num_features=10)

    return explain_fn

