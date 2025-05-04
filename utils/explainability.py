import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer


# SHAP explainability for CNN model


def show_in_notebook(self,
                     labels=None,
                     predict_proba=True,
                     show_predicted_value=True,
                     **kwargs):
    """Shows html explanation in ipython notebook.

    See as_html() for parameters.
    This will throw an error if you don't have IPython installed"""
    
    from IPython.display import display, HTML

    # Ensure predict_proba is JSON serializable only during HTML rendering
    original_predict_proba = self.predict_proba  # Backup the original NumPy array
    if hasattr(self, 'predict_proba') and isinstance(self.predict_proba, np.ndarray):
        self.predict_proba = self.predict_proba.tolist()  # Convert to list for JSON serialization

    try:
        display(HTML(self.as_html(labels=labels,
                                  predict_proba=predict_proba,
                                  show_predicted_value=show_predicted_value,
                                  **kwargs)))
    finally:
        self.predict_proba = original_predict_proba  # Restore the original NumPy array


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
    background_texts = texts[:num_samples]
    background_inputs = np.array([tokenizer(t, max_len) for t in background_texts])
    
    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(x)
            return out.cpu().numpy()
    
    explainer = shap.KernelExplainer(predict_fn, background_inputs, random_state=rng)
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
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)  # Add random_state for reproducibility
    
    def predict_proba(texts):
        inputs = [tokenizer(t, max_len) for t in texts]
        inputs = torch.tensor(inputs).to(device)
        with torch.no_grad():
            probs = model(inputs).cpu().numpy()
        # Model outputs probability of spam; return [P(ham), P(spam)]
        probs = np.stack([1 - probs, probs], axis=1)
        return probs  # Return NumPy array instead of converting to list
    
    def explain_fn(text):
        return explainer.explain_instance(text, predict_proba, num_features=10)
    
    return explain_fn



