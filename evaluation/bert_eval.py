import os
import random
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

from models.bert import SpamBERT
from utils.bert_evaluation import analyze_explanations, visualize_attention, evaluate_adversarial_robustness
from utils.functions import load_glove_embeddings, build_vocab

ROOT_PATH = '../'
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, '../spam-detection-data/'))

GLOVE_PATH = os.path.join(DATA_PATH, 'data/raw/glove.6B/glove.6B.300d.txt')

train_df = pd.read_pickle(DATA_PATH + '/data/processed/train.pkl')
test_df = pd.read_pickle(DATA_PATH + '/data/processed/test.pkl')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    set_seed(42)
    word2idx, idx2word = build_vocab(train_df['text'])
    embedding_dim = 300
    max_len = 200

    print("Loading GloVe embeddings...")
    pretrained_embeddings = load_glove_embeddings(GLOVE_PATH, word2idx, embedding_dim)

    print("Loading BERT tokenizer and model...")
    # Load tokenizer and model
    bert_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = SpamBERT(bert_model_name=bert_model_name)
    model.load(os.path.join(DATA_PATH, 'trained_models', 'spam_bert.pt'))
    model = model.to(device)
    model.eval()

    # Analyze a sample email
    sample_text = test_df['text'].iloc[0]
    print("Analyzing text:", sample_text[:100], "...\n")

    explanation_data, prob = analyze_explanations(model, sample_text, tokenizer)

    # Visualize attention for layer 12
    if 'layer_12' in explanation_data:
        print("Visualizing attention weights for layer 12:")
        # Ensure tensor is detached and moved to CPU before converting to numpy
        attention_weights = explanation_data['layer_12'][0].detach().cpu().numpy()
        visualize_attention(sample_text, attention_weights, tokenizer)
    else:
        print("Warning: Layer 12 attention weights not available")

    # Print prediction and explanation metrics
    print(f"\nPrediction probability: {prob:.4f}")
    if 'metrics' in explanation_data:
        print("\nExplanation metrics:")
        for metric, value in explanation_data['metrics'].items():
            # Ensure value is float before formatting, handle NaN or other non-float types gracefully
            if isinstance(value, (float, np.floating)):
                 print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
            else:
                 print(f"{metric.replace('_', ' ').capitalize()}: {value}")

    # Evaluate adversarial robustness
    print("\n\nEvaluating Adversarial Robustness...")
    # device is already defined globally in bert_eval.py
    # n_samples and k_for_retention can be adjusted as needed
    adv_robustness_metrics = evaluate_adversarial_robustness(model, test_df, tokenizer, device=device, n_samples=10, k_for_retention=5)

    if adv_robustness_metrics:
        print("\n--- Adversarial Robustness Metrics ---")
        for metric_name, value in adv_robustness_metrics.items():
            if isinstance(value, (float, np.floating)):
                 print(f"  {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")
            else:
                 print(f"  {metric_name.replace('_', ' ').capitalize()}: {value}")
        print("------------------------------------")
    else:
        print("Adversarial robustness metrics could not be calculated.")