import os
import random

import numpy as np
import pandas as pd
import torch

from models.bilstm import BiLSTMSpam
from utils.functions import load_glove_embeddings, build_vocab
from utils.bilstm_evaluation import analyze_model_explanations, analyze_adversarial_robustness

ROOT_PATH = '../'
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, '../spam-detection-data/'))

GLOVE_PATH = os.path.join(DATA_PATH, 'data/raw/glove.6B/glove.6B.300d.txt')

train_df = pd.read_pickle(DATA_PATH + '/data/processed/train.pkl')
test_df = pd.read_pickle(DATA_PATH + '/data/processed/test.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    word2idx, idx2word = build_vocab(train_df['text'])
    embedding_dim = 300
    max_len = 200

    print("Loading GloVe embeddings...")
    pretrained_embeddings = load_glove_embeddings(GLOVE_PATH, word2idx, embedding_dim)

    # Load the trained BiLSTM model
    model_path = DATA_PATH + '/trained_models/best_bilstm_model.pt'
    model = BiLSTMSpam(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                    pretrained_embeddings=pretrained_embeddings)
    model.load(model_path)
    model = model.to(device)
    model.eval()

    # Run explainability analysis
    print("Analyzing model explanations...")
    explanation_metrics = analyze_model_explanations(model, test_df, word2idx, max_len, device, num_samples=5)

    if explanation_metrics:
        print("\n\n--- Overall Explanation Quality Metrics ---")
        for metric_name, value in explanation_metrics.items():
            print(f"  {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")
        print("---------------------------------------")
    else:
        print("Explanation metrics could not be calculated.")

    # Optionally, run adversarial robustness analysis if still needed
    print("\n\nAnalyzing adversarial robustness...")
    # The function now has k_for_retention parameter, default is 5
    adversarial_robustness_results = analyze_adversarial_robustness(model, test_df, word2idx, max_len, device, num_samples=5, k_for_retention=5)
    
    if adversarial_robustness_results:
        print("\n\n--- Overall Adversarial Robustness Metrics ---")
        for eps_result in adversarial_robustness_results:
            print(f"\n  Results for Epsilon = {eps_result['epsilon']}:")
            print(f"    Attack Success Rate: {eps_result.get('attack_success_rate', float('nan')):.4f}")
            print(f"    Avg. Explanation Shift (Cosine Sim): {eps_result.get('avg_explanation_shift_cosine_sim', float('nan')):.4f}")
            print(f"    Avg. Top-k Retention: {eps_result.get('avg_top_k_retention', float('nan')):.4f}")
        print("-------------------------------------------")
    else:
        print("Adversarial robustness metrics could not be calculated.")
