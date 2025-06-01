import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bar
from models.cnn import SpamCNN
from utils.functions import encode, load_glove_embeddings, build_vocab
import pandas as pd
from utils.cnn_adversarial_metrics import (
    calculate_attack_success_rate_cnn,
    calculate_explanation_shift_cnn,
    calculate_top_k_retention_cnn
)

ROOT_PATH = '../'
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, '../spam-detection-data/'))
GLOVE_PATH = os.path.join(DATA_PATH, 'data/raw/glove.6B/glove.6B.300d.txt')

# Load the data
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
    # Build vocabulary and load embeddings
    set_seed(42)
    word2idx, idx2word = build_vocab(train_df['text'])
    embedding_dim = 300
    max_len = 200

    print(("Loading pretrained embeddings from: {}".format(GLOVE_PATH)))
    pretrained_embeddings = load_glove_embeddings(GLOVE_PATH, word2idx, embedding_dim)
    print("Pretrained embeddings loaded.")

    # Load the trained CNN model
    model_path = DATA_PATH + '/trained_models/spam_cnn.pt'
    cnn_model = SpamCNN(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                        pretrained_embeddings=pretrained_embeddings)
    cnn_model.load(model_path, map_location=device)  # map_location handled in load()
    cnn_model = cnn_model
    cnn_model.eval()

    # Prepare test data
    X_test_tensor = torch.tensor([encode(t, word2idx, max_len) for t in test_df['text']])
    y_test_tensor = torch.tensor(test_df['label'].values, dtype=torch.float32)

    # Move data to device
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # Generate Grad-CAM explanations
    # NOTE: Do NOT use torch.no_grad() here, as Grad-CAM needs gradients!
    print("\nGenerating explanations...")
    cam_maps = cnn_model.grad_cam(X_test_tensor)

    # Compute explainability metrics
    # Ensure inputs to compute_explanation_metrics are on the correct device, 
    # although the method itself also handles device placement.
    exp_metrics = cnn_model.compute_explanation_metrics(X_test_tensor.to(device), cam_maps.to(device), num_perturbations=10)

    print("\nExplanation Quality Metrics:")
    print(f"AUC-Del (Faithfulness): {exp_metrics.get('auc_del', float('nan')):.4f}")
    print(f"AUC-Ins (Faithfulness): {exp_metrics.get('auc_ins', float('nan')):.4f}")
    print(f"Comprehensiveness (Top-5): {exp_metrics.get('comprehensiveness', float('nan')):.4f}")
    print(f"Jaccard Stability (Top-5): {exp_metrics.get('jaccard_stability', float('nan')):.4f}")
    print(f"Rank Correlation (Spearman ρ): {exp_metrics.get('rank_correlation', float('nan')):.4f}")
    print(f"Explanation Consistency Score (ECS): {exp_metrics.get('ecs', float('nan')):.4f}")

    # --- Adversarial Robustness Evaluation ---
    print("\n\nEvaluating Adversarial Robustness for CNN...")
    n_adv_samples = 50  # Number of samples for adversarial evaluation (can be adjusted)
    if n_adv_samples > len(X_test_tensor):
        n_adv_samples = len(X_test_tensor)
    
    sample_indices = np.random.choice(len(X_test_tensor), n_adv_samples, replace=False)
    X_adv_test_samples = X_test_tensor[sample_indices]
    y_adv_test_samples = y_test_tensor[sample_indices] # True labels for these samples

    epsilon_range = [0.01, 0.05, 0.1, 0.15] # Perturbation strengths to test
    k_for_retention = 5
    all_robustness_metrics = [] # To store metrics for plotting

    print(f"Using {n_adv_samples} samples for adversarial robustness evaluation.")

    for epsilon in epsilon_range:
        print(f"\n--- Epsilon = {epsilon} ---")
        
        # 1. Get original predictions and explanations (CAMs)
        with torch.no_grad(): # Predictions don't need grad
            original_preds_prob = cnn_model.predict(X_adv_test_samples) # (n_adv_samples,)
        # CAMs need gradients to be enabled during their generation
        original_cams = cnn_model.grad_cam_auto(X_adv_test_samples) # (n_adv_samples, seq_len)

        # 2. Generate adversarial examples
        # generate_adversarial_example enables train mode and grads internally
        adversarial_X = cnn_model.generate_adversarial_example(X_adv_test_samples, y_adv_test_samples, epsilon=epsilon, num_steps=10)

        # 3. Get predictions and explanations for adversarial examples
        with torch.no_grad():
            adversarial_preds_prob = cnn_model.predict(adversarial_X)
        adversarial_cams = cnn_model.grad_cam_auto(adversarial_X)

        # 4. Calculate metrics
        asr = calculate_attack_success_rate_cnn(original_preds_prob, adversarial_preds_prob)
        explanation_shift = calculate_explanation_shift_cnn(original_cams, adversarial_cams)
        top_k_ret = calculate_top_k_retention_cnn(original_cams, adversarial_cams, k=k_for_retention)

        print(f"  Attack Success Rate: {asr:.4f}")
        print(f"  Explanation Shift (Cosine Distance): {explanation_shift:.4f}")
        print(f"  Top-{k_for_retention} Retention: {top_k_ret:.4f}")
        
        all_robustness_metrics.append({
            'epsilon': epsilon,
            'asr': asr,
            'explanation_shift': explanation_shift,
            'top_k_retention': top_k_ret
        })

    # Plotting Adversarial Robustness Metrics
    epsilons = [m['epsilon'] for m in all_robustness_metrics]
    asrs = [m['asr'] for m in all_robustness_metrics]
    shifts = [m['explanation_shift'] for m in all_robustness_metrics]
    retentions = [m['top_k_retention'] for m in all_robustness_metrics]

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epsilons, asrs, marker='o', label='ASR')
    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate')
    plt.title('ASR vs. Epsilon')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epsilons, shifts, marker='o', label='Explanation Shift (Cosine Dist.)', color='green')
    plt.xlabel('Epsilon')
    plt.ylabel('Explanation Shift')
    plt.title('Explanation Shift vs. Epsilon')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epsilons, retentions, marker='o', label=f'Top-{k_for_retention} Retention', color='purple')
    plt.xlabel('Epsilon')
    plt.ylabel(f'Top-{k_for_retention} Retention')
    plt.title(f'Top-{k_for_retention} Retention vs. Epsilon')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("---------------------------------------------")
