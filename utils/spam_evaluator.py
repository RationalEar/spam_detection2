import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
from textattack.attack_recipes import TextFoolerAttack, DeepWordBugAttack
from textattack.models.wrappers import HuggingFaceModelWrapper
from scipy.stats import spearmanr
from torch.nn.functional import cosine_similarity


class SpamEvaluator:
    def __init__(self, model, tokenizer):
        """
        Initialize evaluator with model and tokenizer
        Args:
            model: SpamBERT model instance
            tokenizer: BERT tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self._setup_attacks()

    def _setup_attacks(self):
        """Setup adversarial attacks"""
        model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer)
        self.textfooler = TextFoolerAttack.build(model_wrapper)
        self.deepwordbug = DeepWordBugAttack.build(model_wrapper)

    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification metrics
        Args:
            predictions: Model predictions
            labels: Ground truth labels
        Returns:
            dict: Dictionary of metrics
        """
        preds = (predictions > 0.5).float().cpu().numpy()
        labels = labels.cpu().numpy()

        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision_score(labels, preds)),
            'recall': float(recall_score(labels, preds)),
            'f1': float(f1_score(labels, preds)),
            'auc_roc': float(roc_auc_score(labels, predictions.cpu().numpy()))
        }

    def compute_explanation_consistency_score(
            self,
            original_explanations: Dict[str, torch.Tensor],
            perturbed_explanations: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute Explanation Consistency Score (ECS)
        Args:
            original_explanations: Original model explanations
            perturbed_explanations: Explanations after perturbation
        Returns:
            float: ECS score
        """
        faithfulness = self._compute_faithfulness(original_explanations)
        stability = self._compute_stability(original_explanations, perturbed_explanations)
        plausibility = self._compute_plausibility(original_explanations)
        simplicity = self._compute_simplicity(original_explanations)

        # Weighted combination as per thesis Section 3.5.2
        ecs = (0.4 * faithfulness +
               0.3 * stability +
               0.2 * plausibility +
               0.1 * simplicity)

        return float(ecs)

    def _compute_faithfulness(self, explanations: Dict[str, torch.Tensor]) -> float:
        """Compute faithfulness score"""
        # Implementation based on AUC-Del metric
        if 'attributions' not in explanations:
            return 0.0

        attrs = explanations['attributions']
        sorted_indices = torch.argsort(attrs.abs(), descending=True)

        # Compute AUC-Del (area under deletion curve)
        n_steps = min(len(sorted_indices), 100)
        scores = []

        for i in range(n_steps):
            mask = torch.ones_like(attrs)
            mask[sorted_indices[:i]] = 0
            masked_output = (attrs * mask).sum()
            scores.append(float(masked_output))

        return 1.0 - np.trapz(scores) / len(scores)

    def _compute_stability(
            self,
            original: Dict[str, torch.Tensor],
            perturbed: Dict[str, torch.Tensor]
    ) -> float:
        """Compute stability score"""
        if not all(k in original and k in perturbed for k in ['layer_6', 'layer_9', 'layer_12']):
            return 0.0

        similarities = []
        for layer in ['layer_6', 'layer_9', 'layer_12']:
            sim = cosine_similarity(
                original[layer].view(-1),
                perturbed[layer].view(-1),
                dim=0
            )
            similarities.append(float(sim))

        return float(np.mean(similarities))

    def _compute_plausibility(self, explanations: Dict[str, torch.Tensor]) -> float:
        """
        Compute plausibility score
        Note: In practice, this would involve human evaluation
        Here we use a simple heuristic based on attention concentration
        """
        if not any(k.startswith('layer_') for k in explanations):
            return 0.0

        attention_scores = []
        for k in explanations:
            if k.startswith('layer_'):
                # Measure attention concentration
                attn = explanations[k]
                entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
                attention_scores.append(float(entropy))

        # Normalize to [0,1] range
        return float(1.0 - np.mean(attention_scores) / np.log(attn.size(-1)))

    def _compute_simplicity(self, explanations: Dict[str, torch.Tensor]) -> float:
        """Compute simplicity score"""
        if 'attributions' not in explanations:
            return 0.0

        # Count significant attributions (above mean + std)
        attrs = explanations['attributions']
        threshold = attrs.mean() + attrs.std()
        significant = (attrs.abs() > threshold).float().mean()

        # Convert to simplicity score (fewer significant = more simple)
        return float(1.0 - significant)

    def evaluate_adversarial_robustness(
            self,
            texts: List[str],
            labels: List[int],
            n_examples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model's adversarial robustness
        Args:
            texts: List of input texts
            labels: List of labels
            n_examples: Number of examples to test
        Returns:
            dict: Robustness metrics
        """
        # Sample examples
        indices = np.random.choice(len(texts), min(n_examples, len(texts)), replace=False)
        test_texts = [texts[i] for i in indices]
        test_labels = [labels[i] for i in indices]

        # Run attacks
        textfooler_results = self.textfooler.attack_dataset(list(zip(test_texts, test_labels)))
        deepwordbug_results = self.deepwordbug.attack_dataset(list(zip(test_texts, test_labels)))

        return {
            'textfooler_success_rate': float(textfooler_results.attack_success_rate()),
            'deepwordbug_success_rate': float(deepwordbug_results.attack_success_rate())
        }
