# Results and Discussion

Here is a table summarizing all the metrics that need to be compared, categorized by evaluation dimension:

| **Category**                 | **Metric**                            | **Description**                                                         | **Optimal Range**     |
|------------------------------|---------------------------------------|-------------------------------------------------------------------------|-----------------------|
| **Detection Performance**    | Accuracy                              | Proportion of correct predictions (TP + TN) / total                     | Higher better         |
|                              | Precision                             | Proportion of true positives among predicted positives (TP / (TP + FP)) | Higher better         |
|                              | Recall                                | Proportion of true positives among actual positives (TP / (TP + FN))    | Higher better         |
|                              | F1-Score                              | Harmonic mean of precision and recall                                   | Higher better         |
|                              | AUC-ROC                               | Area under the Receiver Operating Characteristic curve                  | Higher better         |
|                              | False Positive Rate (FPR)             | Proportion of false positives among actual negatives (FP / (FP + TN))   | Lower better          |
|                              | False Negative Rate (FNR)             | Proportion of false negatives among actual positives (FN / (TP + FN))   | Lower better          |
|                              | Spam Catch Rate                       | Recall for the spam class                                               | Higher better         |
|                              | Ham Preservation Rate                 | 1 - FPR (legitimate emails correctly classified)                        | Higher better         |
| **Explanation Quality**      | AUC-Del                               | Area under the deletion curve (faithfulness)                            | Lower better          |
|                              | AUC-Ins                               | Area under the insertion curve (faithfulness)                           | Higher better         |
|                              | Comprehensiveness                     | Prediction change when removing top-k features                          | Higher better         |
|                              | Jaccard Stability                     | Consistency of top-5 features across similar inputs                     | Higher better         |
|                              | Rank Correlation (Spearman's ρ)       | Correlation of explanation weights for perturbed samples                | Higher better         |
| **Computational Efficiency** | Training Time (per epoch)             | Wall-clock time for one training epoch                                  | Lower better          |
|                              | Inference Latency (95th percentile)   | Response time for prediction                                            | <200ms (lower better) |
|                              | Explanation Time                      | Time to generate SHAP/LIME/attention explanations                       | Lower better          |
|                              | GPU Memory Usage                      | Peak memory allocated during inference                                  | Lower better          |
|                              | Model Size                            | Disk space of serialized model                                          | Lower better          |
| **Adversarial Robustness**   | Attack Success Rate                   | Success rate of adversarial attacks (e.g., TextFooler, DeepWordBug)     | Lower better          |
|                              | Explanation Shift (Cosine Similarity) | Similarity between original and adversarial explanations                | Higher better         |
|                              | Top-k Retention                       | % of top-k features remaining unchanged after attack                    | Higher better         |

### Notes:
1. **Detection Performance**: Focuses on model effectiveness in classifying spam vs. ham.  
2. **Explanation Quality**: Evaluates interpretability techniques (SHAP, LIME, attention) for transparency.  
3. **Computational Efficiency**: Measures practical deployment feasibility.  
4. **Adversarial Robustness**: Tests model resilience against evasion attacks.

Here’s the comparison table of the three models (CNN, BiLSTM, BERT) based on the experiment results, structured by evaluation dimension:  

| **Category**               | **Metric**                          | **CNN** | **BiLSTM**     | **BERT** |
|----------------------------|-------------------------------------|---------|----------------|----------|
| **Detection Performance**  | Accuracy                            | 0.5562  | 0.94           | 0.9760   |
|                            | Precision                           | 0.3140  | 0.90 (class 1) | 0.9443   |
|                            | Recall                              | 0.3488  | 0.91 (class 1) | 0.9816   |
|                            | F1-Score                            | 0.3305  | 0.91 (class 1) | 0.9626   |
|                            | AUC-ROC                             | 0.9894  | 0.982          | 0.9972   |
|                            | False Positive Rate (FPR)           | 0.3488  | 0.05*          | 0.0265*  |
|                            | False Negative Rate (FNR)           | 0.6512  | 0.09*          | 0.0184*  |
|                            | Spam Catch Rate                     | 0.3488  | 0.91           | 0.9816   |
|                            | Ham Preservation Rate               | 0.6512  | 0.95           | 0.9735   |
| **Explanation Quality**    | AUC-Del (Faithfulness)              | 0.2442  | -0.1537        | -0.0277  |
|                            | AUC-Ins (Faithfulness)              | 0.7367  | -0.4701        | -0.1195  |
|                            | Comprehensiveness (Top-5)           | 0.0207  | -0.1978        | -0.0000  |
|                            | Jaccard Stability (Top-5)           | 0.4462  | 0.2579         | 0.1111   |
|                            | Rank Correlation (Spearman ρ)       | 0.7343  | 0.7245         | 0.3809   |
| **Adversarial Robustness** | Attack Success Rate (ε=0.1)         | 0.0000  | 0.2000         | 0.0000   |
|                            | Explanation Shift (Cosine Sim)      | 0.0000  | 0.7618         | 0.9997   |
|                            | Top-k Retention                     | 1.0000  | 0.5200         | 0.2000   |

Key Findings:
1. **Robustness**:
   - CNN shows perfect robustness (0% attack success, 1.0 top-k retention)
   - BERT also has 0% attack success but lower top-k retention (0.2)
   - BiLSTM is vulnerable (20% attack success)

2. **Explanation Quality**:
   - CNN leads in stability metrics (Jaccard: 0.45, Rank Corr: 0.73)
   - All models show negative values for some faithfulness metrics
   - BERT has the lowest attention stability (Jaccard: 0.11)

3. **Performance**:
   - BERT dominates detection metrics (Accuracy: 0.976, F1: 0.963)
   - BiLSTM balances performance and explainability
   - CNN has lower accuracy but best adversarial robustness

*Notes:
- FPR/FNR for BiLSTM calculated from confusion matrix
- Negative explanation metrics may need investigation
- Perfect scores (1.0) indicate complete resistance to perturbations*
