# Results and Discussion

Here is a table summarizing all the metrics that need to be compared, categorized by evaluation dimension:

| **Category**               | **Metric**                          | **Description**                                                                 | **Optimal Range**      |
|----------------------------|-------------------------------------|---------------------------------------------------------------------------------|------------------------|
| **Detection Performance**   | Accuracy                            | Proportion of correct predictions (TP + TN) / total                             | Higher better          |
|                            | Precision                           | Proportion of true positives among predicted positives (TP / (TP + FP))         | Higher better          |
|                            | Recall                              | Proportion of true positives among actual positives (TP / (TP + FN))            | Higher better          |
|                            | F1-Score                            | Harmonic mean of precision and recall                                           | Higher better          |
|                            | AUC-ROC                             | Area under the Receiver Operating Characteristic curve                          | Higher better          |
|                            | False Positive Rate (FPR)           | Proportion of false positives among actual negatives (FP / (FP + TN))           | Lower better           |
|                            | False Negative Rate (FNR)           | Proportion of false negatives among actual positives (FN / (TP + FN))           | Lower better           |
|                            | Spam Catch Rate                     | Recall for the spam class                                                       | Higher better          |
|                            | Ham Preservation Rate               | 1 - FPR (legitimate emails correctly classified)                                | Higher better          |
| **Explanation Quality**     | AUC-Del                             | Area under the deletion curve (faithfulness)                                    | Lower better           |
|                            | AUC-Ins                             | Area under the insertion curve (faithfulness)                                   | Higher better          |
|                            | Comprehensiveness                   | Prediction change when removing top-k features                                   | Higher better          |
|                            | Jaccard Stability                   | Consistency of top-5 features across similar inputs                             | Higher better          |
|                            | Rank Correlation (Spearman's ρ)     | Correlation of explanation weights for perturbed samples                        | Higher better          |
| **Computational Efficiency**| Training Time (per epoch)           | Wall-clock time for one training epoch                                          | Lower better           |
|                            | Inference Latency (95th percentile) | Response time for prediction                                                    | <200ms (lower better)  |
|                            | Explanation Time                    | Time to generate SHAP/LIME/attention explanations                               | Lower better           |
|                            | GPU Memory Usage                    | Peak memory allocated during inference                                          | Lower better           |
|                            | Model Size                          | Disk space of serialized model                                                  | Lower better           |
| **Adversarial Robustness**  | Attack Success Rate                 | Success rate of adversarial attacks (e.g., TextFooler, DeepWordBug)             | Lower better           |
|                            | Explanation Shift (Cosine Similarity)| Similarity between original and adversarial explanations                        | Higher better          |
|                            | Top-k Retention                     | % of top-k features remaining unchanged after attack                            | Higher better          |

### Notes:
1. **Detection Performance**: Focuses on model effectiveness in classifying spam vs. ham.  
2. **Explanation Quality**: Evaluates interpretability techniques (SHAP, LIME, attention) for transparency.  
3. **Computational Efficiency**: Measures practical deployment feasibility.  
4. **Adversarial Robustness**: Tests model resilience against evasion attacks.

Here’s the compiled comparison table of the three models (CNN, BiLSTM, BERT) based on the experiment results, structured by evaluation dimension and simplified for clarity:  

| **Category**               | **Metric**                          | **CNN**       | **BiLSTM**     | **BERT**      |
|----------------------------|-------------------------------------|---------------|----------------|---------------|
| **Detection Performance**   | Accuracy                            | 0.5562        | 0.94           | 0.9760        |
|                            | Precision                           | 0.3140        | 0.90 (class 1) | 0.9443        |
|                            | Recall                              | 0.3488        | 0.91 (class 1) | 0.9816        |
|                            | F1-Score                            | 0.3305        | 0.91 (class 1) | 0.9626        |
|                            | AUC-ROC                             | 0.9894        | 0.982          | 0.9972        |
|                            | PR AUC                              | N/A           | 0.954          | N/A           |
|                            | False Positive Rate (FPR)           | 0.3488        | 0.05*          | 0.0265*       |
|                            | False Negative Rate (FNR)           | 0.6512        | 0.09*          | 0.0184*       |
|                            | Spam Catch Rate                     | 0.3488        | 0.91           | 0.9816        |
|                            | Ham Preservation Rate               | 0.6512        | 0.95           | 0.9735        |
| **Explanation Quality**     | AUC-Del (Faithfulness)              | 0.2442        | -0.1537        | N/A           |
|                            | AUC-Ins (Comprehensiveness)         | 0.7367        | -0.4701        | N/A           |
|                            | Comprehensiveness (k=5)             | N/A           | -0.1978        | N/A           |
|                            | Jaccard Stability (Attention, k=5)  | N/A           | 0.2579         | N/A           |
|                            | Rank Correlation (Attention)        | N/A           | 0.7245         | N/A           |
|                            | Stability Score                     | 0.4462        | N/A            | N/A           |
| **Adversarial Robustness**  | Clean Accuracy                      | 0.6250        | N/A            | N/A           |
|                            | Attack Success Rate (ε=0.1)         | N/A           | 0.2000         | N/A           |
|                            | Explanation Shift (Cosine Sim, ε=0.1)| 0.0000       | 0.7618         | N/A           |
|                            | Top-k Retention (ε=0.1)             | N/A           | 0.5200         | N/A           |
|                            | Prediction Stability                | N/A           | N/A            | 0.7603        |
|                            | Decision Stability                  | N/A           | N/A            | 0.7600        |

**Key Observations:**
1. **BiLSTM Explanation Quality** now includes:
   - Negative values for AUC-Del/AUC-Ins (suggesting potential issues with explanation faithfulness)
   - Moderate attention stability (Jaccard: 0.26, Rank Correlation: 0.72)

2. **BiLSTM Adversarial Robustness** shows:
   - 20% attack success rate at ε=0.1
   - High explanation shift (0.76 cosine similarity) but improved top-k retention (52%)

3. **BERT** maintains superior detection metrics but lacks detailed explanation/adversarial metrics.

4. **CNN** shows complete resistance to explanation shift (0.0) under perturbations despite lower accuracy.

### Notes:  
1. **BiLSTM FPR/FNR**: Calculated from confusion matrix (FPR = FP/(FP+TN) = 38/(792+38) ≈ 0.05; FNR = FN/(TP+FN) = 33/(347+33) ≈ 0.09).  
2. **BERT FPR/FNR**: Derived from Ham Preservation Rate (FPR = 1 - 0.9735 = 0.0265) and Spam Catch Rate (FNR = 1 - 0.9816 = 0.0184).  
3. **N/A**: Metrics not provided in the results.  
4. **Adversarial Robustness**: CNN shows no explanation shift under perturbations, while BiLSTM has a 20% attack success rate. BERT reports stability scores but lacks attack metrics.  

This table highlights **BERT’s superior performance** in detection metrics and **BiLSTM’s balance** between performance and adversarial robustness. Let me know if you'd like adjustments!