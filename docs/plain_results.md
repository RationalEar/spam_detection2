# CNN Evaluation
## 1. Comprehensive Performance Metrics
False Positive Rate: 0.3488
False Negative Rate: 0.6512
AUC-ROC: 0.9894
Weighted Error (FP=0.3, FN=0.7): 260.0711
Accuracy 0.5562
Precision 0.3140
Recall 0.3488
F1 Score 0.3305
Specificity 0.6512

## 2. Explainability Analysis
Explanation Quality Metrics:
AUC-Del (Faithfulness): 0.2442
AUC-Ins (Comprehensiveness): 0.7367
Stability Score: 0.4462

## 3. Adversarial Robustness Analysis
Adversarial Robustness Results:
Clean Accuracy: 0.6250

Accuracy under different perturbation sizes:
ε=0.01: 0.6250
ε=0.05: 0.6250
ε=0.1: 0.6250

Explanation shift under different perturbation sizes:
ε=0.01: 0.0000
ε=0.05: 0.0000
ε=0.1: 0.0000

Clean vs Adversarial Performance:
Clean Performance:
- Accuracy: -5.0000

Adversarial Performance:
- Accuracy: -5.0000


# BiLSTM Evaluation
## 1. Comprehensive Performance Metrics

Classification Report:
              precision    recall  f1-score   support

         0.0       0.96      0.95      0.96       830
         1.0       0.90      0.91      0.91       380

    accuracy                           0.94      1210
   macro avg       0.93      0.93      0.93      1210
weighted avg       0.94      0.94      0.94      1210

Confusion Matrix:
0    792      38
1     33     347
      0       1

Summary Metrics:
ROC AUC: 0.982
PR AUC: 0.954

## 2. Explainability Analysis

## 3. Adversarial Robustness Analysis

Final adversarial predictions: [8.8056797e-05 1.2742224e-01 9.5726556e-01 4.5300460e-01 3.0288944e-02]
Number of tokens changed: 96
Changes per sequence: [30, 30, 5, 1, 30]
Number of tokens changed: 96
Adversarial predictions: [0. 0. 1. 1. 0.]
Adversarial confidence scores: [2.0011171e-04 2.5880715e-01 9.6534961e-01 8.6443156e-01 9.9148452e-03]
Attack success rate: 0.2000

Successful attacks:
Sample 1:
Original prediction: 0.9989
Adversarial prediction: 0.2588

Adversarial Results:
[{'epsilon': 0.01,
  'success_rate': 0.20000000298023224,
  'original_preds': array([0., 1., 1., 1., 0.], dtype=float32),
  'adversarial_preds': array([0., 0., 1., 1., 0.], dtype=float32)},
 {'epsilon': 0.05,
  'success_rate': 0.20000000298023224,
  'original_preds': array([0., 1., 1., 1., 0.], dtype=float32),
  'adversarial_preds': array([0., 0., 1., 1., 0.], dtype=float32)},
 {'epsilon': 0.1,
  'success_rate': 0.20000000298023224,
  'original_preds': array([0., 1., 1., 1., 0.], dtype=float32),
  'adversarial_preds': array([0., 0., 1., 1., 0.], dtype=float32)}]


# BERT Evaluation
## 1. Comprehensive Performance Metrics
accuracy: 0.9760
precision: 0.9443
recall: 0.9816
f1: 0.9626
auc_roc: 0.9972
spam_catch_rate: 0.9816
ham_preservation_rate: 0.9735

## 2. Explainability Analysis
Prediction probability: 0.0001

Explanation metrics:
convergence_delta: -0.0129
attribution_mean: -0.0000
attribution_std: 0.0005

## 3. Adversarial Robustness Analysis
prediction_stability: 0.7603
decision_stability: 0.7600