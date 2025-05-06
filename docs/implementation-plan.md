# **Spam Detection Project Summary**

---

### **Phase 1: Setup & Data Preparation**
#### **1. Project Setup (Local - PyCharm)**
- **Create Project**:
  - Open PyCharm → New Project → `spam-detection`
  - Set up Python 3.11 environment (Virtualenv).

- **File Structure**:
  ```
  spam-detection/
  ├── data/               # Raw/preprocessed data
  ├── images/             # Visualization output images
  ├── models/             # Model definitions (CNN, BiLSTM, etc.)
  ├── utils/              # Helper functions (preprocessing, metrics)
  ├── trained_models/     # Trained models (CNN, BiLSTM, etc.)
  ├── configs.py          # Hyperparameters
  ├── train.py            # Training script
  └── requirements.txt    # Dependencies
  └── SpamDetection.ipynb # Notebook for all code to be run in Google Colab
  └── local.ipynb         # Notebook for all code to be run locally
  └── ../spam-detection-data/data/raw/glove.6B/glove.6B.300d.txt  # GloVe embeddings
  ```

#### **2. Install Dependencies**
- In PyCharm terminal:
  ```bash
  pip install torch==2.5.1+cu124 transformers==4.48.0 scikit-learn pandas numpy matplotlib
  ```
- Add MLflow for experiment tracking, BeautifulSoup for HTML parsing, and SHAP/LIME for explainability:
  ```bash
  pip install mlflow beautifulsoup4 shap lime
  ```
- Save to `requirements.txt`:
  ```bash
  pip freeze > requirements.txt
  ```

#### **3. Download & Preprocess Data**
- **Download SpamAssassin Dataset**:
  ```python
  # utils/data_loader.py
  import pandas as pd
  from sklearn.model_selection import train_test_split

  def load_data():
      # Load emails, labels (adjust paths)
      emails, labels = ...  
      X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2)
      return X_train, X_test, y_train, y_test
  ```
- **Preprocessing Steps**:
  - Extract email headers (sender, subject, reply-to, etc.)
  - Strip HTML tags using BeautifulSoup
  - Tokenize while preserving entities (emails, currencies, URLs)
  - Feature engineering: TF-IDF, URL count, sender reputation, etc.
  - Redact PII and hash emails with SHA-256 for privacy

---

### **Phase 2: Model Development (Local)**
#### **4. Implement Models**
- **CNN Model** (`models/cnn.py`)
  - Use pretrained GloVe embeddings (frozen during training)
  - Integrate Grad-CAM for explainability
- **BiLSTM Model** (`models/bilstm.py`)
  - Use pretrained GloVe embeddings (frozen)
  - Integrate attention mechanism and attention visualization
- **BERT Model** (`models/bert.py`)
  - Use Hugging Face Transformers
  - Apply layer-wise learning rate decay
  - Integrate attention visualization

#### **5. Training Script** (`train.py`)
  ```python
  from models.cnn import SpamCNN
  from utils.data_loader import load_data

  def train(model, model_name):
      X_train, X_test, y_train, y_test = load_data()
      # Training loop (optimizer, loss, etc.)
      model.save('models/' + model_name + '.pt')
  ```
- Track experiments and parameters with MLflow
- Set fixed random seeds for reproducibility

---

### **Phase 3: Explainability & Evaluation**
#### **6. Integrate Explainability Methods**
- Implement SHAP and LIME for all models
- Visualize explanations: heatmaps, force plots, attention maps
- Compare explanation quality: faithfulness (AUC-Del/Ins), stability (Jaccard), Explanation Cosine Similarity (ECS)

#### **7. Evaluation Metrics**
- Classification: F1, AUC-ROC, false positive/negative rates
- Explanation: faithfulness, stability, ECS
- Efficiency: latency, memory usage

#### **8. Adversarial Robustness**
- Evaluate with adversarial attacks (TextFooler, DeepWordBug)
- Measure explanation shift under attack

---

### **Phase 4: Run on Colab**
#### **9. Upload to GitHub**
- Commit code to GitHub:
  ```bash
  git init
  git remote add origin https://github.com/rationalear/spam-detection.git
  git add .
  git commit -m "Initial commit"
  git push -u origin main
  ```

#### **10. Colab Setup**
- Open [Google Colab](https://colab.research.google.com/) → New Notebook.
- Clone repo and install dependencies:
  ```python
  !git clone https://github.com/rationalear/spam-detection.git
  %cd spam-detection
  !pip install -r requirements.txt
  ```

#### **11. Run Training**
- Execute your script:
  ```python
  %run train.py  # Runs on Colab’s GPU
  ```
- For interactive development:
  ```python
  from train import train
  train()  # Call your functions directly
  ```

---

### **Phase 5: Save, Monitor & Reproducibility**
#### **12. Save Results**
- Mount Google Drive in Colab:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Save models:
  ```python
  torch.save(model, '/content/drive/MyDrive/models/cnn_model.pt')
  ```

#### **13. Monitor Progress**
- Use TensorBoard:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir /content/drive/MyDrive/logs
  ```
- Use Docker/Conda for environment reproducibility
- Document all steps and release anonymized preprocessed data/code

---

### **Phase 6: Iterate & Improve**
1. **Test locally** → **Debug in PyCharm** → **Push changes to GitHub**.
2. **Re-run** in Colab:
   ```python
   !git pull origin main  # Sync latest code
   %run train.py
   ```

---

### **Critical Tips**
1. **Colab GPU**: 
   - Runtime → Change runtime type → GPU (T4/A100).
2. **Data Caching**: 
   - Upload preprocessed data to Drive to avoid re-processing.
3. **Session Management**: 
   - Save checkpoints hourly (Colab may disconnect).

---

### **Troubleshooting**
- **OOM Errors**: Reduce batch size (8 or 16).
- **Slow Training**: Use mixed precision (`torch.cuda.amp`).
- **Version Mismatch**: Pin library versions in `requirements.txt`.
