- Implemented SpamCNN class in models/cnn.py with pretrained GloVe embedding support (frozen), three 1D convolutional layers, global max pooling, two dense layers, and a Grad-CAM method stub for explainability.
- Added load_glove_embeddings function to utils/preprocessor.py to load GloVe vectors and create an embedding matrix for a given vocabulary, enabling the CNN model to use pretrained GloVe embeddings.

- Fixed SpamCNN forward() in models/cnn.py to always return a 2D tensor (batch_size, 1) for SHAP compatibility, removing .squeeze(-1) from the output.

- Implemented SpamBERT model in models/bert.py using Hugging Face's BertModel, with a dropout layer and a classifier head. The model outputs probabilities and attention weights for explainability. Includes save/load methods.

- Added utils/explainability.py with SHAP and LIME helper functions for the CNN model, following coding guidelines.
- Added example code cells to local.ipynb to demonstrate SHAP and LIME explainability for the CNN model, including visualization.
