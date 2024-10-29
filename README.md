
# LSTM-Based Plagiarism Detection System

## Overview

This project focuses on building a **Plagiarism Detection System** using a **Long Short-Term Memory (LSTM)** neural network. The goal is to detect whether a given text is plagiarized by comparing it with previously uploaded documents. This model processes sequences of text data and determines if there is a match or similarity between the newly uploaded content and the existing corpus.

## Model Performance

| Metric               | Value   |
|----------------------|---------|
| **Training Accuracy** | 71.02%  |
| **Training Loss**     | 0.5369  |
| **Validation Accuracy**| 78.14% |
| **Validation Loss**   | 0.4465  |

The results indicate that the model generalizes well, with the validation accuracy outperforming the training accuracy.

## Model Architecture

The **LSTM-based model** consists of the following layers:

1. **Embedding Layer**:
   - Converts input text into dense vectors, capturing semantic meanings.
   - Input: Tokenized text sequences.
   
2. **LSTM Layer**:
   - Learns sequential patterns and dependencies within the text.
   - Captures long-term contextual information, essential for comparing the similarities between texts.
   
3. **Dense Layer**:
   - A fully connected layer using a sigmoid activation function for binary classification (plagiarized vs. original).
   - Output: Probability score (0-1) indicating whether the text is plagiarized.

## Training Configuration

The model was trained using the following configuration:
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam Optimizer
- **Batch Size**: 32
- **Epochs**: 3

## Use Cases

This LSTM-based model can be applied in various domains where plagiarism detection is essential:

- **Academic Institutions**: Detect similarities in student essays, research papers, or theses.
- **Content Creation**: Verify the originality of blogs, articles, or web content.
- **Corporate Documents**: Ensure the uniqueness of business reports, proposals, and marketing materials.

## Future Enhancements

- **Data Augmentation**: Expanding the dataset with more diverse text samples for better model generalization.
- **Hyperparameter Tuning**: Experimenting with different learning rates, batch sizes, and optimizers to optimize performance.
- **Regularization Techniques**: Adding techniques such as dropout to prevent overfitting.
  
## Model Results

The model demonstrates strong generalization capabilities with a validation accuracy of 78.14%. While the training accuracy is lower, the validation accuracy suggests that the model performs well on unseen data.

---

### Example Usage

To run the plagiarism detection:

```python
# Load the model
model = tf.keras.models.load_model('quantized_model.h5')

# Preprocess the text
sequences = tokenizer.texts_to_sequences([input_text])
padded = pad_sequences(sequences, maxlen=200)

# Predict plagiarism
prediction = model.predict(padded)
```

---

### Installation

To install the necessary dependencies, run:

```bash
pip install tensorflow
pip install streamlit
pip install tqdm
```

---
###  To Run the File

To install the necessary dependencies, run:

```bash
streamlit run Myapp.py
```

---

This repository offers an end-to-end solution for detecting plagiarism in text files using deep learning techniques. It is designed for academic, content creation, and business use cases.

