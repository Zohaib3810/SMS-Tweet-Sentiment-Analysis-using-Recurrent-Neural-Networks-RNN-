
# SMS/Tweet Sentiment Analysis using Recurrent Neural Networks (RNN)

This repository contains a comprehensive pipeline for performing sentiment analysis (binary classification) on text datasets (such as tweets or SMS messages) using deep learning techniques. We build, train, and compare two Recurrent Neural Network models using **Keras** and **TensorFlow**:
1. **Simple RNN**
2. **Bidirectional RNN**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
  - [Simple RNN](#simple-rnn)
  - [Bidirectional RNN](#bidirectional-rnn)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)

---

## Project Overview
Recurrent Neural Networks (RNNs) are designed to process sequential data by retaining a hidden state (memory) of past inputs. However, standard one-way RNNs only process sequences from left to right. This project explores whether processing the text in both directions using a **Bidirectional RNN** improves classification accuracy on textual sentiment analysis tasks.

---

## Dataset
The pipeline assumes a dataset containing text with corresponding binary labels (e.g., positive/negative, spam/ham):
* **Labels**: The target binary class (e.g., `0` or `1`).
* **Data**: The raw text content.

Place your custom data in a CSV file named `mydataset.csv` in the root directory.

---

## Preprocessing Pipeline
To clean and prepare the raw text data for deep learning, we implement the following text preprocessing steps using **NLTK** and **Pandas**:
1. **Number Removal**: Removing digit sequences from texts.
2. **Case Normalization**: Converting all characters to lowercase to maintain uniformity.
3. **Punctuation Removal**: Stripping away all standard punctuation marks.
4. **Stopword Removal**: Eliminating common English stopwords (e.g., "the", "is", "at") that do not add semantic value to the classification task.
5. **Tokenization & Padding**: Standardizing text length by converting tokens into numerical sequences and padding them up to a maximum length (`maxlen = 150`).

---

## Model Architectures

### 1. Simple RNN
* **Embedding Layer**: Projects words into a 64-dimensional dense vector space.
* **SimpleRNN Layer**: Consists of 64 recurrent units processing sequences sequentially.
* **Dense Layer**: Single neuron with a `sigmoid` activation function for binary output.

### 2. Bidirectional RNN
* Wraps the `SimpleRNN` layer inside a `Bidirectional` wrapper, enabling the network to learn contextual information from both past (left-to-right) and future (right-to-left) directions simultaneously.

---

## Installation and Setup

### Prerequisites
Make sure you have Python 3.x installed along with the required dependencies:

```bash
pip install tensorflow keras scikit-learn pandas nltk
```

---

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/your-repo-name.git
   cd your-repo-name
   ```

2. **Prepare your dataset:**
   Save your dataset as `mydataset.csv` in the project folder with two columns representing your classification labels and text contents.

3. **Run the Notebook or Python Script:**
   Open the notebook in Google Colab or run your local Python environment to train and evaluate the models.

---

## Results & Evaluation
Both models are evaluated on a test set (20% split) using binary cross-entropy loss and accuracy metrics. 

```python
# Expected Console Output Format:
SimpleRNN Test Accuracy: 0.XXXX
Bidirectional SimpleRNN Test Accuracy: 0.XXXX
```

Compare the final test accuracies to analyze how bidirectional context learning influences your specific dataset's performance!
```
