# Sentimental-analysis

# 📑Project Title  
### Sentimental analysis
# 📌Description

This project performs sentiment classification on IMDB movie reviews using a deep learning approach with Long Short-Term Memory (LSTM) networks. Reviews are categorized into positive or negative sentiments using natural language processing (NLP) techniques combined with a neural network-based text classifier.
The model is built using TensorFlow/Keras, leveraging Tokenizer and pad_sequences for preprocessing. An Embedding layer followed by an LSTM network is used to capture contextual meaning from the reviews. The dataset used contains 50,000 labeled movie reviews sourced from the IMDB dataset.

# 📌Table of contents

- [Project Overview](#project-Overview)
- [Dataset](#datasets)
- [Dependencies](#dependencies)
- [Required Imports & Libraries](#required-imports-Libraries)
- [Project Structure](#project-Structure)
- [Data Preprocessing](#data-Preprocessing)
- [Model Architecture](#model-architecture)
- [Training Details](#training-Details)
- [Performance Metrics](#performance-Metrics)
- [Running the App](#running-the-App)
- [Sample Output](#sample-Output)
- [Future Work](#future-Work)
  
# 📌Project Overview

This sentiment analysis system classifies movie reviews as positive or negative using a Recurrent Neural Network (LSTM). It applies tokenization, padding, and word embedding on the IMDB dataset, followed by LSTM-based training and evaluation.

# 📂Dataset

Data Source:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

## 🛠️Dependencies

Before running the project, ensure the following Python libraries are installed:

- `tensorflow / keras`: For building, training, and loading the LSTM sentiment classification model.

- `numpy`: For efficient numerical computations and handling padded sequences.

- `pandas`: For loading and organizing the IMDb dataset.

- `matplotlib & seaborn`: For plotting training/validation accuracy, loss graphs, and confusion matrix visualizations.

- `scikit-learn`: For splitting data and calculating performance metrics like accuracy, precision, recall, and F1-score.

To install these dependencies, run the following command:

```sh
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```
## 📌Required Imports and Libraries
```sh
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
##  📁Project Structure
```sh
sentiment-analysis/
│
├── SentimentModel.ipynb        # Jupyter notebook with training and evaluation
├── IMDB Dataset.csv            # Dataset file (50,000 reviews)
├── README.md                   # Project documentation
└── requirements.txt            # Dependency list

```
## 🔄Dara Preprocessing

Label Encoding:
```sh
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
```
Tokenization and Padding:
```sh
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=200)
```  
## 🏗️Model Architectures Used

- `Embedding Layer`: Converts each word into a dense vector of fixed size (128).

- `LSTM Layer`: 128 memory units to capture sequential patterns.

- `Dense Layer`: Final sigmoid-activated layer for binary classification.
```sh
model = Sequential()
model.add(Embedding(5000, 128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```

## 🎯Training Details

- Optimizer: Adam

- Loss: Binary Crossentropy

- Metrics: Accuracy

- Epochs: 5–10 (configurable)

- Batch Size: 64

## 📊Performance Metrics

- Best Model: LSTM-based sentiment classifier (trained on IMDb dataset and evaluated based on highest validation accuracy)

- Metrics Used: Accuracy (monitored during training and validation)

- Accuracy and Validation Loss: Displayed during training over each epoch using matplotlib for visual interpretation

- Performance can be further evaluated using:

- Confusion matrix (generated using scikit-learn)

- Precision, recall, F1-score (calculated via classification_report from sklearn.metrics)

- Sample predictions on random test sentences for qualitative validation of the model's sentiment classification

## ▶️Running the app

This is currently a Jupyter Notebook-based project. To run the code:

```sh
jupyter notebook SentimentModel.ipynb
```
You can visualize results through:

- Model accuracy/loss plots

- Confusion matrix

- Classification report

## 🖼️Sample Output

| Review                                 | Sentiment |
| -------------------------------------- | --------- |
| "I loved this movie, great acting!"    | Positive  |
| "It was a waste of time, very boring." | Negative  |

## 🚀Future Work

- Deploy as a Streamlit or Flask web app.

- Integrate with real-time data sources like Twitter or product reviews.

- Experiment with Bidirectional LSTM or Transformer-based models.

- Hyperparameter tuning for performance boost.

- Add sentiment visualization using word clouds or attention maps.

