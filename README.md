# WhatTheGlove-

<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/50px-TensorFlowLogo.svg.png" alt="TensorFlow" width="50" height="50"/>
</a>

<a href="https://keras.io/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/50px-Keras_logo.svg.png" alt="Keras" width="50" height="50"/>
</a>

<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/50px-Pandas_logo.svg.png" alt="pandas" width="50" height="50"/>
</a>

<a href="https://numpy.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/NumPy_logo.svg/50px-NumPy_logo.svg.png" alt="NumPy" width="50" height="50"/>
</a>

<a href="https://matplotlib.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/50px-Created_with_Matplotlib-logo.svg.png" alt="Matplotlib" width="50" height="50"/>
</a>

<a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer">
    <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn" width="150" height="50"/>
</a>

<a href="https://www.nltk.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/NLTK_logo.svg/50px-NLTK_logo.svg.png" alt="NLTK" width="50" height="50"/>
</a>


# Project Overview

This project aims to classify news articles into five categories: sport, business, tech, entertainment, and politics. It utilizes GloVe embeddings for text representation and employs a neural network model for classification.

## Technologies Used 🛠️
- TensorFlow
- Keras
- pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK

## Instructions for GloVe Embeddings
Head over to [this link](https://nlp.stanford.edu/data/glove.6B.zip) to get access to GloVe embeddings.

## Dataset Exploration 📊
The dataset consists of news articles with corresponding categories. Key steps in exploring the dataset include:
- Loading and examining the data
- Handling missing values
- Visualizing the distribution of categories using pie chart

## Data Preprocessing 🧹
Data preprocessing involves:
- Extracting text and labels
- One-hot encoding categorical labels
- Tokenizing text data and converting to sequences of integers
- Padding sequences to ensure uniform length
- Splitting the data into training and validation sets

## GloVe Embeddings 🌐
The project utilizes pre-trained GloVe word embeddings for text representation:
- Loading GloVe embeddings from a file
- Creating an embedding matrix for words in the vocabulary
- Initializing an embedding layer with GloVe embeddings in the neural network model

## Neural Network Training 🧠
The neural network training process includes:
- Implementing k-fold cross-validation for hyperparameter tuning
- Experimenting with different learning rates and optimizers
- Defining the neural network model architecture
- Compiling the model with appropriate loss function and metrics
- Training the model with training data and validating with validation data
- Saving the best model based on validation accuracy using ModelCheckpoint callback

## Results 📊
The trained model achieves high accuracy in classifying news articles into their respective categories.

## Future Work 🔍
- Fine-tuning hyperparameters for improved performance
- Exploring other GloVe embeddings to try for better accuracies
- Deploying the model for real-time classification of news articles

## Acknowledgments 🙏
Thanks to GloVe for providing pre-trained word embeddings.
