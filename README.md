# WhatTheGlove-

This project aims to classify news articles into five categories: sport, business, tech, entertainment, and politics. It utilizes GloVe embeddings for text representation and employs a neural network model for classification.

Head over to https://nlp.stanford.edu/data/glove.6B.zip to get access to GloVe embeddings.

Technologies Used 🛠️

TensorFlow
Keras
pandas
NumPy
Matplotlib
Seaborn
NLTK
Dataset Exploration 📊

The dataset consists of news articles with corresponding categories. Key steps in exploring the dataset include:

Loading and examining the data
Handling missing values
Visualizing the distribution of categories using pie chart
Data Preprocessing 🧹

Data preprocessing involves:

Extracting text and labels
One-hot encoding categorical labels
Tokenizing text data and converting to sequences of integers
Padding sequences to ensure uniform length
Splitting the data into training and validation sets
GloVe Embeddings 🌐

The project utilizes pre-trained GloVe word embeddings for text representation:

Loading GloVe embeddings from a file
Creating an embedding matrix for words in the vocabulary
Initializing an embedding layer with GloVe embeddings in the neural network model
Neural Network Training 🧠

The neural network training process includes:

Implementing k-fold cross-validation for hyperparameter tuning
Experimenting with different learning rates and optimizers
Defining the neural network model architecture
Compiling the model with appropriate loss function and metrics
Training the model with training data and validating with validation data
Saving the best model based on validation accuracy using ModelCheckpoint callback
Results 📊

The trained model achieves a training accuracy of 1.00 and a validation accuracy of 0.91

Acknowledgments 🙏

Special thanks to GloVe for providing pre-trained word embeddings.
