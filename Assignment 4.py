# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:04:32 2024

@author: barab
"""

#Problem 1(iv)

#Pseudo-code

#1. Initialize β (coefficients) randomly with small values.
#2. Choose a learning rate α (a small positive value, e.g., 0.01).
#3. Choose the number of epochs (an epoch is a full pass through the entire dataset).
#4. Repeat for each epoch until convergence or the maximum number of epochs is reached:
    #a. Shuffle the dataset randomly to ensure diverse mini-batches.
    #b. For each example (x_i, y_i) in the dataset:
        #i. Calculate the predicted probability for the current example:
           #p_i = 1 / (1 + exp(-(β_0 + β_1*x_i1 + ... + β_n*x_in)))
        #ii. Calculate the gradient Δ of the loss function with respect to each β_j:
           #For j = 0 (the intercept, if included):
              #Δβ_j = -(y_i - p_i)
           #For j = 1 to n (each feature's coefficient):
               #Δβ_j = -(y_i - p_i) * x_ij
           #Note: This gradient calculation is for the logistic loss function.
        #iii. Update each coefficient β_j:
             #β_j := β_j - α * Δβ_j
    #c. Optionally, evaluate the performance on a validation set to monitor convergence.
#5. After training, use β to make predictions on new data or test data.


#Problem 2(iii)

import nltk
import numpy as np
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

# Ensure the necessary NLTK datasets are downloaded
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')

# Load the tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Preprocess the tweets
def preprocess_tweet(tweet):
    tokenized = word_tokenize(tweet)
    cleaned = [word.lower() for word in tokenized if word.lower() not in stopwords.words('english')
               and word not in string.punctuation]
    return ' '.join(cleaned)

tweets = [preprocess_tweet(tweet) for tweet in positive_tweets + negative_tweets]
labels = np.array([1] * len(positive_tweets) + [0] * len(negative_tweets))

# Split data
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=0)

# Feature extraction: Combine BoW and TF-IDF features
vectorizer = FeatureUnion([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfVectorizer())
])
X_combined_train = vectorizer.fit_transform(X_train)
X_combined_test = vectorizer.transform(X_test)

# Train a logistic regression model with the combined features
lr = LogisticRegression(max_iter=1000)
lr.fit(X_combined_train, y_train)

# Evaluate the model
y_pred = lr.predict(X_combined_test)
print("Accuracy (Combined BoW and TF-IDF):", accuracy_score(y_test, y_pred))


#Problem 2(iv)

# SGD Logistic Regression Implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_regularized_gradient(X, y, theta, lambda_):
    m = X.shape[0]
    predictions = sigmoid(X.dot(theta))
    gradient = (1/m) * X.T.dot(predictions - y) + (lambda_/m) * theta
    return gradient

# Feature scaling
scaler = StandardScaler(with_mean=False)
X_scaled_train = scaler.fit_transform(X_combined_train)

# Perform the logistic regression using SGD manually with regularization
def sgd_logistic_regression_regularized(X, y, learning_rate=0.01, n_iterations=1000, lambda_=0.1):
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    theta = np.zeros((X_with_intercept.shape[1], 1))
    for i in range(n_iterations):
        idx = np.random.randint(X_with_intercept.shape[0])
        X_i = X_with_intercept[idx:idx+1]
        y_i = y[idx:idx+1].reshape(-1, 1)
        gradients = compute_regularized_gradient(X_i, y_i, theta, lambda_)
        theta -= learning_rate * gradients
    return theta.flatten()

# Set the regularization strength and learning rate
lambda_ = 0.1  # Regularization strength
learning_rate = 0.01  # Learning rate
n_iterations = 10000  # Number of iterations

theta_sgd_reg = sgd_logistic_regression_regularized(X_scaled_train.toarray(), y_train, learning_rate, n_iterations, lambda_)

# Predict and evaluate using the manual SGD model
y_pred_sgd_reg = (sigmoid(np.hstack([np.ones((X_scaled_train.shape[0], 1)), X_scaled_train.toarray()]).dot(theta_sgd_reg)) >= 0.5).astype(int)
print("Manual SGD Accuracy:", accuracy_score(y_train, y_pred_sgd_reg))
print(classification_report(y_train, y_pred_sgd_reg))

# Compare the coefficients
coef_sklearn = lr.coef_.flatten()
intercept_sklearn = lr.intercept_
coef_sgd_reg = theta_sgd_reg[1:]  # Exclude intercept term

# Plot
plt.figure(figsize=(10, 5))
plt.plot(coef_sklearn, label='Scikit-learn Coefficients', marker='o')
plt.plot(coef_sgd_reg, label='Manual SGD Regularized Coefficients', marker='x')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Logistic Regression Coefficients')
plt.legend()
plt.show()

# Comparing the intercept terms
# Extract the intercept from the scikit-learn model
intercept_sklearn = lr.intercept_[0]  # Assuming binary classification

# The first element of theta from the manual SGD model is the intercept
intercept_sgd_reg = theta_sgd_reg[0]

# Print both intercepts for comparison
print("Intercept (scikit-learn):", intercept_sklearn)
print("Intercept (Manual SGD Regularized):", intercept_sgd_reg)
