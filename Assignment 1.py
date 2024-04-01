# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:09:05 2024

@author: barab
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/barab/OneDrive/Documents/McGill MMA/Courses/MGSC 695/advertising.csv")

X = data[['TV', 'Radio', 'Newspaper']]
Y = data['Sales']

### Manual Linear Regression ###
X_manual = np.c_[np.ones(X.shape[0]), X]
beta_star_manual = np.linalg.inv(X_manual.T @ X_manual) @ X_manual.T @ Y

# Parameter estimates
print("Parameter estimates from manual linear regression:")
print(beta_star_manual)

### Scikit-learn Linear Regression ###
model_sklearn = LinearRegression()
model_sklearn.fit(X, Y)

beta_star_sklearn = np.append(model_sklearn.intercept_, model_sklearn.coef_)

# Parameter estimates
print("\nParameter estimates from scikit-learn's Linear Regression:")
print(beta_star_sklearn)


