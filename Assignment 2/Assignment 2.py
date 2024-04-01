# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:25:48 2024

@author: barab
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

#Problem 1

df= pd.read_csv("C:/Users/barab/OneDrive/Documents/McGill MMA/Courses/MGSC 695/advertising.csv")

X = df[['TV', 'Radio', 'Newspaper']].values
y = df['Sales'].values

model = gp.Model('L1Regression')

# Add variables
beta = model.addVars(X.shape[1], name='beta')
beta_0 = model.addVar(name='beta_0')

# Add absolute deviation variables
u = model.addVars(len(y), name='u', vtype=GRB.CONTINUOUS)

# Constraints
for i in range(len(y)):
    model.addConstr(gp.quicksum(beta[j] * X[i, j] for j in range(X.shape[1])) + beta_0 + u[i] >= y[i])
    model.addConstr(-(gp.quicksum(beta[j] * X[i, j] for j in range(X.shape[1])) + beta_0) + u[i] >= -y[i])

# Set objective function (minimize sum of absolute deviations)
model.setObjective(gp.quicksum(u[i] for i in range(len(y))), GRB.MINIMIZE)

# Optimize
model.optimize()

# Display results 
print('Optimal coefficients:')
for j in range(X.shape[1]):
    print(f'beta_{j+1}: {beta[j].X:.6f}')

print(f'beta_0: {beta_0.X:.6f}')


#Problem 2
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from gurobipy import Model, GRB

# Generate dataset
X, y = make_blobs(n_samples=195, centers=2, n_features=2, random_state=42)

# Convert y to -1 and 1
y = 2 * y - 1

# Visualize the data
plt.figure(figsize=(4, 3))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='Class -1')
plt.title("Scatter Plot of Simulated Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

def find_optimal_linear_classifier(X, y):
    model = Model("LinearClassifier")

    # Add variables
    intercept = model.addVar(name="B0")
    coefficient = model.addVars(2, name="B1")
    slack = model.addVars(len(X), lb=0, name="slack")

    # Objective function
    model.setObjective(sum(coefficient[i]*coefficient[i] for i in range(2)) + sum(slack[i] for i in range(len(X))), GRB.MINIMIZE)

    # Add constraints
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        model.addConstr(yi * (coefficient[0]*xi[0] + coefficient[1]*xi[1] + intercept) >= 1 - slack[i], name=f"constraint_{i}")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        return intercept.X, [coefficient[0].X, coefficient[1].X], model.ObjVal
    else:
        return None, None, None

# Optimal linear classifier
intercept, betas, optimal_value = find_optimal_linear_classifier(X, y)

# Results
if intercept is not None and betas is not None:
    print("Intercept (B0):", intercept)
    print("Betas (B1):", betas)
    print("Optimal Value (Objective Function):", optimal_value)
    print("Linear Classifier Function: y = {:.2f} + {:.2f} * x1 + {:.2f} * x2".format(intercept, betas[0], betas[1]))
else:
    print("No feasible solution found.")
    
import matplotlib.pyplot as plt

# Given results from the optimization
intercept = 1.3753370137125657e-11
betas = [1.0752862747425174, 1.4644115138024693e-12]

# Generate dataset again
X, y = make_blobs(n_samples=195, centers=2, n_features=2, random_state=42)
y = 2 * y - 1 

# Plot the data points
plt.figure(figsize=(5, 4))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='Class -1')

x_decision_boundary = -intercept / betas[0]

# Plot the decision boundary
plt.axvline(x=x_decision_boundary, color="green", label='Decision Boundary')

# Set plot limits
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot with Decision Boundary')
plt.legend()
plt.show()
