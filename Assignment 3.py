# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:00:03 2024

@author: barab
"""

#Problem 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

df = pd.read_csv("C:/Users/barab/OneDrive/Documents/McGill MMA/Courses/MGSC 695/sales.csv")

# Visual Inspection: Plotting histogram of the data
plt.hist(df, bins=30, density=True, alpha=0.6, color='g') 
plt.title('Histogram of Sales Data')
plt.show()

# Q-Q plot
probplot(df['DAILY SALES'], dist="norm", plot=plt) 
plt.title('Q-Q Plot of Sales Data')
plt.show()

# Negative Log-Likelihood Function
def neg_log_likelihood(params):
    mu, sigma = params
    likelihood = norm(mu, sigma).logpdf(df['DAILY SALES'])
    return -np.sum(likelihood)

# Gradient Descent Function
def gradient_descent(lr, epochs, start_params):
    params = np.array(start_params)
    for _ in range(epochs):
        # Compute gradients
        mu, sigma = params
        dmu = np.sum((df['DAILY SALES'] - mu) / sigma**2) 
        dsigma = np.sum(((df['DAILY SALES'] - mu)**2 - sigma**2) / sigma**3) 
        
        # Update parameters
        params[0] += lr * dmu / len(df['DAILY SALES']) 
        params[1] += lr * dsigma / len(df['DAILY SALES'])
        if params[1] < 0:  
            params[1] = 0.0001

    return params

# Parameters initialization
initial_params = [np.mean(df['DAILY SALES']), np.std(df['DAILY SALES'])] 

# Run Gradient Descent
optimized_params = gradient_descent(lr=0.001, epochs=1000, start_params=initial_params)

# Print estimated parameters
print(f"Estimated Parameters: mu = {optimized_params[0]}, sigma = {optimized_params[1]}")

nll_value = neg_log_likelihood(optimized_params)
print(f"Negative Log-Likelihood: {nll_value}")

#Problem 2
import numpy as np
import matplotlib.pyplot as plt

# Objective functions
def f(x, y):
    return (x - 5)**2 + 2 * (y + 3)**2 + x * y

def g(x, y):
    return (1 - (y - 3))**2 + 10 * ((x + 4) - (y - 3)**2)**2

# Gradients of the objective functions
def grad_f(x, y):
    df_dx = 2 * (x - 5) + y
    df_dy = 4 * (y + 3) + x
    return np.array([df_dx, df_dy])

def grad_g(x, y):
    dg_dx = 20 * ((x + 4) - (y - 3)**2)
    dg_dy = -2 * (1 - (y - 3)) - 40 * ((x + 4) - (y - 3)**2) * (y - 3)
    return np.array([dg_dx, dg_dy])

# Gradient descent function with learning rate schedule
def gradient_descent(grad, start, learning_rate, lr_schedule, n_steps=50):
    path = [start]
    x = np.array(start, dtype=float)

    for t in range(n_steps):
        current_lr = lr_schedule(learning_rate, t)
        x = x - current_lr * grad(*x)
        path.append(x)

    return np.array(path)

# Learning rate schedules
def constant_lr(learning_rate, iteration):
    return learning_rate

def exp_decay_lr(initial_lr, iteration, decay_rate=0.1):
    return initial_lr * np.exp(-decay_rate * iteration)

def inv_decay_lr(initial_lr, iteration, decay_rate=0.1):
    return initial_lr / (1 + decay_rate * iteration)

# Starting point and learning rates
start_point = [0, 2]
learning_rate_f = 0.05
learning_rate_g = 0.0015

# Apply gradient descent with constant, exponential decay, and inverse decay learning rates for function f
path_f_const = gradient_descent(grad_f, start_point, learning_rate_f, constant_lr)
path_f_exp = gradient_descent(grad_f, start_point, learning_rate_f, lambda lr, t: exp_decay_lr(lr, t, decay_rate=0.1))
path_f_inv = gradient_descent(grad_f, start_point, learning_rate_f, lambda lr, t: inv_decay_lr(lr, t, decay_rate=0.1))

# Apply gradient descent with constant, exponential decay, and inverse decay learning rates for function g
path_g_const = gradient_descent(grad_g, start_point, learning_rate_g, constant_lr)
path_g_exp = gradient_descent(grad_g, start_point, learning_rate_g, lambda lr, t: exp_decay_lr(lr, t, decay_rate=0.1))
path_g_inv = gradient_descent(grad_g, start_point, learning_rate_g, lambda lr, t: inv_decay_lr(lr, t, decay_rate=0.1))

# Calculate objective function values for each path
values_f_const = [f(*coords) for coords in path_f_const]
values_f_exp = [f(*coords) for coords in path_f_exp]
values_f_inv = [f(*coords) for coords in path_f_inv]

values_g_const = [g(*coords) for coords in path_g_const]
values_g_exp = [g(*coords) for coords in path_g_exp]
values_g_inv = [g(*coords) for coords in path_g_inv]

# Plotting the results
plt.figure(figsize=(14, 6))

# Plot for objective function f
plt.subplot(1, 2, 1)
plt.plot(values_f_const, label='Constant LR', color='blue')
plt.plot(values_f_exp, label='Exponential Decay LR', color='orange')
plt.plot(values_f_inv, label='Inverse Decay LR', color='green')
plt.title('Objective Function f(x, y) Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('f(x, y)')
plt.legend()

# Plot for objective function g
plt.subplot(1, 2, 2)
plt.plot(values_g_const, label='Constant LR', color='blue')
plt.plot(values_g_exp, label='Exponential Decay LR', color='orange')
plt.plot(values_g_inv, label='Inverse Decay LR', color='green')
plt.title('Objective Function g(x, y) Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('g(x, y)')
plt.legend()

plt.tight_layout()
plt.show()

# Gradient descent function with learning rate schedule
def gradient_descent(grad, start, learning_rate, lr_schedule, threshold=1e-5, max_steps=1000):
    x = np.array(start, dtype=float)
    prev_value = grad(*x)
    for t in range(max_steps):
        current_lr = lr_schedule(learning_rate, t)
        x = x - current_lr * grad(*x)
        current_value = grad(*x)
        # Check if the change in value is less than the threshold
        if np.linalg.norm(current_value - prev_value) < threshold:
            return t  # Return the number of iterations taken
        prev_value = current_value
    return max_steps

# Function to test all combinations of learning rates and decay rates
def test_lr_combinations(objective_grad, lr_schedule):
    learning_rates = np.linspace(0.001, 0.1, 10)  
    decay_rates = np.linspace(0.01, 0.2, 10) 
    best_lr = None
    best_decay_rate = None
    min_iterations = np.inf

    for lr in learning_rates:
        for decay_rate in decay_rates:
            num_iterations = gradient_descent(
                objective_grad, 
                start=[0, 2], 
                learning_rate=lr, 
                lr_schedule=lambda lr, t: lr_schedule(lr, t, decay_rate)
            )
            if num_iterations < min_iterations:
                min_iterations = num_iterations
                best_lr = lr
                best_decay_rate = decay_rate

    return best_lr, best_decay_rate, min_iterations

# Testing for both functions
best_lr_f_exp, best_decay_rate_f_exp, min_iter_f_exp = test_lr_combinations(grad_f, exp_decay_lr)
best_lr_g_exp, best_decay_rate_g_exp, min_iter_g_exp = test_lr_combinations(grad_g, exp_decay_lr)

best_lr_f_inv, best_decay_rate_f_inv, min_iter_f_inv = test_lr_combinations(grad_f, inv_decay_lr)
best_lr_g_inv, best_decay_rate_g_inv, min_iter_g_inv = test_lr_combinations(grad_g, inv_decay_lr)

# Print the results
print("Function f, Exponential Decay:", best_lr_f_exp, best_decay_rate_f_exp, min_iter_f_exp)
print("Function g, Exponential Decay:", best_lr_g_exp, best_decay_rate_g_exp, min_iter_g_exp)
print("Function f, Inverse Decay:", best_lr_f_inv, best_decay_rate_f_inv, min_iter_f_inv)
print("Function g, Inverse Decay:", best_lr_g_inv, best_decay_rate_g_inv, min_iter_g_inv)

