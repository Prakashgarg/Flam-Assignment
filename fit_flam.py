"""
FLAM R&D Assignment — Parameter Estimation Script
Author: Prakash Kumar Garg
Description: Fits θ, M, X in parametric curve equations using L1 loss minimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

df = pd.read_csv('xy_data.csv')
x_obs = df['x'].values
y_obs = df['y'].values
N = len(x_obs)
t = np.linspace(6.0, 60.0, N)

def predict(params):
    theta_deg, M, X = params
    th = np.deg2rad(theta_deg)
    expo = np.exp(M * np.abs(t))
    sin03t = np.sin(0.3 * t)
    x_pred = t * np.cos(th) - expo * sin03t * np.sin(th) + X
    y_pred = 42.0 + t * np.sin(th) + expo * sin03t * np.cos(th)
    return x_pred, y_pred

def l1_loss(params):
    x_pred, y_pred = predict(params)
    return np.sum(np.abs(x_pred - x_obs) + np.abs(y_pred - y_obs))

bounds = [(0.001, 50.0), (-0.05, 0.05), (0.0, 100.0)]
result_global = differential_evolution(l1_loss, bounds, tol=1e-6, maxiter=500, polish=False)
result_local = minimize(l1_loss, result_global.x, bounds=bounds, method='L-BFGS-B')

theta_hat, M_hat, X_hat = result_local.x
print("Estimated Parameters:")
print("Theta (deg):", theta_hat)
print("M:", M_hat)
print("X:", X_hat)

x_fit, y_fit = predict(result_local.x)
plt.figure(figsize=(8,6))
plt.scatter(x_obs, y_obs, s=10, label='Observed Data', alpha=0.7)
plt.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Curve')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('FLAM Assignment: Curve Fitting Result')
plt.grid(True)
plt.tight_layout()
plt.show()
