import numpy as np

x = np.array([2, 3, 4])
y = np.array([1, 2, 3])

lam = 1

X = np.column_stack([np.ones(len(x)), x])
I = np.eye(X.shape[1])

beta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

intercept, slope = beta[0], beta[1]

print(f"intercept : {intercept:.4f}")
print(f"slope : {slope:.4f}")
print(f"equation y = {intercept:.4f} + {slope:.4f}x")
