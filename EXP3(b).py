import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([2, 3, 4])
y = np.array([1, 2, 3])

lam = 1

X = np.column_stack([np.ones(len(x1)), x1, x2])

XTX = X.T @ X
XTy = X.T @ y

I = np.eye(X.shape[1])

beta = np.linalg.inv(XTX + lam * I) @ XTy

b0, b1, b2 = beta

print(f"intercept : {b0:.4f}")
print(f"slope x1 : {b1:.4f}")
print(f"slope x2 : {b2:.4f}")

print(f"equation: y = {b0:.4f} + {b1:.4f}x1 + {b2:.4f}x2")
