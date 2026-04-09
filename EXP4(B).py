import numpy as np

# Alpha values
alpha = np.array([1, 1, 1])

# Support vectors
support_vectors = np.array([
    [0, 1, 1],
    [0, 2, -1],
    [-1, 0, 2]
])

# Class labels
y = np.array([1, -1, -1])

# Input vector
x = np.array([0.2, 0.1, 0.4])

# Decision function
f = 0

for i in range(len(alpha)):
    dot_product = np.dot(support_vectors[i], x)
    contribution = alpha[i] * y[i] * dot_product
    f += contribution

print("Decision function value:", f)

predicted_label = np.sign(f)
print("Predicted label:", int(predicted_label))
