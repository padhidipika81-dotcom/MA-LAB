import numpy as np

alpha = [1, 1, 1]
y = [1, -1, -1]

sv = [[1, 0, -1],
      [-1, 2, 0],
      [0, -1, -1]]

x = [0.5, 1, -0.5]

f = 0

for i in range(3):
    f += alpha[i] * y[i] * np.dot(sv[i], x)

print("Decision value:", f)
print("Predicted class:", np.sign(f))
