import numpy as np
from sklearn import svm

x = np.array([[1,1],
              [2,1],
              [2,3],
              [3,3]])

y = np.array([1,1,-1,-1])

model = svm.SVC(kernel='linear')
model.fit(x, y)

w = model.coef_[0]
b = model.intercept_[0]

print("Weight vector:", w)
print("Bias:", b)
