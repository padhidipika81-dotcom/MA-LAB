import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x = np.array([1, 2, 3]).reshape(-1, 1)
y1 = np.array([2, 4, 6])
y2 = np.array([3, 5, 7])

Y = np.column_stack((y1, y2))


model = LinearRegression()
model.fit(x, Y)

Y_pred = model.predict(x)

print("Intercepts:", model.intercept_)
print("Coefficients:", model.coef_)

print("\nRegression Equations:")
print(f"y1 = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}x")
print(f"y2 = {model.intercept_[1]:.2f} + {model.coef_[1][0]:.2f}x")


mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("\nPerformance Metrics:")
print("MSE  :", mse)
print("RMSE :", rmse)
print("MAE  :", mae)
print("R^2  :", r2)


plt.figure(figsize=(8, 5))

plt.scatter(x, y1, color='blue', label='Actual y1')
plt.scatter(x, y2, color='green', label='Actual y2')


plt.plot(x, Y_pred[:, 0], color='blue', linestyle='--', label='Predicted y1')
plt.plot(x, Y_pred[:, 1], color='green', linestyle='--', label='Predicted y2')

plt.xlabel("x")
plt.ylabel("Output")
plt.title("Multivariate Linear Regression with Error Metrics")
plt.legend()
plt.grid(True)
plt.show()
