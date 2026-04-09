import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# x & y data
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

# reshape x
x_reshaped = x.reshape(-1,1)

# create regression model
model = LinearRegression()
model.fit(x_reshaped, y)

score = model.coef_[0]
intercept = model.intercept_

print(f"Calculated slope (b1): {score:.4f}")
print(f"Calculated intercept (b0): {intercept:.4f}")

# predicted values
y_pred = model.predict(x_reshaped)

print("Predicted y values (y_pred) using model.predict():")
print(y_pred)

# error calculations
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

r_squared = model.score(x_reshaped, y)

n = len(y)
p = 1

adj_r_squared = 1 - (1 - r_squared)*(n - 1)/(n - p - 1)

print(f"\nMAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R-Squared: {r_squared:.4f}")
print(f"Adj R-Squared: {adj_r_squared:.4f}")

# plot graph
plt.figure(figsize=(8,6))
plt.scatter(x, y, color='blue', label='Actual Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line')

plt.title("Linear Regression with Data Points")
plt.xlabel("X values")
plt.ylabel("Y values")

plt.legend()
plt.grid(True)
plt.show()
