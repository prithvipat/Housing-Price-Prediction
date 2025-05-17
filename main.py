# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# Load data
data = pd.read_csv('temp.csv')

X = np.array(data[['area','bedrooms', 'bathrooms', 'stories']].values) # Data with 2 features
y = data['price'].values # Actual Price

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

w = np.zeros(X.shape[1]) # Weights initially set to 0
b = 0 # Bias initially set to 0
lr = 0.01 # Learning rate
n = X.shape[0] # Number of samples
losses = []


def fit(X, w, y, b, lr, n):
    for i in range(10000):
        y_pred = np.dot(X,w) + b

        dw = (1/n) * np.dot(X.T, (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)
        w = w - lr * dw
        b = b - lr * db

        loss = (1/n) * np.sum((y_pred - y)**2)
        losses.append(loss)  # Save loss for plotting
        
    return w, b


w, b = fit(X, w, y, b, lr, n)

print(w)
print(b)




y_pred = np.dot(X,w)+b
mse = mean_squared_error(y, y_pred)

print("Mean Squared Error:", mse)
r2 = r2_score(y, y_pred)
print("R2 Score:", r2)


# Plot the loss over iterations
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curve (Training Progress)')
plt.show()