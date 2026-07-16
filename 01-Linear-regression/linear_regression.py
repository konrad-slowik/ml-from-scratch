import pandas as pd
import numpy as np
import math
import os

np.random.seed(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'apartments.csv')
df = pd.read_csv(file_path)

df = pd.get_dummies(df, columns=['condition'], drop_first=True, dtype=int)

y = df['price_k'].values
X = df.drop('price_k', axis=1).values


perm = np.random.permutation(X.shape[0])
X_shuffled = X[perm]
y_shuffled = y[perm]

split = math.floor(X.shape[0]*0.8)
X_train = X_shuffled[:split]
X_test = X_shuffled[split:]
y_train = y_shuffled[:split]
y_test = y_shuffled[split:]

mask = np.isnan(X_train[:, 0])
area_mean_train = np.nanmean(X_train[:, 0])
X_train[mask, 0] = area_mean_train

mask = np.isnan(X_train[:, 1])
room_median_train = np.nanmedian(X_train[:, 1])
X_train[mask, 1] = room_median_train

mask = np.isnan(X_test[:, 0])
X_test[mask, 0] = area_mean_train

mask = np.isnan(X_test[:, 1])
X_test[mask, 1] = room_median_train

area_std_train =  np.nanstd(X_train[:,0])
room_std_train =  np.nanstd(X_train[:,1])

room_mean_train = np.nanmean(X_train[:,1])

X_train[:,0] = (X_train[:,0] - area_mean_train) / area_std_train  
X_train[:,1] = (X_train[:,1] - room_mean_train) / room_std_train  
X_test[:,0] = (X_test[:,0] - area_mean_train) / area_std_train  
X_test[:,1] = (X_test[:,1] - room_mean_train) / room_std_train  

ones = np.ones(X_train.shape[0])
X_train = np.c_[ones, X_train]

W = np.random.rand(X_train.shape[1])

epochs = 10000
learning_rate = 0.01
n = X_train.shape[0]

for i in range(epochs):
    
    y_hat = X_train @ W
    loss_train = np.mean(((y_hat - y_train) **2))
    gradient = (2/n) * (X_train.T @ (y_hat - y_train))
    W = W - (gradient * learning_rate)

    if i % 100 == 0:
        print(f"Epoch {i} Loss: {loss_train:.2f}")

print(f"Weights: {W}")

ones = np.ones(X_test.shape[0])
X_test = np.c_[ones, X_test]

y_hat = X_test @ W
loss_test = np.mean(((y_hat - y_test) **2))
mape = np.mean(np.abs((y_test - y_hat) / y_test)) * 100

print(f"Loss: {loss_test}")
print(f"Real prices: {y_test}")
print(f"Predicted prices after learning {y_hat}")
print(f"MAPE: {mape:.2f}%")