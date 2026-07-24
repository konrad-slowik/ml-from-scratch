import pandas as pd
import numpy as np
import os
import math

np.random.seed(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'admission.csv')
df = pd.read_csv(file_path)

X = df.drop('zdal',axis=1).values
y = df['zdal'].values

perm = np.random.permutation(X.shape[0])
X_shuffled = X[perm]
y_shuffled = y[perm] 

split = math.floor(X.shape[0]*0.8)
X_train = X_shuffled[:split] 
X_test = X_shuffled[split:]
y_train = y_shuffled[:split]
y_test = y_shuffled[split:]

mask = np.isnan(X_train[:,0])
hours_studied_mean = np.nanmean(X_train[:,0])
X_train[mask,0] = hours_studied_mean

mask = np.isnan(X_test[:,0])
X_test[mask,0] = hours_studied_mean

mask = np.isnan(X_train[:,1])
exam_score_mean = np.nanmean(X_train[:,1])
X_train[mask,1] = exam_score_mean

mask = np.isnan(X_test[:,1])
X_test[mask,1] = exam_score_mean

hours_studied_std = np.std(X_train[:,0])
exam_score_std = np.std(X_train[:,1])

X_train[:,0] = (X_train[:,0] - hours_studied_mean) / hours_studied_std
X_train[:,1] = (X_train[:,1] - exam_score_mean) / exam_score_std
X_test[:,0] = (X_test[:,0] - hours_studied_mean) / hours_studied_std
X_test[:,1] = (X_test[:,1] - exam_score_mean) / exam_score_std

ones = np.ones(X_train.shape[0])
X_train = np.c_[ones,X_train]
ones = np.ones(X_test.shape[0])
X_test = np.c_[ones,X_test]

W = np.random.rand(X_train.shape[1])

def sigmoid(z):
    prob = 1 / (1+ np.exp(-z))
    return prob

epochs = 10000
n = X_train.shape[0]
learning_rate = 0.001

for i in range(epochs):
    
    z = X_train @ W
    sigmoid_z = sigmoid(z)
    loss = -np.mean(y_train * np.log(sigmoid_z) + (1-y_train) * np.log(1-sigmoid_z))
    if i % 1000 == 0:
        print(f"Loss:{loss} Epoch: {i}")
    gradient = (1/n) * (X_train.T @ (sigmoid_z - y_train)) 
    W = W - gradient * learning_rate

z = X_test @ W
sigmoid_z = sigmoid(z)
y_pred = (sigmoid_z >= 0.5).astype(int)
print(f"Test values: {y_test}")
print(f"Pred values: {y_pred}")

correct_percent = (np.count_nonzero(y_test == y_pred) / X_test.shape[0])*100
print(f"Accuracy: {correct_percent:.2f}%")
