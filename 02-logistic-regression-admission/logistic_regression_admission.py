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


