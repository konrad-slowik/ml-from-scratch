import pandas as pd
import numpy as np


raw_data = {
    'area': [45.0, 60.5, np.nan, 85.0, 120.0, 55.0],
    'rooms': [2, 3, 3, np.nan, 5, 2],
    'condition': ['good', 'needs_renovation', 'good', 'premium', 'premium', 'good'],
    'price_k': [400, 480, 500, 750, 1200, 450] 
}

df = pd.DataFrame(raw_data)

df['area'] = df['area'].fillna(df['area'].mean())
df['rooms'] = df['rooms'].fillna(df['rooms'].median())

df = pd.get_dummies(df, columns=['condition'], drop_first=True, dtype=int)

df['area'] = ((df['area'] - df['area'].mean()) / df['area'].std())
df['rooms'] = ((df['rooms'] - df['rooms'].mean()) / df['rooms'].std())

y = df['price_k'].values
X = df.drop('price_k', axis=1).values

ones = np.ones(X.shape[0])

X = np.c_[ones, X]

print(X.shape)


W = np.random.rand(X.shape[1])

epochs = 50000
learning_rate = 0.01
n = X.shape[0]

for i in range(epochs):
    
    y_hat = X @ W

    loss = np.mean(((y_hat - y) **2))

    gradient = (2/n) * (X.T @ (y_hat - y))
    
    W = W - (gradient * learning_rate)

    if i % 1000 == 0:
        print(f"Epoch {i} Loss: {loss:.2f}")


print(f"Weights: {W}")

final_predictions = X @ W
print(f"\nReal prices: {y}")
print(f"Predicted prices after learning: {final_predictions}")