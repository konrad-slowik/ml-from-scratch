import pandas as pd
import numpy as np


dane_surowe = {
    'metraz': [45.0, 60.5, np.nan, 85.0, 120.0, 55.0],
    'pokoje': [2, 3, 3, np.nan, 5, 2],
    'stan': ['dobry', 'do_remontu', 'dobry', 'premium', 'premium', 'dobry'],
    'cena_tys': [400, 480, 500, 750, 1200, 450] 
}

df = pd.DataFrame(dane_surowe)

df['metraz'] = df['metraz'].fillna(df['metraz'].mean())
df['pokoje'] = df['pokoje'].fillna(df['pokoje'].median())

df = pd.get_dummies(df, columns=['stan'], dtype=int)

y = df['cena_tys'].values
X = df.drop('cena_tys', axis=1).values

jedynki = np.ones(X.shape[0])

X = np.c_[jedynki, X]

liczba_kolumn = X.shape[1]

W = np.random.rand(liczba_kolumn)

epoki = 1000000
learning_rate = 0.0001
n = X.shape[0]

for i in range(epoki):
    
    y_hat = X @ W

    loss = np.mean(((y_hat - y) **2))

    gradient = (2/n) * (X.T @ (y_hat - y))
    
    W = W - (gradient * learning_rate)

    if i % 10000 == 0:
        print(f"Epoka {i} Błąd: {loss:.2f}")


print(f"Ostateczne wagi: {W}")

ostateczne_predykcje = X @ W
print(f"\nPrawdziwe ceny: {y}")
print(f"Przewidywania po nauce: {ostateczne_predykcje}")