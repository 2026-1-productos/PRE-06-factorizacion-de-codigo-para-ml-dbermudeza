"""Script para entrenar y guardar el modelo ElasticNet."""

import pickle

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

# Descarga de datos
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Preparación de datos
y = df["quality"]
x = df.copy()
x.pop("quality")

# Dividir los datos en entrenamiento y testing
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=123456,
)

# Entrenar el modelo
estimator = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=12345)
estimator.fit(x_train, y_train)

# Guardar el modelo
with open("models/estimator.pkl", "wb") as f:
    pickle.dump(estimator, f)
print("Modelo entrenado y guardado en models/estimator.pkl")
