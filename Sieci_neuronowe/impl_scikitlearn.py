from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

import tensorflow as tf
from tensorflow import keras

nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam',
                         learning_rate_init=0.001, random_state=42, max_iter=2000)

housing = fetch_california_housing()
num_test = 10
scaler = StandardScaler()
X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
Y_train = housing.target[:-num_test]
X_test = scaler.transform(housing.data[-num_test:, :])
Y_test = housing.target[-num_test:]

nn_scikit.fit(X_train, Y_train)
predictions = nn_scikit.predict(X_test)
print("Zbior klas testowych")
print(Y_test)
print("Prognozy modelu z scikit-learn")
print(predictions)
print(np.mean((Y_test - predictions) ** 2))     # Błąd sredniokwadratowy prognoz


# implementacja z tensorflow

tf.random.set_seed(42)
model = keras.Sequential([keras.layers.Dense(units=20, activation='relu'),
                          keras.layers.Dense(units=8, activation='relu'),
                          keras.layers.Dense(units=1)])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.02))
model.fit(X_train, Y_train, epochs=40)
predictions = model.predict(X_test)[:, 0]
print("Prognozy modelu z tensorflow")
print(predictions)
print(np.mean((Y_test - predictions) ** 2))

