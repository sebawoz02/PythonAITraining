
import numpy as np
from sklearn import datasets

"""Implementacja regresji liniowej"""


def compute_prediction(x, weights):
    """
    Funkcja wyliczajaca prognoze y_hat na podstawie biezacych wag
    """
    predictions = np.dot(x, weights)
    return predictions


def update_weights_gd(x_train, y_train, weights, learning_rate):
    """
    Funkcja zwracajaca zaktualizowane wagi w jednym kroku
    """
    predictions = compute_prediction(x_train, weights)
    weights_delta = np.dot(x_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights


def compute_cost(x, y, weights):
    """
    Funkcja wyliczajaca koszt J(w)
    """
    predictions = compute_prediction(x, weights)
    cost = np.mean((predictions - y) ** 2 / 2.0)
    return cost


def train_linear_regression(x_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """
    Funkcja trenujaca model regresji liniowej z wykorzystaniem
    gradientu prostego i zwracajaca przetrenowany model
    """
    if fit_intercept:
        intercept = np.ones((x_train.shape[0], 1))
        x_train = np.hstack((intercept, x_train))
    weights = np.zeros(x_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(x_train, y_train, weights, learning_rate)
        if iteration % 100 == 0:
            print(compute_cost(x_train, y_train, weights))
    return weights


def predict(x, weights):
    if x.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((x.shape[0], 1))
        x = np.hstack((intercept, x))
    return compute_prediction(x, weights)


"""
import matplotlib.pyplot as plt
# test modelu na malym zbiorze danych
X_train = np.array([[6], [2], [3], [4], [1], [5], [2], [6], [4], [7]])
Y_train = np.array([5.5, 1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8])
model = train_linear_regression(X_train, Y_train, max_iter=100, learning_rate=0.01, fit_intercept=True)
X_test = np.array([[1.3], [3.5], [5.2], [2.8]])
predctions = predict(X_test, model)
plt.scatter(X_train[:, 0], Y_train, marker='o', c='b')
plt.scatter(X_test[:, 0], predctions, marker='*', c='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""

diabetes = datasets.load_diabetes()
num_test = 30
X_train = diabetes.data[:-num_test, :]
Y_train = diabetes.target[:-num_test]
model = train_linear_regression(X_train, Y_train, max_iter=5000, learning_rate=1, fit_intercept=True)
X_test = diabetes.data[-num_test:, :]
Y_test = diabetes.target[-num_test:]
predictions = predict(X_test, model)
print(Y_test)
print("Prognoza modelu wlasnego:")
print(predictions)


# ta sama implementacja z uzyciem modelu z pakieru scikit-learn
from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001,
                         learning_rate='constant', eta0=0.01, max_iter=1000)
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_test)
print("Prognoza modelu SGDRegressor:")
print(predictions)

# implementacja z uzyciem TensorFlow
import tensorflow as tf
layer0 = tf.keras.layers.Dense(units=1, input_shape=[X_train.shape[1]])
model = tf.keras.Sequential(layer0)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1))
model.fit(X_train, Y_train, epochs=100, verbose=True)
predictions = model.predict(X_test)[:, 0]
print("Prognoza modelu TensorFlow:")
print(predictions)
