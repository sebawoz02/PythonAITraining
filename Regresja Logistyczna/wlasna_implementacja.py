import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import timeit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier


def sigmoid(inp):
    return 1.0 / (1 + np.exp(-inp))


def compute_prediction(x, weights):
    """
    Funkcja wyliczająca prognozę y_hat z wykorzystaniem bieżących wag
    """
    z = np.dot(x, weights)
    predictions = sigmoid(z)
    return predictions


def update_weights_sgd(x_train, y_train, weights, learning_rate):
    """
    Pojedyncza iteracja modyfikujaca wagi
    """
    for X_each, Y_each in zip(x_train, y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (Y_each - prediction)
        weights += learning_rate * weights_delta
    return weights


def compute_cost(x, y, weights):
    """
    Funkcja wyliczająca wagi J(w)
    """
    predictions = compute_prediction(x, weights)
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    return cost


def train_logistic_regression_sgd(x_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """
    Funkcja trenująca model regresji logistycznej
    :param x_train: numpy.ndarray, treningowy zbior danych ( cechy )
    :param y_train: numpy.ndarray, treningowy zbior danych ( klasy )
    :param max_iter: int, liczba iteracji
    :param learning_rate: float
    :param fit_intercept: bool, flaga z przechwyceniem w0 czy bez niego
    :return: numpy.ndarray, wyliczone wagi
    """
    if fit_intercept:
        intercept = np.ones((x_train.shape[0], 1))
        x_train = np.hstack((intercept, x_train))
    weights = np.zeros(x_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_sgd(x_train, y_train, weights, learning_rate)
        if iteration % 10 == 0:
            print(compute_cost(x_train, y_train, weights))
    return weights


def predict(x, weights):
    if x.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((x.shape[0], 1))
        x = np.hstack((intercept, x))
    return compute_prediction(x, weights)


n_rows = 300000
df = pd.read_csv("C:/Users/mario/PycharmProjects/pythonAITraining/Drzewo decyzyjne/train.csv", nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
n_train = 10000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

start_time = timeit.default_timer()
weights = train_logistic_regression_sgd(X_train_enc.toarray(), Y_train, max_iter=100,
                                        learning_rate=0.01, fit_intercept=True)
print(f"--- {(timeit.default_timer() - start_time):.3f} s ---")

pred = predict(X_test_enc.toarray(), weights)
print(f"Liczba probek treningowych: {n_train}, pole pod krzywą ROC dla zbioru treningowego: "
      f"{roc_auc_score(Y_test, pred):.3f}")

sgd_lr = SGDClassifier(loss='log_loss', penalty=None, fit_intercept=True, max_iter=100,
                       learning_rate='constant', eta0=0.01)
sgd_lr.fit(X_train_enc.toarray(), Y_train)
pred = sgd_lr.predict_proba(X_test_enc.toarray())[:, 1]
print(f"Liczba probek treningowych: {n_train}, pole pod krzywą ROC dla zbioru treningowego: "
      f"{roc_auc_score(Y_test, pred):.3f}")

# selekcja cech w regularyzacji L1
sgd_lr_l1 = SGDClassifier(loss='log_loss', penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=100,
                          learning_rate='constant', eta0=0.01)
sgd_lr_l1.fit(X_train_enc.toarray(), Y_train)
coef_abs = np.abs(sgd_lr_l1.coef_)

