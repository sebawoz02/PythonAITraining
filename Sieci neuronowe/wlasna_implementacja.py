import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


# użyta funkcja aktywacji to sigmoida
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_deriative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


# funkcja trenujaca model
def train(x, y, n_hidden, learning_rate, n_iter):
    m, n_input = x.shape
    w1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    w2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))
    """
     W kazdej iteracji wszystkim warstwom przypisywane są ostatnio wyliczone wagi i obciążenia
     Nastepnie wyliczane są gradienty, które w procesie propagacji wstecz 
     są wykorzystywane do modyfikowania wag i obciazen.
    """
    for i in range(1, n_iter + 1):
        z2 = np.matmul(x, w1) + b1
        a2 = sigmoid(z2)
        z3 = np.matmul(a2, w2) + b2
        a3 = z3
        dz3 = a3 - y
        dw2 = np.matmul(a2.T, dz3)
        db2 = np.sum(dz3, axis=0, keepdims=True)
        dz2 = np.matmul(dz3, w2.T) * sigmoid_deriative(z2)
        dw1 = np.matmul(x.T, dz2)
        db1 = np.sum(dz2, axis=0)
        w2 = w2 - learning_rate * dw2 / m
        b2 = b2 - learning_rate * db2 / m
        w1 = w1 - learning_rate * dw1 / m
        b1 = b1 - learning_rate * db1 / m
        if i % 100 == 0:
            cost = np.mean((y - a3) ** 2)
            print(f'Iteracja: {i}, strata: {cost}')
    model = {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
    return model


def predict(x, model):
    w1 = model['W1']
    b1 = model['b1']
    w2 = model['W2']
    b2 = model['b2']
    a2 = sigmoid(np.matmul(x, w1) + b1)
    a3 = np.matmul(a2, w2) + b2
    return a3


housing = fetch_california_housing()
num_test = 10
scaler = StandardScaler()
X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
Y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = scaler.transform(housing.data[-num_test:, :])
Y_test = housing.target[-num_test:]

n_hidden = 20   # warstwa ukryta ma 20 węzłów
learning_rate = 0.1
n_iter = 2000
model = train(X_train, Y_train, n_hidden, learning_rate, n_iter)

predictions = predict(X_test, model)
print(predictions)
print(Y_test)
