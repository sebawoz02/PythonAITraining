from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import timeit
from sklearn.metrics import roc_auc_score

n_rows = 100000 * 11
df = pd.read_csv("C:/Users/mario/PycharmProjects/pythonAITraining/Drzewo decyzyjne/train.csv", nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
n_train = 100000 * 10
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
sgd_lr_online = SGDClassifier(loss='log_loss', penalty=None, fit_intercept=True, max_iter=1, learning_rate='constant',
                              eta0=0.01)
start_time = timeit.default_timer()
for i in range(10):
    x_train = X_train[i * 100000:(i + 1) * 100000]
    y_train = Y_train[i * 100000:(i + 1) * 100000]
    x_train_enc = enc.transform(x_train)
    sgd_lr_online.partial_fit(x_train_enc.toarray(), y_train, classes=[0, 1])
print(f"--- {(timeit.default_timer() -  start_time):.3f} s---")

x_test_enc = enc.transform(X_test)
pred = sgd_lr_online.predict_proba(x_test_enc.toarray())[:, 1]
print(f"Liczba próbek treningowych: {n_train * 10}, pole pod krzywą ROC dla zbioru treningowego: "
      f"{roc_auc_score(Y_test, pred):.3f}")
