from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import multilabel_confusion_matrix

digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
Y = digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
parameters = {'penalty': ['l2', None],
              'alpha': [1e-07, 1e-06, 1e-05, 1e-04],
              'eta0': [0.01, 0.1, 1, 10],
              'max_iter': [50, 70, 100]}
sgl_lr = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, fit_intercept=True)
grid_search = GridSearchCV(sgl_lr, parameters, n_jobs=-1, cv=3)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)

sgd_lr_best = grid_search.best_estimator_
accuracy = sgd_lr_best.score(X_test, Y_test)
print(f"Dokładność modelu dla zbioru testowego: {accuracy*100:.1f}%")

pred = sgd_lr_best.predict(X_test)

print(multilabel_confusion_matrix(Y_test, pred))
