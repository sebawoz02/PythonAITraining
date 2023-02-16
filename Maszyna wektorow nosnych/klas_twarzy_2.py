from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

face_data = fetch_lfw_people(min_faces_per_person=50)
X = face_data.data
Y = face_data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

svc = LinearSVC(class_weight='balanced', random_state=42, dual=True, max_iter=100000)
pca = PCA(n_components=100, whiten=True, random_state=42)
model = Pipeline([('pca', pca), ('svc', svc)])

parameters = {
    'svc__C': (1e-3, 1e-2, 1e-1, 1, 10)
}
grid_search = GridSearchCV(model, parameters, n_jobs=-1)
grid_search.fit(X_train, Y_train)

print('Najlepszy model:\n', grid_search.best_params_)
print('Najlepsza srednia skutecznosc:', grid_search.best_score_)

best_svc = grid_search.best_estimator_
print(f"Dokładność: {best_svc.score(X_test, Y_test) * 100:.1f}%")
pred = best_svc.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))
