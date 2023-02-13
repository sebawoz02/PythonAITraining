from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as metrics

cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target

n_pos = (Y == 1).sum()  # probki pozytywne
n_neg = (Y == 0).sum()  # probki negatywne

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = SVC(kernel='linear', C=1.0, random_state=42, gamma='auto', tol=0.001)  # maszyna wektorow nosnych
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print("Rozpoznawanie raka piersi.")
print(f"Dokładność modelu: {accuracy*100:.1f}%")

wine_data = load_wine()
X = wine_data.data
Y = wine_data.target

n_class1 = (Y == 0).sum()
n_class2 = (Y == 1).sum()
n_class3 = (Y == 2).sum()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = SVC(kernel='linear', C=1.0, random_state=42, gamma='auto', tol=0.001)
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print("Rozpoznawanie rodzaju wina.")
print(f"Dokładność modelu: {accuracy*100:.1f}%")

pred = clf.predict(X_test)
print(metrics.classification_report(Y_test, pred))
