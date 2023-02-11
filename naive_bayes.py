# system rekomendacji filmów na podstawie zbioru danych ml-1m , grouplens.org

from sklearn.naive_bayes import MultinomialNB
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_rating_data(data_path, n_users, n_movies):
    """
    Odczytanie ocen z pliku.
    :param data_path: scieżka do pliku z ocenami
    :param n_users: liczba osób
    :param n_movies: liczba ocenionych filmow
    :return: data - oceny [osoba, film], movie_n_rating {id_filmu : liczba_ocen}, movie_id_mapping {id_filmu: index}
    """
    data = np.zeros([n_users, n_movies], dtype=float)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = float(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


data_path = 'ml-1m/ratings.dat'
n_users = 6040
n_movies = 3952

data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)

"""
# Oceny i ich ilosc
values, counts = np.unique(data, return_counts=True)
for value, count in zip(values, counts):
    if value != 0.0:
        print(f"{value} : {count}")
"""
# Najczesciej oceniany film
movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f"Film o ID {movie_id_most} uzyskal {n_rating_most} ocen.")

X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

"""
values, counts = np.unique(Y, return_counts=True)
for value, count in zip(values, counts):
    if value != 0.0:
        print(f"{value} : {count}")
"""
recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f"Liczba pozytywnych probek: {n_pos}, Liczba negatywnych: {n_neg}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = MultinomialNB(alpha=1.0, fit_prior=True)  # wspolczynnik wygladzajacy 1.0
clf.fit(X_train, Y_train)

prediction_proba = clf.predict_proba(X_test)
print(prediction_proba[:10])

prediction = clf.predict(X_test)
print(prediction[:10])

accuracy = clf.score(X_test, Y_test)
print(f"\033[31;1mDokładnosc modelu: {accuracy*100:.1f}%")
