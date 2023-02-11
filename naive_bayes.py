# system rekomendacji filmów na podstawie zbioru danych ml-latest-small , grouplens.org

from sklearn.naive_bayes import BernoulliNB
import numpy as np
from collections import defaultdict


def load_rating_data(data_path, n_users, n_movies):
    """
    Odczytanie ocen z pliku.
    :param data_path: scieżka do pliku z ocenami
    :param n_users: liczba osób
    :param n_movies: liczba ocenionych filmow
    :return: data - oceny [osoba, film], movie_n_rating {id_filmu : liczba_ocen}, movie_id_mapping {id_filmu: index}
    """
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


clf = BernoulliNB(alpha=1.0, fit_prior=True)    # wspolczynnik wygładzający 1.0
data_path = 'ml-latest-small/ratings.csv'
n_users = 6040
n_movies = 3706
