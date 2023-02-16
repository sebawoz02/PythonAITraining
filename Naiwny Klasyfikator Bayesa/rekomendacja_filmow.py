# System rekomendacji filmów na bazie naiwnego klasyfikatora Bayesa napisanego od podstaw
import numpy as np
from collections import defaultdict


def group_labels(labels):
    """
    Funkcja grupująca próbki na podstawie ich etykiet.
    :param labels: lista etykiet
    :return: słownik w którym kluczami są etykiety a wartościami indexy na których się znadują
    """
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


def calculate_a_priori(label_indices):
    """
    Funkcja licząca prawodpodobieństwo a priori na podstawie próbek treningowych.
    :param label_indices: indexy próbek z funkcji group_labels
    :return: słownik w którym kluczami są etykiety, a wartości to prawdopobieństwo a priori
    """
    priori = {label: len(indieces) for label, indieces in label_indices.items()}
    total_count = sum(priori.values())
    for label in priori:
        priori[label] /= total_count
    return priori


def calculate_likelihood(features, label_indices, smoothing=0):
    """
    Wyliczanie szansy na podstawie próbek treningowych.
    :param features: Macierz cech
    :param label_indices: indexy probek pogupowane wedlug etykiet ( klas )
    :param smoothing: wspołczynnik wygładzający
    :return: słownik w których kluczem jest etykieta, a wartością wektor P( cecha | klasa )
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood


def calculate_a_posteriori(x_test, priori, likelihood):
    """
    Funkcja licząca a posteriori na podstawie prori i likelihood
    :param x_test: próbki treningowe
    :param priori: słownik z etykietami klasy oraz przypisanym do nich prawdopodobienstwem a priori
    :param likelihood: słownik z etykietami klasy oraz przypisanymi wektorami P( cecha | klasa )
    :return: słownik z etykietami klasy oraz przypisanym prawodpodobieństwem a posteriori
    """
    posteriors = []
    for x in x_test:
        posterior = priori.copy()
        for label, likelihood_val in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_val[index] if bool_value else (1 - likelihood_val[index])
        # Normalizacja
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


X_train = np.array([        # przykładowe małe dane treningowe o cechah X = [x1, x2, x3]
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
                [1, 1, 0]])
Y_train = ['Y', 'N', 'Y', 'Y']      # Y - film sie podobal, N - film sie nie podobal
X_test = np.array([[1, 1, 0]])      # dane testowe, P(T | X_test) = 1458/1583

label_idx = group_labels(Y_train)
a_priori = calculate_a_priori(label_idx)

smoothing = 1
likelihood = calculate_likelihood(X_train, label_idx, smoothing)

posteriori = calculate_a_posteriori(X_test, a_priori, likelihood)
print(posteriori)
