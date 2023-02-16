import numpy as np


def gini_impurity(labels):
    if labels.size == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)


def entropy(labels):
    if labels.size == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return np.sum(fractions * np.log2(fractions))


criterion_function = {'gini': gini_impurity,
                      'entropy': entropy}


def weighted_impurity(groups, criterion='gini'):
    """
    Funkcja wyliczajace ważone zanieczyszczenie zbioru po podziale
    :param groups: lista węzłów potomnych z których kazdy zawiera liste etykiet i klas
    :param criterion: 'gini' oznacza zanieczyszczenie Giniego, a 'entropy' przyrost informacji
    :return: float, weighted impurity
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum


def split_node(x, y, index, value):
    """
    Funkcja dzielaca zbior x na podstawie cechy i wartosci
    :param x: numpy.ndarray, cechy zbioru
    :param y: numpy.ndarray, docelowy zbior
    :param index: int, indeks cechy wykorzystywanej do dzielenia
    :param value: wartosc cechy wykorzystywanej do dzielenia
    :return: dwie listy reprezentujace wezly potomne lewy i prawy
    """
    x_index = x[:, index]
    # cecha liczbowa
    if x[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= value
    else:
        mask = x_index == value
    left = [x[~mask, :], y[~mask]]
    right = [x[mask, :], y[mask]]
    return left, right


def get_best_split(x, y, criterion):
    """
    Funkcja wyszukujaca najlepszy punkt podzialu zbioru x, y
    :param x: numpy.ndarray, cechy zbioru
    :param y: numpy.ndarray, docelowy zbior
    :param criterion: kryterium gini lub entropy
    :return: dict {index: indeks cechy, value: wartosc cechy,
    children: wezly potomne lewy i prawy}
    """
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len(x[0])):
        for value in np.sort(np.unique(x[:, index])):
            groups = split_node(x, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}


def get_leaf(labels):
    return np.bincount(labels).argmax()


def split(node, max_depth, min_size, depth, criterion):
    """
    Funkcja dzieląca węzeł lub przypisująca mu końcową etykietę
    :param node: słownik z informacjami o węzle
    :param max_depth: maksymalna glebokosc drzewa
    :param min_size: minimalna liczba probek wymagana do podzialu
    :param depth: glebokosc aktualnego wezla
    :param criterion: kryterium gini lub entropy
    """
    left, right = node['children']
    del (node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1, criterion)
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        result = get_best_split(right[0], right[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['right'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth + 1, criterion)


def train_tree(x_train, y_train, max_depth, min_size, criterion='gini'):
    """
    Funkcja inicjująca budowanie drzewa
    :param x_train: lista probek treningowych - cechy
    :param y_train: lista probek treningowych - cel
    :param max_depth: int, maksymalna glebokosc drzewa
    :param min_size: int, minimalna liczba probek wymagana do podzialu wezla
    :param criterion: kryterium gini lub entropy
    """
    x = np.array(x_train)
    y = np.array(y_train)
    root = get_best_split(x, y, criterion)
    split(root, max_depth, min_size, 1, criterion)
    return root


CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'to', 'no': 'to nie'}}


def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if node['value'].dtype.kind in ['i', 'f']:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print(f"{depth * ' '}[{node}]")


X_train = [['technika', 'specjalista'],
           ['moda', 'student'],
           ['sport', 'student'],
           ['moda', 'specjalista'],
           ['technika', 'student'],
           ['technika', 'emeryt'],
           ['sport', 'specjalista']]
Y_train = [1, 0, 0, 0, 1, 0, 1]
tree = train_tree(X_train, Y_train, 2, 2)
visualize_tree(tree)

X_train_n = [[6, 7], [2, 4],
             [7, 2], [3, 6],
             [4, 7], [5, 2],
             [1, 6], [2, 0],
             [6, 3], [4, 1]]
Y_train_n = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
tree_n = train_tree(X_train_n, Y_train_n, 2, 2)
visualize_tree(tree_n)
