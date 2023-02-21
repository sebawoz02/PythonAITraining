import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

"""Przeksztalcenie danych"""


def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


def add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']


def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']


def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']


def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)


def generate_features(df):
    """
    Funkcja generujaca cechy na podstawie histroycznych wartosci indeksu i jego zmiennosci
    :param df: obiekt DataFrame zawierajacy kolumny 'Open', 'Close', 'High', 'Low', 'Volume', 'Adjusted Close'
    :return: Obiekt DataFrame zawierajcy zbior danych z nowymi cechami
    """
    df_n = pd.DataFrame()
    # 6 oryginalnych cech
    add_original_feature(df, df_n)
    # 31 wygenerowanych cech
    add_avg_volume(df, df_n)
    add_avg_price(df, df_n)
    add_std_price(df, df_n)
    add_std_volume(df, df_n)
    add_return_feature(df, df_n)
    df_n['close'] = df['Close']
    df_n = df_n.dropna(axis=0)
    return df_n


raw_data = pd.read_csv("19880101_20230118.csv", index_col='Date')
data = generate_features(raw_data)
start_train = '1988-01-04'
end_train = '2021-12-31'
start_test = '2022-01-01'
end_test = '2023-01-18'
data_train = data.loc[start_train:end_train]
X_train = data_train.drop('close', axis=1).values
Y_train = data_train['close'].values
data_test = data.loc[start_test:end_test]
X_test = data_test.drop('close', axis=1).values
Y_test = data_test['close'].values

# Skalowanie danych
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
# regresja liniowa
# Przeszukiwanie siatki w celu znalezienia najlepszych parametrow
param_grid = {
    "alpha": [1e-4, 3e-4, 1e-3],
    "eta0": [0.01, 0.03, 0.1]
}
lr = SGDRegressor(penalty='l2', max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled_train, Y_train)
print(grid_search.best_params_)
lr_best = grid_search.best_estimator_
predictions_lr = lr_best.predict(X_scaled_test)
print(f"Blad sredniokwadratowy: {mean_squared_error(Y_test, predictions_lr):.3f}")
print(f"Blad bezwzgledny: {mean_absolute_error(Y_test, predictions_lr):.3f}")
print(f"R^2: {r2_score(Y_test, predictions_lr):.3f}")
# {'alpha': 0.0001, 'eta0': 0.03}
# Blad sredniokwadratowy: 67063.325
# Blad bezwzgledny: 211.560
# R^2: 0.962

# las losowy

param_grid = {
    'max_depth': [30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [3, 5]
}
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
rf_best = grid_search.best_estimator_
predictions_rf = rf_best.predict(X_test)

print(f"Blad sredniokwadratowy: {mean_squared_error(Y_test, predictions_rf):.3f}")
print(f"Blad bezwzgledny: {mean_absolute_error(Y_test, predictions_rf):.3f}")
print(f"R^2: {r2_score(Y_test, predictions_rf):.3f}")

# {'max_depth': 30, 'min_samples_leaf': 3, 'min_samples_split': 10}
# Blad sredniokwadratowy: 84691.299
# Blad bezwzgledny: 237.754
# R^2: 0.952


# regresja wektorow nosnych

param_grid = [{'kernel': ['linear'], 'C': [100, 300, 500], 'epsilon': [0.00003, 0.0001]},
              {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 100, 1000], 'epsilon': [0.00003, 0.0001]}]
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_scaled_train, Y_train)
print(grid_search.best_params_)
svr_best = grid_search.best_estimator_
predictions_svr = svr_best.predict(X_scaled_test)
print(f"Blad sredniokwadratowy: {mean_squared_error(Y_test, predictions_svr):.3f}")
print(f"Blad bezwzgledny: {mean_absolute_error(Y_test, predictions_svr):.3f}")
print(f"R^2: {r2_score(Y_test, predictions_svr):.3f}")

plt.plot(data_test.index, Y_test, c='k')
plt.plot(data_test.index, predictions_lr, c='b')
plt.plot(data_test.index, predictions_rf, c='r')
plt.plot(data_test.index, predictions_svr, c='g')
plt.xticks(range(0, 252, 10), rotation=60)
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia dnia')
plt.legend(['Wartosci rzeczywiste', 'Regresja liniowa', 'Las losowy', 'Regresja wektorów nośnych'])
plt.show()

# {'C': 500, 'epsilon': 0.0001, 'kernel': 'linear'}
# Blad sredniokwadratowy: 43647.545
# Blad bezwzgledny: 169.576
# R^2: 0.975
