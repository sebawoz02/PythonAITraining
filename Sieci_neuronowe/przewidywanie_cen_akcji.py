import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

data_raw = pd.read_csv("C:/Users/mario/PycharmProjects/pythonAITraining/Algorytmy_regresji/19880101_20230118.csv",
                       index_col='Date')


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


data = generate_features(data_raw)
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
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
"""
model = keras.Sequential([
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1)
])
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
model.fit(X_scaled_train, Y_train, epochs=100, verbose=True)

predictions = model.predict(X_scaled_test)[:, 0]

print(f"Blad sredniokwadratowy: {mean_squared_error(Y_test, predictions):.3f}")
print(f"Blad bezwzgledny: {mean_absolute_error(Y_test, predictions):.3f}")
print(f"R^2: {r2_score(Y_test, predictions):.3f}")
"""
HP_HIDDEN = hp.HParam('hidden_size', hp.Discrete([64, 32, 16]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300, 1000]))
HP_LEARINING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.4))


def train_test_model(hparams, logdir):
    model = keras.Sequential([
        keras.layers.Dense(units=hparams[HP_HIDDEN], activation='relu'),
        keras.layers.Dense(units=1)
    ])
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(hparams[HP_LEARINING_RATE]),
                  metrics=['mean_squared_error'])
    model.fit(X_scaled_train, Y_train, validation_data=(X_scaled_test, Y_test),
              epochs=hparams[HP_EPOCHS], verbose=False,
              callbacks=[
                  keras.callbacks.TensorBoard(logdir),
                  hp.KerasCallback(logdir, hparams),
                  keras.callbacks.EarlyStopping(
                      monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto'
                  )
              ])
    _, mse = model.evaluate(X_scaled_test, Y_test)
    pred = model.predict(X_scaled_test)
    r2 = r2_score(Y_test, pred)
    return mse, r2


def run(hparams, logdir):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams=[HP_HIDDEN, HP_EPOCHS, HP_LEARINING_RATE],
            metrics=[hp.Metric('mean_squared_error', display_name='mse'),
                     hp.Metric('r2', display_name='r2')],
        )
        mse, r2 = train_test_model(hparams, logdir)
        tf.summary.scalar('mean_squared_error', mse, step=1)
        tf.summary.scalar('r2', r2, step=1)


session_num = 1
for hidden in HP_HIDDEN.domain.values:
    for epochs in HP_EPOCHS.domain.values:
        for learning_rate in tf.linspace(HP_LEARINING_RATE.domain.min_value,
                                         HP_LEARINING_RATE.domain.max_value, 5):
            hparams = {
                HP_HIDDEN: hidden,
                HP_EPOCHS: epochs,
                HP_LEARINING_RATE: float("%.2f" % float(learning_rate)),
            }
            run_name = "run-%d" % session_num
            print('---Proba: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(hparams, 'logs/hparam_tuning/' + run_name)
            session_num += 1
