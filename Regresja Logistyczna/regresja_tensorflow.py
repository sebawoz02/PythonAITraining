import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder


n_rows = 300000
df = pd.read_csv("C:/Users/mario/PycharmProjects/pythonAITraining/Drzewo decyzyjne/train.csv", nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train).toarray().astype('float32')
X_test_enc = enc.transform(X_test).toarray().astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

batch_size = 1000
train_data = tf.data.Dataset.from_tensor_slices((X_train_enc, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

n_features = int(X_train_enc.shape[1])
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))
learning_rate = 0.0008
optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        logits = tf.add(tf.matmul(x, W), b)[:, 0]
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        gradients = g.gradient(cost, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))


training_steps = 6000
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % 500 == 0:
        logits = tf.add(tf.matmul(batch_x, W), b)[:, 0]
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=logits))
        print(f"Liczba kroków: {step}, strata: {loss}")

logits = tf.add(tf.matmul(X_test_enc, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)
auc_metrics = tf.keras.metrics.AUC()
auc_metrics.update_state(Y_test, pred)
print(f"Pole pod krzywą ROC dla zbioru testowego: {auc_metrics.result().numpy():.3f}")
