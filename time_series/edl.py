import datetime
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from dataset import WindowGenerator, column_indices


tfd = tfp.distributions
from tensorflow.keras.layers import Dense, Layer


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)

    nll = (
        0.5 * tf.math.log(np.pi / v)
        - alpha * tf.math.log(twoBlambda)
        + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
        + tf.math.lgamma(alpha)
        - tf.math.lgamma(alpha + 0.5)
    )

    return tf.reduce_mean(nll) if reduce else nll


def compile_and_fit(model, window, optimizer, metrics, MAX_EPOCHS=20, patience=2):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
            loss_value = NIG_NLL(y, mu, v, alpha, beta)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        metrics["train/loss"].update_state(loss_value)
        return loss_value

    @tf.function
    def test_step(x, y):
        y_pred = model(x, training=False)
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        loss_value = NIG_NLL(y, mu, v, alpha, beta)
        metrics["val/loss"].update_state(loss_value)

    wait = 0
    best = float("inf")
    for epoch in range(MAX_EPOCHS):
        for step, (x_batch_train, y_batch_train) in enumerate(window.train):
            loss_value = train_step(x_batch_train, y_batch_train)
        train_loss = metrics["train/loss"].result()
        print("Training loss over epoch: %.4f" % (float(train_loss),))
        metrics["train/loss"].reset_states()

        for x_batch_val, y_batch_val in window.val:
            test_step(x_batch_val, y_batch_val)

        val_loss = metrics["val/loss"].result()
        metrics["val/loss"].reset_states()
        print("Validation loss: %.4f" % (float(val_loss),))

        wait += 1
        if val_loss < best:
            best = val_loss
            wait = 0
        if wait >= patience:
            break


def test(model, test_set, metrics):
    list_x_test, list_y_hats, list_y_true, list_out_nstd, list_out_pstd = [], [], [], [], []
    for x_batch, y_batch in test_set:
        y_pred = model(x_batch, training=False)
        mean, v, alpha, beta = tf.split(y_pred, 4, axis=-1)

        std = np.sqrt((beta / (alpha - 1) + beta / (v * (alpha - 1))))
        # std = np.sqrt((beta/(alpha-1) + beta/(v*(alpha-1))))

        out_nstd = mean - 2 * std
        out_pstd = mean + 2 * std

        y_hats = np.array([mean, std])
        log_std = tf.math.log(std)
        loss_value = tf.reduce_mean(2 * log_std + ((y_batch - mean) / tf.exp(log_std)) ** 2)

        list_x_test.append(x_batch.numpy())
        list_y_hats.append(y_hats)
        list_y_true.append(y_batch.numpy())
        list_out_nstd.append(out_nstd.numpy())
        list_out_pstd.append(out_pstd.numpy())
        metrics["test/loss"].update_state(loss_value)
    test_loss = metrics["test/loss"].result()
    metrics["test/loss"].reset_states()
    print("Test loss: %.4f" % (float(test_loss),))

    list_x_test = np.concatenate(list_x_test, axis=0)
    list_y_true = np.concatenate(list_y_true, axis=0)
    list_y_hats = np.concatenate(list_y_hats, axis=1)
    list_out_nstd = np.concatenate(list_out_nstd, axis=0)
    list_out_pstd = np.concatenate(list_out_pstd, axis=0)

    return list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd


class DenseNormalGamma(Layer):
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = Dense(4 * self.units, activation=None)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config["units"] = self.units
        return base_config


def main():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            DenseNormalGamma(1),
        ]
    )

    metrics = {
        "train/loss": tf.keras.metrics.Mean(),
        "val/loss": tf.keras.metrics.Mean(),
        "test/loss": tf.keras.metrics.Mean(),
    }
    optimizer = tf.keras.optimizers.Adam()

    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["T (degC)"])

    compile_and_fit(model, single_step_window, optimizer, metrics)

    wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=["T (degC)"])

    list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(model, wide_window.test, metrics)

    with open("out/pickle/edl_probs_iid.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/edl_labels_iid.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

    plt.savefig("out/edl_iid.png")
    plt.close()

    list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(model, wide_window.test_ood, metrics)

    with open("out/pickle/edl_probs_ood.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/edl_labels_ood.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

    plt.savefig("out/edl_ood.png")


if __name__ == "__main__":
    main()
