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

alpha = 0.1
alpha_low = alpha / 2.0
alpha_high = 1.0 - alpha / 2.0


def compile_and_fit(model, window, optimizer, metrics, MAX_EPOCHS=20, patience=2):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            low_q_preds, high_q_preds = model(x, training=True)
            loss_value = (
                tf.reduce_mean(tfa.losses.pinball_loss(y, low_q_preds, alpha_low))
                + tf.reduce_mean(tfa.losses.pinball_loss(y, high_q_preds, alpha_high))
            ) / 2.0

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        metrics["train/loss"].update_state(loss_value)
        return loss_value

    @tf.function
    def test_step(x, y):
        low_q_preds, high_q_preds = model(x, training=False)
        mean = low_q_preds + (high_q_preds - low_q_preds) / 2.0
        log_std = tf.math.log((high_q_preds - low_q_preds) / 4.0)
        loss_value = tf.reduce_mean(2 * log_std + ((y - mean) / tf.exp(log_std)) ** 2)
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
        low_q_preds, high_q_preds = model(x_batch, training=False)
        mean = low_q_preds + (high_q_preds - low_q_preds) / 2.0

        std = (high_q_preds - low_q_preds) / 4.0
        y_hats = np.array([mean, std])

        out_nstd = low_q_preds
        out_pstd = high_q_preds
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


def main():
    dense = tf.keras.Sequential([tf.keras.layers.Dense(units=64, activation="relu")])
    low_q_dense = tf.keras.Sequential(
        [tf.keras.layers.Dense(units=64, activation="relu"), tf.keras.layers.Dense(units=1)]
    )
    up_q_dense = tf.keras.Sequential(
        [tf.keras.layers.Dense(units=64, activation="relu"), tf.keras.layers.Dense(units=1)]
    )

    inputs = tf.keras.Input(shape=(None, None, 17))
    latents = dense(inputs)
    low_q = low_q_dense(latents)
    up_q = up_q_dense(latents)
    model = tf.keras.Model(inputs=inputs, outputs=[low_q, up_q])

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

    with open("out/pickle/quantile_probs_iid.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/quantile_labels_iid.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

    plt.savefig("out/quantile_iid.png")
    plt.close()

    list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(model, wide_window.test_ood, metrics)

    with open("out/pickle/quantile_probs_ood.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/quantile_labels_ood.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

    plt.savefig("out/quantile_ood.png")


if __name__ == "__main__":
    main()
