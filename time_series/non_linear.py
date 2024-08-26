import datetime
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dataset import WindowGenerator, column_indices


def compile_and_fit(model, window, loss_fn, optimizer, metrics, MAX_EPOCHS=20, patience=2):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_hats = model(x, training=True)
            loss_value = loss_fn(y, y_hats)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        metrics["train/loss"].update_state(y, y_hats)
        return loss_value

    @tf.function
    def test_step(x, y):
        y_hats = model(x, training=False)
        metrics["val/loss"].update_state(y, y_hats)

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
    list_x_test, list_y_hats, list_y_true = [], [], []
    for x_batch, y_batch in test_set:
        y_hats = model(x_batch, training=False)
        list_x_test.append(x_batch.numpy())
        list_y_hats.append(y_hats.numpy())
        list_y_true.append(y_batch.numpy())
        metrics["test/loss"].update_state(y_batch, y_hats)

    test_loss = metrics["test/loss"].result()
    metrics["test/loss"].reset_states()
    print("Test loss: %.4f" % (float(test_loss),))

    return (
        np.concatenate(list_x_test, axis=0),
        np.concatenate(list_y_true, axis=0),
        np.concatenate(list_y_hats, axis=0),
    )


def main():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=1),
        ]
    )

    metrics = {
        "train/loss": tf.keras.metrics.MeanAbsoluteError(),
        "val/loss": tf.keras.metrics.MeanAbsoluteError(),
        "test/loss": tf.keras.metrics.MeanAbsoluteError(),
    }
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["T (degC)"])

    compile_and_fit(model, single_step_window, loss_fn, optimizer, metrics)

    wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=["T (degC)"])

    list_x_test, list_y_true, list_y_hats = test(model, wide_window.test, metrics)

    with open("out/pickle/non_linear_probs_iid.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/non_linear_labels_iid.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats)

    plt.savefig("out/non_linear_iid.png")
    plt.close()

    list_x_test, list_y_true, list_y_hats = test(model, wide_window.test_ood, metrics)

    with open("out/pickle/non_linear_probs_ood.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/non_linear_labels_ood.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats)

    plt.savefig("out/non_linear_ood.png")


if __name__ == "__main__":
    main()
