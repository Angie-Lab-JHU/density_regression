import datetime
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import official.nlp.modeling.layers as nlp_layers
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from dataset import WindowGenerator, column_indices
from sklearn import cluster


def _get_initial_lengthscale(f_X_samples):
    f_X_samples = torch.from_numpy(f_X_samples)
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu().numpy()


def initial_values(window, model, n_inducing_points=64):
    f_Z_samples = []
    for step, (x_batch_train, y_batch_train) in enumerate(window.train):
        f_Z_samples.append(model.extract(x_batch_train).numpy())

    f_Z_samples = np.concatenate(f_Z_samples, axis=0)
    f_Z_samples = tf.reshape(f_Z_samples, [f_Z_samples.shape[0] * f_Z_samples.shape[1], f_Z_samples.shape[2]])
    f_Z_samples = f_Z_samples.numpy()

    list_f_Z_samples = []
    for step in range(10):
        np.random.shuffle(f_Z_samples)
        list_f_Z_samples.append(f_Z_samples[:1000])
    f_Z_samples = np.concatenate(list_f_Z_samples, axis=0)

    kmeans = cluster.MiniBatchKMeans(n_clusters=n_inducing_points, batch_size=n_inducing_points * 10)
    kmeans.fit(f_Z_samples)
    initial_inducing_points = tf.convert_to_tensor(kmeans.cluster_centers_, dtype=tf.float32)
    initial_lengthscale = _get_initial_lengthscale(f_Z_samples)

    return initial_inducing_points, initial_lengthscale


def compile_and_fit(model, window, optimizer, metrics, MAX_EPOCHS=20, patience=2):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_hats = model(x, training=True)
            mean = y_hats[:, :, :1]
            log_std = y_hats[:, :, 1:]
            loss_value = tf.reduce_mean(2 * log_std + ((y - mean) / tf.exp(log_std)) ** 2)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        metrics["train/loss"].update_state(loss_value)
        return loss_value

    @tf.function
    def test_step(x, y):
        yhats = [model(x, training=True) for _ in range(4)]
        mus, log_sigmas = tf.split(yhats, 2, axis=-1)
        sigmas = tf.exp(log_sigmas)
        mean_mu = tf.reduce_mean(mus, axis=0)
        var = tf.reduce_mean(sigmas**2 + tf.square(mus), axis=0) - tf.square(mean_mu)
        log_std = tf.math.log(tf.sqrt(var))
        loss_value = tf.reduce_mean(2 * log_std + ((y - mean_mu) / tf.exp(log_std)) ** 2)
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
        yhats = [model(x_batch, training=True) for _ in range(4)]
        mus, log_sigmas = tf.split(yhats, 2, axis=-1)
        sigmas = tf.exp(log_sigmas)
        mean_mu = tf.reduce_mean(mus, axis=0)
        var = tf.reduce_mean(sigmas**2 + tf.square(mus), axis=0) - tf.square(mean_mu)

        mean = mean_mu
        std = tf.sqrt(var)
        y_hats = np.array([mean, std])

        out_nstd = mean - 2 * std
        out_pstd = mean + 2 * std

        log_std = tf.math.log(std)
        loss_value = tf.reduce_mean(2 * log_std + ((y_batch - mean_mu) / tf.exp(log_std)) ** 2)

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


class DeepResNet(tf.keras.Model):
    def __init__(self, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.input_layer = tf.keras.layers.Dense(units=64, activation="relu")
        self.dense_layers = [
            nlp_layers.SpectralNormalization(tf.keras.layers.Dense(units=64, activation="relu"), norm_multiplier=0.9)
            for _ in range(num_layers)
        ]

    def add_regressor(self, initial_inducing_points, initial_lengthscale):
        mean = tf.reduce_mean(initial_inducing_points, axis=0)
        std = tf.math.reduce_std(initial_inducing_points, axis=0)
        custom_random_features_initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=std)
        self.regressor = nlp_layers.RandomFeatureGaussianProcess(
            units=2,
            num_inducing=64,
            custom_random_features_initializer=(custom_random_features_initializer),
            gp_kernel_scale=initial_lengthscale,
        )

    def extract(self, inputs):
        hidden = self.input_layer(inputs)
        hidden = tf.keras.layers.Dropout(0.1)(hidden)
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = tf.keras.layers.Dropout(0.1)(resid)
            hidden += resid

        return hidden

    def call(self, inputs):
        hidden = self.input_layer(inputs)
        hidden = tf.keras.layers.Dropout(0.1)(hidden)
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = tf.keras.layers.Dropout(0.1)(resid)
            hidden += resid

        org_shape_0 = hidden.shape[0]
        org_shape_1 = hidden.shape[1]
        hidden = tf.reshape(hidden, [hidden.shape[0] * hidden.shape[1], hidden.shape[2]])
        outs, covmat = self.regressor(hidden)
        outs = tf.reshape(outs, [org_shape_0, org_shape_1, 2])

        return outs


class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            self.model.regressor.reset_covariance_matrix()


class DeepResNetdueWithCovReset(DeepResNet):
    def fit(self, *args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks", []))
        kwargs["callbacks"].append(ResetCovarianceCallback())

        return super().fit(*args, **kwargs)


def main():
    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["T (degC)"])

    model = DeepResNetdueWithCovReset()
    initial_inducing_points, initial_lengthscale = initial_values(single_step_window, model)
    model.add_regressor(initial_inducing_points, initial_lengthscale)

    metrics = {
        "train/loss": tf.keras.metrics.Mean(),
        "val/loss": tf.keras.metrics.Mean(),
        "test/loss": tf.keras.metrics.Mean(),
    }
    optimizer = tf.keras.optimizers.Adam()

    compile_and_fit(model, single_step_window, optimizer, metrics)

    wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=["T (degC)"])

    list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(model, wide_window.test, metrics)

    with open("out/pickle/due_probs_iid.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/due_labels_iid.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

    plt.savefig("out/due_iid.png")
    plt.close()

    list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(model, wide_window.test_ood, metrics)

    with open("out/pickle/due_probs_ood.pkl", "wb") as fp:
        pickle.dump(list_y_hats, fp)
    with open("out/pickle/due_labels_ood.pkl", "wb") as fp:
        pickle.dump(list_y_true, fp)

    wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

    plt.savefig("out/due_ood.png")


if __name__ == "__main__":
    main()
