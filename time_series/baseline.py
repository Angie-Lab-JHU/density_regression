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


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["T (degC)"])

test_loss_metric = tf.keras.metrics.MeanAbsoluteError()

baseline = Baseline(label_index=column_indices["T (degC)"])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=["T (degC)"])

list_x_test, list_y_hats, list_y_true = [], [], []
for x_batch, y_batch in wide_window.test:
    # for x_batch, y_batch in wide_window.test_ood:
    y_hats = baseline(x_batch, training=False)
    list_x_test.append(x_batch.numpy())
    list_y_hats.append(y_hats.numpy())
    list_y_true.append(y_batch.numpy())
    test_loss_metric.update_state(y_batch, y_hats)

list_x_test = np.concatenate(list_x_test, axis=0)
list_y_hats = np.concatenate(list_y_hats, axis=0)
list_y_true = np.concatenate(list_y_true, axis=0)

test_loss = test_loss_metric.result()
test_loss_metric.reset_states()
print("Test loss: %.4f" % (float(test_loss),))

with open("out/pickle/baseline_probs_iid.pkl", "wb") as fp:
    pickle.dump(list_y_hats, fp)
with open("out/pickle/baseline_labels_iid.pkl", "wb") as fp:
    pickle.dump(list_y_true, fp)

wide_window.plot(list_x_test, list_y_true, list_y_hats)

plt.savefig("out/baseline_iid.png")

# with open("out/pickle/baseline_probs_ood.pkl", "wb") as fp:
# 	pickle.dump(list_y_hats, fp)
# with open("out/pickle/baseline_labels_ood.pkl", "wb") as fp:
# 	pickle.dump(list_y_true, fp)

# wide_window.plot(list_x_test, list_y_true, list_y_hats)

# plt.savefig("out/baseline_ood.png")
