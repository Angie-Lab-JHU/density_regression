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


def save_x_y(window, name):
    list_x, list_y = [], []
    for x_batch, y_batch in window:
        list_x.append(x_batch)
        list_y.append(y_batch)
    list_x = np.vstack(list_x)
    list_x = np.squeeze(list_x)
    with open(name + "_x.pkl", "wb") as fp:
        pickle.dump(list_x, fp)
    list_y = np.vstack(list_y)
    list_y = np.squeeze(list_y)
    with open(name + "_y.pkl", "wb") as fp:
        pickle.dump(list_y, fp)


def save(single_step_window, wide_window):
    save_x_y(single_step_window.train, "train")
    save_x_y(single_step_window.val, "val")
    save_x_y(wide_window.test, "test")
    save_x_y(wide_window.test_ood, "test_ood")


def main():
    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["T (degC)"])

    wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=["T (degC)"])

    save(single_step_window, wide_window)


if __name__ == "__main__":
    main()
