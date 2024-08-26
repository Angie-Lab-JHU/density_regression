import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity


# with open("z_train.pkl", "rb") as fp:
# 	x_train = np.squeeze(pickle.load(fp))

# with open("z_test.pkl", "rb") as fp:
# 	x_test = np.squeeze(pickle.load(fp))

# with open("z_test_ood.pkl", "rb") as fp:
# 	x_test_ood = np.squeeze(pickle.load(fp))

# with open("z_val.pkl", "rb") as fp:
# 	x_val = np.squeeze(pickle.load(fp))

# x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], x_test.shape[2])
# x_test_ood = x_test_ood.reshape(x_test_ood.shape[0] * x_test_ood.shape[1], x_test_ood.shape[2])

# density_func = KernelDensity(kernel='gaussian', bandwidth = 1.5)
# density_func.fit(x_train)

# train_nll = np.exp(density_func.score_samples(x_train))
# max_train_nll = max(train_nll)
# train_nll = train_nll/max_train_nll
# val_nll = np.exp(density_func.score_samples(x_val))/max_train_nll
# test_nll = np.exp(density_func.score_samples(x_test))/max_train_nll
# test_ood_nll = np.exp(density_func.score_samples(x_test_ood))/max_train_nll


# print(train_nll.shape)
# print(val_nll.shape)
# print(test_nll.shape)
# exit()
def rescale_normal(X, x_min, x_max, y_min=-1, y_max=0):
    X = (X - x_min) / (x_max - x_min)
    X = X * (y_max - y_min)
    X = X + y_min
    return X


with open("tmp/train_nll.pkl", "rb") as fp:
    train_nll = np.squeeze(pickle.load(fp))
min_nll, max_nll = min(train_nll), max(train_nll)
train_nll = rescale_normal(train_nll, min_nll, max_nll)
train_nll = np.exp(train_nll)
max_train_nll = max(train_nll)
train_nll = train_nll / max_train_nll
with open("tmp/test_nll.pkl", "rb") as fp:
    test_nll = np.squeeze(pickle.load(fp))
test_nll = rescale_normal(test_nll, min_nll, max_nll)
test_nll = np.exp(test_nll)
test_nll = test_nll / max_train_nll
with open("tmp/test_ood_nll.pkl", "rb") as fp:
    test_ood_nll = np.squeeze(pickle.load(fp))
test_ood_nll = rescale_normal(test_ood_nll, min_nll, max_nll)
test_ood_nll = np.exp(test_ood_nll)
test_ood_nll = test_ood_nll / max_train_nll

bins = int(100 / 1)
plt.xlabel("Likelihood value p(x)", fontsize=11)
plt.ylabel("Normalized density", fontsize=11)
# plt.xlim([0.4, 1])
plt.hist(train_nll, label="train", alpha=0.8, density=True, bins=bins)
# plt.hist(val_nll, label="val", alpha=0.8, density=True, bins=bins)
plt.hist(test_nll, label="test", alpha=0.8, density=True, bins=bins)
plt.hist(test_ood_nll, label="test_ood", alpha=0.8, density=True, bins=bins)
plt.legend()
plt.tight_layout()
plt.savefig("tmp/density_histogram.png")
