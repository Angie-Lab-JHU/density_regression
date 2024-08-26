import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from tqdm.notebook import tqdm

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(constrained_layout=True)
ax.plot([0, 1], [0, 1], "--", color="grey", label = "Ideal calibration")

conf_level_lower_bounds = np.arange(start=0.025, stop=0.5, step=0.025)
conf_levels = 1 - 2 * conf_level_lower_bounds


def test(labels, rba_outputs, method, color):
    if rba_outputs.shape[0] == 2:
        targets_pred = rba_outputs[0].reshape(-1)
        stdevs = rba_outputs[1].reshape(-1)
    else:
        targets_pred = rba_outputs.reshape(-1)
        stdevs = np.zeros(targets_pred.shape)
    
    targets_test = labels.reshape(-1)
    residuals = targets_pred - targets_test.reshape(-1)
    norm = stats.norm(loc=0, scale=1)

    # Computing calibration
    def calculate_density(percentile):
        """
        Calculate the fraction of the residuals that fall within the lower
        `percentile` of their respective Gaussian distributions, which are
        defined by their respective uncertainty estimates.
        """
        # Find the normalized bounds of this percentile
        upper_bound = norm.ppf(percentile)

        # Normalize the residuals so they all should fall on the normal bell curve
        normalized_residuals = residuals.reshape(-1) / stdevs.reshape(-1)

        # Count how many residuals fall inside here
        num_within_quantile = 0
        for resid in normalized_residuals:
            if resid <= upper_bound:
                num_within_quantile += 1

        # Return the fraction of residuals that fall within the bounds
        density = num_within_quantile / len(residuals)
        return density

    predicted_pi = np.linspace(0, 1, 20)
    observed_pi = [calculate_density(quantile) for quantile in tqdm(predicted_pi, desc="Calibration")]

    calibration_error = ((predicted_pi - observed_pi) ** 2).sum()
    ax.plot(predicted_pi, observed_pi, '-o', label=method, color = color)


with open("out/pickle/robust_regression_ood1.pkl", "rb") as fp:
    edl_linear_outputs = np.squeeze(pickle.load(fp))
with open("out/pickle/quantile_probs_ood.pkl", "rb") as fp:
    ensembles_outputs = np.squeeze(pickle.load(fp))
with open("out/pickle/dropout_probs_ood.pkl", "rb") as fp:
    dropout_outputs = np.squeeze(pickle.load(fp))
with open("out/pickle/baseline_labels_ood.pkl", "rb") as fp:
    labels = np.squeeze(pickle.load(fp))
    #4s
with open("out/pickle/robust_regression_ood5.pkl", "rb") as fp:
    rba_outputs = np.squeeze(pickle.load(fp))
test(labels, dropout_outputs, "MC Dropout", "tab:green")
test(labels, edl_linear_outputs, "EDL", "tab:olive")
test(labels, ensembles_outputs, "Ensembles", "tab:brown")
test(labels, rba_outputs, "Ours", "blue")
# test(labels, density_regression_outputs, "density_regression")

ax.legend()
# ax.set_title("Reliability diagram on OOD")
ax.set_xlabel("Predicted Confidence Level")
ax.set_ylabel("Observed Confidence Level")
plt.savefig("out/calib_ood.png")
