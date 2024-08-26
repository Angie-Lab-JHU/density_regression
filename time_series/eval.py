import pickle
import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from tqdm.notebook import tqdm


model_name = "Density Regression"

with open("out/pickle/baseline_labels_ood.pkl", "rb") as fp:
    labels = np.squeeze(pickle.load(fp))

with open("out/pickle/robust_regression_ood10.pkl", "rb") as fp:
    rba_outputs = np.squeeze(pickle.load(fp))

# targets_pred = rba_outputs.reshape(-1)
# stdevs = np.zeros(targets_pred.shape)

targets_pred = rba_outputs[0].reshape(-1)
stdevs = rba_outputs[1].reshape(-1)

targets_test = labels.reshape(-1)
residuals = targets_pred - targets_test.reshape(-1)
norm = stats.norm(loc=0, scale=1)

width = 7.5 / 3  # 1/3 of a page
fontsize = 20
rc = {
    "figure.figsize": (width, width),
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
}
sns.set(rc=rc)
sns.set_style("ticks")

lims = [-4, 4]
grid = sns.jointplot(x=targets_test.reshape(-1), y=targets_pred, kind="hex", bins="log", extent=lims + lims)
ax = grid.ax_joint
_ = ax.set_xlim(lims)
_ = ax.set_ylim(lims)
_ = ax.plot(lims, lims, "--")
_ = ax.set_xlabel("actual")
_ = ax.set_ylabel("%s predicted" % model_name)

# Calculate the error metrics
mae = mean_absolute_error(targets_test, targets_pred)
rmse = np.sqrt(mean_squared_error(targets_test, targets_pred))
mdae = median_absolute_error(targets_test, targets_pred)
marpd = np.abs(2 * residuals / (np.abs(targets_pred) + np.abs(targets_test.reshape(-1)))).mean() * 100
r2 = r2_score(targets_test, targets_pred)
corr = np.corrcoef(targets_test.reshape(-1), targets_pred)[0, 1]
NLL = np.mean(2 * np.log(stdevs) + ((targets_test - targets_pred) / np.exp(np.log(stdevs))) ** 2)

# Report
text = "  MDAE = %.4f\n" % mdae + "  MAE = %.4f\n" % mae + "  RMSE = %.4f\n" % rmse

print("NLL = %.4f" % NLL)
print("RMSE = %.4f" % rmse)
print("MAE = %.4f" % mae)
print("MDAE = %.4f" % mdae)
print("MARPD = %.4f" % marpd)
print("R2 = %.4f" % r2)
print("PPMCC = %.4f" % corr)
_ = ax.text(x=lims[0], y=lims[1], s=text, horizontalalignment="left", verticalalignment="top", fontsize=fontsize)

# Save figure
plt.savefig("tmp/parity.png", dpi=300, bbox_inches="tight", transparent=True)


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


predicted_pi = np.linspace(0, 1, 100)
observed_pi = [calculate_density(quantile) for quantile in tqdm(predicted_pi, desc="Calibration")]

calibration_error = ((predicted_pi - observed_pi) ** 2).sum()
print("Calibration error = %.4f" % calibration_error)

# Set figure defaults
width = 4  # Because it looks good
fontsize = 12
rc = {
    "figure.figsize": (width, width),
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
}
sns.set(rc=rc)
sns.set_style("ticks")

# Plot the calibration curve
fig_cal = plt.figure(figsize=(4, 4))
ax_ideal = sns.lineplot(x=[0, 1], y=[0, 1], label="ideal")
_ = ax_ideal.lines[0].set_linestyle("--")
ax_gp = sns.lineplot(x=predicted_pi, y=observed_pi, label=model_name)
ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi, alpha=0.2, label="miscalibration area")
_ = ax_ideal.set_xlabel("Expected cumulative distribution")
_ = ax_ideal.set_ylabel("Observed cumulative distribution")
_ = ax_ideal.set_xlim([0, 1])
_ = ax_ideal.set_ylim([0, 1])

# Calculate the miscalibration area.
polygon_points = []
for point in zip(predicted_pi, observed_pi):
    polygon_points.append(point)
for point in zip(reversed(predicted_pi), reversed(predicted_pi)):
    polygon_points.append(point)
polygon_points.append((predicted_pi[0], observed_pi[0]))
polygon = Polygon(polygon_points)
x, y = polygon.exterior.xy  # original data
ls = LineString(np.c_[x, y])  # closed, non-simple
lr = LineString(ls.coords[:] + ls.coords[0:1])
mls = unary_union(lr)
polygon_area_list = [poly.area for poly in polygonize(mls)]
miscalibration_area = np.asarray(polygon_area_list).sum()

# Annotate the plot with the miscalibration area
plt.text(
    x=0.95,
    y=0.05,
    # s="Miscalibration area = %.3f" % miscalibration_area,
    s="Calibration error = %.4f" % calibration_error,
    verticalalignment="bottom",
    horizontalalignment="right",
    fontsize=fontsize,
)
plt.title(model_name)

# Save
plt.savefig("tmp/calibration.png", dpi=300, bbox_inches="tight", transparent=True)

# Plot sharpness curve
xlim = [0.0, 1.0]
fig_sharp = plt.figure(figsize=(4, 4))
ax_sharp = sns.distplot(stdevs, kde=False, norm_hist=True)
ax_sharp.set_xlim(xlim)
ax_sharp.set_xlabel("Predicted standard deviations")
ax_sharp.set_ylabel("Normalized frequency")
ax_sharp.set_yticklabels([])
ax_sharp.set_yticks([])

# Calculate and report sharpness/dispersion
sharpness = np.sqrt(np.mean(stdevs**2))
_ = ax_sharp.axvline(x=sharpness, label="sharpness")
dispersion = np.sqrt(((stdevs - stdevs.mean()) ** 2).sum() / (len(stdevs) - 1)) / stdevs.mean()
if sharpness < (xlim[0] + xlim[1]) / 2:
    text = '\n  Sharpness = %.4f\n  C$_v$ = %.4f' % (sharpness, dispersion)
    # text = "\n  Sharpness = %.4f" % (sharpness)
    h_align = "left"
else:
    text = "\nSharpness = %.4f" % (sharpness)
    h_align = "right"
_ = ax_sharp.text(
    x=sharpness,
    y=ax_sharp.get_ylim()[1],
    s=text,
    verticalalignment="top",
    horizontalalignment=h_align,
    fontsize=fontsize,
)
print("Sharpness = %.4f" % sharpness)
# Save
plt.title(model_name)
plt.savefig("tmp/sharpness.png", dpi=300, bbox_inches="tight", transparent=True)
# plt.savefig('tmp/sharpness.pdf', dpi=300, bbox_inches='tight', transparent=True)

# Pull a random sample of the data, because plotting thousands of these at once would look absurd
all_predictions = list(zip(targets_pred, targets_test.reshape(-1), stdevs))
samples = random.sample(all_predictions, k=20)

# Parse the samples
_preds, _targets, _stdevs = zip(*samples)
_preds = np.array(_preds)
_targets = np.array(_targets)
_stdevs = np.array(_stdevs)

# Plot
fig = plt.figure(figsize=(4, 4))
_ = plt.errorbar(_targets, _preds, yerr=2 * _stdevs, fmt="o")
ax = plt.gca()

# Make a parity line
lims = [-2, 2]
_ = ax.plot(lims, lims, "--")

# Format
_ = ax.set_xlim(lims)
_ = ax.set_ylim(lims)
_ = ax.set_xticks(list(range(-2, 3)))
_ = ax.set_yticks(list(range(-2, 3)))
_ = ax.set_xlabel("actual")
_ = ax.set_ylabel("predicted")

plt.title(model_name)

# Save
_ = plt.savefig("tmp/error_bar_parity.png", dpi=300, bbox_inches="tight", transparent=True)
