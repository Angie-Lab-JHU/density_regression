import functools
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.neighbors import KernelDensity

def rescale_normal(X, x_min, x_max, y_min=0, y_max=0.4):
	X = (X - x_min) / (x_max - x_min)
	X = X * (y_max - y_min)
	X = X + y_min
	return X

def generate_cubic(x, noise=False):
	x = x.astype(np.float32)
	y = x**3

	if noise:
		sigma = 3 * np.ones_like(x)
	else:
		sigma = np.zeros_like(x)
	r = np.random.normal(0, sigma).astype(np.float32)
	return y+r, sigma

def main():
	train_bounds = [[-4, 4]]
	x_train = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in train_bounds]).reshape(-1, 1)
	y_train, sigma_train = generate_cubic(x_train, noise=True)

	test_bounds = [[-7, +7]]
	x_test = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in test_bounds]).reshape(-1, 1)
	y_test, sigma_test = generate_cubic(x_test, noise=False)

	# kde = KernelDensity().fit(x_train)

	dim_x = 100
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(100, activation="relu"),
		tf.keras.layers.Dense(100, activation="relu"),
		tf.keras.layers.Dense(100, activation="relu"),
		tf.keras.layers.Dense(100, activation="relu"),
	])
	regressor = tf.keras.layers.Dense(2)

	# initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
	# # initializer = tf.keras.initializers.GlorotUniform()

	# M_yy = tf.Variable(initializer(shape=[1, dim_x + 1]))
	# M_yx1 = tf.Variable(initializer(shape=[1, dim_x + 1]))

	metrics = {
		'train/loss': tf.keras.metrics.Mean(),
		'val/loss': tf.keras.metrics.Mean(),
		'test/loss': tf.keras.metrics.Mean()
	}

	lr = 5e-4
	optimizer = tf.keras.optimizers.Adam(lr)

	def compute_grad(y_pred, mean, var, y_true):
		y_vec = tf.concat([y_true, y_pred, tf.ones([y_true.shape[0], 1])], 1)
		emp = y_vec * y_true
		y_mean = tf.concat([mean, y_pred, tf.ones([mean.shape[0], 1])], 1)
		var_vec = tf.concat([var, tf.zeros([var.shape[0], dim_x + 1])], 1)
		exp = y_mean * mean + var_vec
		grad = tf.reduce_mean(exp, 0) - tf.reduce_mean(emp, 0)
		return grad

	@tf.function
	def pre_train_step(x, y):
		with tf.GradientTape() as tape:
			y_pred = regressor(model(x, training=True), training = True)
			M_yy_out, M_yx1_out = tf.split(y_pred, 2, axis=-1)
			log_std = -1/2 * (tf.math.log(2.) + M_yy_out)
			var = tf.exp(log_std)**2
			mean = var * (-2 * M_yx1_out)
			loss_value = tf.reduce_mean(2 * log_std + ((y - mean) / tf.exp(log_std)) ** 2)

		list_weights = model.trainable_weights + regressor.trainable_weights
		grads = tape.gradient(loss_value, list_weights)
		optimizer.apply_gradients(zip(grads, list_weights))
		metrics['train/loss'].update_state(loss_value)
		return loss_value

	@tf.function
	def train_step(z, y, iid_loglikelihood):
		with tf.GradientTape() as tape:
			y_pred = regressor(z, training = True)
			M_yy_out, M_yx1_out = tf.split(y_pred, 2, axis=-1)
			log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + M_yy_out)
			var = tf.exp(log_std)**2
			mean = var * (-2 * tf.exp(iid_loglikelihood) * M_yx1_out)
			loss_value = tf.reduce_mean(2 * log_std + ((y - mean) / tf.exp(log_std)) ** 2)

		list_weights = regressor.trainable_weights
		grads = tape.gradient(loss_value, list_weights)
		optimizer.apply_gradients(zip(grads, list_weights))
		metrics['train/loss'].update_state(loss_value)
		return loss_value

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.shuffle(128).batch(128)

	ite = 0
	while(ite < 4000):
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			loss_value = pre_train_step(x_batch_train, y_batch_train)
			ite += 1
		train_loss = metrics['train/loss'].result()
		if ite % 100 == 0:
			print("Training loss over iteration " + str(ite) + ": %.4f" % (float(train_loss)))
		metrics['train/loss'].reset_states()
	
	latents = []
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
		latents.append(model(x_batch_train, training=False).numpy())
	latents = np.concatenate(latents, axis=0)
	kde = KernelDensity(kernel = 'exponential', metric = "l1").fit(latents)
	train_nll = kde.score_samples(latents)
	min_nll, max_nll = min(train_nll), max(train_nll)

	while(ite < 5000):
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			z_batch_train = model(x_batch_train, training=True)
			iid_loglikelihood = kde.score_samples(z_batch_train.numpy())
			iid_loglikelihood = rescale_normal(iid_loglikelihood, min_nll, max_nll)
			iid_loglikelihood = tf.reshape(iid_loglikelihood, [x_batch_train.shape[0], 1])
			iid_loglikelihood = tf.cast(iid_loglikelihood, tf.float32)
			loss_value = train_step(z_batch_train, y_batch_train, iid_loglikelihood)
			ite += 1
		train_loss = metrics['train/loss'].result()
		if ite % 100 == 0:
			print("Training loss over iteration " + str(ite) + ": %.4f" % (float(train_loss)))
		metrics['train/loss'].reset_states()

	@tf.function
	def test_step(z, iid_loglikelihood):
		y_pred = regressor(z, training = False)
		M_yy_out, M_yx1_out = tf.split(y_pred, 2, axis=-1)
		log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + M_yy_out)
		var = tf.exp(log_std)**2
		mean = var * (-2 * tf.exp(iid_loglikelihood) * M_yx1_out)
		y_pred = tf.concat([mean, var], 1)
		return y_pred

	# Predict and plot using the trained model
	z_test = model(x_test, training=True)
	iid_loglikelihood = kde.score_samples(z_test.numpy())
	iid_loglikelihood = rescale_normal(iid_loglikelihood, min_nll, max_nll)
	iid_loglikelihood = tf.reshape(iid_loglikelihood, [x_test.shape[0], 1])
	iid_loglikelihood = tf.cast(iid_loglikelihood, tf.float32)
	y_pred = test_step(z_test, iid_loglikelihood)
	plot_predictions(x_train, y_train, x_test, y_test, y_pred)

	# Done!!

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, iid_loglikelihood = None, n_stds=4, kk=0):
	x_test = x_test[:, 0]

	mu, var = tf.split(y_pred, 2, axis=-1)
	mu = mu[:, 0]
	var = np.sqrt(var)[:, 0]

	plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
	plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
	plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
	plt.plot([-4, -4], [-350, 350], 'k--', alpha=0.4, zorder=0)
	plt.plot([+4, +4], [-350, 350], 'k--', alpha=0.4, zorder=0)
	for k in np.linspace(0, n_stds, 4):
		plt.fill_between(
			x_test, (mu - k * var), (mu + k * var),
			alpha=0.3,
			edgecolor=None,
			facecolor='#00aeef',
			linewidth=0,
			antialiased=True,
			zorder=1,
			label="Unc." if k == 0 else None)
	plt.gca().set_ylim(-350, 350)
	plt.gca().set_xlim(-7, 7)
	plt.title("Density-Regression", fontsize = 20)
	plt.legend(loc="upper left", fontsize = 16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()
	plt.savefig('out.png')

if __name__ == "__main__":
	main()