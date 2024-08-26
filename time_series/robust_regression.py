import datetime
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from dataset import WindowGenerator, column_indices
from tensorflow import keras
from tensorflow.keras import regularizers


tfd = tfp.distributions
import argparse


dim_x = 64


def compile_and_fit(
	model,
	flow_model,
	max_train_nll,
	min_train_nll,
	window,
	optimizer,
	metrics,
	MAX_EPOCHS=20,
	# MAX_EPOCHS=40,
	patience=2,
):
	initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.1)
	# M_yy = tf.Variable(initializer(shape=[1, 1]))
	M_yy = tf.Variable(initializer(shape=[1, dim_x + 1]))
	tmp = M_yy.numpy()
	tmp[0][dim_x] = 1
	M_yy.assign(tmp)
	M_yx1 = tf.Variable(initializer(shape=[1, dim_x + 1]))
	tmp = M_yx1.numpy()
	tmp[0][dim_x] = 0
	M_yx1.assign(tmp)

	def compute_grad(y_pred, mean, var, y_true):
		y_vec = tf.concat([y_true, y_pred, tf.ones([y_true.shape[0], y_true.shape[1], 1])], 2)
		emp = y_vec * y_true
		y_mean = tf.concat([mean, y_pred, tf.ones([mean.shape[0], mean.shape[1], 1])], 2)
		var_vec = tf.concat([var, tf.zeros([var.shape[0], var.shape[1], dim_x + 1])], 2)
		exp = y_mean * mean + var_vec
		grad = tf.reduce_mean(exp, 0) - tf.reduce_mean(emp, 0)
		return grad[0]

	@tf.function
	def train_step(x, y):
		with tf.GradientTape() as tape:
			y_pred = model(x, training=True)
			x_batch_clone = tf.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
			x_batch_clone = tf.concat([x_batch_clone, tf.ones([x_batch_clone.shape[0], 1])], 1)
			iid_loglikelihood = flow_model.score_samples(x_batch_clone)
			iid_loglikelihood = rescale_normal(iid_loglikelihood, min_train_nll, max_train_nll)
			iid_loglikelihood = tf.reshape(iid_loglikelihood, [x.shape[0], 1])

			log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + tf.tensordot(M_yy,
			            tf.concat([y_pred, tf.ones([y_pred.shape[0], y_pred.shape[1], 1], tf.float32)], 2),
			            axes=[1, 2])[0])
			# log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + tf.math.log(M_yy))
			var = tf.exp(log_std)**2

			mean = var * (
				-2
				* tf.exp(iid_loglikelihood)
				* tf.tensordot(
					M_yx1,
					tf.concat([y_pred, tf.ones([y_pred.shape[0], y_pred.shape[1], 1], tf.float32)], 2),
					axes=[1, 2],
				)[0]
			)
			mean = tf.expand_dims(mean, axis=2)
			var = tf.expand_dims(var, axis=2)
			# grad_M = compute_grad(y_pred, mean, var, y)
			loss_value = tf.reduce_mean(2 * log_std + ((y - mean) / tf.exp(log_std)) ** 2)
			# loss_value = tf.reduce_mean(tf.math.log(tf.math.sqrt(var)) + 0.5*tf.math.log(2*np.pi) + 0.5*((y-mean)/tf.math.sqrt(var))**2)

		# M_yy.assign(M_yy + 1e-3 * grad_M[0])
		# M_yx1.assign(M_yx1 + 1e-3 * grad_M[1:])

		list_weights = model.trainable_weights
		list_weights.append(M_yx1)
		list_weights.append(M_yy)

		grads = tape.gradient(loss_value, list_weights)
		optimizer.apply_gradients(zip(grads, list_weights))

		metrics["train/loss"].update_state(loss_value)

	@tf.function
	def test_step(x, y):
		y_pred = model(x, training=False)
		x_batch_clone = tf.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
		x_batch_clone = tf.concat([x_batch_clone, tf.ones([x_batch_clone.shape[0], 1])], 1)
		iid_loglikelihood = flow_model.score_samples(x_batch_clone)
		iid_loglikelihood = rescale_normal(iid_loglikelihood, min_train_nll, max_train_nll)
		iid_loglikelihood = tf.reshape(iid_loglikelihood, [x.shape[0], 1])
		
		log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + tf.tensordot(M_yy,
			            tf.concat([y_pred, tf.ones([y_pred.shape[0], y_pred.shape[1], 1], tf.float32)], 2),
			            axes=[1, 2])[0])
		# log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + tf.math.log(M_yy))
		var = tf.exp(log_std)**2
		mean = var * (
			-2
			* tf.exp(iid_loglikelihood)
			* tf.tensordot(
				M_yx1, tf.concat([y_pred, tf.ones([y_pred.shape[0], y_pred.shape[1], 1], tf.float32)], 2), axes=[1, 2]
			)[0]
		)
		mean = tf.expand_dims(mean, axis=2)
		var = tf.expand_dims(var, axis=2)
		loss_value = tf.reduce_mean(2 * log_std + ((y - mean) / tf.exp(log_std)) ** 2)
		metrics["val/loss"].update_state(loss_value)

	wait = 0
	best = float("inf")
	for epoch in range(MAX_EPOCHS):
		for step, (x_batch_train, y_batch_train) in enumerate(window.train):
			train_step(x_batch_train, y_batch_train)
		train_loss = metrics["train/loss"].result()
		print("Training loss over epoch: %.4f" % (float(train_loss),))
		metrics["train/loss"].reset_states()

		for x_batch_val, y_batch_val in window.val:
			test_step(x_batch_val, y_batch_val)

		val_loss = metrics["val/loss"].result()
		metrics["val/loss"].reset_states()
		print("Validation loss: %.4f" % (float(val_loss),))

		# wait += 1
		# if val_loss < best:
		# 	best = val_loss
		# 	wait = 0
		# if wait >= patience:
		# 	break

	return M_yy, M_yx1


def test(model, M_yy, M_yx1, flow_model, max_train_nll, min_train_nll, test_set, metrics):
	list_x_test, list_y_hats, list_y_true, list_out_nstd, list_out_pstd = [], [], [], [], []
	for x_batch, y_batch in test_set:
		y_pred = model(x_batch, training=False)

		x_batch_clone = tf.reshape(x_batch, [x_batch.shape[0] * x_batch.shape[1], x_batch.shape[2]])
		x_batch_clone = tf.concat([x_batch_clone, tf.ones([x_batch_clone.shape[0], 1])], 1)
		iid_loglikelihood = flow_model.score_samples(x_batch_clone)
		iid_loglikelihood = rescale_normal(iid_loglikelihood, min_train_nll, max_train_nll)
		iid_loglikelihood = tf.reshape(iid_loglikelihood, [x_batch.shape[0], x_batch.shape[1], 1])
		
		log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + tf.expand_dims(tf.tensordot(M_yy,
			            tf.concat([y_pred, tf.ones([y_pred.shape[0], y_pred.shape[1], 1], tf.float32)], 2),
			            axes=[1, 2])[0], axis=2))
		# log_std = -1/2 * (tf.math.log(2.) + iid_loglikelihood + tf.math.log(M_yy))
		var = tf.exp(log_std)**2
		mean = var * (
			-2
			* tf.exp(iid_loglikelihood)
			* tf.expand_dims(
				tf.tensordot(
					M_yx1,
					tf.concat([y_pred, tf.ones([y_pred.shape[0], y_pred.shape[1], 1], tf.float32)], 2),
					axes=[1, 2],
				)[0],
				axis=2,
			)
		)

		std = tf.math.sqrt(var)

		out_nstd = mean - 2 * std
		out_pstd = mean + 2 * std

		y_hats = np.array([mean, std])

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


def rescale_normal(X, x_min, x_max, y_min=1, y_max=3):
	X = (X - x_min) / (x_max - x_min)
	X = X * (y_max - y_min)
	X = X + y_min
	return X


def train_flow(density_model, window):
	metrics = {"train/flow_nll": tf.keras.metrics.Mean()}

	def compute_loss(density_model, x):
		y, logdet = density_model(x)
		log_likelihood = density_model.distribution.log_prob(y) + logdet
		return -tf.reduce_mean(log_likelihood)

	@tf.function
	def get_loss_and_grads(x_train, density_model):
		with tf.GradientTape() as tape:
			loss = compute_loss(density_model, x_train)
			grads = tape.gradient(loss, density_model.trainable_variables)
		return loss, grads

	optimizer = keras.optimizers.Adam()
	num_steps = 10

	for i in range(num_steps):
		for step, (x_batch_train, _) in enumerate(window.train):
			x_batch_train = tf.reshape(x_batch_train, [x_batch_train.shape[0], 17])
			x_batch_train = tf.concat([x_batch_train, tf.ones([x_batch_train.shape[0], 1])], 1)
			loss, grads = get_loss_and_grads(x_batch_train, density_model)
			metrics["train/flow_nll"].update_state(loss)
			optimizer.apply_gradients(zip(grads, density_model.trainable_variables))
		print("Step {:03d}: Loss: {:.3f}".format(i, metrics["train/flow_nll"].result()))
		metrics["train/flow_nll"].reset_states()

	train_nll = []
	for step, (x_batch, _) in enumerate(window.train):
		x_batch = tf.reshape(x_batch, [x_batch.shape[0], 17])
		x_batch = tf.concat([x_batch, tf.ones([x_batch.shape[0], 1])], 1)
		train_nll.append(density_model.score_samples(x_batch))
	train_nll = np.concatenate(train_nll, axis=0)
	with open("tmp/train_nll.pkl", "wb") as fp:
		pickle.dump(train_nll, fp)

	test_nll = []
	for step, (x_batch, _) in enumerate(window.test):
		x_batch = tf.reshape(x_batch, [x_batch.shape[0], 17])
		x_batch = tf.concat([x_batch, tf.ones([x_batch.shape[0], 1])], 1)
		test_nll.append(density_model.score_samples(x_batch))
	test_nll = np.concatenate(test_nll, axis=0)
	with open("tmp/test_nll.pkl", "wb") as fp:
		pickle.dump(test_nll, fp)

	test_ood_nll = []
	for step, (x_batch, _) in enumerate(window.test_ood):
		x_batch = tf.reshape(x_batch, [x_batch.shape[0], 17])
		x_batch = tf.concat([x_batch, tf.ones([x_batch.shape[0], 1])], 1)
		test_ood_nll.append(density_model.score_samples(x_batch))
	test_ood_nll = np.concatenate(test_ood_nll, axis=0)
	with open("tmp/test_ood_nll.pkl", "wb") as fp:
		pickle.dump(test_ood_nll, fp)

	min_nll, max_nll = min(train_nll), max(train_nll)
	train_nll = rescale_normal(train_nll, min_nll, max_nll)
	
	return max_nll, min_nll


def Coupling(input_shape):
	output_dim = 16
	reg = 0.01
	input = keras.layers.Input(shape=input_shape)

	t_layer_1 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(input)
	t_layer_2 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(t_layer_1)
	t_layer_3 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(t_layer_2)
	t_layer_4 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(t_layer_3)
	t_layer_5 = keras.layers.Dense(input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg))(
		t_layer_4
	)

	s_layer_1 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(input)
	s_layer_2 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(s_layer_1)
	s_layer_3 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(s_layer_2)
	s_layer_4 = keras.layers.Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(s_layer_3)
	s_layer_5 = keras.layers.Dense(input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg))(s_layer_4)

	return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])


class RealNVP(keras.Model):
	def __init__(self, num_coupling_layers, input_dim):
		super(RealNVP, self).__init__()

		self.num_coupling_layers = num_coupling_layers

		self.distribution = tfp.distributions.MultivariateNormalDiag(
			loc=np.zeros(input_dim, dtype="float32"), scale_diag=np.ones(input_dim, dtype="float32")
		)
		self.masks = np.array(
			[
				np.concatenate((np.zeros(input_dim // 2), np.ones(input_dim // 2))),
				np.concatenate((np.ones(input_dim // 2), np.zeros(input_dim // 2))),
			]
			* (num_coupling_layers // 2),
			dtype="float32",
		)

		self.loss_tracker = keras.metrics.Mean(name="loss")
		self.layers_list = [Coupling(input_dim) for i in range(num_coupling_layers)]

	@property
	def metrics(self):
		return [self.loss_tracker]

	def call(self, x, training=True):
		log_det_inv = 0
		direction = 1
		if training:
			direction = -1
		for i in range(self.num_coupling_layers)[::direction]:
			x_masked = x * self.masks[i]
			reversed_mask = 1 - self.masks[i]
			s, t = self.layers_list[i](x_masked)
			s *= reversed_mask
			t *= reversed_mask
			gate = (direction - 1) / 2
			x = reversed_mask * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s)) + x_masked
			log_det_inv += gate * tf.reduce_sum(s, [1])

		return x, log_det_inv

	def score_samples(self, x):
		y, logdet = self(x, training=False)
		log_likelihood = self.distribution.log_prob(y) + logdet
		return log_likelihood


def main(idx):
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Dense(units=64, activation="relu"),
			tf.keras.layers.Dense(units=64, activation="relu"),
		]
	)

	metrics = {
		"train/loss": tf.keras.metrics.Mean(),
		"val/loss": tf.keras.metrics.Mean(),
		"test/loss": tf.keras.metrics.Mean(),
	}
	optimizer = tf.keras.optimizers.Adam()

	single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["T (degC)"])

	flow_model = RealNVP(num_coupling_layers=2, input_dim=18)
	flow_model.build(input_shape=(None, 18))
	max_train_nll, min_train_nll = train_flow(flow_model, single_step_window)
	M_yy, M_yx1 = compile_and_fit(
		model, flow_model, max_train_nll, min_train_nll, single_step_window, optimizer, metrics
	)

	wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=["T (degC)"])

	list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(
		model, M_yy, M_yx1, flow_model, max_train_nll, min_train_nll, wide_window.test, metrics
	)

	with open("out/pickle/robust_regression_iid" + str(idx) + ".pkl", "wb") as fp:
		pickle.dump(list_y_hats, fp)
	with open("out/pickle/robust_regression_labels_iid" + str(idx) + ".pkl", "wb") as fp:
		pickle.dump(list_y_true, fp)

	wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

	plt.savefig("out/robust_regression_iid.png")
	plt.close()

	list_x_test, list_y_true, list_y_hats, list_out_nstd, list_out_pstd = test(
		model, M_yy, M_yx1, flow_model, max_train_nll, min_train_nll, wide_window.test_ood, metrics
	)

	with open("out/pickle/robust_regression_ood" + str(idx) + ".pkl", "wb") as fp:
		pickle.dump(list_y_hats, fp)
	with open("out/pickle/robust_regression_labels_ood" + str(idx) + ".pkl", "wb") as fp:
		pickle.dump(list_y_true, fp)

	wide_window.plot(list_x_test, list_y_true, list_y_hats[0], list_out_nstd, list_out_pstd)

	plt.savefig("out/robust_regression_ood.png")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--exp_idx", help="Index of experiment")
	bash_args = parser.parse_args()
	main(bash_args.exp_idx)
