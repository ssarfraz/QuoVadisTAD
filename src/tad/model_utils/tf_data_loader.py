import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import numpy as np


def create_tf_dataset(
	data_array: np.ndarray,
	input_sequence_length: int,
	forecast_horizon: int,
	batch_size: int = 128,
	reconstruction_dataset=False,
	shuffle=False,
	multi_horizon=False,
	input_for_gnn=False,
):
	"""Creates tensorflow dataset from numpy array.

	This function creates a dataset where each element is a tuple `(inputs, targets)`.
	`inputs` is a Tensor
	of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
	the `input_sequence_length` past values of the timeseries for each node.
	`targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
	containing the `forecast_horizon`
	future values of the timeseries for each node.

	Args:
	    data_array: np.ndarray with shape `(num_time_steps, num_routes)`
	    input_sequence_length: Length of the input sequence (in number of timesteps).
	    forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
	        `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
	        timeseries `forecast_horizon` steps ahead (only one value).

	    reconstruction_dataset: False for forecasting mode. if True will return the input batch as target (for reconstruction tasks, auto encoders etc.,)
	    batch_size: Number of timeseries samples in each batch.
	    shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
	    multi_horizon: See `forecast_horizon`.

	Returns:
	    A tf.data.Dataset instance.
	"""
	if reconstruction_dataset:
		shuffle_if = shuffle
	else:
		shuffle_if = False

	inputs = timeseries_dataset_from_array(
		data_array[:-forecast_horizon],
		# np.expand_dims(data_array[:-forecast_horizon], axis=-1),
		None,
		sequence_length=input_sequence_length,
		shuffle=shuffle_if,
		batch_size=batch_size,
	)

	if not reconstruction_dataset:
		target_offset = (
			input_sequence_length
			if multi_horizon
			else input_sequence_length + forecast_horizon - 1
		)
		target_seq_length = forecast_horizon if multi_horizon else 1
		targets = timeseries_dataset_from_array(
			data_array[target_offset:],
			None,
			sequence_length=target_seq_length,
			shuffle=False,
			batch_size=batch_size,
		)
	else:
		targets = inputs
	# labels = label_array[target_offset:]
	dataset = tf.data.Dataset.zip((inputs, targets))
	if shuffle:
		dataset = dataset.shuffle(100)

	return dataset.prefetch(16).cache()
