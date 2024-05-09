from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def add_anomalies_to_univariate_series(
	x: np.ndarray,
	normal_duration_rate: float,
	anomaly_duration_rate: float,
	anomaly_size_range: tuple[float, float],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
	"""Add anomalies to a given time series.

	Args:
		x: The series to add anomalies to.
		normal_duration_rate: Average duration of a normal interval.
		anomaly_duration_rate: Average duration of an anomalous interval.
		anomaly_size_range: A range where the magnitude of the anomaly lies.
			E.g. if this is (0.5, 0.8), then a random value in that interval with be
			added or subtracted from the series in the anomaly interval.

	Returns:
		x: A copy of the original array which has anomalies added to it.
		anomaly_intervals: A list of tuples which represent the (start, end) of the anomaly intervals.
	"""
	# Validate the anomaly size range.
	if anomaly_size_range[0] >= anomaly_size_range[1]:
		raise ValueError(
			f"The anomaly size range {anomaly_size_range} should be strictly increasing."
		)

	# Copy x in order to not overwrite it.
	x = x.copy()
	N = len(x)
	# Define two exponential distributions which describe the lengths of normal and anomalous intervals.
	# So e.g. stats.expon(scale=20) will sample a duration of an anomalous interval with mean 20.
	distr_duration_normal = stats.expon(scale=normal_duration_rate)
	distr_duration_anomalous = stats.expon(scale=anomaly_duration_rate)

	# Loop over a max number of intervals and add the anomalies.
	max_number_of_intervals = 8
	location = 0
	anomaly_intervals = []
	for _ in range(max_number_of_intervals):
		# First sample a normal interval. The anomaly will start at the end of it.
		random_states = np.random.randint(0, np.iinfo(np.int32).max, size=2)
		anom_start = location + int(
			distr_duration_normal.rvs(random_state=random_states[0])
		)
		# Then sample an anomalous interval. The anomaly will end at the end of it.
		anom_end = anom_start + int(
			distr_duration_anomalous.rvs(random_state=random_states[1])
		)
		# Make sure we don't exceed the length of the series.
		anom_end = min(N, anom_end)

		if anom_start >= N:
			break

		# The anomaly shifts the signal up or down to the interval [-0.8, -0.5] or [0.5, 0.8].
		shift_sign = 1 if np.random.randint(low=0, high=2) == 1 else -1
		shift = shift_sign * np.random.uniform(
			anomaly_size_range[0], anomaly_size_range[1], size=anom_end - anom_start
		)
		x[anom_start:anom_end] += shift
		# Update the location to the end of the anomaly.
		location = anom_end

		# mark the indices of anomaly for creating labels
		anomaly_intervals.append((anom_start, anom_end))

	return x, anomaly_intervals


def synthetic_dataset_with_out_of_range_anomalies(
	number_of_sensors: int = 1,
	train_size: int = 5_000,
	test_size: int = 1000,
	nominal_data_mean: float = 0.0,
	nominal_data_std: float = 0.1,
	normal_duration_rate: float = 400.0,
	anomaly_duration_rate: float = 20.0,
	anomaly_size_range: tuple = (0.5, 0.8),
	ratio_of_anomalous_sensors: float = 0.2,
	seed: Optional[int] = None
) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
	"""Generate a synthetic dataset with out-of-range anomalies. Normal data are i.i.d. distributed in time based on
	a normal distribution. The test data are generated the same way and then anomalies are added to some randomly
	selected sensors. The anomalies appear as shifts away of the mean of the normal distribution in some intervals
	whose starts are selected based on an exponential distribution. All those parameters can be controlled in the
	function input and are set to some reasonable defaults.

	Args:
		number_of_sensors: The number of sensors of the dataset. To generate univariate datasets, just set this to 1.
		train_size: The size of the nominal training series in timestamps.
		test_size: The size of the anomalous test series in timestamps.
		nominal_data_mean: The mean of the normal distribution defining nominal data.
		nominal_data_std: The standard deviation of the normal distribution defining nominal data.
		normal_duration_rate: Average duration of a normal interval in the anomalous test data.
		anomaly_duration_rate: Average duration of an anomalous interval in the anomalous test data.
		anomaly_size_range: A range where the magnitude of the anomaly lies.
			E.g. if this is (0.5, 0.8), then a random value in that interval with be
			added or subtracted from the series in the anomaly interval.
		ratio_of_anomalous_sensors: The ratio of sensors which have anomalies in the test set.
		seed: Random seed for reproducibility.

	Returns:
		dataset: A dictionary of the form {'train': train, 'test': test, 'labels': labels} containing all the
			information of the generated dataset.
		anomaly_intervals: Lists of tuples which represent the (start, end) of the anomaly intervals. They are in a
			dictionary which maps the anomalous sensor indices to the corresponding anomaly intervals.
	"""
	# Fix the random state of numpy.
	np.random.seed(seed)

	# Generate the nominal train data. Just a multivariate series of length `train_size` with `number_of_sensors`
	# features which are independently sampled from the same normal distribution.
	train = np.random.normal(
		nominal_data_mean,
		nominal_data_std,
		size=(train_size, number_of_sensors)
	)

	# Generate the test data the same way as the train data.
	test = np.random.normal(
		nominal_data_mean,
		nominal_data_std,
		size=(test_size, number_of_sensors)
	)

	# Add some anomalies to randomly selected sensors.
	number_of_sensors_with_anomalies = max(1, int(round(number_of_sensors * ratio_of_anomalous_sensors)))
	sensors_with_anomalies = np.random.choice(number_of_sensors, number_of_sensors_with_anomalies, replace=False)

	# Create labels which capture the anomalies. Also capture the locations as intervals for visualization purposes.
	all_locations = {}
	labels = np.zeros_like(test)
	for idx in sensors_with_anomalies:
		test[:, idx], anomaly_locations = add_anomalies_to_univariate_series(
			test[:, idx],
			normal_duration_rate=normal_duration_rate,
			anomaly_duration_rate=anomaly_duration_rate,
			anomaly_size_range=anomaly_size_range,
		)

		for start, end in anomaly_locations:
			labels[start:end, idx] = 1

		all_locations[idx] = anomaly_locations

	dataset = {'train': train, 'test': test, 'labels': labels}
	anomaly_intervals = [all_locations.get(i, []) for i in range(test.shape[1])]

	return dataset, anomaly_intervals


def plot_series_and_predictions(
	series: np.ndarray,
	gt_anomaly_intervals: list[list[tuple[int, int]]],
	predictions: Optional[dict[str, np.ndarray]] = None,
	single_series_figsize: tuple[int, int] = (10, 1),
	gt_ylim: tuple[int, int] = (-1, 1),
	gt_color: str = 'steelblue',
	prediction_color: str = 'orange',
	prediction_circle_size: float = 0.2,
	prediction_ylim: tuple[int, int] = (-1, 1)
) -> None:
	n_sensors = series.shape[1]
	fig, ax = plt.subplots(
		n_sensors,
		ncols=1,
		figsize=(single_series_figsize[0], (n_sensors + 1)*single_series_figsize[1]),
		squeeze=False
	)
	fig.suptitle('Ground truth test data.', fontsize=14)
	ax = [a[0] for a in ax]

	for i in range(n_sensors):
		ax[i].set_title(f'Sensor {i}', fontsize=8)
		ax[i].plot(series[:, i], color=gt_color)
		ax[i].set_ylim(gt_ylim)

		for start, end in gt_anomaly_intervals[i]:
			ax[i].axvspan(start, end, alpha=0.2, color=gt_color)

	plt.tight_layout()
	plt.show()

	if predictions is not None:
		n_methods = len(predictions)
		fig, ax = plt.subplots(
			n_methods,
			ncols=1,
			figsize=(single_series_figsize[0], (n_methods + 1) * single_series_figsize[1]),
			squeeze=False
		)
		fig.suptitle('Predictions', fontsize=14)
		ax = [a[0] for a in ax]

		for i, [method_name, pred] in enumerate(predictions.items()):
			ax[i].set_title(method_name, fontsize=8)
			ax[i].scatter(
				np.arange(len(pred)),
				pred,
				s=prediction_circle_size,
				color=prediction_color
			)
			ax[i].set_ylim(prediction_ylim)

		plt.tight_layout()
		plt.show()
