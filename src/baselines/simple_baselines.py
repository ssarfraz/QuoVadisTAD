from typing import Optional, Union

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA

from baselines.utils import check_timeseries_shape


class SensorRangeDeviation:
    def __init__(
            self,
            sensor_range: Optional[Union[tuple[np.ndarray, np.ndarray], tuple[int, int]]] = None,
            count_sensors: bool = False
    ) -> None:
        """Deviation from the usual range of the sensors. The ranges are either provided or else are computed on the
        training dataset as the minimum and maximum values of each sensor. The total anomaly score across sensors is
        defined as the existence or not of a deviating sensor or as number of sensors deviating from their range.

        Args:
            sensor_range: An optional range the user can input. Its form is a tuple of arrays of the form:
                (sensors_minima, sensors_maxima) or just a tuple (min, max) defining a shared range for all sensors.
            count_sensors: If true, the total count of deviating sensors is used to compute the anomaly.
        """
        self.sensor_range = sensor_range
        self.count_sensors = count_sensors

    def fit(self, x: np.ndarray):
        check_timeseries_shape(x)

        if self.sensor_range is None:
            self.sensor_range = (x.min(axis=0), x.max(axis=0))

    def transform(self, x: np.ndarray):
        check_timeseries_shape(x)

        sensor_min, sensor_max = self.sensor_range

        outside_range = (x < sensor_min) | (sensor_max < x)

        any_sensor_outside_range = (
            outside_range.sum(axis=1)
            if self.count_sensors
            else
            outside_range.max(axis=1)
        )

        return any_sensor_outside_range


class LNorm:
    def __init__(self, ord: int = 2):
        """Computes the L^n norm as the anomaly score of a sequence. This is directly applied to the test set. The
        method defaults to the L2 norm, but the power can be changed if needed.

        Args:
            ord: The power n of the L^n norm. Defaults to 2.
        """
        self.ord = ord

    def fit(self, x: np.ndarray):
        check_timeseries_shape(x)

        print(f'No fitting applied in the L2-Norm method.')

    def transform(self, x: np.ndarray):
        check_timeseries_shape(x)

        return np.linalg.norm(x, ord=self.ord, axis=1)


class NNDistance:
    def __init__(self, distance: str = 'euclidean'):
        """Computes an anomaly score as the distance to the nearest neighbor in the train set.

        Args:
            distance: The distance metric to be used, defaults to Euclidean.
        """
        self.distance = distance
        self.train_data = None

    def fit(self, x: np.ndarray):
        check_timeseries_shape(x)

        self.train_data = x

    def transform(self, x: np.ndarray):
        check_timeseries_shape(x)

        neighbor_distances = metrics.pairwise.pairwise_distances(
            x,
            self.train_data,
            metric=self.distance
        )

        return neighbor_distances.min(axis=1)


class PCAError:
    def __init__(self, pca_dim: int = 30, svd_solver: str = 'full'):
        """Evaluates an anomaly score as the reconstruction error from a PCA projection. Only a small number of components
            is used in order to remove information and the PCA parameters are fit on the train set.

        Args:
            pca_dim: The number of pca components to keep.
            svd_solver: The solver used to estimate the pca parameters.
        """
        self.pca = PCA(n_components=pca_dim, svd_solver=svd_solver)

    def fit(self, x: np.ndarray):
        self.pca.fit(x)

    def transform(self, x: np.ndarray):
        latent = self.pca.transform(x)
        reconstructed = self.pca.inverse_transform(latent)

        return np.abs(x - reconstructed)


class Random:
    def __init__(self, seed: Optional[int] = None):
        """The method randomly selects an anomaly value of 0 or 1 per timestamp. The method does not make sense in any
        practical setting, but it is useful to expose issues in scoring, for example when using F1 score with
        point adjust (PA).

        Args:
            seed: An optional random seed for repeatability.
        """
        self.seed = seed

    def fit(self, x: np.ndarray):

        check_timeseries_shape(x)

        print(f'No fitting applied in the random method.')
        pass

    def transform(self, x: np.ndarray):
        np.random.seed(self.seed)
        return np.random.randint(low=0, high=2, size=x.shape[0])
