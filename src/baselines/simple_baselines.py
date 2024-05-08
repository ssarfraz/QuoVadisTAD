from typing import Optional, Union

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA

from src.baselines.utils import check_timeseries_shape, TADMethodEstimator


class SensorRangeDeviation(TADMethodEstimator):
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

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape(x)

        if self.sensor_range is None:
            self.sensor_range = (x.min(axis=0), x.max(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_timeseries_shape(x)

        sensor_min, sensor_max = self.sensor_range

        outside_range = ((x < sensor_min) | (sensor_max < x)).astype(np.float32)

        any_sensor_outside_range = (
            outside_range.sum(axis=1)
            if self.count_sensors
            else
            outside_range.max(axis=1)
        )

        return any_sensor_outside_range


class LNorm(TADMethodEstimator):
    def __init__(self, ord: int = 2) -> None:
        """Computes the L^n norm as the anomaly score of a sequence. This is directly applied to the test set. The
        method defaults to the L2 norm, but the power can be changed if needed.

        Args:
            ord: The power n of the L^n norm. Defaults to 2.
        """
        self.ord = ord

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape(x)
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_timeseries_shape(x)

        return np.linalg.norm(x, ord=self.ord, axis=1)


class NNDistance(TADMethodEstimator):
    def __init__(self, distance: str = 'euclidean') -> None:
        """Computes an anomaly score as the distance to the nearest neighbor in the train set.

        Args:
            distance: The distance metric to be used, defaults to Euclidean.
        """
        self.distance = distance
        self.train_data = None

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape(x)

        self.train_data = x

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_timeseries_shape(x)

        neighbor_distances = metrics.pairwise.pairwise_distances(
            x,
            self.train_data,
            metric=self.distance
        )

        return neighbor_distances.min(axis=1)


class PCAError(TADMethodEstimator):
    def __init__(self, pca_dim: Union[int, str] = 'auto', svd_solver: str = 'full') -> None:
        """Evaluates an anomaly score as the reconstruction error from a PCA projection. Only a small number of
            components is used in order to remove information and the PCA parameters are fit on the train set.

        Args:
            pca_dim: The number of pca components to keep. If set to auto, it will use 10 components on datasets with
                less than 50 features, else 30 components and 2 components on univariate datasets.
            svd_solver: The solver used to estimate the pca parameters.
        """
        self.pca_dim = pca_dim
        self.svd_solver = svd_solver

        if pca_dim == 'auto':
            self.pca = None
        else:
            self.pca = PCA(n_components=self.pca_dim, svd_solver=self.svd_solver)

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape(x)

        if self.pca_dim == 'auto':
            n_features = x.shape[1]
            if univariate:
                dim = 2
            elif n_features <= 50:
                dim = 10
            else:
                dim = 30

            if dim > x.shape[1]:
                # If the number of components is too large, then use a simple heuristic to select it.
                old_dim = dim
                dim = min(max(2, x.shape[1] // 5), x.shape[1])
                print(f'Adjusting estimated number of PCA components from {old_dim} to {dim}.')

            self.pca = PCA(n_components=dim, svd_solver=self.svd_solver)
        self.pca.fit(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_timeseries_shape(x)

        latent = self.pca.transform(x)
        reconstructed = self.pca.inverse_transform(latent)

        return np.abs(x - reconstructed)


class Random(TADMethodEstimator):
    def __init__(self, seed: Optional[int] = None) -> None:
        """The method randomly selects an anomaly value of 0 or 1 per timestamp. The method does not make sense in any
        practical setting, but it is useful to expose issues in scoring, for example when using F1 score with
        point adjust (PA).

        Args:
            seed: An optional random seed for repeatability.
        """
        self.seed = seed

    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        check_timeseries_shape(x)
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_timeseries_shape(x)
        np.random.seed(self.seed)

        return np.random.randint(
            low=0,
            high=2,
            size=x.shape[0]
        )
