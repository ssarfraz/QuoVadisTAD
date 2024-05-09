import abc

import numpy as np


def check_timeseries_shape(timeseries: np.ndarray):
    """Check that the input series is a 2D array."""
    if len(timeseries.shape) != 2:
        raise ValueError(
            f'Expected a 2D timeseries array with timestamps and features, '
            f'instead received an input of shape: {timeseries.shape}')


class TADMethodEstimator:
    @abc.abstractmethod
    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        pass

    @abc.abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass
