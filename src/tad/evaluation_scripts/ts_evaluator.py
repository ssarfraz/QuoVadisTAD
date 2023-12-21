from typing import List, Callable, Tuple, Dict, Optional, Any, Union
from enum import Enum

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc



def _constant_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a constant bias function that assigns the same weight to all positions.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{1}{n} \sum_{i = 1}^{n} \text{inputs}_i,

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    return torch.sum(inputs).item() / inputs.shape[0]


def _back_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a bias function that assigns the more weight to predictions towards the back of a
    ground-truth anomaly window.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{2}{n * (n + 1)} \sum_{i = 1}^{n} \text{inputs}_i \cdot i,

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    n = inputs.shape[0]
    res = torch.dot(inputs, torch.arange(1, n + 1, dtype=inputs.dtype, device=inputs.device)).item()
    res /= (n * (n + 1)) // 2  # sum of numbers 1, ..., n
    return res


def _front_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a bias function that assigns the more weight to predictions towards the front of a
    ground-truth anomaly window.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{2}{n * (n + 1)} \sum_{i = 1}^{n} \text{inputs}_i \cdot (n + 1 - i),

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    n = inputs.shape[0]
    res = torch.dot(inputs, torch.arange(n, 0, -1, dtype=inputs.dtype, device=inputs.device)).item()
    res /= (n * (n + 1)) // 2  # sum of numbers 1, ..., n
    return res


def _middle_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a bias function that assigns the more weight to predictions in the middle of a
    ground-truth anomaly window.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{2}{m * (m + 1) + (n - m) * (n - m + 1)} \sum_{i = 1}^{n} \text{inputs}_i \cdot
        \begin{cases}
            i & \text{if } i \leq m\\
            (n + 1 - i) & \text{otherwise}
        \end{cases},

    where :math:`n = \lvert \text{inputs} \rvert` and :math:`m = \lceil \frac{n}{2} \rceil`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    n = inputs.shape[0]
    result = torch.empty_like(inputs)
    middle, remainder = divmod(n, 2)
    middle2 = middle + remainder
    torch.arange(1, middle + 1, out=result[:middle], dtype=result.dtype, device=result.device)
    torch.arange(middle2, 0, -1, out=result[-middle2:], dtype=result.dtype, device=result.device)
    result = torch.dot(inputs, result).item()
    result /= (middle * (middle + 1) + middle2 * (middle2 + 1)) // 2
    return result


class BiasFunction(Enum):
    r"""
    An enumeration of possible Bias functions, that can be used in the `ts_precision_and_recall` function.
    The functions differ in how they assign weights to predictions within a single ground-truth window.
    """
    __func_sig = Callable[[torch.Tensor], float]
    CONSTANT: __func_sig = _constant_bias_fn
    FRONT: __func_sig = _front_bias_fn
    MIDDLE: __func_sig = _middle_bias_fn
    BACK: __func_sig = _back_bias_fn


def _inverse_proportional_cardinality_fn(cardinality: int, gt_length: int) -> float:
    r"""
    Cardinality function that assigns an inversely proportional weight to predictions within a single ground-truth
    window.

    This is the default cardinality function recommended in [Tatbul2018]_.

    .. note::
       This function leads to a metric that is not recall-consistent! Please see [Wagner2023]_ for more details.

    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window (unused).
    :return: The cardinality factor :math:`\frac{1}{\text{cardinality}}`.

    .. [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
        Precision and recall for time series. Advances in neural information processing systems. 2018;31.
    .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
        TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
        Transactions on Machine Learning Research (TMLR), (to appear) 2023.
    """
    return 1 / max(1, cardinality)


def _improved_cardinality_fn(cardinality: int, gt_length: int):
    r"""
    Recall-consistent cardinality function introduced by [Wagner2023]_ that assigns lower weight to ground-truth windows
    that overlap with many predicted windows.

    This function computes

    .. math::
        \left(\frac{\text{gt_length} - 1}{\text{gt_length}}\right)^{\text{cardinality} - 1}.

    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window.
    :return: The cardinality factor.
    """
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)


class CardinalityFunction(Enum):
    r"""
    An enumeration of possible cardinality functions, that can be used in the `ts_precision_and_recall` function.
    """
    __func_sig = Callable[[int, int], float]
    INVERSE_PROPORTIONAL: __func_sig = _inverse_proportional_cardinality_fn
    IMPROVED: __func_sig = _improved_cardinality_fn


def compute_window_indices(binary_labels: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Compute a list of indices where anomaly windows begin and end.

    :param binary_labels: A 1-D :class:`~torch.Tensor` containing ``1`` for an anomalous time step or ``0`` otherwise.
    :return: A list of tuples ``(start, end)`` for each anomaly window in ``binary_labels``, where ``start`` is the
        index at which the window starts and ``end`` is the first index after the end of the window.
    """
    boundaries = torch.empty_like(binary_labels)
    boundaries[0] = 0
    boundaries[1:] = binary_labels[:-1]
    boundaries *= -1
    boundaries += binary_labels
    # boundaries will be 1 where a window starts and -1 at the end of a window

    indices = torch.nonzero(boundaries, as_tuple=True)[0].tolist()
    if len(indices) % 2 != 0:
        # Add the last index as the end of a window if appropriate
        indices.append(binary_labels.shape[0])
    indices = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]

    return indices


def _compute_overlap(preds: torch.Tensor, pred_indices: List[Tuple[int, int]],
                     gt_indices: List[Tuple[int, int]], alpha: float,
                     bias_fn: Callable, cardinality_fn: Callable,
                     use_window_weight: bool = False) -> float:
    n_gt_windows = len(gt_indices)
    n_pred_windows = len(pred_indices)
    total_score = 0.0
    total_gt_points = 0

    i = j = 0
    while i < n_gt_windows and j < n_pred_windows:
        gt_start, gt_end = gt_indices[i]
        window_length = gt_end - gt_start
        total_gt_points += window_length
        i += 1

        cardinality = 0
        while j < n_pred_windows and pred_indices[j][1] <= gt_start:
            j += 1
        while j < n_pred_windows and pred_indices[j][0] < gt_end:
            j += 1
            cardinality += 1

        if cardinality == 0:
            # cardinality == 0 means no overlap at all, hence no contribution
            continue

        # The last predicted window that overlaps our current window could also overlap the next window.
        # Therefore, we must consider it again in the next loop iteration.
        j -= 1

        cardinality_multiplier = cardinality_fn(cardinality, window_length)

        prediction_inside_ground_truth = preds[gt_start:gt_end]
        # We calculate omega directly in the bias function, because this can greatly improve running time
        # for the constant bias, for example.
        omega = bias_fn(prediction_inside_ground_truth)

        # Either weight evenly across all windows or based on window length
        weight = window_length if use_window_weight else 1

        # Existence reward (if cardinality > 0 then this is certainly 1)
        total_score += alpha * weight
        # Overlap reward
        total_score += (1 - alpha) * cardinality_multiplier * omega * weight

    denom = total_gt_points if use_window_weight else n_gt_windows

    return total_score / denom


def ts_precision_and_recall(anomalies: torch.Tensor, predictions: torch.Tensor, alpha: float = 0,
                            recall_bias_fn: BiasFunction = BiasFunction.CONSTANT,
                            recall_cardinality_fn: CardinalityFunction = CardinalityFunction.INVERSE_PROPORTIONAL,
                            precision_bias_fn: Optional[BiasFunction] = None,
                            precision_cardinality_fn: Optional[CardinalityFunction] = None,
                            anomaly_ranges: Optional[List[Tuple[int, 2]]] = None,
                            prediction_ranges: Optional[List[Tuple[int, 2]]] = None,
                            weighted_precision: bool = False) -> Tuple[float, float]:
    """
    Computes precision and recall for time series as defined in [Tatbul2018]_.

    .. note::
       The default parameters for this function correspond to the defaults recommended in [Tatbul2018]_. However,
       those might not be desirable in most cases, please see [Wagner2023]_ for a detailed discussion.

    :param anomalies: Binary 1-D :class:`~torch.Tensor` of shape ``(length,)`` containing the true labels.
    :param predictions: Binary 1-D :class:`~torch.Tensor` of shape ``(length,)`` containing the predicted labels.
    :param alpha: Weight for existence term in recall.
    :param recall_bias_fn: Function that computes the bias term for a given ground-truth window.
    :param recall_cardinality_fn: Function that compute the cardinality factor for a given ground-truth window.
    :param precision_bias_fn: Function that computes the bias term for a given predicted window.
        If ``None``, this will be the same as ``recall_bias_function``.
    :param precision_cardinality_fn: Function that computes the cardinality factor for a given predicted window.
        If ``None``, this will be the same as ``recall_cardinality_function``.
    :param weighted_precision: If True, the precision score of a predicted window will be weighted with the
        length of the window in the final score. Otherwise, each window will have the same weight.
    :param anomaly_ranges: A list of tuples ``(start, end)`` for each anomaly window in ``anomalies``, where ``start``
        is the index at which the window starts and ``end`` is the first index after the end of the window. This can
        be ``None``, in which case the list is computed automatically from ``anomalies``.
    :param prediction_ranges: A list of tuples ``(start, end)`` for each anomaly window in ``predictions``, where
        ``start`` is the index at which the window starts and ``end`` is the first index after the end of the window.
        This can be ``None``, in which case the list is computed automatically from ``predictions``.
    :return: A tuple consisting of the time-series precision and recall for the given labels.
    """
    has_anomalies = torch.any(anomalies > 0).item()
    has_predictions = torch.any(predictions > 0).item()

    # Catch special cases which would cause a division by zero
    if not has_predictions and not has_anomalies:
        # In this case, the classifier is perfect, so it makes sense to set precision and recall to 1
        return 1, 1
    elif not has_predictions or not has_anomalies:
        return 0, 0

    # Set precision functions to the same as recall functions if they are not given
    if precision_bias_fn is None:
        precision_bias_fn = recall_bias_fn
    if precision_cardinality_fn is None:
        precision_cardinality_fn = recall_cardinality_fn

    if anomaly_ranges is None:
        anomaly_ranges = compute_window_indices(anomalies)
    if prediction_ranges is None:
        prediction_ranges = compute_window_indices(predictions)

    recall = _compute_overlap(predictions, prediction_ranges, anomaly_ranges, alpha, recall_bias_fn,
                              recall_cardinality_fn)
    precision = _compute_overlap(anomalies, anomaly_ranges, prediction_ranges, 0, precision_bias_fn,
                                 precision_cardinality_fn, use_window_weight=weighted_precision)

    return precision, recall


## Main Evaluator


class Evaluator:
    """
    A class that can compute several evaluation metrics for a dataset. Each method returns the score as a single float,
    but it can also return additional information in a dict.
    """

    def rocauc(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the classic point-wise area under the receiver operating characteristic curve.
        
        This will return a value between 0 and 1 where 1 indicates a perfect classifier.

        .. seealso::
            Scikit-learns's :func:`~sklearn.metrics.roc_auc_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :return: A tuple consisting of the AUC score and an empty dict.
        """

        return roc_auc_score(labels.numpy(), scores.numpy()), {}

    def f1_score(self, labels: torch.Tensor, scores: torch.Tensor, pos_label: int = 1) -> Tuple[float, Dict[str, Any]]:
        """Compute the classic point-wise F1 score.
        
        This will return a value between 0 and 1 where 1 indicates a perfect classifier.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.f1_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing binary predictions of whether a point is an anomaly or
            not.
        :param pos_label: Class to report.
        :return: A tuple consisting of the F1 score and an empty dict.
        """

        return f1_score(labels.numpy(), scores.numpy(), pos_label=pos_label).item(), {}

    def best_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float) -> Dict[str, Union[float, Any]]:
        r"""
        Compute the classic point-wise :math:`F_{\beta}` score.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{\beta}`
        score for the resulting binary predictions. It then returns the highest score.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.fbeta_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :param beta: Positive number that determines the trade-off between precision and recall when computing the
            F-score. :math:`\beta = 1` assigns equal weight to both while :math:`\beta < 1` emphasizes precision and
            vice versa.
        :return: A tuple consisting of the best :math:`F_{\beta}` score and a dict containing the threshold that
            produced the maximal score.
        """
        precision, recall, thresholds = precision_recall_curve(labels.numpy(), scores.numpy())

        f_score = np.nan_to_num((1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall), nan=0)
        best_index = np.argmax(f_score)
        area = self.auprc(labels, scores)

        return dict(f1=f_score[best_index].item(),
                    precision=precision[best_index].item(),
                    recall=recall[best_index].item(),
                    auprc=area[0],
                    threshold=thresholds[best_index].item())

    def best_f1_score(self, labels: torch.Tensor, scores: torch.Tensor) -> Dict[str, Union[float, Any]]:
        r"""
        Compute the classic point-wise :math:`F_{1}` score.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{1}`
        score for the resulting binary predictions. It then returns the highest score.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.f1_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :return: A tuple consisting of the best :math:`F_{1}` score and a dict containing the threshold that
            produced the maximal score.
        """

        return self.best_fbeta_score(labels, scores, 1)

    def auprc(self, labels: torch.Tensor, scores: torch.Tensor, integration: str = 'trapezoid') -> Tuple[
        float, Dict[str, Any]]:
        r"""
        Compute the classic point-wise area under the precision-recall curve.

        This will return a value between 0 and 1 where 1 indicates a perfect classifier.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.average_precision` function.

            Scikit-learn's :func:`~sklearn.metrics.precision_recall_curve` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :param integration: Method to use for computing the area under the curve. ``'riemann'`` corresponds to a simple
            Riemann sum, whereas ``'trapezoid'`` uses the trapezoidal rule.
        :return: A tuple consisting of the AuPRC score and an empty dict.
        """
        precision, recall, thresholds = precision_recall_curve(labels.numpy(), scores.numpy())
        # recall is nan in the case where all ground-truth labels are 0. Simply set it to zero here
        # so that it does not contribute to the area
        recall = np.nan_to_num(recall, nan=0)

        if integration == 'riemann':
            area = -np.sum(np.diff(recall) * precision[:-1])
        else:
            area = auc(recall, precision)

        return area.item(), {}

    def average_precision(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the classic point-wise average precision score.

        .. note::
            This is just a shorthand for :meth:`auprc` with ``integration='riemann'``.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.average_precision` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the average precision score and an empty dict.
        """
        return self.auprc(labels, scores, integration='riemann')

    def ts_auprc(self, labels: torch.Tensor, scores: torch.Tensor, integration='trapezoid',
                 weighted_precision: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the area under the precision-recall curve using precision and recall for time series [Tatbul2018]_.

        .. note::
            This function uses the improved cardinality function described in [Wagner2023]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param integration: Method to use for computing the area under the curve. ``'riemann'`` corresponds to a simple
            Riemann sum, whereas ``'trapezoid'`` uses the trapezoidal rule.
        :param weighted_precision: If ``True``, the precision score of a predicted window will be weighted with the
            length of the window in the final score. Otherwise, each window will have the same weight.
        :return: A tuple consisting of the AuPRC score and an empty dict.

        .. [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
            Precision and recall for time series. Advances in neural information processing systems. 2018;31.
        .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
            TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
            Transactions on Machine Learning Research (TMLR), (to appear) 2023.
        """
        thresholds = torch.unique(input=scores, sorted=True)

        precision = torch.empty(thresholds.shape[0] + 1, dtype=torch.float, device=thresholds.device)
        recall = torch.empty(thresholds.shape[0] + 1, dtype=torch.float, device=thresholds.device)
        predictions = torch.empty_like(scores, dtype=torch.long)

        # Set last values when threshold is at infinity so that no point is predicted as anomalous.
        # Precision is not defined in this case, we set it to 1 to stay consistent with scikit-learn
        precision[-1] = 1
        recall[-1] = 0

        label_ranges = compute_window_indices(labels)

        for i, t in enumerate(thresholds):
            torch.greater_equal(scores, t, out=predictions)
            prec, rec = ts_precision_and_recall(labels, predictions, alpha=0,
                                                recall_cardinality_fn=CardinalityFunction.IMPROVED,
                                                anomaly_ranges=label_ranges,
                                                weighted_precision=weighted_precision)
            precision[i] = prec
            recall[i] = rec

        if integration == 'riemann':
            area = -torch.sum(torch.diff(recall) * precision[:-1])
        else:
            area = auc(recall.numpy(), precision.numpy())

        return area.item(), {}

    def ts_average_precision(self, labels: torch.Tensor, scores: torch.Tensor, weighted_precision: bool = True) \
            -> Tuple[float, Dict[str, Any]]:
        """
        Compute the average precision score using precision and recall for time series [Tatbul2018]_.

        .. note::
            This is just a shorthand for :meth:`ts_auprc` with ``integration='riemann'``.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param weighted_precision: If ``True``, the precision score of a predicted window will be weighted with the
            length of the window in the final score. Otherwise, each window will have the same weight.
        :return: A tuple consisting of the average precision score and an empty dict.
        """

        return self.ts_auprc(labels, scores, integration='riemann', weighted_precision=weighted_precision)

    def ts_auprc_unweighted(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the area under the precision-recall curve using precision and recall for time series [Tatbul2018]_.

        .. note::
            This is just a shorthand for :meth:`ts_auprc` with ``integration='riemann'`` and
            ``weighted_precision=False``.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the AuPRC score and an empty dict.
        """
        return self.ts_auprc(labels, scores, integration='trapezoid', weighted_precision=False)

    def __best_ts_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float,
                              recall_cardinality_fn: CardinalityFunction = CardinalityFunction.IMPROVED,
                              weighted_precision: bool = True) -> Dict[str, Union[float, Any]]:
        thresholds = torch.unique(input=scores, sorted=True)

        precision = torch.empty_like(thresholds, dtype=torch.float)
        recall = torch.empty_like(thresholds, dtype=torch.float)
        predictions = torch.empty_like(scores, dtype=torch.long)

        label_ranges = compute_window_indices(labels)
        # label_ranges = None

        for i, t in enumerate(thresholds):
            torch.greater(scores, t, out=predictions)
            prec, rec = ts_precision_and_recall(labels, predictions, alpha=0,
                                                recall_cardinality_fn=recall_cardinality_fn,
                                                anomaly_ranges=label_ranges,
                                                weighted_precision=weighted_precision)

            # We need to handle the case where precision and recall are both 0. This can either happen for an
            # extremely bad classifier or if all predictions are 0
            if prec == rec == 0:
                # We simply set rec = 1 to avoid dividing by zero. The F-score will still be 0
                rec = 1

            precision[i] = prec
            recall[i] = rec

        f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        max_score_index = torch.argmax(f_score)
        area = self.ts_auprc(labels, scores, integration='trapezoid', weighted_precision=weighted_precision)  #

        return dict(f1=f_score[max_score_index].item(),
                    precision=precision[max_score_index].item(),
                    recall=recall[max_score_index].item(),
                    auprc=area[0],
                    threshold=thresholds[max_score_index].item())

    def best_ts_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float) -> Dict[
        str, Union[float, Any]]:
        r"""
        Compute the :math:`F_{\beta}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{\beta}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the improved cardinality function and weighted precision as described in [Wagner2023]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param beta: Positive number that determines the trade-off between precision and recall when computing the
            F-score. :math:`\beta = 1` assigns equal weight to both while :math:`\beta < 1` emphasizes precision and
            vice versa.
        :return: A tuple consisting of the best :math:`F_{\beta}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.__best_ts_fbeta_score(labels, scores, beta, recall_cardinality_fn=CardinalityFunction.IMPROVED,
                                          weighted_precision=True)

    def best_ts_fbeta_score_classic(self, labels: torch.Tensor, scores: torch.Tensor, beta: float) -> Dict[
        str, Union[float, Any]]:
        r"""
        Compute the :math:`F_{\beta}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{\beta}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the default cardinality function (:math:`\frac[1}{x}`) and unweighted precision, i.e.,
            the default parameters described in [Tatbul2018]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param beta: Positive number that determines the trade-off between precision and recall when computing the
            F-score. :math:`\beta = 1` assigns equal weight to both while :math:`\beta < 1` emphasizes precision and
            vice versa.
        :return: A tuple consisting of the best :math:`F_{\beta}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.__best_ts_fbeta_score(labels, scores, beta,
                                          recall_cardinality_fn=CardinalityFunction.INVERSE_PROPORTIONAL,
                                          weighted_precision=False)

    def best_ts_f1_score(self, labels: torch.Tensor, scores: torch.Tensor) -> Dict[str, Union[float, Any]]:
        r"""
        Compute the :math:`F_{1}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{1}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the improved cardinality function and weighted precision as described in [Wagner2023]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the best :math:`F_{1}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.best_ts_fbeta_score(labels, scores, 1)

    def best_ts_f1_score_classic(self, labels: torch.Tensor, scores: torch.Tensor) -> Dict[str, Union[float, Any]]:
        r"""
        Compute the :math:`F_{1}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{1}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the default cardinality function (:math:`\frac[1}{x}`) and unweighted precision, i.e.,
            the default parameters described in [Tatbul2018]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the best :math:`F_{1}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.best_ts_fbeta_score_classic(labels, scores, 1)
