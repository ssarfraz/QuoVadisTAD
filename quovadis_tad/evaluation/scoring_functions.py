import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from typing import List, Any, Dict, Tuple, Callable, Optional

'''
Numpy implementation of torch based evualtion from TiemseAD (https://github.com/wagner-d/TimeSeAD)
'''

"""
MIT License

Copyright (c) 2023 TimeSeAD authors (https://github.com/wagner-d/TimeSeAD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def constant_bias_fn(inputs: np.ndarray) -> float:
    """
    Compute the overlap size for a constant bias function that assigns the same weight to all positions.

    This function computes the average of the input values.

    :param inputs: A 1-D numpy array containing the predictions inside a ground-truth window.
    :return: The overlap, which is the average of the input values.
    """
    return np.sum(inputs) / inputs.shape[0]

def back_bias_fn(inputs: np.ndarray) -> float:
    """
    Compute the overlap size for a bias function that assigns more weight to predictions towards the back of a
    ground-truth anomaly window.

    This function computes the weighted average of the input values, where the weights increase linearly towards
    the end of the input array.
    
    :param inputs: A 1-D numpy array containing the predictions inside a ground-truth window.
    :return: The overlap, which is the weighted average of the input values.
    """
    n = inputs.shape[0]
    weights = np.arange(1, n + 1)
    res = np.dot(inputs, weights)
    return res / ((n * (n + 1)) // 2)

def front_bias_fn(inputs: np.ndarray) -> float:
    """
        Compute the overlap size for a bias function that assigns more weight to predictions towards the front of a
    ground-truth anomaly window.

    This function computes the weighted average of the input values, where the weights decrease linearly towards
    the end of the input array.

    :param inputs: A 1-D numpy array containing the predictions inside a ground-truth window.
    :return: The overlap, which is the weighted average of the input values
    """
    n = inputs.shape[0]
    weights = np.arange(n, 0, -1)
    res = np.dot(inputs, weights)
    return res / ((n * (n + 1)) // 2)

def middle_bias_fn(inputs: np.ndarray) -> float:
    """
    Compute the overlap size for a bias function that assigns more weight to predictions in the middle of a
    ground-truth anomaly window.

    This function computes the weighted average of the input values, where the weights are higher in the middle
    of the input array and decrease towards the ends.

    :param inputs: A 1-D numpy array containing the predictions inside a ground-truth window.
    :return: The overlap, which is the weighted average of the input values.
    """
    n = inputs.shape[0]
    middle = n // 2
    weights = np.concatenate((np.arange(1, middle + 1), np.arange(n // 2 + n % 2, 0, -1)))
    weighted_sum = np.dot(inputs, weights)
    normalization_factor = (middle * (middle + 1) + (middle + n % 2) * (middle + n % 2 + 1)) // 2
    return weighted_sum / normalization_factor

def inverse_proportional_cardinality_fn(cardinality: int, gt_length: int) -> float:
    """
    Cardinality function that assigns an inversely proportional weight to predictions within a single ground-truth
    window.
    
    This is the default cardinality function recommended in [Tatbul2018]
    .. [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
        Precision and recall for time series. Advances in neural information processing systems. 2018;31.
    .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
        TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
        Transactions on Machine Learning Research (TMLR), (to appear) 2023.
    
    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window (unused).
    :return: The cardinality factor 1/cardinality, with a minimum value of 1.
    """
    return 1 / max(1, cardinality)

def improved_cardinality_fn(cardinality: int, gt_length: int):
    """
    Recall-consistent cardinality function introduced by [Wagner2023]_ that assigns lower weight to ground-truth windows
    that overlap with many predicted windows.

    This function computes

    .. math::
        \left(\frac{\text{gt_length} - 1}{\text{gt_length}}\right)^{\text{cardinality} - 1}.
    
    .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
        TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
        Transactions on Machine Learning Research (TMLR), (to appear) 2023.
        
    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window.
    :return: The cardinality factor.
    """
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)

def compute_window_indices(binary_labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute a list of indices where anomaly windows begin and end.

    :param binary_labels: A 1-D numpy array containing 1 for an anomalous time step or 0 otherwise.
    :return: A list of tuples (start, end) for each anomaly window in binary_labels, where start is the
        index at which the window starts and end is the first index after the end of the window.
    """
    # Compute the differences between consecutive elements
    differences = np.diff(binary_labels, prepend=0)
    # Find the indices where the differences are non-zero (start and end of windows)
    indices = np.nonzero(differences)[0]
    # If the number of indices is odd, append the last index
    if len(indices) % 2 != 0:
        indices = np.append(indices, binary_labels.size)
    # Pair the start and end indices
    window_indices = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]

    return window_indices

def _compute_overlap(preds: np.ndarray, pred_indices: List[Tuple[int, int]],
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

def ts_precision_and_recall(anomalies: np.ndarray, predictions: np.ndarray, alpha: float = 0,
                            recall_bias_fn: Callable[[np.ndarray], float] = constant_bias_fn,
                            recall_cardinality_fn: Callable[[int, int], float] = inverse_proportional_cardinality_fn,
                            precision_bias_fn: Optional[Callable[[np.ndarray], float]] = None,
                            precision_cardinality_fn: Optional[Callable[[int, int], float]] = None,
                            anomaly_ranges: Optional[List[Tuple[int, int]]] = None,
                            prediction_ranges: Optional[List[Tuple[int, int]]] = None,
                            weighted_precision: bool = False) -> Tuple[float, float]:
    """
    Computes precision and recall for time series as defined in [Tatbul2018]_.

    :param anomalies: Binary 1-D numpy array containing the true labels.
    :param predictions: Binary 1-D numpy array containing the predicted labels.
    :param alpha: Weight for existence term in recall.
    :param recall_bias_fn: Function that computes the bias term for a given ground-truth window.
    :param recall_cardinality_fn: Function that computes the cardinality factor for a given ground-truth window.
    :param precision_bias_fn: Function that computes the bias term for a given predicted window.
        If None, this will be the same as recall_bias_function.
    :param precision_cardinality_fn: Function that computes the cardinality factor for a given predicted window.
        If None, this will be the same as recall_cardinality_function.
    :param weighted_precision: If True, the precision score of a predicted window will be weighted with the
        length of the window in the final score. Otherwise, each window will have the same weight.
    :param anomaly_ranges: A list of tuples (start, end) for each anomaly window in anomalies, where start
        is the index at which the window starts and end is the first index after the end of the window. This can
        be None, in which case the list is computed automatically from anomalies.
    :param prediction_ranges: A list of tuples (start, end) for each anomaly window in predictions, where
        start is the index at which the window starts and end is the first index after the end of the window.
        This can be None, in which case the list is computed automatically from predictions.
    :return: A tuple consisting of the time-series precision and recall for the given labels.
    """
    has_anomalies = np.any(anomalies > 0)
    has_predictions = np.any(predictions > 0)

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


class Evaluator:
    """
    A class that can compute several evaluation metrics for a dataset using NumPy arrays.
    Each method returns the score as a single float, but it can also return additional information in a dict.
    """

    def rocauc(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the classic point-wise area under the receiver operating characteristic curve.
        """
        return roc_auc_score(labels, scores), {}

    def f1_score(self, labels: np.ndarray, scores: np.ndarray, pos_label: int = 1) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the classic point-wise F1 score.
        """
        
        return f1_score(labels, scores, pos_label=pos_label), {}

    def best_fbeta_score(self, labels: np.ndarray, scores: np.ndarray, beta: float) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the classic point-wise F_beta score.
        """
        precision, recall, thresholds = precision_recall_curve(labels, scores)

        numerator = (1 + beta ** 2) * precision * recall
        denominator = beta ** 2 * precision + recall
        f_score = np.where(
            denominator == 0,
            0,
            numerator / np.where(denominator == 0, 1, denominator)
        )

        best_index = np.argmax(f_score)
        area = self.auprc(labels, scores)
        if labels.sum() > 0:
            auc_roc = self.rocauc(labels, scores)     
        else:
            print('AUC-ROC cannot be computed as true labels only have one class, computing AUC-PR instead')
            auc_roc = area
        
        return dict(f1=f_score[best_index],                    
                    precision=precision[best_index],
                    recall=recall[best_index],                   
                    auprc=area[0],
                    auroc= auc_roc[0],
                    threshold=thresholds[best_index])

    def best_f1_score(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
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

    def auprc(self, labels: np.ndarray, scores: np.ndarray, integration: str = 'trapezoid') -> Tuple[float, Dict[str, Any]]:
        """
        Compute the classic point-wise area under the precision-recall curve.

        :param labels: A 1-D numpy array containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D numpy array containing the scores.
        :param integration: Method to use for computing the area under the curve. 'riemann' corresponds to a simple
            Riemann sum, whereas 'trapezoid' uses the trapezoidal rule.
        :return: A tuple consisting of the AuPRC score and an empty dict.
        """
        precision, recall, _ = precision_recall_curve(labels, scores)
        # recall is nan in the case where all ground-truth labels are 0. Simply set it to zero here
        # so that it does not contribute to the area
        recall = np.nan_to_num(recall, nan=0)

        if integration == 'riemann':
            area = -np.sum(np.diff(recall) * precision[:-1])
        else:
            area = auc(recall, precision)

        return area, {}

    def average_precision(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
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


    def ts_auprc(self, labels: np.ndarray, scores: np.ndarray, integration='trapezoid',
                 weighted_precision: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the area under the precision-recall curve using precision and recall for time series.
        """
        thresholds = np.unique(scores)

        precision = np.empty(len(thresholds) + 1)
        recall = np.empty(len(thresholds) + 1)
        predictions = np.empty_like(scores, dtype=int)

        # Set last values when threshold is at infinity so that no point is predicted as anomalous.
        # Precision is not defined in this case, we set it to 1 to stay consistent with scikit-learn
        precision[-1] = 1
        recall[-1] = 0

        label_ranges = compute_window_indices(labels)

        for i, t in enumerate(thresholds):
            predictions = scores >= t
            prec, rec = ts_precision_and_recall(labels, predictions, alpha=0,
                                                recall_cardinality_fn=improved_cardinality_fn,
                                                anomaly_ranges=label_ranges,
                                                weighted_precision=weighted_precision)
            precision[i] = prec
            recall[i] = rec

        if integration == 'riemann':
            area = -np.sum(np.diff(recall) * precision[:-1])
        else:
            area = auc(recall, precision)

        return area, {}

    def ts_average_precision(self, labels: np.ndarray, scores: np.ndarray, weighted_precision: bool = True) \
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

    def ts_auprc_unweighted(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
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

    def __best_ts_fbeta_score(self, labels: np.ndarray, scores: np.ndarray, beta: float,
                              recall_cardinality_fn: Callable = improved_cardinality_fn,
                              weighted_precision: bool = True) -> Dict[str, Any]:
        thresholds = np.unique(scores)

        precision = np.empty(len(thresholds))
        recall = np.empty(len(thresholds))
        predictions = np.empty_like(scores, dtype=int)

        label_ranges = compute_window_indices(labels)

        for i, t in enumerate(thresholds):
            predictions = scores > t
            prec, rec = ts_precision_and_recall(labels, predictions.astype(int), alpha=0,
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

        f_score = (1 + beta**2) * precision * recall / (np.maximum(beta**2 * precision + recall, 1e-15))
        max_score_index = np.argmax(f_score)
        area = self.ts_auprc(labels, scores, integration='trapezoid', weighted_precision=weighted_precision)
        area_roc = self.auprc(labels, scores)
        if labels.sum() > 0:
            auc_roc = self.rocauc(labels, scores)
        else:
            print('AUC-ROC cannot be computed as true labels only have one class, computing AUC-PR instead')
            auc_roc = area_roc

        return {
            'f1': f_score[max_score_index],
            'precision': precision[max_score_index],
            'recall': recall[max_score_index],
            'auprc': area[0],
            'auroc': auc_roc[0],
            'threshold': thresholds[max_score_index]
        }

    def best_ts_fbeta_score(self, labels: np.ndarray, scores: np.ndarray, beta: float) -> Tuple[float, Dict[str, Any]]:
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
        return self.__best_ts_fbeta_score(labels, scores, beta, recall_cardinality_fn=improved_cardinality_fn,
                                        weighted_precision=True)

    def best_ts_fbeta_score_classic(self, labels: np.ndarray, scores: np.ndarray, beta: float) -> Tuple[float, Dict[str, Any]]:
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
                                          recall_cardinality_fn=inverse_proportional_cardinality_fn,
                                          weighted_precision=False)

    def best_ts_f1_score(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
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

    def best_ts_f1_score_classic(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
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