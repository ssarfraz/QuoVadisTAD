from typing import Any, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.single_series_evaluation import evaluate_ts
from src.evaluation.evaluation_utils import get_results_for_all_score_normalizations
from src.dataset_utils.data_utils import preprocess_data, concatenate_windows_feat


def evaluate_methods(
        methods,
        train_array,
        test_array,
        labels,
        score_normalization: str = 'optimal',
        eval_method='point_wise',
        univariate: bool = False,
        verbose=False
):
    """Evaluates multiple methods on a single dataset. For each method:

    - The method is fitted on the train_array.
    - It is evaluated on the test array using the provided labels.

    If the dataset contains labels per sensor, then the maximum anomaly value per timestamp is used.
    """
    if len(labels.shape) > 1:
        test_labels = labels.max(1)
    else:
        test_labels = labels

    dfs_per_method: list[tuple[Optional[str], pd.DataFrame]] = []

    for name, method in methods.items():
        method.fit(train_array, verbose=verbose, univariate=univariate)
        anomaly_scores = method.transform(test_array)

        score_normalizations_applicable = (len(anomaly_scores.shape) > 1) and (anomaly_scores.shape[1] > 1)

        if score_normalizations_applicable:
            dfs_all_normalizations, df_best_normalization = get_results_for_all_score_normalizations(
                anomaly_scores,
                test_labels,
                eval_method=eval_method
            )

            if score_normalization == 'all':
                dfs_per_method.extend([
                    (f'{name} ({normalization_name} norm)', df)
                    for normalization_name, df in dfs_all_normalizations.items()
                ])
            elif score_normalization == 'optimal':
                dfs_per_method.append((str(name), df_best_normalization))
            else:
                dfs_per_method.append((str(name), dfs_all_normalizations[score_normalization]))

        else:
            _, df_no_normalization = evaluate_ts(
                anomaly_scores,
                test_labels,
                eval_method=eval_method,
                verbose=verbose
            )

            dfs_per_method.append((str(name), df_no_normalization))

    extended_method_names, score_dfs = zip(*dfs_per_method)

    concatenated_score_dfs = pd.concat(score_dfs, ignore_index=False, axis=1)

    score_dict = {
        'F1': concatenated_score_dfs[eval_method].iloc[0].tolist(),
        'P': concatenated_score_dfs[eval_method].iloc[1].tolist(),
        'R': concatenated_score_dfs[eval_method].iloc[2].tolist(),
        'AUPRC': concatenated_score_dfs[eval_method].iloc[3].tolist(),
    }

    return pd.DataFrame(score_dict, index=extended_method_names)


def evaluate_methods_on_dataset_or_traces(
    methods: dict[str, Any],
    data: dict[str, Union[np.ndarray, list[np.ndarray]]],
    data_normalization: str = '0-1',
    score_normalization: str = 'optimal',
    eval_method: str = 'point_wise',
    series_are_split_in_traces: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """Evaluate multiple methods on a single dataset which might consist of a single timeseries or multiple traces.
    In the latter case, the scores are averaged among traces."""
    train, test, labels = data['train'], data['test'], data['labels']

    if series_are_split_in_traces:
        df = []

        for i in tqdm(list(range(len(train))), disable=not verbose):
            trace_train, _, trace_test = preprocess_data(
                train[i],
                test[i],
                train_size=1.0,
                val_size=0.,
                normalization=data_normalization,
            )
            if trace_train.shape[1] == 1:  # check if univariate timeseries
                univariate = True
                # use windowed_features
                trace_train = concatenate_windows_feat(trace_train, window_size=5)
                trace_test = concatenate_windows_feat(trace_test, window_size=5)
            else:
                univariate = False

            df_i = evaluate_methods(
                methods,
                trace_train,
                trace_test,
                labels[i],
                score_normalization=score_normalization,
                eval_method=eval_method,
                univariate=univariate,
                verbose=verbose
            )

            df.append(df_i)
        df = pd.concat(df, ignore_index=False, axis=1)

        df = df.T.groupby(level=0, sort=False).mean().T

    else:
        train, _, test = preprocess_data(
            train,
            test,
            train_size=0.98,
            val_size=0.02,
            normalization=data_normalization
        )
        if train.shape[1] == 1:  # check if univariate timeseries
            univariate = True
            train = concatenate_windows_feat(train, window_size=5)
            test = concatenate_windows_feat(test, window_size=5)
        else:
            univariate = False

        df = evaluate_methods(
            methods,
            train,
            test,
            labels,
            score_normalization=score_normalization,
            eval_method=eval_method,
            univariate=univariate,
            verbose=verbose
        )

    return df


def evaluate_methods_on_datasets(
        methods: dict[str, Any],
        datasets: dict[str, dict[str, Union[np.ndarray, list[np.ndarray]]]],
        data_normalization: str = '0-1',
        score_normalization: str = 'optimal',
        eval_method: str = 'point_wise',
        verbose: bool = True
) -> pd.DataFrame:
    """Evaluates multiple methods on multiple datasets and returns the results in a nested dataframe. Look in the
    argument definitions for the input/output formats.

    Args:
        methods: A dictionary of anomaly detection methods. The keys are names, the values must be classes implementing:
            - fit: 2D_train_array -> None
            - transform: 2D_test_array -> 2D_anomaly_scores_array
        datasets: A dictionary of datasets. Keys are names and each value is of a dictionary with the following keys:
            - 'train': A 2D series array of nominal data for training or a list of them, in case the dataset contains
                multiple traces. The shape of each array is (timestamp sensor).
            - 'test': A 2D series array of nominal data for testing or a list of them, in case the dataset contains
                multiple traces. The shape of each array is (timestamp sensor).
            - 'labels': A 1D/2D series array with 0/1 values indicating the presence of anomaly per timestamp,
                optionally also per sensor,  or a list of them if there are multiple traces.
        data_normalization: Normalize the data. Can take the following values:
            '0-1': Min-max scaling to [0, 1].
            'mean-std': A scaling to mean 0 and standard deviation 1.
            'none': It performs no normalization.

            'optimal': For each dataset, it picks the best scoring one of the above in the test set.
            'all': It tries all methods when applicable and returns all three scores.
        score_normalization: Normalize the scores outputted by the methods per sensor. Can take the following values:
            'median-iqr': Robust normalization by subtracting the mean and dividing by the interquartile range.
            'mean-std': A scaling to mean 0 and standard deviation 1.
            None: It performs no normalization.
            'optimal': For each dataset, it picks the best scoring one of the above in the test set (including None).
            'all': It tries all methods when applicable and returns all three scores.
        eval_method: The evaluation protocol to apply. Can take the following values:
            - 'point_wise': Scores on each timestamp of the timeseries.
            - 'point_adjust': WARNING!! Do not use this protocol, it provides unreliable scores. It only exists here for
                demonstration purposes and reproducibility. It is the same as 'point_wise' but applies the point adjust
                (PA) to update the predictions.
        verbose: If true, it prints the names of the datasets and a progress bar on the methods.
    Returns:
        A nested dataframe with one method per row, datasets as top column names and scores as second column names.
    """
    scores_dfs = []
    for name, data in datasets.items():
        if verbose:
            print(f'Evaluating on {name}.')

        train = data['train']
        series_are_split_in_traces = type(train) is list

        scores = evaluate_methods_on_dataset_or_traces(
            methods=methods,
            data=data,
            data_normalization=data_normalization,
            score_normalization=score_normalization,
            eval_method=eval_method,
            series_are_split_in_traces=series_are_split_in_traces,
            verbose=verbose
        )

        multi_col = [(name, col) for col in scores.columns]
        scores.columns = pd.MultiIndex.from_tuples(multi_col)
        scores_dfs.append(scores)
        if verbose:
            if series_are_split_in_traces:
                print(f'Evaluation on {name} finished: on {len(train)} traces,  results are averaged')
            else:
                print(f'Evaluation on {name} finished')

    return pd.concat(scores_dfs, ignore_index=False, axis=1)
