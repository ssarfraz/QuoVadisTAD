import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA

from tad.dataset.reader import GeneralDataset, dataset_loader_map
from tad.evaluation_scripts.evaluate_ts import evaluate_ts as evaluate
from tad.utils.data_utils import preprocess_data, median_iqr_norm, concatenate_windows_feat

warnings.filterwarnings("ignore")


def run_baselines(data,
                  preprocessing="0-1",
                  eval_method='standard',
                  distance="euclidean",
                  pca_dim=30,
                  use_window_feats=False):
    train, test, labels = data

    if type(train) is list:
        res = []
        df = []
        for i in range(len(train)):
            tr, _, te = preprocess_data(train[i], test[i], 1.0, 0., normalization=preprocessing)
            if tr.shape[1] == 1:  # check if univariate timeseries
                use_window_feats = True
                pca_dim = 2
            if use_window_feats:
                tr = concatenate_windows_feat(tr, window_size=5)
                te = concatenate_windows_feat(te, window_size=5)

            res_i, df_i = evaluate_baselines(tr, te, labels[i],
                                             eval_method=eval_method,
                                             pca_dim=pca_dim,
                                             distance=distance)

            res.append(res_i)
            df.append(df_i)
        df = pd.concat(df, ignore_index=False, axis=1)
        df = df.groupby(level=0, axis=1, sort=False).apply(lambda x: x.mean(1))
    else:
        train, _, test = preprocess_data(train, test, 0.98, 0.02, normalization=preprocessing)
        if train.shape[1] == 1:  # check if univariate timeseries
            use_window_feats = True
            pca_dim = 2

        if use_window_feats:
            train = concatenate_windows_feat(train, window_size=5)
            test = concatenate_windows_feat(test, window_size=5)
        # print(f'Train shape : {train.shape} - Test shape: {test.shape}')
        res, df = evaluate_baselines(train, test, labels,
                                     eval_method=eval_method,
                                     pca_dim=pca_dim,
                                     distance=distance)

    return res, df


def evaluate_baselines(train_array,
                       test_array,
                       labels,
                       eval_method='standard',
                       distance='euclidean',
                       pca_dim=30,
                       verbose=False):
    if len(labels.shape) > 1:
        test_labels = labels.max(1)
    else:
        test_labels = labels

    baselines = ['Random',
                 'Sensor Range Deviation',
                 'Distance_to_train_1-NN ',
                 'Distance_to train_avg',
                 'PCA_Error(median-iqr norm)',
                 'PCA_Error(mean-std norm)',
                 'PCA_Error(no norm)',
                 'PCA_Error(no norm with Smoothing)',
                 'Simple L2_norm']

    # Baseline-1: Random predictions
    results_0, df_0 = evaluate(np.random.rand(test_labels.shape[0]),
                               test_labels, eval_method=eval_method, verbose=verbose)

    # Sensor range deviation
    m1 = 1 * (test_array > 1)
    m2 = 1 * (test_array < 0)
    mask = m1 + m2
    ano_sens_cand = mask.sum(1)
    results_01, df_01 = evaluate(ano_sens_cand > 0, test_labels,
                                 eval_method=eval_method, verbose=verbose)

    # Baseline-2: closest distance to train timestamp
    scores = metrics.pairwise.pairwise_distances(test_array, train_array, metric=distance)
    results_1, df_1 = evaluate(scores.min(1), test_labels, eval_method=eval_method, verbose=verbose)

    # Baseline 3: difference to train dist mean
    scores = metrics.pairwise.pairwise_distances(test_array, train_array.mean(0)[None, :],
                                                 metric=distance)
    results_2, df_2 = evaluate(scores, test_labels,
                               eval_method=eval_method, verbose=verbose)

    # Baseline-4: PCA as prediction and GDN deviation scoring
    dim = pca_dim  # test_array.shape[1]
    # if dim >= 30:
    #    dim = 15
    pca = PCA(n_components=dim, svd_solver='full')
    pca.fit(train_array)
    t_pca = pca.transform(test_array)
    inv_pca = pca.inverse_transform(t_pca)

    # median-IQR normalization on deviation scores
    results_3, df_3 = evaluate(median_iqr_norm(np.abs(inv_pca - test_array)).max(1),
                               test_labels, eval_method=eval_method, verbose=verbose)
    # mean-std normalization on deviation scores
    results_3_1, df_3_1 = evaluate(median_iqr_norm(np.abs(inv_pca - test_array), norm="mean-std").max(1),
                                   test_labels, eval_method=eval_method, verbose=verbose)

    # Baseline-5: PCA as prediction and direct scoring without norm or smoothing
    results_4, df_4 = evaluate(np.abs(inv_pca - test_array).max(1),
                               test_labels, eval_method=eval_method, verbose=verbose)

    # Baseline-5-5: PCA prediction no norm , only smoothing
    results_4_1, df_4_1 = evaluate(median_iqr_norm(np.abs(inv_pca - test_array), norm=None).max(1),
                                   test_labels, eval_method=eval_method, verbose=verbose)

    # Baseline-6: L2 norm
    results_5, df_5 = evaluate(np.linalg.norm(test_array, 2, axis=1),
                               test_labels, eval_method=eval_method, verbose=verbose)

    ## Baseline-6: L2 norm (over moving_avg window)
    # test_array_windowed = smooth_with_movingAvg(test_array, window_size=5)
    # results_6, df_6 = evaluate(np.linalg.norm(test_array_windowed, 2, axis=1), test_labels, pa=False, interval=10, k=50, verbose=verbose)

    res = [results_0, results_01, results_1, results_2, results_3, results_3_1, results_4, results_4_1, results_5]
    dfs = [df_0, df_01, df_1, df_2, df_3, df_3_1, df_4, df_4_1, df_5]

    cf = pd.concat(dfs, ignore_index=False, axis=1)

    score_dict = {'F1': cf[eval_method].iloc[0].tolist(),
                  'P': cf[eval_method].iloc[1].tolist(),
                  'R': cf[eval_method].iloc[2].tolist(),
                  'AUPRC': cf[eval_method].iloc[3].tolist()}
    df_f = pd.DataFrame(score_dict, index=baselines)
    return res, df_f


def evaluate_datasets(preprocessing="0-1",
                      eval_method='standard',
                      distance="euclidean",
                      pca_dim=30,
                      use_window_feats=False,
                      datasets_ordered=None,
                      skip_useless_datasets=False,
                      verbose=True):
    df_comb = []
    results = []
    if verbose:
        if datasets_ordered is not None:
            print(f'[INFO]: Evaluating {len(datasets_ordered)} datasets')
        else:
            print(f'[INFO]: Evaluating {len(GeneralDataset)} datasets')

    if datasets_ordered is None:
        datasets_ordered = [GeneralDataset.MSL,
                            GeneralDataset.UCR_1,
                            GeneralDataset.SWAT,
                            GeneralDataset.WADI,
                            GeneralDataset.WADI_GDN,
                            GeneralDataset.SMAP,
                            GeneralDataset.SMD]  # TODO why not all of them (like UCR_2, UCR_3, etc.)

    for dataset in datasets_ordered:
        dataset_name = dataset.name
        if dataset in [GeneralDataset.SMD, GeneralDataset.MSL, GeneralDataset.SMAP]:
            pca_dim = 10
            if skip_useless_datasets:
                print(f'Skipping evaluation of {dataset_name}')
                continue

        # for dataset, loader in datasets.items():
        train, test, labels = dataset_loader_map[dataset]()

        if use_window_feats:
            print(f'using subset of WADI train')
            train = train[:60_000]

        result, df = run_baselines([train, test, labels],
                                   preprocessing=preprocessing,
                                   eval_method=eval_method,
                                   distance=distance,
                                   pca_dim=pca_dim,
                                   use_window_feats=use_window_feats)

        multi_col = [(dataset_name, col) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(multi_col)
        df_comb.append(df)
        results.append(result)
        if verbose:
            if type(train) is list:
                print(f' Evaluation on {dataset_name} finished: on {len(train)} traces,  results are averaged')
            else:
                print(f' Evaluation on {dataset_name} finished')

    return results, pd.concat(df_comb, ignore_index=False, axis=1)


if __name__ == '__main__':
    _, df = evaluate_datasets(preprocessing="0-1",
                              eval_method='wagner2023',
                              distance="euclidean",
                              skip_useless_datasets=False,
                              datasets_ordered=[GeneralDataset.SWAT],  # 'MSL', 'SMAP',
                              pca_dim=10)
    print(df.to_string(index=False))
