import os
import numpy as np
import pandas as pd
import random
import glob

from sklearn import metrics

from src.dataset_utils.dataset_reader import datasets
from src.dataset_utils.data_utils import preprocess_data, normalise_scores, concatenate_windows_feat

from src.evaluation_scripts.evaluate_ts import evaluate_ts
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")




def get_normalise_scores(scores, test_labels, eval_method=None):
    # get score under all three normalizations
    result_list = []
    df_list = []
    tmp = []
    normalisations = ["median-iqr", "mean-std", None]
    for n in normalisations:
        r, d = evaluate_ts(normalise_scores(scores, norm=n).max(1), test_labels, eval_method=eval_method, verbose=False)
        tmp.append(r['f1'])
        result_list.append(r)
        df_list.append(d)
    best_score_idx = np.array(tmp).argmax()
    return result_list, df_list, result_list[best_score_idx], df_list[best_score_idx]


def run_baselines(data,
                  preprocessing="0-1",
                  eval_method='point_wise',
                  distance="euclidean",
                  pca_dim=30,
                  show_norm_impact=False):
    
    train, test, labels = data
    
    if type(train) is list:
        res = []
        df = []
        for i in range(len(train)):
            tr, _, te = preprocess_data(train[i], test[i], 1.0, 0., normalization=preprocessing)
            if tr.shape[1] == 1:   # check if univariate timeseries
                pca_dim = 2
                # use windowed_features
                tr = concatenate_windows_feat(tr, window_size=5)
                te = concatenate_windows_feat(te, window_size=5)             
                
            res_i, df_i = evaluate_baselines(tr, te, labels[i],
                                             eval_method=eval_method,
                                             pca_dim=pca_dim,
                                             show_norm_impact=show_norm_impact,
                                             distance=distance)
            
            res.append(res_i)
            df.append(df_i)
        df = pd.concat(df, ignore_index=False, axis=1)
        df = df.groupby(level=0, axis=1, sort=False).apply(lambda x: x.mean(1))
    else:
        train, _, test = preprocess_data(train, test, 0.98, 0.02, normalization=preprocessing)
        if train.shape[1] == 1:   # check if univariate timeseries
                pca_dim = 2
                train = concatenate_windows_feat(train, window_size=5)
                test = concatenate_windows_feat(test, window_size=5)
    
        res, df = evaluate_baselines(train, test, labels,
                                     eval_method=eval_method,
                                     pca_dim=pca_dim,
                                     show_norm_impact=show_norm_impact,
                                     distance=distance)

    return res, df




def evaluate_baselines(train_array,
                       test_array,
                       labels,
                       eval_method='point_wise',
                       distance='euclidean',
                       pca_dim=30,
                       show_norm_impact=False,
                       verbose=False):

    if len(labels.shape) > 1:
        test_labels = labels.max(1)
    else:
        test_labels = labels
    
    
    
    # Baseline-1: Sensor range deviation
    m1 = 1 * (test_array > 1)
    m2 = 1 * (test_array < 0)
    mask = m1 + m2
    ano_sens_cand = mask.sum(1)   
    results_1, df_1 = evaluate_ts(ano_sens_cand > 0, test_labels,
                               eval_method=eval_method, verbose=verbose)
    
    # Baseline-2: L2 norm
    results_2, df_2 = evaluate_ts(np.linalg.norm(test_array, 2, axis=1),
                               test_labels, eval_method=eval_method, verbose=verbose)
    
    # Baseline-3: 1-NN distance - closest distance to train timestamp
    scores = metrics.pairwise.pairwise_distances(test_array, train_array, metric=distance)
    results_3, df_3 = evaluate_ts(scores.min(1), test_labels, eval_method=eval_method, verbose=verbose)
    
    # Baseline 3_3: distance to train mean (fast and often very good as well)
    #scores = metrics.pairwise.pairwise_distances(test_array, train_array.mean(0)[None, :], metric=distance)                                             
    #results_2, df_2 = evaluate_ts(scores, test_labels, eval_method=eval_method, verbose=verbose)
    
    
    # Baseline-4: PCA-Error
    pca = PCA(n_components=pca_dim, svd_solver='full')
    pca.fit(train_array)
    t_pca = pca.transform(test_array)
    inv_pca = pca.inverse_transform(t_pca)
    scores_pca = np.abs(inv_pca - test_array)
        
    result_list, df_list, results_4, df_4 = get_normalise_scores(scores_pca, test_labels, eval_method=eval_method)
    
    if show_norm_impact:
        res = [results_1, results_2, results_3] + result_list
        dfs = [df_1, df_2, df_3] + df_list
        baselines = ['Sensor Range Deviation',
                     'Simple L2_norm',
                     '1-NN Distance',
                     #'Distance_to train_avg',
                     'PCA_Error(median-iqr norm)',
                     'PCA_Error(mean-std norm)',
                     'PCA_Error(no norm)',
                 ]
    else:
        res = [results_1, results_2, results_3] + [results_4]
        dfs = [df_1, df_2, df_3] + [df_4]
        baselines = ['Sensor Range Deviation',
                     'Simple L2_norm',
                     '1-NN Distance',
                     #'Distance_to train_avg',
                     'PCA_Error',
                 ]


    cf = pd.concat(dfs, ignore_index=False, axis=1)

    
    score_dict = {'F1': cf[eval_method].iloc[0].tolist(),
                  'P':cf[eval_method].iloc[1].tolist(),
                  'R':cf[eval_method].iloc[2].tolist(),
                  'AUPRC': cf[eval_method].iloc[3].tolist(),
                 }
    df_f = pd.DataFrame(score_dict, index=baselines)
    return res, df_f




def evaluate_datasets(root_path,
                      preprocessing="0-1",
                      eval_method='point_wise',
                      distance="euclidean",
                      pca_dim=30,
                      show_norm_impact=False,
                      dataset_names_ordered=None,
                      verbose=True):
    df_comb = []
    results = []
    
    if dataset_names_ordered is None:
        dataset_names_ordered = ['swat', 'wadi_127', 'wadi_112', 'smd', 'msl', 'smap', 'ucr_IB']  
    
    if verbose:
        print(f'[INFO]: Evaluating {len(dataset_names_ordered)} datasets')

    
    for dataset_name in dataset_names_ordered:
        if dataset_name in ['smd', 'smap']: # We use pca_dim=10 for datasets with less than 50 sensors
            pca_dim = 10
            
         
        #for dataset_name, loader in datasets.items():
        train, test, labels = datasets[dataset_name](root_path)
                 
        dataset_name = dataset_name.upper()    
        result, df = run_baselines([train, test, labels],
                                    preprocessing=preprocessing,
                                    eval_method=eval_method,
                                    distance=distance,
                                    pca_dim=pca_dim,
                                    show_norm_impact=show_norm_impact
                                    )

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
    root_path = '../QuoVadisTAD'
    _, df = evaluate_datasets(root_path,
                      preprocessing="0-1",
                      eval_method='point_wise',
                      distance="euclidean",
                      show_norm_impact=False,
                      dataset_names_ordered=['swat'],
                      pca_dim=30)
    print(df.to_string(index=False))

