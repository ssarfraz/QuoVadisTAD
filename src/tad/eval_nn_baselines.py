import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from tad.evaluation_scripts.evaluate_ts import evaluate_ts
import warnings

warnings.filterwarnings("ignore")


def run_nn_eval(module_path,
                dataset_name,
                eval_method='standard',
                verbose=False):
    sota_index = ['1-Layer MLP',
                  'Single block MLPMixer',
                  'Single Transformer block',
                  '1-Layer GCN-LSTM']

    sota = ['mlp',
            'MLPmixer',
            'Transformer',
            'gcn']

    res = []
    dfs = []
    for model_name in sota:
        # mlp
        # dataset_name = dataset_name.upper()

        if dataset_name == 'SMD':
            gt_path = Path(module_path, "resources", 'NN_baselines_predictions', model_name + '_' + dataset_name.lower() + '_gt.pkl')
            score_path = Path(module_path, "resources", 'NN_baselines_predictions',
                              model_name + '_' + dataset_name.lower() + '_preds.pkl')
            with open(gt_path, 'rb') as f:
                gt_dict = pickle.load(f)
            with open(score_path, 'rb') as s:
                score_dict = pickle.load(s)
            df_i = []
            results_i = []
            for i in range(len(gt_dict)):
                res_i, dfs_i = evaluate_ts(score_dict[i], gt_dict[i], eval_method=eval_method, verbose=verbose)
                results_i.append(res_i)
                df_i.append(dfs_i)
            df_i = pd.concat(df_i, ignore_index=False, axis=1)
            df_i = df_i.groupby(level=0, axis=1, sort=False).apply(lambda x: x.mean(1))

        else:
            arr = np.load(
                Path(module_path, "resources", 'NN_baselines_predictions', model_name + '_' + dataset_name.lower() + '.npy'))
            scores = arr[0]
            gt_labels = arr[1]

            results_i, df_i = evaluate_ts(scores, gt_labels, eval_method=eval_method, verbose=verbose)

        res.append(results_i)
        dfs.append(df_i)

    cf = pd.concat(dfs, ignore_index=False, axis=1)

    score_dict = {'F1': cf[eval_method].iloc[0].tolist(),
                  'P': cf[eval_method].iloc[1].tolist(),
                  'R': cf[eval_method].iloc[2].tolist(),
                  'AUPRC': cf[eval_method].iloc[3].tolist()}
    df_f = pd.DataFrame(score_dict, index=sota_index)
    return res, df_f


def evaluate_nn_baselines(root_path,
                          eval_method='standard',
                          dataset_names=None,
                          ):
    df_comb = []
    results = []

    if dataset_names is None:
        dataset_names = ['SWAT', 'WADI', 'WADI_gdn', 'SMD']

    print(f'[INFO]: Evaluating all SOTA on {len(dataset_names)} datasets')
    for dataset_name in dataset_names:
        if dataset_name in ['msl', 'smap']:
            continue

        result, df = run_nn_eval(root_path, dataset_name, eval_method=eval_method, verbose=False)

        multi_col = [(dataset_name, col) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(multi_col)
        df_comb.append(df)
        results.append(result)
        print(f' Evaluation on {dataset_name} finished')

    return results, pd.concat(df_comb, ignore_index=False, axis=1)
