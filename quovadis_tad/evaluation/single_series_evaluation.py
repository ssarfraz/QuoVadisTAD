import numpy as np
import torch

import pandas as pd
from quovadis_tad.evaluation.point_adjust import evaluate as evaluate_pa
from quovadis_tad.evaluation.scoring_functions import Evaluator

def evaluate_ts(scores, targets, eval_method='point_wise', verbose=False):
    # eval_method one of {'point_wise', 'point_adjust', 'range_wise'}
    if eval_method == 'point_adjust':
        # point adjust eval
        results, df = evaluate_pa(scores, targets, pa=True, verbose=False)
        df = df.drop(['without_PA'], axis=1)
        df = df.drop(3)
        df = df.rename(columns={'with_PA': eval_method})
        
    else:
       results, df = get_ts_eval(scores, targets, eval_method=eval_method, verbose=verbose)
    if verbose: 
            print(df.to_string(index=False)) 
    return results, df


def get_ts_eval(scores, targets, eval_method='point_wise', verbose=False):
    ts_evalator = Evaluator()
    #targets = torch.from_numpy(targets)
    #scores = torch.from_numpy(scores)
    
    if eval_method == 'point_wise':
        # point wise standard metrics
        results = ts_evalator.best_f1_score(targets, scores)
    elif eval_method == 'range_wise':
        # recall-consistant [wagner et al. 2023]
        results = ts_evalator.best_ts_f1_score(targets, scores)
    else:        
        raise ValueError('Evaluation method not implemented. use one of {"point_wise", "range_wise", "point_adjust"} ')
    
    ## dataframe to display
    metrics_name= ['F1', 'Precision', 'Recall', 'AUPRC', 'AUROC']
    raw = [results['f1'], results['precision'], results['recall'], results['auprc'], results['auroc']]        
    score_dict= {'': metrics_name, eval_method: raw}    
          
    df = pd.DataFrame(score_dict)

    return results, df
