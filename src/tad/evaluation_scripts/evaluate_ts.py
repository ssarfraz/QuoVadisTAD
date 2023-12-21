import torch

import pandas as pd
from tad.evaluation_scripts.evaluate import evaluate as evaluate_pa
from tad.evaluation_scripts.ts_evaluator import Evaluator

def evaluate_ts(scores, targets, eval_method='standard', verbose=True):
    # method ={'standard', 'point_adjust', 'wagner2023', 'tabul2018'}
    if eval_method == 'point_adjust':
        # point adjust eval
        results, df = evaluate_pa(scores, targets, pa=True, interval=10, k=0, verbose=False)
        df = df.drop(['without_PA'], axis=1)
        df = df.drop(3)
        df = df.rename(columns={'with_PA': eval_method})
        if verbose: 
            print(df.to_string(index=False)) 
            
    else:
       results, df = get_ts_eval(scores, targets, eval_method=eval_method, verbose=verbose) 
      
    
    return results, df


def get_ts_eval(scores, targets, eval_method='standard', verbose=True):
    ts_evaluator = Evaluator()
    targets = torch.from_numpy(targets)
    scores = torch.from_numpy(scores)
    
    if eval_method == 'standard':
        # point wise standard metrics
        results = ts_evaluator.best_f1_score(targets, scores)
    elif eval_method == 'wagner2023':
        # recall-consistant [wagner2023]
        results = ts_evaluator.best_ts_f1_score(targets, scores)
    elif eval_method == 'tabul2018':
        # recall-consistant [Tatbul2018]
        results = ts_evaluator.best_ts_f1_score_classic(targets, scores)
    else:        
        raise ValueError('Evaluation method not implemented.')
    
    ## datframe to display
    metrics_name= ['F1', 'Precision', 'Recall', 'AUPRC']
    raw = [results['f1'], results['precision'], results['recall'], results['auprc']]        
    score_dict= {'': metrics_name, eval_method: raw}    
          
    df = pd.DataFrame(score_dict)
    if verbose: 
        print(df.to_string(index=False))   
    return results, df