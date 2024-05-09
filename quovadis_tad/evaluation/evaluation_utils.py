from typing import Optional

import numpy as np
import pandas as pd

from quovadis_tad.dataset_utils.data_utils import normalise_scores
from quovadis_tad.evaluation.single_series_evaluation import evaluate_ts


def get_results_for_all_score_normalizations(
        scores: np.ndarray,
        test_labels: np.ndarray,
        eval_method: Optional[str] = None
) -> tuple[dict[Optional[str], pd.DataFrame], pd.DataFrame]:
    # get score under all three normalizations
    df_list = []
    f1_scores = []
    normalisations = ["median-iqr", "mean-std", None]
    for n in normalisations:
        r, d = evaluate_ts(
            normalise_scores(scores, norm=n).max(1),
            test_labels,
            eval_method=eval_method,
            verbose=False
        )
        f1_scores.append(r['f1'])
        if n is None:
            n = 'no'
        df_list.append((n, d))
    best_score_idx = np.array(f1_scores).argmax()
    return dict(df_list), df_list[best_score_idx][1]
