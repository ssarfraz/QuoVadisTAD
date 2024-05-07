from typing import Optional

import numpy as np
import pandas as pd

from src.dataset_utils.data_utils import normalise_scores
from src.evaluation.single_series_evaluation import evaluate_ts


def get_normalise_scores(
        scores: np.ndarray,
        test_labels: np.ndarray,
        eval_method: Optional[str] = None
) -> tuple[dict[Optional[str], pd.DataFrame], pd.DataFrame]:
    if len(scores.shape) == 1:
        scores = scores[:, None]
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
        df_list.append((n, d))
    best_score_idx = np.array(f1_scores).argmax()
    return dict(df_list), df_list[best_score_idx][1]
