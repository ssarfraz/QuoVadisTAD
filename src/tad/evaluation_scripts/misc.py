import numpy as np
from sklearn import metrics


def convert_cluster_labels_to_scores(
	labels, test_labels
):  # TODO add docstring and / or make private or remove
	pred = np.zeros_like(labels)
	un, cnts = np.unique(labels, return_counts=True)
	# cluster label of of smaller # of samples should be assigned 1 (anomolous)
	anomaly_label = un[np.argmin(cnts)]
	pred[labels == anomaly_label] = 1

	pr, rec, f1, _ = metrics.precision_recall_fscore_support(
		test_labels, pred, average="binary"
	)
	score_norm = np.random.uniform(low=0.0, high=0.5, size=len(pred))
	score_anamol = np.random.uniform(low=0.6, high=0.999, size=len(pred))
	pred_scores_n = pred + score_norm
	pred_scores_n[pred_scores_n > 1] = 0
	pred_scores_an = (1 - pred) + score_anamol
	pred_scores_an[pred_scores_an > 1] = 0
	scores = pred_scores_n + pred_scores_an
	return scores, pred, {"precision": pr, "recall": rec, "F1-score": f1}


def assign_cluster_labels():  # TODO add docstring and / or make private or remove
	test_cluster_centers = cool_mean(test_array, req_c_tw)
	train_array_center = train_array[: test_array.shape[0]].mean(0)[None, :]
	d_proto = metrics.pairwise.pairwise_distances(
		test_cluster_centers, train_array_center, metric="cosine"
	)
	normal_cluster_label = np.argmin(d_proto)
	labels = np.ones_like(req_c_tw)
	labels[req_c_tw == normal_cluster_label] = 0
