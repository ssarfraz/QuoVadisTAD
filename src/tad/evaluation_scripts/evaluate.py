"""
Evaluation from: Towards a Rigorous Evaluation of Time-series Anomaly Detection (AAAI 2022)
https://github.com/tuslkkk/tadpak
"""
import numpy as np
from sklearn import metrics
import pandas as pd


def evaluate(scores, targets, pa=True, interval=10, k=0, verbose=True):
	"""
	TODO add description

	:param scores: list or np.array or tensor, anomaly score
	:param targets: list or np.array or tensor, target labels
	:param pa: True/False
	:param interval: threshold search interval
	:param k: PA%K threshold
	:return: results dictionary
	"""
	assert len(scores) == len(targets)

	results = {}

	try:
		scores = np.asarray(scores)
		targets = np.asarray(targets)
	except TypeError:
		scores = np.asarray(scores.cpu())
		targets = np.asarray(targets.cpu())

	precision, recall, threshold = metrics.precision_recall_curve(targets, scores)
	f1_score = 2 * precision * recall / (precision + recall + 1e-12)

	results["best_f1_wo_pa"] = np.max(f1_score)
	results["best_precision_wo_pa"] = precision[np.argmax(f1_score)]
	results["best_recall_wo_pa"] = recall[np.argmax(f1_score)]
	# results['auprc_wo_pa'] = metrics.average_precision_score(targets, scores)
	results["auprc_wo_pa"] = metrics.auc(recall, precision)
	results["auc_wo_pa"] = metrics.roc_auc_score(targets, scores)
	## datframe to display
	metrics_name = ["F1", "Precision", "Recall", "AUC", "AUPRC"]
	raw = [
		results["best_f1_wo_pa"],
		results["best_precision_wo_pa"],
		results["best_recall_wo_pa"],
		results["auc_wo_pa"],
		results["auprc_wo_pa"],
	]
	score_dict = {"": metrics_name, "without_PA": raw}

	if pa:
		if verbose:
			print(
				"[INFO]: Computing scores with Point Adjust (PA) method by finding best threshold... this may take a while ..."
			)

		# find F1 score with optimal threshold of best_f1_wo_pa
		# pa_scores = _pak(scores, targets, threshold[np.argmax(f1_score)], k)
		# results['raw_f1_w_pa'] = metrics.f1_score(targets, pa_scores)
		# results['raw_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
		# results['raw_recall_w_pa'] = metrics.recall_score(targets, pa_scores)

		# find best F1 score with varying thresholds
		if len(scores) // interval < 1:
			ths = threshold
		elif len(threshold) // interval < 2:
			ths = threshold
		else:
			ths = [threshold[interval * i] for i in range(len(threshold) // interval)]

		## find once the convention PA evaluation with K=0
		pa_f1_scores = [
			metrics.f1_score(targets, _pak(scores, targets, th, 0)) for th in ths
		]  # tqdm(ths)
		pa_f1_scores = np.asarray(pa_f1_scores)
		results["best_f1_w_pa"] = np.max(pa_f1_scores)
		results["best_f1_th_w_pa"] = ths[np.argmax(pa_f1_scores)]

		pa_scores = _pak(scores, targets, ths[np.argmax(pa_f1_scores)], 0)
		results["best_precision_w_pa"] = metrics.precision_score(targets, pa_scores)
		results["best_recall_w_pa"] = metrics.recall_score(targets, pa_scores)
		results["auprc_w_pa"] = metrics.average_precision_score(targets, pa_scores)
		results["auc_w_pa"] = metrics.roc_auc_score(targets, pa_scores)
		results["pa_f1_scores"] = pa_f1_scores
		## datframe to display
		with_pa = [
			results["best_f1_w_pa"],
			results["best_precision_w_pa"],
			results["best_recall_w_pa"],
			results["auc_w_pa"],
			results["auprc_w_pa"],
		]
		score_dict = {"": metrics_name, "with_PA": with_pa, "without_PA": raw}

		if k != 0:
			if verbose:
				print("computing PA@k scores")
			pak_f1_scores = [
				metrics.f1_score(targets, _pak(scores, targets, th, k)) for th in ths
			]  # tqdm(ths)
			pak_f1_scores = np.asarray(pak_f1_scores)
			results["best_f1_w_pak"] = np.max(pak_f1_scores)
			results["best_f1_th_w_pak"] = ths[np.argmax(pak_f1_scores)]

			pak_scores = _pak(scores, targets, ths[np.argmax(pak_f1_scores)], k)
			results["best_precision_w_pak"] = metrics.precision_score(
				targets, pak_scores
			)
			results["best_recall_w_pak"] = metrics.recall_score(targets, pak_scores)
			results["auprc_w_pak"] = metrics.average_precision_score(
				targets, pak_scores
			)
			results["auc_w_pak"] = metrics.roc_auc_score(targets, pak_scores)
			results["pak_f1_scores"] = pak_f1_scores
			## dataframe to display
			with_pak = [
				results["best_f1_w_pak"],
				results["best_precision_w_pak"],
				results["best_recall_w_pak"],
				results["auc_w_pak"],
				results["auprc_w_pak"],
			]
			score_dict = {
				"": metrics_name,
				"with_PA": with_pa,
				"with_PA@K": with_pak,
				"without_PA": raw,
			}

	df = pd.DataFrame(score_dict)
	if verbose:
		print(df.to_string(index=False))

	return results, df


def _pak(scores, targets, thres, k=20):
	"""
	:param scores: anomaly scores
	:param targets: target labels
	:param thres: anomaly threshold
	:param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
	:return: point_adjusted predictions
	"""
	predicts = scores > thres
	actuals = targets > 0.01

	one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
	zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

	assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(
		zero_start_idx
	)

	if len(one_start_idx) == len(zero_start_idx) + 1:
		zero_start_idx = np.append(zero_start_idx, len(predicts))

	for i in range(len(one_start_idx)):
		if predicts[one_start_idx[i] : zero_start_idx[i]].sum() > k / 100 * (
			zero_start_idx[i] - one_start_idx[i]
		):
			predicts[one_start_idx[i] : zero_start_idx[i]] = 1

	return predicts
