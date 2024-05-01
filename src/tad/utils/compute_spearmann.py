from tad.dataset.reader import GeneralDataset, dataset_loader_map
from tad.utils.data_utils import preprocess_data
from scipy import stats
import pdb

if __name__ == "__name__":
	train, test, labels = dataset_loader_map[GeneralDataset.SWAT]()

	if len(labels.shape) > 1:
		test_labels = labels.max(1)
	else:
		test_labels = labels

	train_array, val_array, test_array = preprocess_data(
		train, test, 0.85, 0.15, normalization="0-1"
	)
	print(f"train set size: {train_array.shape}")
	print(f"validation set size: {val_array.shape}")
	print(f"test set size: {test_array.shape}")

	score = stats.spearmanr(test_array, train_array[10000, :], axis=1)
	pdb.set_trace()
