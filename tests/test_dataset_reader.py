from tad.dataset.reader import GeneralDataset, dataset_loader_map


def test_loader_existance():
	"""
	Test that all datasets from the `GeneralDataset` enum have a loader function.
	"""
	assert all([x in dataset_loader_map.keys() for x in GeneralDataset])
