"""
This module contains the dataset readers for the datasets used in the paper.
"""
from pathlib import Path
from typing import Union, Tuple, Callable, List, Dict
from enum import Enum

import tad
import os
import numpy as np

from tad import DEFAULT_DATA_DIR


class GeneralDataset(Enum):
    """
    Dataset Enum

    usage:
    ```python
    from tad.dataset.dataset_reader import GeneralDataset
    dataset_name, load = GeneralDataset.WADI.value
    train, test, labels = load(data_dir)
    ```
    """

    WADI = "wadi"
    WADI_GDN = "wadi_gdn"
    SWAT = "swat"
    SMD = "smd"
    MSL = "msl"
    SMAP = "smap"
    UCR_1 = "ucr_1"
    UCR_2 = "ucr_2"
    UCR_3 = "ucr_3"
    UCR_4 = "ucr_4"
    UCR_ALL = "ucr_all"









def find_files_in_path(directory: str, file_ending: str = ".npy") -> List[str]:
    """list all the files with ending in a path.

    :param directory: the directory where you'd like to find the files

    :param file_ending: the type of file. Default: .npy

    :return: a list of matched files
    """
    # initialize the output list
    found_files = []

    # Walk through the files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_ending):
                # add a found file to the list
                found_files.append(file)
    return found_files





def _load_wadi(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "WADI") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def _load_wadi_gdn(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "WADI_gdn") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def _load_swat(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "SWaT") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


# load only the trace used in TransAD
def _load_ucr(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "UCR") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'136_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def _load_ucr_2(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "UCR") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'135_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels

def _load_ucr_3(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "UCR") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'137_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def _load_ucr_4(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "UCR") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'138_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels

def _load_ucr_all(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "UCR") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # load all machines subsets as in orignal OmniAnomly paper and Kime et al"Towards a Rigorous Evaluation of Time-series Anomaly Detection" 2022 paper
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] UCR contains {len(data_traces)} data traces')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)  
    # Please note that the `labels` is a 2D array
    # returning split 1 of UCR as used in TransAD
    return data[0], data[1], data[2]


def _load_smd(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "SMD") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # load all machines subsets as in orignal OmniAnomly paper and Kime et al"Towards a Rigorous Evaluation of Time-series Anomaly Detection" 2022 paper
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] SMD contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)  
    # Please note that the `labels` is a 2D array
    return data[0], data[1], data[2]


def _load_msl(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "MSL") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path

    # data_traces = ['C-1_', 'C-2_'] ## as specified in TransAD paper (in their main script they only load 'C-1_') -- TODO check. Checked as below:
    # Notes after reading the TransAD paper:
    # The text descriptions (4.1.5) and the number of data points shown in Table 1 do not match.
    # In Table 1, they loaded all the 27 traces but "P-2". This seems to be the common practice in the literature.
    # For example: https://github.com/NetManAIOps/OmniAnomaly/blob/master/data_preprocess.py
    # And according to the label.csv file, "P-2" belongs to the SMAP dataset.

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    if "P-2_" in data_traces:
        data_traces.remove("P-2_")
    print(f'[INFO:] MSL contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)

    # Ensure the number of data points
    assert np.concatenate(data[0]).shape[0] == 58317
    assert np.concatenate(data[1]).shape[0] == 73729
    assert np.concatenate(data[2]).shape[0] == 73729
    assert np.concatenate(
        data[0]).shape[1] == np.concatenate(
        data[1]).shape[1] == np.concatenate(
            data[2]).shape[1] == 55

    # Please notes that the `labels` is a 2D array
    return data[0], data[1], data[2]


def _load_smap(data_path: Union[str, Path] = DEFAULT_DATA_DIR / "SMAP") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    files_path = data_path

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] SMAP contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))
        data.append(trace_data)
    data_tuple: Tuple[np.ndarray, 3] = tuple(data)

    # Please note that the `labels` is a 2D array
    return data_tuple[0], data_tuple[1], data_tuple[2]


dataset_loader_map: Dict[GeneralDataset, Callable[[str | Path], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        GeneralDataset.WADI: _load_wadi,
        GeneralDataset.WADI_GDN: _load_wadi_gdn,
        GeneralDataset.SWAT: _load_swat,
        GeneralDataset.UCR_1: _load_ucr,
        GeneralDataset.UCR_2: _load_ucr_2,
        GeneralDataset.UCR_3: _load_ucr_3,
        GeneralDataset.UCR_4: _load_ucr_4,
        GeneralDataset.UCR_ALL: _load_ucr_all,
        GeneralDataset.SMD: _load_smd,
        GeneralDataset.MSL: _load_msl,
        GeneralDataset.SMAP: _load_smap
}
"""
A dictionary mapping the dataset enum to the loader function of the dataset.
"""