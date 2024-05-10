"""
functions for reading in benchmark datasets:
    - WADI
    - SWaT
    - SMD
    - SMAP
    - MSL
    - UCR/IB
"""

# Python pacakges
from pathlib import Path
from typing import Union, Tuple, Dict, Callable
from enum import Enum
import pandas as pd
import numpy as np
# import h5py


# QuoVadis packages
from quovadis_tad.dataset_utils.data_utils import find_files_in_path, extract_ucr_internal_bleeding_dataset


# Configuration

DATA_DIR = 'resources/processed_datasets'
"""the name of the data directory"""


class GeneralDatasetNames(Enum):
    wadi_127 = 'wadi_127'
    wadi_112 = 'wadi_112'
    swat = 'swat'
    smd = 'smd'
    msl = 'msl'
    smap = 'smap'
    ucr_IB_16 = 'ucr_IB_16'
    ucr_IB_17 = 'ucr_IB_17'
    ucr_IB_18 = 'ucr_IB_18'
    ucr_IB_19 = 'ucr_IB_19'
    ucr_IB = 'ucr_IB'


def load_wadi_127(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "WADI_127"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_wadi_112(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "WADI_112"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_swat(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "SWaT"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_ucr_IB_16(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'135_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_ucr_IB_17(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'136_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_ucr_IB_18(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'137_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels


def load_ucr_IB_19(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"
    data = []
    for file in ['train', 'test', 'labels']:
        data.append(np.load(Path(files_path, f'138_{file}.npy')))
    trainset = data[0]
    testset = data[1]
    test_labels = data[2]

    return trainset, testset, test_labels

def load_ucr_IB(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UCR/Internal Bleeding dataset loader. 
    """
    # load all four UCR/IB traces
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "UCR"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] UCR contains {len(data_traces)} data traces')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)  
    
    return data[0], data[1], data[2]


def load_smd(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "SMD"

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


def load_msl(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "MSL"

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


def load_smap(data_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    files_path = data_path / DATA_DIR / "SMAP"

    data_traces = list(set([f'{filename.split("_")[0]}_' for filename in find_files_in_path(directory=files_path)]))
    
    print(f'[INFO:] SMAP contains {len(data_traces)} data traces.')

    data = []

    for file in ['train', 'test', 'labels']:
        trace_data = []
        for traces in data_traces:
            file_ = traces + file
            trace_data.append(np.load(Path(files_path, f'{file_}.npy')))

        data.append(trace_data)

    # Please note that the `labels` is a 2D array
    return data[0], data[1], data[2]


datasets:  \
    Dict[str, Callable[[Union[str, Path]], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        GeneralDatasetNames.wadi_127.value: load_wadi_127,
        GeneralDatasetNames.wadi_112.value: load_wadi_112,
        GeneralDatasetNames.swat.value: load_swat,
        GeneralDatasetNames.smd.value: load_smd,
        GeneralDatasetNames.msl.value: load_msl,
        GeneralDatasetNames.smap.value: load_smap,
        GeneralDatasetNames.ucr_IB_17.value: load_ucr_IB_17,
        GeneralDatasetNames.ucr_IB_16.value: load_ucr_IB_16,
        GeneralDatasetNames.ucr_IB_18.value: load_ucr_IB_18,
        GeneralDatasetNames.ucr_IB_19.value: load_ucr_IB_19,    
        GeneralDatasetNames.ucr_IB.value: load_ucr_IB
    }

