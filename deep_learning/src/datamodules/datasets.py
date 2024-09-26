from typing import Tuple
import logging
import os 
import pandas as pd
import torch 
from torch.utils.data import Dataset, TensorDataset, random_split


VALID_DTYPES = {"float32": torch.float32, "float64": torch.float64}

logger = logging.getLogger(__name__)


def split_dataset(ds: Dataset, train_fraction: float, seed: int = 0) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing datasets.
    
    Args:
    ds (Dataset): The dataset to split.
    train_fraction (float): The fraction of the dataset to use for training.
    seed (int): The seed for random generator for reproducibility.
    
    Returns:
    Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    assert 0.0 <= train_fraction <= 1.0, "train_fraction must be between 0 and 1"
    n_full = len(ds)
    n_train = int(train_fraction * n_full)
    n_test = n_full - n_train 
    generator = torch.Generator().manual_seed(seed)
    ds_train, ds_test = random_split(ds, [n_train, n_test], generator=generator)
    return ds_train, ds_test 


def downsample_dataset(ds: Dataset, factor: float, seed: int = 0) -> Dataset:
    """
    Downsamples the dataset by a specified factor.
    
    Args:
    ds (Dataset): The dataset to downsample.
    factor (float): The fraction of the dataset to retain.
    seed (int): The seed for random generator for reproducibility.
    
    Returns:
    Dataset: The downsampled dataset.
    """
    assert 0.0 < factor <= 1.0, "factor must be between 0 (exclusive) and 1 (inclusive)"
    if factor == 1.0:
        return ds
    else:
        ds_down, _ = split_dataset(ds, factor, seed)
        return ds_down


def get_sequence_data(config: dict) -> Dataset:
    """
    Prepares a sequence dataset based on a configuration.
    """
    required_keys = ['data_dir', 'sequence_len', 'dtype', 'downsample_factor']
    if not all(key in config for key in required_keys):
        missing_keys = ', '.join(key for key in required_keys if key not in config)
        raise KeyError(f"Missing keys in config: {missing_keys}")
    ds = SequenceData(config["data_dir"], config["sequence_len"], config["dtype"], config.get("Dt", None),
                      config.get("Dt_range", None), config.get("exclude", False))
    return downsample_dataset(ds, config["downsample_factor"])


class SequenceData(Dataset):
    """Custom dataset for sequence data.""" 

    def __init__(self, data_dir: str, sequence_len: int, dtype: str, Dt: float = None,
                 Dt_range: list = None, exclude: bool = False):

        # Validate dtype
        if dtype not in VALID_DTYPES:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {', '.join(VALID_DTYPES.keys())}")

        # Prepare to load sequence files
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.dtype = VALID_DTYPES[dtype]
        
        # Load sequence data
        filenames = [f"U{n}.csv" for n in range(sequence_len)]
        self.target_sequences = [self.load_data(filename) for filename in filenames]
        self.inputs = self.target_sequences[0]

        # Load or set Dt
        self.Dt = self.load_or_set_Dt(Dt)

        # Apply filter based on Dt range if provided
        if Dt_range is not None:
            min_Dt, max_Dt = Dt_range
            if exclude:
                mask = (self.Dt.squeeze() < min_Dt) | (self.Dt.squeeze() > max_Dt)
            else:
                mask = (self.Dt.squeeze() >= min_Dt) & (self.Dt.squeeze() <= max_Dt)
            self.filter_data(mask)

    def load_data(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return torch.tensor(pd.read_csv(file_path).to_numpy(), dtype=self.dtype)

    def load_or_set_Dt(self, Dt):
        Dt_file = os.path.join(self.data_dir, "Dt.csv")
        if os.path.exists(Dt_file):
            Dt = torch.tensor(pd.read_csv(Dt_file).to_numpy(), dtype=self.dtype)
        elif Dt is not None:
            Dt = torch.tensor(Dt, dtype=self.dtype).repeat(len(self.inputs), 1)
        else:
            raise ValueError("Stepsize Dt not provided and 'Dt.csv' not found")
        return Dt

    def filter_data(self, mask):
        self.inputs = self.inputs[mask]
        self.target_sequences = [ts[mask] for ts in self.target_sequences]
        self.Dt = self.Dt[mask]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input": self.inputs[idx],
            "target_seq": [u[idx] for u in self.target_sequences],
            "Dt": self.Dt[idx]
        }


# def get_supervised_data(config: dict) -> TensorDataset:
#     """
#     Prepares a supervised dataset by fetching sequence data and downsampling it based on a configuration.

#     Args:
#     config (dict): Configuration dictionary with keys 'data_dir', 'sequence_len', 'dtype', and 'downsample_factor'.

#     Returns:
#     TensorDataset: The prepared supervised dataset.
    
#     Raises:
#     KeyError: If a required key is missing in the config.
#     """
#     required_keys = ['data_dir', 'sequence_len', 'dtype', 'downsample_factor']
#     if not all(key in config for key in required_keys):
#         missing_keys = ', '.join(key for key in required_keys if key not in config)
#         raise KeyError(f"Missing keys in config: {missing_keys}")
#     ds = SequenceData(config["data_dir"], config["sequence_len"], config["dtype"], config.get("Dt"))
#     return downsample_dataset(ds, config["downsample_factor"])


# def get_unsupervised_data(config: dict) -> TensorDataset:
#     """
#     Prepares an unsupervised dataset by fetching initial points of sequence data and downsampling it based on a configuration.

#     Args:
#     config (dict): Configuration dictionary with keys 'data_dir', 'dtype', and 'downsample_factor'. The sequence length is fixed at 1.

#     Returns:
#     TensorDataset: The prepared unsupervised dataset.
    
#     Raises:
#     KeyError: If a required key is missing in the config.
#     """
#     required_keys = ['data_dir', 'dtype', 'downsample_factor']
#     if not all(key in config for key in required_keys):
#         missing_keys = ', '.join(key for key in required_keys if key not in config)
#         raise KeyError(f"Missing keys in config: {missing_keys}")
#     ds = get_sequence_data(config["data_dir"], 1, config["dtype"])
#     return downsample_dataset(ds, config["downsample_factor"])


