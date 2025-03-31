from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import os 
import pandas as pd
import torch 
from torch.utils.data import Dataset, random_split
from omegaconf import DictConfig, ListConfig


VALID_DTYPES = {"float32": torch.float32, "float64": torch.float64}

logger = logging.getLogger(__name__)


def split_dataset(ds: Dataset, train_fraction: float, seed: int = 0) -> Tuple[Dataset, Dataset]:
    """Splits the dataset into training and testing datasets."""

    if train_fraction < 0.0 or train_fraction > 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    n_full = len(ds)
    n_train = int(train_fraction * n_full)
    n_test = n_full - n_train 
    generator = torch.Generator().manual_seed(seed)
    ds_train, ds_test = random_split(ds, [n_train, n_test], generator=generator)

    return ds_train, ds_test 


def downsample_dataset(ds: Dataset, factor: float, seed: int = 0) -> Dataset:
    """Downsamples the dataset by a specified factor."""

    if factor <= 0.0 or factor > 1.0:
        raise ValueError("factor must be between 0 (exclusive) and 1 (inclusive)")

    return ds if factor == 1.0 else split_dataset(ds, factor, seed)[0]


def get_sequence_data(config: dict) -> Dataset:
    """Prepares a sequence dataset based on a configuration."""

    required_keys = ["data_dir", "sequence_len", "dtype", "downsample_factor"]
    if not all(key in config for key in required_keys):
        missing_keys = ', '.join(key for key in required_keys if key not in config)
        raise KeyError(f"Missing keys in config: {missing_keys}")
    
    ds = SequenceData(
        config["data_dir"], 
        config["sequence_len"], 
        config["dtype"], 
        config.get("Dt", None),
        config.get("Dt_filter", None),
        config.get("required_params", None),
        config.get("param_filters", None)
    )

    return downsample_dataset(ds, config["downsample_factor"])


@dataclass
class ParameterFilter:
    """Dataclass for specifying parameter filtering criteria."""
    min_val: float = -float('inf')
    max_val: float = float('inf')
    exclude: bool = False


class SequenceData(Dataset):
    """Custom dataset for loading and processing sequence data.""" 

    def __init__(
            self, 
            data_dir: str, 
            sequence_len: int,
            dtype: str, 
            Dt: Optional[float] = None,
            Dt_filter: Optional[Union[ListConfig, DictConfig]] = None, 
            required_params: Optional[List[str]] = None,
            param_filters: Optional[Dict[str, Union[ListConfig, DictConfig]]] = None
        ):
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self._validate_and_set_dtype(dtype)
        
        # Load sequence data
        self._load_sequences()
        
        # Handle Dt (time step) information
        self.Dt = self._load_or_set_Dt(Dt)
        
        # Load state parameters
        self.params = self._load_parameters(required_params)
        
        # Apply Dt range filtering if specified
        if Dt_filter is not None:
            self._apply_Dt_filter(Dt_filter)

        # Apply parameter filtering if specified
        if param_filters is not None:
            self._apply_param_filters(param_filters)

    def _validate_and_set_dtype(self, dtype: str) -> None:
        if dtype not in VALID_DTYPES:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {', '.join(VALID_DTYPES.keys())}")
        self.dtype = VALID_DTYPES[dtype]
    
    def _load_sequences(self) -> None:
        try:
            filenames = [f"U{n}.csv" for n in range(self.sequence_len)]
            self.target_sequences = [self._load_file(filename) for filename in filenames]
            self.inputs = self.target_sequences[0]
        except Exception as e:
            raise RuntimeError(f"Error loading sequence files: {str(e)}") from e

    def _load_file(self, filename: str) -> torch.Tensor:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            data = pd.read_csv(filepath)
            return torch.tensor(data.to_numpy(), dtype=self.dtype)
        except Exception as e:
            raise RuntimeError(f"Error loading {filename}: {str(e)}") from e

    def _load_or_set_Dt(self, Dt: float) -> torch.Tensor:
        Dt_file = os.path.join(self.data_dir, "Dt.csv")
        if os.path.exists(Dt_file):
            return self._load_file("Dt.csv")
        elif Dt is not None:
            return torch.tensor(Dt, dtype=self.dtype).repeat(len(self.inputs), 1)
        raise ValueError("Stepsize Dt not provided and 'Dt.csv' not found")

    def _load_parameters(self, required_params: List[str]) -> Dict[str, torch.Tensor]:
        params_file = os.path.join(self.data_dir, "params.csv")
        if not required_params:
            return {}
        elif not os.path.exists(params_file):
            raise FileNotFoundError(f"params.csv not found in {self.data_dir} but required parameters were specified")
        try:
            params_df = pd.read_csv(params_file)
            missing_params = set(required_params) - set(params_df.columns)
            if missing_params:
                raise ValueError(f"Required parameters {missing_params} not found in params.csv")
            params_df = params_df[required_params]
            return {
                col: torch.tensor(params_df[col].values, dtype=self.dtype).unsqueeze(-1)
                for col in params_df.columns
            }
        except Exception as e:
            raise RuntimeError(f"Error loading parameters: {str(e)}") from e

    def _convert_to_parameter_filter(self, filter_spec: Union[ListConfig, DictConfig]) -> ParameterFilter:
        if isinstance(filter_spec, ListConfig) or isinstance(filter_spec, list):
            if len(filter_spec) != 2:
                raise ValueError("List filter spec must contain exactly [min_val, max_val]")
            return ParameterFilter(min_val=filter_spec[0], max_val=filter_spec[1])
        elif isinstance(filter_spec, DictConfig) or isinstance(filter_spec, dict):
            return ParameterFilter(
                min_val=filter_spec.get("min", -float("inf")),
                max_val=filter_spec.get("max", float("inf")),
                exclude=filter_spec.get("exclude", False)
            )
        else:
            raise ValueError(f"Unsupported filter specification type: {type(filter_spec)}")

    def _filter_data(self, mask: torch.Tensor) -> None:
        logger.info(f"Filtering data: {len(self.inputs)} -> {mask.sum()}")
        self.inputs = self.inputs[mask]
        self.target_sequences = [ts[mask] for ts in self.target_sequences]
        self.Dt = self.Dt[mask]
        if self.params:
            self.params = {name: values[mask] for name, values in self.params.items()}

    def _apply_Dt_filter(self, Dt_filter: Union[ListConfig, DictConfig]) -> None:
        Dt_filter = self._convert_to_parameter_filter(Dt_filter)
        Dt_values = self.Dt.squeeze()
        mask = (Dt_values >= Dt_filter.min_val) & (Dt_values <= Dt_filter.max_val)
        if Dt_filter.exclude:
            mask = ~mask
        self._filter_data(mask)

    def _apply_param_filters(self, param_filters: Dict[str, Union[ListConfig, DictConfig]]) -> None:
        if not self.params:
            raise ValueError("No parameters loaded but parameter filtering was requested")
        mask = torch.ones(len(self.inputs), dtype=torch.bool)
        for param_name, filter_spec in param_filters.items():
            if param_name not in self.params:
                raise ValueError(f"Filter parameter '{param_name}' not found in loaded parameters")
            param_filter = self._convert_to_parameter_filter(filter_spec)
            param_values = self.params[param_name].squeeze()            
            param_mask = (param_values >= param_filter.min_val) & (param_values <= param_filter.max_val)
            if param_filter.exclude:
                param_mask = ~param_mask
            mask &= param_mask
        self._filter_data(mask)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "input": self.inputs[idx],
            "target_seq": [u[idx] for u in self.target_sequences],
            "Dt": self.Dt[idx]
        }
        if self.params:
            item["params"] = {name: values[idx] for name, values in self.params.items()}
        return item
