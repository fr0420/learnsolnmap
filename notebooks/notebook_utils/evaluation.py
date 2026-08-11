"""
Model evaluation utilities for machine learning models.

This module provides utilities for evaluating machine learning models across different 
physical problems. It includes functions for model predictions, derivatives, residuals, 
checkpoint management, and benchmarking.

The module is organized into several categories:
- One-step prediction functions (independent predictions from same u0)
- Recursive prediction functions (dependent predictions using previous output)
- Derivative computation functions  
- Residual computation functions
- Checkpoint and data management utilities
- Benchmarking utilities
"""

import copy
import glob
import re
import sys
import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple


# =============================================================================
# Deep Learning Module Imports
# =============================================================================

def _setup_deep_learning_imports():
    """Setup imports from deep learning modules. Called once at module load."""
    global load_model_from_ckpt_dl, time_forward_dl
    
    deep_learning_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "deep_learning", "src"))
    
    if deep_learning_src not in sys.path:
        sys.path.insert(0, deep_learning_src)
    
    try:
        from utils.checkpoint_utils import load_model_from_ckpt as load_model_from_ckpt_dl
        from utils.benchmark_utils import time_forward as time_forward_dl
    except ImportError as e:
        # Set to None if imports fail - will be handled in wrapper functions
        load_model_from_ckpt_dl = None
        time_forward_dl = None
        print(f"Warning: Could not import deep learning utilities: {e}")

# Initialize deep learning imports
_setup_deep_learning_imports()


# =============================================================================
# Core Tensor Preparation Utilities
# =============================================================================

def prepare_model_inputs(u0: Union[np.ndarray, torch.Tensor, List], 
                        t: Union[float, int, np.ndarray, torch.Tensor, List],
                        p: Dict[str, Any], 
                        model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Prepare inputs for model evaluation with proper tensor handling and broadcasting.
    
    Handles 4 input cases:
    1. Single u0 vector (dim,), single t -> u0: (1, dim), t: (1, 1)
    2. Single u0 vector (dim,), multiple t -> u0: (len(t), dim), t: (len(t), 1) 
    3. Multiple u0 vectors (bs, dim), single t -> u0: (bs, dim), t: (bs, 1)
    4. Multiple u0 vectors (bs, dim), multiple t (bs,) -> u0: (bs, dim), t: (bs, 1)
    
    Args:
        u0: Initial condition(s)
        t: Time(s) 
        p: Parameters dictionary
        model: Model instance (for dtype/device)
        
    Returns:
        tuple: (u0_torch, t_torch, p_torch) with proper shapes and types
    """
    # Convert u0 to torch tensor
    if isinstance(u0, np.ndarray):
        u0_torch = torch.tensor(u0, dtype=model.dtype).to(model.device)
    else:
        u0_torch = u0.to(model.device).to(model.dtype)
    
    # Ensure u0_torch is 2D: (batch_size, dim)
    if u0_torch.dim() == 1:
        u0_torch = u0_torch.unsqueeze(0)
    
    # Convert t to torch tensor and ensure it's 2D: (batch_size, 1)
    if isinstance(t, (int, float)):
        t_torch = torch.tensor([[t]], dtype=model.dtype).to(model.device)
    elif isinstance(t, np.ndarray):
        if t.ndim == 0:  # 0D numpy array (scalar)
            t_torch = torch.tensor([[t.item()]], dtype=model.dtype).to(model.device)
        else:
            t_torch = torch.tensor(t, dtype=model.dtype).to(model.device)
            if t_torch.dim() == 1:
                t_torch = t_torch.unsqueeze(-1)
    elif isinstance(t, list):
        t_torch = torch.tensor(t, dtype=model.dtype).to(model.device)
        if t_torch.dim() == 1:
            t_torch = t_torch.unsqueeze(-1)
    else:  # torch.Tensor
        t_torch = t.to(model.device).to(model.dtype)
        if t_torch.dim() == 0:
            t_torch = t_torch.unsqueeze(0).unsqueeze(-1)
        elif t_torch.dim() == 1:
            t_torch = t_torch.unsqueeze(-1) 
    
    # Handle broadcasting cases
    u0_batch_size = u0_torch.shape[0]
    t_batch_size = t_torch.shape[0]
    
    if u0_batch_size == 1 and t_batch_size == 1:
        # Case 1: Single u0, single t - already correct shapes
        pass
    elif u0_batch_size == 1 and t_batch_size > 1:
        # Case 2: Single u0, multiple t - broadcast u0
        u0_torch = u0_torch.repeat(t_batch_size, 1)
    elif u0_batch_size > 1 and t_batch_size == 1:
        # Case 3: Multiple u0, single t - broadcast t
        t_torch = t_torch.repeat(u0_batch_size, 1)
    else:
        # Case 4: Multiple u0, multiple t - check batch sizes match
        if u0_batch_size != t_batch_size:
            raise ValueError(f"Batch size mismatch: u0 has {u0_batch_size} samples, "
                           f"but t has {t_batch_size} values")
    
    # Handle parameters - each parameter gets shape (batch_size, 1)
    batch_size = u0_torch.shape[0]
    p_torch = {}
    for k, v in p.items():
        if isinstance(v, (int, float)):
            p_torch[k] = torch.full((batch_size, 1), v, dtype=model.dtype).to(model.device)
        else:
            v_torch = torch.tensor(v, dtype=model.dtype).to(model.device)
            if v_torch.dim() == 0:
                p_torch[k] = torch.full((batch_size, 1), v_torch.item(), dtype=model.dtype).to(model.device)
            else:
                p_torch[k] = v_torch.unsqueeze(-1) if v_torch.dim() == 1 else v_torch

    return u0_torch, t_torch, p_torch


# =============================================================================
# One-Step Prediction Functions
# =============================================================================

def one_step_predict(model: torch.nn.Module, u0: np.ndarray, t: Union[float, List[float]], params: Dict[str, Any]) -> np.ndarray:
    """
    Compute one-step prediction Phi(u0, t) with parameters p.
    For scalar t: single prediction. For list/array t: multiple independent predictions.
    
    Args:
        model: Model instance
        u0: Initial condition vector
        t: Time(s) - scalar or list/array of floats
        params: Parameters dictionary
        
    Returns:
        np.ndarray: Predicted state(s): scalar t -> (state_dim,), array t -> (len(t), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, t, params, model)
    u = model(u0_torch, t_torch, p_torch)
    
    # Return scalar if single time, array if multiple times
    if isinstance(t, (int, float)) or (isinstance(t, np.ndarray) and t.ndim == 0):
        return u.squeeze(0).detach().cpu().numpy()
    else:
        return u.detach().cpu().numpy()


def one_step_predict_ignore_net(model: torch.nn.Module, u0: np.ndarray, t: Union[float, List[float]], params: Dict[str, Any]) -> np.ndarray:
    """
    Compute one-step prediction ignoring the network component.
    For scalar t: single prediction. For list/array t: multiple independent predictions.
    
    Args:
        model: Model instance
        u0: Initial condition vector  
        t: Time(s) - scalar or list/array of floats
        params: Parameters dictionary
        
    Returns:
        np.ndarray: Predicted state(s): scalar t -> (state_dim,), array t -> (len(t), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, t, params, model)
    u = model.forward_ignore_net(u0_torch, t_torch, p_torch)
    
    # Return scalar if single time, array if multiple times
    if isinstance(t, (int, float)) or (isinstance(t, np.ndarray) and t.ndim == 0):
        return u.squeeze(0).detach().cpu().numpy()
    else:
        return u.detach().cpu().numpy()


def one_step_predict_no_velocity_preservation(model: torch.nn.Module, u0: np.ndarray, t: Union[float, List[float]], params: Dict[str, Any]) -> np.ndarray:
    """
    Compute one-step prediction with velocity norm preservation turned off.
    For scalar t: single prediction. For list/array t: multiple independent predictions.
    
    Args:
        model: Model instance
        u0: Initial condition vector
        t: Time(s) - scalar or list/array of floats
        params: Parameters dictionary
        
    Returns:
        np.ndarray: Predicted state(s): scalar t -> (state_dim,), array t -> (len(t), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, t, params, model)
    
    # Create a copy of the model and turn off velocity preservation
    model_copy = copy.deepcopy(model)
    model_copy.preserve_velocity_norm = None
    u = model_copy(u0_torch, t_torch, p_torch)
    
    # Return scalar if single time, array if multiple times
    if isinstance(t, (int, float)) or (isinstance(t, np.ndarray) and t.ndim == 0):
        return u.squeeze(0).detach().cpu().numpy()
    else:
        return u.detach().cpu().numpy()


# =============================================================================
# Recursive Prediction Functions
# =============================================================================

def recursive_predict(model: torch.nn.Module, u0: Union[np.ndarray, List[np.ndarray]], dt: float, params: Dict[str, Any], sequence_len: int) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Predict a sequence using recursive application of the model.
    Each step uses the output of the previous step as input: u_{i+1} = Phi(u_i, dt).
    
    Efficiently handles both single initial conditions and batches of initial conditions.
    For batches, processes all conditions simultaneously for better performance.
    
    Args:
        model: Model instance
        u0: Initial condition(s) - single vector or list of vectors
        dt: Time step
        params: Parameters dictionary
        sequence_len: Number of recursive steps
        
    Returns:
        np.ndarray or List[np.ndarray]: 
            - Single u0: (sequence_len, state_dim) array
            - Multiple u0: List of (sequence_len, state_dim) arrays
    """
    # Handle single initial condition
    if isinstance(u0, np.ndarray):
        u0_torch, t_torch, p_torch = prepare_model_inputs(u0, dt, params, model)
        
        # Use model's built-in predict_sequence for single case
        u_seq = model.predict_sequence(u0_torch, t_torch, p_torch, sequence_len)  
        return torch.stack(u_seq, dim=0).squeeze(1).detach().cpu().numpy()
    
    # Handle multiple initial conditions (batch processing)
    elif isinstance(u0, list):
        return batch_recursive_predict(model, u0, dt, params, sequence_len)
    
    else:
        raise TypeError(f"u0 must be np.ndarray or list of np.ndarray, got {type(u0)}")


def batch_recursive_predict(model: torch.nn.Module, u0_list: List[np.ndarray], dt: float, params: Dict[str, Any], sequence_len: int) -> List[np.ndarray]:
    """
    Predict sequences for multiple initial conditions using efficient batch processing.
    All initial conditions are processed simultaneously at each recursive step.
    
    Args:
        model: Model instance
        u0_list: List of initial condition vectors
        dt: Time step
        params: Parameters dictionary
        sequence_len: Number of recursive steps
        
    Returns:
        List[np.ndarray]: List of (sequence_len, state_dim) arrays, one for each initial condition
    """
    if not u0_list:
        return []
    
    # Convert list of initial conditions to batch tensor
    u0_batch = np.array(u0_list)  # (batch_size, state_dim)
    u0_torch = torch.tensor(u0_batch, dtype=model.dtype).to(model.device)
    
    # Prepare time and parameter tensors for batch
    t_torch = torch.ones_like(u0_torch[..., :1]) * dt  # (batch_size, 1)
    p_torch = {k: torch.full_like(t_torch, v) for k, v in params.items()}
    
    # Manual recursive prediction for batch efficiency
    u_seq_list = [u0_batch]  # Start with initial conditions
    with torch.no_grad():
        u = u0_torch
        for _ in range(sequence_len - 1):
            u = model(u, t_torch, p_torch)
            u_seq_list.append(u.detach().cpu().numpy())
    
    # Convert to list of individual sequences
    u_seq_array = np.array(u_seq_list)  # (sequence_len, batch_size, state_dim)
    return [u_seq_array[:, i] for i in range(len(u0_list))]


# =============================================================================
# Batch Prediction Functions
# =============================================================================

def predict_batch(model: torch.nn.Module, u0_batch: torch.Tensor, t_batch: Union[torch.Tensor, float], params: Dict[str, Any]) -> torch.Tensor:
    """
    Predict for a batch of initial conditions and times.
    
    Args:
        model: Model instance
        u0_batch: Initial conditions tensor (batch_size, state_dim)
        t_batch: Times tensor (batch_size,) or float
        params: Parameters dictionary
        
    Returns:
        torch.Tensor: Predicted states (batch_size, state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0_batch, t_batch, params, model)
    with torch.no_grad():
        u = model(u0_torch, t_torch, p_torch)
    return u


# =============================================================================
# Derivative Computation Functions
# =============================================================================

def compute_first_derivative(model: torch.nn.Module, u0: np.ndarray, times: List[float], params: Dict[str, Any]) -> np.ndarray:
    """
    Compute first derivative dPhi/dt(u0, t) for each t in times.
    
    Args:
        model: Model instance
        u0: Initial condition vector
        times: List of time values
        params: Parameters dictionary
        
    Returns:
        np.ndarray: First derivatives (len(times), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, times, params, model)
    dPhidt = model._calc_dPhidt(u0_torch, t_torch, p_torch)
    return dPhidt.detach().cpu().numpy()


def compute_second_derivative(model: torch.nn.Module, u0: np.ndarray, times: List[float], params: Dict[str, Any]) -> np.ndarray:
    """
    Compute second derivative d²Phi/dt²(u0, t) for each t in times.
    
    Args:
        model: Model instance
        u0: Initial condition vector
        times: List of time values
        params: Parameters dictionary
        
    Returns:
        np.ndarray: Second derivatives (len(times), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, times, params, model)
    d2Phidt2 = model._calc_d2Phidt2(u0_torch, t_torch, p_torch)
    return d2Phidt2.detach().cpu().numpy()


# =============================================================================
# Residual Computation Functions
# =============================================================================

def compute_exact_residual(model: torch.nn.Module, u0: np.ndarray, times: List[float], params: Dict[str, Any]) -> np.ndarray:
    """
    Compute exact residual R(u0, t) for each t in times.
    
    Args:
        model: Model instance
        u0: Initial condition vector
        times: List of time values
        params: Parameters dictionary
        
    Returns:
        np.ndarray: Exact residuals (len(times), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, times, params, model)
    residual = model._calc_residual_error(u0_torch, t_torch, p_torch, reduction="none", use_mse=True)
    return residual.detach().cpu().numpy()


def compute_numerical_residual(model: torch.nn.Module, u0: np.ndarray, times: List[float], params: Dict[str, Any], 
                              integrator_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Compute numerical residual R_h(u0, t) using a numerical integrator.
    
    Args:
        model: Model instance
        u0: Initial condition vector
        times: List of time values
        params: Parameters dictionary
        integrator_config: Optional integrator configuration
        
    Returns:
        np.ndarray: Numerical residuals (len(times), state_dim)
    """
    u0_torch, t_torch, p_torch = prepare_model_inputs(u0, times, params, model)
    
    if integrator_config is not None:
        # Temporarily override the model's integrator config
        original_config = model.loss_hparams.get("numerical_residual", {}).get("integrator")
        model.loss_hparams.setdefault("numerical_residual", {})["integrator"] = integrator_config
    else:
        integrator_config = model.loss_hparams.get("numerical_residual", {}).get("integrator", 
                                                                                {"method": "ImplicitMidpoint", "stepsize": 0.1, "nsteps": 1})
    
    integrator = model._instantiate_integrator(integrator_config)
    residual = model._calc_numerical_residual_error(u0_torch, t_torch, p_torch, integrator, reduction="none", use_mse=True)
    
    # Restore original config if it was overridden
    if integrator_config is not None and 'original_config' in locals() and original_config is not None:
        model.loss_hparams["numerical_residual"]["integrator"] = original_config
        
    return residual.detach().cpu().numpy()


# =============================================================================
# Checkpoint and Data Management Utilities
# =============================================================================

def extract_epoch_from_checkpoint_path(ckpt_path: str) -> Optional[int]:
    """
    Extract epoch number from checkpoint file path.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        int or None: Epoch number if found, None otherwise
    """
    match = re.search(r'epoch(\d+)', ckpt_path)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint_history(ckpt_dir: str, model_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all epoch checkpoints from a directory.
    
    Args:
        ckpt_dir: Directory containing checkpoint files
        model_name: Name of the model
        
    Returns:
        dict: Dictionary mapping epoch names to checkpoint info
    """
    ckpt_paths = glob.glob(ckpt_dir + "/*epoch*.ckpt")
    ckpt_paths = sorted(ckpt_paths, key=lambda ckpt: extract_epoch_from_checkpoint_path(ckpt))

    model_ckpts = {}
    for path in ckpt_paths:
        epoch = extract_epoch_from_checkpoint_path(path)
        if epoch is not None:
            model_ckpts[f"epoch{epoch}"] = {
                "model_name": model_name,
                "ckpt_path": path
            }
    return model_ckpts


def find_latest_checkpoint(ckpt_dir: str) -> str:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        ckpt_dir: Directory containing checkpoint files
        
    Returns:
        str: Path to the latest checkpoint file
        
    Raises:
        FileNotFoundError: If no latest checkpoint is found
    """
    latest_ckpt = glob.glob(f"{ckpt_dir}/latest*.ckpt")
    if latest_ckpt:
        return latest_ckpt[0]
    else:
        raise FileNotFoundError(f"No latest checkpoint found in {ckpt_dir}")


def load_model_from_checkpoint(ckpt_path: str, model_name: str = "T0CenteredSolutionMap", strict: bool = True,
load_net_T0_only: bool = False) -> Tuple[torch.nn.Module, torch.dtype]:
    """
    Load a model from a checkpoint file.
    
    Args:
        ckpt_path: Path to the checkpoint file
        model_name: Name of the model class to instantiate
        strict: Whether to strictly load the state dict
        load_net_T0_only: Whether to load only the net_T0 module for TaylorBasedT0CenteredSolutionMap. Temporary fix.
    Returns:
        tuple: (model, dtype) where model is the loaded model and dtype is torch.dtype
        
    Raises:
        ImportError: If deep learning modules are not available
    """
    if load_model_from_ckpt_dl is None:
        raise ImportError("Deep learning checkpoint utilities not available. "
                         "Make sure the deep_learning module is properly installed.")
    
    return load_model_from_ckpt_dl(ckpt_path, model_name, strict, load_net_T0_only)


def load_batch_data(batch_size: int, data_dir: str, t_range: Optional[Tuple[float, float]] = None, 
                   t: Optional[float] = None, required_params: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load a batch of data for model evaluation.
    
    Args:
        batch_size: Number of samples to load
        data_dir: Directory containing the data
        t_range: Time range filter (min, max)
        t: Specific time value filter
        required_params: Required parameter names
        
    Returns:
        dict: Batch data dictionary
        
    Raises:
        ImportError: If datamodules is not available
    """
    try:
        from datamodules.datasets import get_sequence_data
    except ImportError:
        raise ImportError("datamodules.datasets.get_sequence_data not available. "
                         "Make sure you're running this from the correct environment.")
    
    data_config = {
        "data_dir": data_dir,
        "sequence_len": 2, 
        "dtype": "float64",
        "downsample_factor": 1,
        "Dt_filter": t_range,
        "Dt": t,
        "required_params": required_params,
    }
    ds = get_sequence_data(data_config)
    return ds[:batch_size]


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_forward_pass(model: torch.nn.Module, nsteps_list: List[int] = [0, 1, 2]) -> Any:
    """
    Benchmark forward pass time of a model.
    
    Args:
        model: The model to benchmark
        nsteps_list: List of sequence lengths to benchmark
        
    Returns:
        Benchmark comparison results
        
    Raises:
        ImportError: If deep learning modules are not available
    """
    if time_forward_dl is None:
        raise ImportError("Deep learning benchmark utilities not available. "
                         "Make sure the deep_learning module is properly installed.")
    
    return time_forward_dl(model, nsteps_list)


# =============================================================================
# Convenience Functions for Backward Compatibility
# =============================================================================

# Legacy function names for backward compatibility
def forward(model, u0, t, p):
    """Legacy: Compute one-step prediction Phi(u0, t) with parameters p."""
    return one_step_predict(model, u0, t, p)

def forward_t_list(model, u0, t_list, p):
    """Legacy: Compute one-step predictions Phi(u0, t) for a list of times."""
    return one_step_predict(model, u0, t_list, p)

def forward_t_list_ignore_net(model, u0, t_list, p):
    """Legacy: Compute one-step predictions ignoring the network component."""
    return one_step_predict_ignore_net(model, u0, t_list, p)

def forward_t_list_turn_off_preserve_velocity_norm(model, u0, t_list, p):
    """Legacy: Compute one-step predictions with velocity norm preservation turned off."""
    return one_step_predict_no_velocity_preservation(model, u0, t_list, p)

def numerical_residual_t_list(model, u0, t_list, p, integrator_config=None):
    """Legacy: Compute numerical residual R_h(u0, t) for a list of times."""
    return compute_numerical_residual(model, u0, t_list, p, integrator_config)

def exact_residual_t_list(model, u0, t_list, p):
    """Legacy: Compute exact residual R(u0, t) for a list of times."""
    return compute_exact_residual(model, u0, t_list, p)

def dPhidt_t_list(model, u0, t_list, p):
    """Legacy: Compute dPhidt(u0, t) for a list of times."""
    return compute_first_derivative(model, u0, t_list, p)

def d2Phidt2_t_list(model, u0, t_list, p):
    """Legacy: Compute d2Phidt2(u0, t) for a list of times."""
    return compute_second_derivative(model, u0, t_list, p)

def batch_forward(model, u0_batch, t_batch, p):
    """Legacy: Predict for a batch of initial conditions and times."""
    return predict_batch(model, u0_batch, t_batch, p)

def predict_sequence(model, u0, dt, p, sequence_len):
    """Legacy: Predict a sequence using recursive application of the model."""
    return recursive_predict(model, u0, dt, p, sequence_len)

def get_epoch_from_ckpt(ckpt_path):
    """Legacy: Extract epoch number from checkpoint file path."""
    return extract_epoch_from_checkpoint_path(ckpt_path)

def find_latest_ckpt(ckpt_dir):
    """Legacy: Find the latest checkpoint in a directory."""
    return find_latest_checkpoint(ckpt_dir)

def get_batch(batch_size, data_dir, t_range=None, t=None, required_params=None):
    """Legacy: Load a batch of data for model evaluation."""
    return load_batch_data(batch_size, data_dir, t_range, t, required_params)

def load_model_from_ckpt(ckpt_path, model_name="T0CenteredSolutionMap", strict=True, load_net_T0_only=False):
    """Legacy: Load a model from a checkpoint file."""
    return load_model_from_checkpoint(ckpt_path, model_name, strict, load_net_T0_only)

def time_forward(model, nsteps_list=[0, 1, 2]):
    """Legacy: Benchmark forward pass time of a model."""
    return benchmark_forward_pass(model, nsteps_list)