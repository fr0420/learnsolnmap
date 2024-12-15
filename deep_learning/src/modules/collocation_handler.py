from dataclasses import dataclass
from typing import Dict, Optional, Union, Callable
import torch
from omegaconf import DictConfig


@dataclass
class CollocationPoints:
    u0: torch.Tensor
    t: torch.Tensor
    s: Optional[torch.Tensor] = None


class CollocationPointsGenerator:
    LOSS_TYPES = {
        # Losses requiring both 's' and 't' timesteps
        "temporal_pair": {"additive", "commutative"},
        
        # Losses requiring only 't' timesteps
        "single_time": {
            "numerical_residual", "residual", "additive_vv",
            "commutative_vv", "dyadic", "reversibility"
        }
    }

    def __init__(
            self, 
            device: Union[str, torch.device], 
            dtype: torch.dtype, 
            initial_states_generator_func: Optional[Callable] = None
        ) -> None:
        """Initialize the generator."""

        self.device = torch.device(device)
        self.dtype = dtype
        self._prepare_random_initial_states = initial_states_generator_func

    def generate_collocation_points(
            self,
            hparams: DictConfig,
            u0: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
        ) -> Dict[str, CollocationPoints]:
        """Generate collocation points for different loss types."""

        if u0 is not None:
            if u0.device != self.device or u0.dtype != self.dtype:
                u0 = u0.to(device=self.device, dtype=self.dtype)
        if t is not None:
            if t.device != self.device or t.dtype != self.dtype:
                t = t.to(device=self.device, dtype=self.dtype)

        active_losses = set(hparams.keys()).intersection(
            self.LOSS_TYPES["temporal_pair"].union(self.LOSS_TYPES["single_time"])
        )
        
        return {
            loss_name: self._generate_points_for_loss(loss_name, hparams[loss_name], u0, t)
            for loss_name in active_losses
        }

    def _generate_points_for_loss(
            self,
            loss_name: str,
            params: DictConfig,
            u0: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
        ) -> CollocationPoints:
        """Generate collocation points for a specific loss type."""
        
        initial_states = self._get_initial_states(params.get("batch_size", None), params["u0_dist"], u0)
        batch_size = initial_states.shape[0]
        
        if loss_name in self.LOSS_TYPES["temporal_pair"]:
            return CollocationPoints(
                u0=initial_states,
                s=self._get_timesteps(batch_size, params["s_dist"], t),
                t=self._get_timesteps(batch_size, params["t_dist"], t)
            )
        else:  # single_time losses
            return CollocationPoints(
                u0=initial_states,
                t=self._get_timesteps(batch_size, params["t_dist"], t)
            )

    def _get_initial_states(self, batch_size: int, dist_config: DictConfig, precomputed_u0: torch.Tensor) -> torch.Tensor:
        """Get initial states based on distribution configuration."""
        
        if "type" not in dist_config:
            raise ValueError("dist_config should have a 'type' key specifying the distribution type.")
        
        if dist_config["type"] == "precomputed":
            if precomputed_u0 is None:
                raise ValueError("precomputed_u0 must be provided for precomputed distribution.")
            return precomputed_u0
        elif dist_config["type"] == "random":
            if batch_size is None:
                raise ValueError("batch_size must be provided for random distribution.")
            try: 
                return self._prepare_random_initial_states(batch_size).to(device=self.device, dtype=self.dtype)
            except Exception as e:
                raise ValueError("Error while generating random initial states.") from e
        else:
            raise ValueError(f"Unsupported distribution type: {dist_config['type']}")

    def _get_timesteps(self, batch_size: int, dist_config: DictConfig, precomputed_t: torch.Tensor) -> torch.Tensor:
        """Get timesteps based on distribution configuration."""

        if "type" not in dist_config:
            raise ValueError("dist_config should have a 'type' key specifying the distribution type.")
        
        if dist_config["type"] == "precomputed":
            if precomputed_t is None:
                raise ValueError("precomputed_t must be provided for precomputed distribution.")
            if precomputed_t.shape[0] != batch_size:
                raise ValueError((
                    f"Shape mismatch: batch_size = {batch_size} does not match"
                    "precomputed_t shape: {precomputed_t.shape}"
                ))
            return precomputed_t
        else:
            return self._prepare_random_timesteps(batch_size, dist_config)

    def _prepare_random_timesteps(self, batch_size: int, dist_config: DictConfig) -> torch.Tensor:
        """Generate random timesteps based on distribution configuration."""

        dist_type = dist_config["type"]
        
        distribution_map = {
            "uniform": {
                "keys": ["min", "max"],
                "generator": lambda batch_size, config: torch.rand(batch_size, 1, device=self.device, dtype=self.dtype) \
                    * (config["max"] - config["min"]) + config["min"]
            },
            "normal": {
                "keys": ["mean", "std"],
                "generator": lambda batch_size, config: torch.randn(batch_size, 1, device=self.device, dtype=self.dtype) \
                    * config["std"] + config["mean"]
            },
        }

        if dist_type not in distribution_map:
            raise ValueError(f"Unsupported distribution type: {dist_type}. Supported types: {list(distribution_map.keys())}.")

        def validate_config(config, required_keys):
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(f"Missing keys in dist_config for {dist_type} distribution: {missing_keys}")

        required_keys = distribution_map[dist_type]["keys"]
        validate_config(dist_config, required_keys)
        generator_func = distribution_map[dist_type]["generator"]
        return generator_func(batch_size, dist_config)
