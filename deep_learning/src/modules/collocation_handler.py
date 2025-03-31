from dataclasses import dataclass
from typing import Dict, Optional, Union, Callable
import torch
from omegaconf import DictConfig


@dataclass
class CollocationPoints:
    u0: torch.Tensor
    t: torch.Tensor
    s: Optional[torch.Tensor] = None
    p: Optional[Dict[str, torch.Tensor]] = None


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
            initial_states_generator_func: Optional[Callable] = None,
            state_params_generator_func: Optional[Callable] = None
        ) -> None:
        """Initialize the generator."""

        self.device = torch.device(device)
        self.dtype = dtype
        self._prepare_random_initial_states = initial_states_generator_func
        self._prepare_random_state_params = state_params_generator_func

    def generate_collocation_points(
            self,
            hparams: DictConfig,
            u0: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            state_params: Optional[Dict[str, torch.Tensor]] = None
        ) -> Dict[str, CollocationPoints]:
        """Generate collocation points for different loss types."""

        if u0 is not None:
            if u0.device != self.device or u0.dtype != self.dtype:
                u0 = u0.to(device=self.device, dtype=self.dtype)
        if t is not None:
            if t.device != self.device or t.dtype != self.dtype:
                t = t.to(device=self.device, dtype=self.dtype)
        if state_params is not None:
            state_params = {
                key: value.to(device=self.device, dtype=self.dtype)
                for key, value in state_params.items()
            }

        active_losses = set(hparams.keys()).intersection(
            self.LOSS_TYPES["temporal_pair"].union(self.LOSS_TYPES["single_time"])
        )
        
        return {
            loss_name: self._generate_points_for_loss(loss_name, hparams[loss_name], u0, t, state_params)
            for loss_name in active_losses
        }

    def _generate_points_for_loss(
            self,
            loss_name: str,
            params: DictConfig,
            u0: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            state_params: Optional[Dict[str, torch.Tensor]] = None,
        ) -> CollocationPoints:
        """Generate collocation points for a specific loss type."""
        
        initial_states, state_params = self._get_initial_states(params.get("batch_size", None), params["u0_dist"], u0, state_params)
        batch_size = initial_states.shape[0]
        
        if loss_name in self.LOSS_TYPES["temporal_pair"]:
            return CollocationPoints(
                u0=initial_states,
                s=self._get_timesteps(batch_size, params["s_dist"], t),
                t=self._get_timesteps(batch_size, params["t_dist"], t),
                p=state_params
            )
        else:  # single_time losses
            return CollocationPoints(
                u0=initial_states,
                t=self._get_timesteps(batch_size, params["t_dist"], t),
                p=state_params
            )

    def _get_initial_states(
            self, 
            batch_size: int, 
            dist_config: DictConfig, 
            precomputed_u0: torch.Tensor, 
            precomputed_state_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get initial states based on distribution configuration."""
        
        if "type" not in dist_config:
            raise ValueError("dist_config should have a 'type' key specifying the distribution type.")
        
        if dist_config["type"] == "precomputed":
            if precomputed_u0 is None:
                raise ValueError("precomputed_u0 must be provided for precomputed distribution.")
            return precomputed_u0, precomputed_state_params
        elif dist_config["type"] == "augment_precomputed":
            if precomputed_u0 is None:
                raise ValueError("precomputed_u0 must be provided for augment_precomputed distribution.")
            if precomputed_state_params is None:
                raise ValueError("precomputed_state_params must be provided for augment_precomputed distribution.")
            return self._augment_precomputed_initial_states(precomputed_u0, precomputed_state_params, dist_config)
        elif dist_config["type"] == "random":
            if batch_size is None:
                raise ValueError("batch_size must be provided for random distribution.")
            try: 
                u0 = self._prepare_random_initial_states(batch_size).to(device=self.device, dtype=self.dtype)
                p = self._prepare_random_state_params(batch_size)
                p = {key: value.to(device=self.device, dtype=self.dtype) for key, value in p.items()}
                return u0, p
            except Exception as e:
                raise ValueError("Error while generating random initial states.") from e
        else:
            raise ValueError(f"Unsupported distribution type: {dist_config['type']}")

    def _augment_precomputed_initial_states(
            self, 
            precomputed_u0: torch.Tensor, 
            precomputed_state_params: Dict[str, torch.Tensor],
            config: DictConfig
        ) -> torch.Tensor:
        # Augment precomputed_u0 with additional samples in high stiffness region 
        # based on the largest eigenvalue of the precomputed state parameters
        stiffness_threshold = config.get("stiffness_threshold", 10.0)
        augmentation_factor = config.get("augmentation_factor", 10)
        perturbation_scale = config.get("perturbation_scale", 0.05)

        if "largest_eigval" not in precomputed_state_params:
            raise ValueError("precomputed_state_params must have 'largest_eigval' key.")
        mask = precomputed_state_params["largest_eigval"].squeeze() > stiffness_threshold

        if mask.sum() == 0:
            return precomputed_u0, precomputed_state_params
        else:
            augmented_u0 = precomputed_u0[mask].repeat(augmentation_factor, 1)
            augmented_state_params = {
                key: value[mask].repeat(augmentation_factor, 1)
                for key, value in precomputed_state_params.items()
            }
            augmented_v0, augmented_x0 = augmented_u0.chunk(2, dim=-1)
            perturbation = torch.randn_like(augmented_v0) * perturbation_scale
            augmented_v0 *= 1 + perturbation
            augmented_u0 = torch.cat([augmented_v0, augmented_x0], dim=-1)
            augmented_u0 = torch.cat([precomputed_u0, augmented_u0], dim=0)
            for key, value in augmented_state_params.items():
                augmented_state_params[key] = torch.cat([precomputed_state_params[key], value], dim=0)
            return augmented_u0, augmented_state_params

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
