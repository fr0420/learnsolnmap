from typing import List

import hydra
from omegaconf import DictConfig
import torch

from modules.solnmap import BaseSolutionMap


class FixedStepSolutionMap(BaseSolutionMap):
    """
    Solution map with a fixed timestep 'T0'.

    For single prediction:
        Phi(u0, T0) = h2o(h2h(i2h(u0)))

    For sequence prediction:
        Phi(u0, k*T0) = h2o(h2h^k(i2h(u0))) for k = 0, 1, 2, ..., sequence_len-1
    """

    def __init__(self, T0: float, network: DictConfig, **kwargs):
        super(FixedStepSolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        self.register_buffer("T0", torch.tensor(T0))

        self.i2h = hydra.utils.instantiate(network.i2h)
        self.h2h = hydra.utils.instantiate(network.h2h)
        self.h2o = hydra.utils.instantiate(network.h2o)

        if self.weight_init is not None:
            self._init_weights()
    
    def predict_sequence(self, u0: torch.Tensor, t: torch.Tensor, sequence_len: int) -> List[torch.Tensor]:
        return self(u0, t, sequence_len=sequence_len)
    
    def forward(self, u0: torch.Tensor, t: torch.Tensor, sequence_len: int) -> List[torch.Tensor]:
        res = []

        u0 = self._apply_nondim(u0)

        hidden = self.i2h(u0)
        out = self.h2o(hidden)
        out = self._apply_dim(out)
        res.append(out)

        for _ in range(sequence_len-1):
            hidden = self.h2h(hidden)
            out = self.h2o(hidden)
            out = self._apply_dim(out)
            res.append(out)

        return res
    
    def freeze_encoder_decoder(self):
        for param in self.i2h.parameters():
            param.requires_grad = False
        for param in self.h2o.parameters():
            param.requires_grad = False
    
    def extra_repr(self):
        return super(FixedStepSolutionMap, self).extra_repr() + f"\nT0: {self.T0.item()}"