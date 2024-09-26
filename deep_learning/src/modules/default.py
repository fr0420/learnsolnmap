import hydra
import pytorch_lightning as pl
import torch
from torch import nn

from omegaconf import DictConfig


class BaseLitModel(pl.LightningModule):
    """
    Base lightning model. 

    Reference: 
    https://github.com/gorodnitskiy/yet-another-lightning-hydra-template/blob/main/src/modules/components/lit_module.py
    """
    def __init__(
            self, 
            optimizer: DictConfig, 
            scheduler: DictConfig, 
            weight_init: str
        ) -> None:
        super(BaseLitModel, self).__init__()
        
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.weight_init = weight_init

        self.training_step_outputs = [] 
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _init_weights(self):
        weight_init_fn = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            }[self.weight_init]

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                weight_init_fn(m.weight)

        self.apply(init_weights)

    def configure_optimizers(self):
        optimizer: torch.optim = hydra.utils.instantiate(
            self.opt_params, params=self.parameters(), _convert_="partial"
        )
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_params.get("extras"):
                for key, value in self.slr_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}
    
    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.test_step_outputs.clear()
