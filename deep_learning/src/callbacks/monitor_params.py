import logging
import numpy as np
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


class MonitorParameters(Callback):
    def __init__(self, param_names: list, every_n_epochs: int = 1):
        super(MonitorParameters, self).__init__()
        self.param_names = param_names
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            for name in self.param_names:
                param = pl_module.state_dict().get(name, None)
                if param is not None:
                    if name == "temp_enc.log_frequencies":
                        freqs = param.detach().exp().cpu().numpy()
                        logger.info(f"Epoch {trainer.current_epoch}: {name} = {param.detach().cpu().numpy()} => frequencies = {freqs}")
                    else: 
                        logger.info(f"Epoch {trainer.current_epoch}: {name} = {param.detach().cpu().numpy()}")
                else:
                    logger.info(f"Warning: Parameter '{name}' not found in the LightningModule.")

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module)
    
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     self.on_epoch_end(trainer, pl_module)

    # def on_test_epoch_end(self, trainer, pl_module):
    #     self.on_epoch_end(trainer, pl_module)