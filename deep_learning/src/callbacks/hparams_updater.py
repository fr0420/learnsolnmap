from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig


class HyperparamsUpdater(Callback):
    def __init__(self, loss_hparams: DictConfig, metric_hparams: DictConfig):
        self.loss_hparams = loss_hparams
        self.metric_hparams = metric_hparams
        
    def on_fit_start(self, trainer, pl_module):
        pl_module.update_loss_hparams(self.loss_hparams)
        pl_module.update_metric_hparams(self.metric_hparams)
    
    def on_test_start(self, trainer, pl_module):
        pl_module.update_loss_hparams(self.loss_hparams)
        pl_module.update_metric_hparams(self.metric_hparams)
