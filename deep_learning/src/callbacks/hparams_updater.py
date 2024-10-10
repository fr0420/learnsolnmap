from typing import List
import logging
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class HyperparamsUpdater(Callback):
    def __init__(self, loss_hparams: DictConfig, metric_hparams: DictConfig):
        """
        Custom callback to update loss and metric hyperparameters at the start of training and testing.
        """
        self.loss_hparams = loss_hparams
        self.metric_hparams = metric_hparams
        
    def on_fit_start(self, trainer, pl_module):
        pl_module.update_loss_hparams(self.loss_hparams)
        pl_module.update_metric_hparams(self.metric_hparams)
    
    def on_test_start(self, trainer, pl_module):
        pl_module.update_loss_hparams(self.loss_hparams)
        pl_module.update_metric_hparams(self.metric_hparams)


class IntegratorStepsizeScheduler(Callback):
    def __init__(self, base_stepsize: float, schedule_keys: List[int], schedule_values: List[float], by_epoch: bool = True, 
                 adjust_batch_size: bool = True, base_batch_size: int = 100):
        """
        Custom callback to update the stepsize parameter of the numerical_residual_loss integrator based on a schedule.

        Args:
            base_stepsize (float): The base stepsize `h0` to scale.
            schedule_keys (list): A list of epoch or global step values at which to update the stepsize.
            schedule_values (list): A list of scaling factors to apply to the base stepsize at the corresponding schedule key.
            by_epoch (bool): If True, use `epoch` for scheduling. If False, use `global_step`.
            adjust_batch_size (bool): If True, adjust the batch size to maintain batch_size * stepsize = const.
            base_batch_size (int): The base batch size to scale.
        
        Example:
            schedule_keys=[5, 10], schedule_values=[0.5, 0.1] means set stepsize to `0.5 * h0` at epoch 5 and `0.1 * h0` at epoch 10.
        """
        if len(schedule_keys) != len(schedule_values):
            raise ValueError("schedule_keys and schedule_values must have the same length.")
        self.base_stepsize = base_stepsize
        self.schedule = dict(zip(schedule_keys, schedule_values))
        self.by_epoch = by_epoch
        self.adjust_batch_size = adjust_batch_size
        self.base_batch_size = base_batch_size

    def on_train_epoch_start(self, trainer, pl_module):
        if self.by_epoch:
            self._update_stepsize(trainer.current_epoch, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if not self.by_epoch:
            self._update_stepsize(trainer.global_step, pl_module)

    def _update_stepsize(self, current_step, pl_module):
        if current_step in self.schedule:
            factor = self.schedule[current_step]
            new_stepsize = factor * self.base_stepsize
            updated_hparams = {
                "numerical_residual": {"integrator": {"stepsize": new_stepsize}},
            }
            logger.info(f"Updated numerical_residual_loss integrator stepsize to {new_stepsize} at {'epoch' if self.by_epoch else 'global step'} {current_step}")

            if self.adjust_batch_size:
                new_batch_size = int(self.base_batch_size / factor)
                updated_hparams["numerical_residual"]["batch_size"] = new_batch_size
                logger.info(f"Updated numerical_residual_loss batch size to {new_batch_size} at {'epoch' if self.by_epoch else 'global step'} {current_step}")

            pl_module.update_loss_hparams(updated_hparams)
            pl_module.reinitialize_loss_integrators()


class TimeDistributionScheduler(Callback):
    def __init__(self, schedule_keys: List[int], schedule_values: List[DictConfig], by_epoch: bool = True):
        """
        Custom callback to update the time distribution of the numerical_residual_loss based on a schedule.

        Args:
            schedule_keys (list): A list of epoch or global step values at which to update the time distribution.
            schedule_values (list): A list of time distribution configurations to apply at the corresponding schedule key.
            by_epoch (bool): If True, use `epoch` for scheduling. If False, use `global_step`.
        
        Example:
            schedule_keys=[5, 10], schedule_values=[0.5, 0.1] means set time distribution to `0.5` at epoch 5 and `0.1` at epoch 10.
        """
        if len(schedule_keys) != len(schedule_values):
            raise ValueError("schedule_keys and schedule_values must have the same length.")
        self.schedule = dict(zip(schedule_keys, schedule_values))
        self.by_epoch = by_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if self.by_epoch:
            self._update_time_distribution(trainer.current_epoch, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if not self.by_epoch:
            self._update_time_distribution(trainer.global_step, pl_module)

    def _update_time_distribution(self, current_step, pl_module):
        if current_step in self.schedule:
            time_distribution = self.schedule[current_step]
            updated_hparams = {
                "numerical_residual": {"t_dist": time_distribution},
            }
            logger.info(f"Updated numerical_residual_loss time distribution to {time_distribution} at {'epoch' if self.by_epoch else 'global step'} {current_step}")
            pl_module.update_loss_hparams(updated_hparams)
