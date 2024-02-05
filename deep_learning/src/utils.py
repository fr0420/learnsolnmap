"""
Utilities

https://github.com/gorodnitskiy/yet-another-lightning-hydra-template/blob/main/src/utils/utils.py
"""

import os
import logging
import datetime
import random
import string
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
import torch 
from callbacks.plot_prediction import plot_energy_profile, plot_trajectory_argoncrystal

from omegaconf import DictConfig
from typing import List
from torch import Tensor


logger = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[pl.Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[pl.Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_litloggers(loggers_cfg: DictConfig) -> List[pl.loggers.Logger]:
    """Instantiates lightning loggers from config."""

    litloggers: List[pl.loggers.Logger] = []

    if not loggers_cfg:
        logger.warning("No logger configs found! Skipping...")
        return litloggers

    for _, lg_conf in loggers_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            litloggers.append(hydra.utils.instantiate(lg_conf))

    return litloggers


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Saves additionally:
    - Number of model parameters

    Args:
        object_dict (dict): Dict object with all parameters.
    """

    hparams = {}

    cfg = object_dict["cfg"]
    cfg = OmegaConf.to_container(cfg, resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["module"] = cfg["module"]

    # save number of model parameters
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    hparams["module"]["params_count"] = {
        "total": n_total,
        "trainable": n_trainable,
        "non_trainable": n_non_trainable
    }

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["profiler"] = cfg.get("profiler")
    hparams["task_name"] = cfg.get("task_name")
    hparams["run_name"] = cfg.get("run_name")
    hparams["run_id"] = cfg.get("run_id")
    hparams["init_model_ckpt"] = cfg.get("init_model_ckpt")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for l in trainer.loggers:
        l.log_hyperparams(hparams)


def generate_random_string(n: int = 5) -> str:
    """Generates a random string of given length in lowercase."""

    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def get_run_name(ckpt_path: str) -> str:
    """Gets the run name of an existing run or generates a new one using current time."""

    if not ckpt_path: 
        run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logger.info(f"Setting run name to current time {run_name}")
    else:
        run_name = ckpt_path.split("/")[-3].split("_")[0]
        logger.info(f"Extracting run name from the given ckpt path: {run_name}")

    return run_name


def get_run_id(ckpt_path: str) -> str:
    """Gets the run id of an existing run or generates a new unique id."""

    if not ckpt_path: 
        run_id = generate_random_string(6)
        logger.info(f"Setting run id to {run_id}")
    else:
        run_id = ckpt_path.split("/")[-3].split("_")[-1]
        logger.info(f"Extracting run id from the given ckpt path: {run_id}")

    return run_id


def save_metrics(metrics: dict, dirname: str) -> None:
    """Save metrics dict returned by `Trainer.test` method."""

    path = os.path.join(dirname, f"test_metrics.csv")
    df = pd.DataFrame.from_dict(metrics).T
    df.to_csv(path)
    
    logger.info(f"Saved test metrics to: {path}")


def save_predictions(predictions: List[List[Tensor]], dirname: str) -> None:
    """Save predictions returned by `Trainer.predict` method."""

    if not predictions:
        logger.warning("Predictions is empty! Saving was cancelled ...")
        return

    path = os.path.join(dirname, "predictions")
    os.makedirs(path, exist_ok=True)

    n_traj = len(predictions)
    traj_len = len(predictions[0])
    dof = len(predictions[0][0]) // 2

    cols = [f"v{i}" for i in range(1, dof+1)] + [f"x{i}" for i in range(1, dof+1)]

    for i in range(n_traj):
        data = torch.stack(predictions[i]).numpy()
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(os.path.join(path, f"traj{i+1}.csv"), index=False)
    
    logger.info(f"Saved {n_traj} predicted trajectories (traj_len = {traj_len}) to: {path}")


def save_energy_plots(predictions: List[List[Tensor]], model: pl.LightningModule, dirname: str) -> None:
    """Plot and save energy profiles computed from predictions returned by `Trainer.predict` method."""

    if not predictions:
        logger.warning("Predictions is empty! Saving was cancelled ...")
        return

    path = os.path.join(dirname, "predictions")
    os.makedirs(path, exist_ok=True)

    n_traj = len(predictions)

    for i in range(n_traj):
        data = torch.stack(predictions[i])
        plot_energy_profile(data, model, filepath=os.path.join(path, f"traj{i+1}.pdf"))
        plot_trajectory_argoncrystal(data, filepath=os.path.join(path, f"x_traj{i+1}.pdf"))
            
    logger.info(f"Saved {n_traj} energy plots to: {path}")
