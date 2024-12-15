from typing import List

import hydra
import logging
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch

from utils.utils import (
    instantiate_callbacks,
    instantiate_litloggers,
    get_run_name,
    get_run_id
)
from utils.checkpoint_utils import load_model_from_ckpt
from utils.saving_utils import save_test_metrics, save_test_predictions
from utils.benchmark_utils import time_forward
from callbacks.predict_trajectory import predict_and_plot, save_predictions, save_figures


logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "get_run_name", lambda ckpt_path: get_run_name(ckpt_path), use_cache=True
)
OmegaConf.register_new_resolver(
    "get_run_id", lambda ckpt_path: get_run_id(ckpt_path), use_cache=True
)

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> pl.Trainer:

    assert cfg.ckpt_path

    # Print package versions
    logger.info(f"Using pytorch {torch.__version__}")
    logger.info(f"Using pytorch lightning {pl.__version__}")
    logger.info(f"Using hydra {hydra.__version__}")

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        logger.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    logger.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Load lightning model
    logger.info(f"Loading lightning model <{cfg.module._target_}> from checkpoint {cfg.ckpt_path}")
    # load_model_from_ckpt(cfg.ckpt_path, model_name=cfg.module._target_)
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    
    # Temporary fix for the loss function in hyperparameters
    hyper_params = checkpoint.get("hyper_parameters", {})
    loss_target = hyper_params["loss"]["_target_"]
    if loss_target.startswith("losses") and loss_target.split(".")[1] != "losses":
        hyper_params["loss"]["_target_"] = "losses." + hyper_params["loss"]["_target_"]

    model: pl.LightningModule = hydra.utils.instantiate(cfg.module, **hyper_params, _recursive_=False)
    if datamodule.get_dtype() == "float64":
        model = model.double()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    logger.info(f"{model}")

    # Init callbacks
    logger.info("Instantiating callbacks...")
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Init lightning loggers
    logger.info("Instantiating lightning loggers...")
    litloggers: List[pl.loggers.Logger] = instantiate_litloggers(cfg.get("loggers"))

    # Init profiler
    if cfg.get("profiler"):
        logger.info(f"Instantiating profiler <{cfg.profiler._target_}>")
        profiler: pl.profilers.Profiler = hydra.utils.instantiate(cfg.profiler)
    else:
        profiler = None

    # Init lightning trainer
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=litloggers, profiler=profiler)

    # Test the model
    logger.info("Start testing!")
    metrics = trainer.test(
        model=model, 
        datamodule=datamodule
    )
    save_test_metrics(metrics, dirpath=cfg.paths.output_dir)
    if cfg.get("save_test_predictions"):
        predictions = trainer.predict(model=model, datamodule=datamodule)
        save_test_predictions(predictions, dirpath=cfg.paths.output_dir)

    # Benchmark forward time
    logger.info("Start benchmarking forward time!")
    t_forward = time_forward(model)
    logger.info(t_forward)

    # Make predictions
    if cfg.get("predict"):
        predict_samples = model.problem.default_initial_states()
        logger.info(f"Predict samples = {predict_samples}")
        if isinstance(cfg.predict_Dt, float):
            assert isinstance(cfg.predict_nsteps, int)
            dirpath = cfg.paths.output_dir+f"/predictions/{cfg.predict_nsteps}x{cfg.predict_Dt}"
            predictions, figures = predict_and_plot(predict_samples, model, nsteps=cfg.predict_nsteps, t=cfg.predict_Dt)
            save_predictions(predictions.cpu(), dirpath=dirpath)
            save_figures(figures, dirpath=dirpath)
        else:
            assert len(cfg.predict_nsteps) == len(cfg.predict_Dt)
            for nsteps, Dt in zip(cfg.predict_nsteps, cfg.predict_Dt):
                dirpath = cfg.paths.output_dir + f"/predictions/{nsteps}x{Dt}"
                predictions, figures = predict_and_plot(predict_samples, model, nsteps=nsteps, t=Dt)
                save_predictions(predictions.cpu(), dirpath=dirpath)
                save_figures(figures, dirpath=dirpath)

    return trainer 


if __name__ == "__main__":
    main()
