import logging
import os
import pandas as pd
import pytorch_lightning as pl
import torch

from typing import Dict, Union
from torch import Tensor
from matplotlib.figure import Figure
from modules.solnmap import BaseSolutionMap


logger = logging.getLogger(__name__)


def predict_and_plot(samples: Tensor, model: BaseSolutionMap, nsteps: int, t: float, plot: bool = True):
    """Predict and plot trajectories."""

    logger.info("Start predicting!")

    # Move samples to the same device as the model
    samples = samples.to(model.dtype).to(model.device)  # shape: (n_traj, 2*dof)

    # Set model to eval mode for prediction
    model.eval()
    with torch.no_grad():
        t = torch.tensor(t, dtype=model.dtype).repeat(samples.shape[0], 1).to(model.device)
        predictions = model.predict_sequence(samples, t, sequence_len=nsteps+1)
        # if isinstance(model, BaseVariableDtSolutionMap):
        #     if Dt is None:
        #         raise ValueError("Dt must be provided for prediction.")
        #     else:
        #         Dt = torch.tensor(Dt, dtype=model.dtype).repeat(samples.shape[0], 1).to(model.device)
        #     predictions = model.predict_sequence(samples, Dt, sequence_len=nsteps+1)
        # else:
        #     predictions = model(samples, sequence_len=nsteps+1)

    # Set model back to train mode
    model.train()

    # Reshape predictions
    predictions = torch.stack(predictions)              # shape: (traj_len, n_traj, 2*dof)
    predictions = torch.transpose(predictions, 0, 1)    # shape: (n_traj, traj_len, 2*dof)

    # Optionally plot the predicted trajectories
    if plot:
        figures = model.problem.plot_trajectories(predictions.cpu())
        
    return predictions, figures if plot else None


def save_predictions(predictions: Tensor, dirpath: str) -> None:
    """Save predictions as csv files."""

    if predictions is None:
        logger.warning("Predictions is empty! Saving was cancelled ...")
        return

    os.makedirs(dirpath, exist_ok=True)

    n_traj = len(predictions)
    traj_len = len(predictions[0])
    dof = len(predictions[0][0]) // 2

    cols = [f"v{i}" for i in range(1, dof+1)] + [f"x{i}" for i in range(1, dof+1)]

    for i in range(n_traj):
        data = predictions[i].numpy()
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(os.path.join(dirpath, f"traj{i+1}.csv"), index=False)
    
    logger.info(f"Saved {n_traj} predicted trajectories (traj_len = {traj_len}) to: {dirpath}")


def save_figures(figures: Dict[str, Figure], dirpath: str) -> None:
    """Save figures as pdf files."""
    
    if not figures:
        logger.warning("Figures is empty! Saving was cancelled ...")
        return
    
    os.makedirs(dirpath, exist_ok=True)

    for name, figure in figures.items():
        filepath = os.path.join(dirpath, f"{name}.pdf")
        figure.savefig(filepath, dpi=150)
    
    logger.info(f"Saved {len(figures)} figures to: {dirpath}")


class PredictAndPlotTrajectory(pl.Callback):
    """Predict and plot trajectories."""

    def __init__(self, nsteps: int = 100, Dt: float = 0.1, log_freq: int = 2, save_predictions: bool = False, save_figures: bool = False, 
                 output_dir: str = "") -> None:
        self.nsteps = nsteps
        self.Dt = Dt
        self.log_freq = log_freq
        self.save_predictions = save_predictions
        self.save_figures = save_figures
        self.output_dir = output_dir

    def setup(self, trainer: pl.Trainer, pl_module: BaseSolutionMap, stage: str) -> None:
        self.predict_samples: Tensor = pl_module.problem.default_initial_states()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch % self.log_freq == 0:
            self.predict_and_plot(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        self.predict_and_plot(trainer, pl_module)
    
    def predict_and_plot(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        predictions, figures = predict_and_plot(self.predict_samples, pl_module, self.nsteps, self.Dt)
        
        # Add epoch number to the figure titles
        for name, fig in figures.items():
            fig.suptitle(f"epoch {trainer.current_epoch}")
            for ax in fig.get_axes():
                ax.text(0.95, 0.01, f"epoch {trainer.current_epoch}",
                        verticalalignment="bottom", horizontalalignment="right",
                        transform=ax.transAxes, fontsize=15)  # workaround for showing epoch number in wandb panel 

        if trainer.logger:
            log = {f"predict/{name}": fig for name, fig in figures.items()}
            trainer.logger.experiment.log(log, commit=False)

        output_dir = os.path.join(self.output_dir, f"epoch{trainer.current_epoch:05d}")
        
        if self.save_predictions:
            save_predictions(predictions.cpu(), dirpath=output_dir)

        if self.save_figures:
            save_figures(figures, dirpath=output_dir)  
