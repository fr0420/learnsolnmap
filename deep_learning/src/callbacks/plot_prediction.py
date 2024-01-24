import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from typing import List
from torch import Tensor
from models import BaseSolutionMap


def reshape_predictions(predictions: List[Tensor]) -> List[Tensor]:
    traj_len = len(predictions)
    n_traj = len(predictions[0])

    trajectories = []
    for i in range(n_traj):
        trajectories.append(torch.stack([predictions[j][i] for j in range(traj_len)]))
    
    return trajectories


def plot_energy_profile(trajectory: Tensor, model: BaseSolutionMap, filepath: str = "", title: str = ""):

    I = model.problem.compute_I(*trajectory.chunk(2, dim=-1)).detach().cpu().numpy()
    H = model.problem.compute_Hamiltonian(*trajectory.chunk(2, dim=-1)).detach().cpu().numpy()
    t = np.arange(len(trajectory)) * model.Delta_t

    fig, ax = plt.subplots()
    
    ax.plot(t, H, linewidth=2, label="H")
    ax.plot(t, I[:, 0], linewidth=2, label="I_1")
    ax.plot(t, I[:, 1], linewidth=2, label="I_2")
    ax.plot(t, I[:, 2], linewidth=2, label="I_3")
    ax.plot(t, I[:, 3], linewidth=2, label="I_tot")
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
    ax.set_title(title)
    ax.text(0.95, 0.01, title,
        verticalalignment="bottom", horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=15)  # workaround for showing figure title in wandb panel 
    ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=150)

    return fig


class PlotEnergyProfile(pl.Callback):

    def __init__(self, nsteps: int = 100, log_freq: int = 2) -> None:
        self.nsteps = nsteps
        self.log_freq = log_freq

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
        predict_samples = self.predict_samples.to(pl_module.dtype).to(pl_module.device)
        predictions = pl_module.predict_step(predict_samples, batch_idx=0, sequence_len=self.nsteps)
        trajectories = reshape_predictions(predictions)

        log = {}
        for i, traj in enumerate(trajectories):
            fig = plot_energy_profile(traj, pl_module, title=f"epoch {trainer.current_epoch}")
            log[f"predict/sample_{i+1}"] = fig
        trainer.logger.experiment.log(log, commit=False)

  