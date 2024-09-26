import logging
import os
import pandas as pd
import torch 

from typing import List


logger = logging.getLogger(__name__)


def save_test_metrics(metrics: dict, dirpath: str) -> None:
    """Save metrics dict returned by `Trainer.test` method."""

    path = os.path.join(dirpath, f"test_metrics.csv")
    df = pd.DataFrame.from_dict(metrics).T
    df.to_csv(path)
    
    logger.info(f"Saved test metrics to: {path}")


def save_test_predictions(predictions: List[dict], dirpath: str, save_targets: bool = True) -> None:
    """Save predictions returned by `Trainer.predict` method."""

    if not predictions:
        logger.warning("Predictions is empty! Saving was cancelled ...")
        return

    dirpath = os.path.join(dirpath, "test_predictions")
    os.makedirs(dirpath, exist_ok=True)

    num_samples = sum([batch["input"].shape[0] for batch in predictions])
    traj_len = len(predictions[0]["pred_seq"])
    dof = predictions[0]["input"].shape[1] // 2

    cols = [f"v{i}" for i in range(1, dof+1)] + [f"x{i}" for i in range(1, dof+1)]

    for i in range(traj_len):
        data = torch.cat([batch["pred_seq"][i] for batch in predictions]).numpy()
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(os.path.join(dirpath, f"U{i}_pred.csv"), index=False)
    
    if save_targets:
        for i in range(traj_len):
            data = torch.cat([batch["target_seq"][i] for batch in predictions]).numpy()
            df = pd.DataFrame(data, columns=cols)
            df.to_csv(os.path.join(dirpath, f"U{i}.csv"), index=False)

    Dt = torch.cat([batch["Dt"] for batch in predictions]).numpy()
    pd.DataFrame(Dt, columns=["Dt"]).to_csv(os.path.join(dirpath, "Dt.csv"), index=False)

    logger.info(f"Saved test predictions (num_samples = {num_samples}, traj_len = {traj_len}) to: {dirpath}")
