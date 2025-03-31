#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data1/3body/version3/max_Dt=1e1_min_Dt=-1e1

# python3 ../src/train.py debug=runtime &

python3 ../src/train.py experiment=threebody_idenf trainer.devices=[0] 
# python3 ../src/train.py experiment=threebody_idenf trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241106-225616_ytnp6i/checkpoints/latest-epoch01415.ckpt &

# python3 ../src/eval.py experiment_eval=threebody ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241107-154952_5tlzlz/checkpoints/latest-epoch00639.ckpt &
# python3 ../src/eval.py experiment_eval=threebody output_dir_suffix=_epoch399 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241107-030411_vca0yr/checkpoints/epoch00955-val_loss1.255e-01.ckpt &

wait