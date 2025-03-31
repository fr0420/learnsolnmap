#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data1/3body-2d-equalmass/version6/max_Dt=1e0_min_Dt=-1e0_filtered

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=threebody2d_equalmass_fixedstep trainer.devices=[0] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250127-153711_dgcgja/checkpoints/latest-epoch00140.ckpt

# python3 ../src/train.py experiment=threebody2d_equalmass_t0centered trainer.devices=[0] & 

# python3 ../src/train.py experiment=threebody2d_equalmass_idenf trainer.devices=[1] &
# python3 ../src/train.py experiment=threebody2d_equalmass_idenf trainer.devices=[0] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241217-142608_4l7rs0/checkpoints/latest-epoch00284.ckpt

python3 ../src/train.py experiment=threebody2d_equalmass_stacked trainer.devices=[0] &

# python3 ../src/eval.py experiment_eval=threebody2d ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241114-005933_5dagsn/checkpoints/latest-epoch00063.ckpt
# python3 ../src/eval.py experiment_eval=threebody2d ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241114-235331_bzbs5o/checkpoints/latest-epoch00636.ckpt &
# python3 ../src/eval.py experiment_eval=threebody output_dir_suffix=_epoch399 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241107-030411_vca0yr/checkpoints/epoch00955-val_loss1.255e-01.ckpt &

wait