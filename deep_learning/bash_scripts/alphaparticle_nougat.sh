#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data1/alphaparticle/eps=5e-2/version1/Dt=1e0

# python3 ../src/train.py debug=runtime &
# python3 ../src/train.py debug=overfit &

# python3 ../src/train.py experiment=alphaparticle_t0centered trainer.devices=[0] & 
# python3 ../src/train.py experiment=alphaparticle_t0centered trainer.devices=[0] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250228-135702_9pk4ru/checkpoints/latest-epoch00124.ckpt & 

python3 ../src/train.py experiment=alphaparticle_t0centered_taylor trainer.devices=[1] & 

# python3 ../src/train.py experiment=alphaparticle_idenf trainer.devices=[1] &
# python3 ../src/train.py experiment=alphaparticle_idenf trainer.devices=[0] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250314-205000_pytkto/checkpoints/latest-epoch00249.ckpt &

# python3 ../src/train.py experiment=alphaparticle_idenf_taylor trainer.devices=[0] &
# python3 ../src/train.py experiment=alphaparticle_idenf_taylor trainer.devices=[1] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250321-015954_uogpe7/checkpoints/latest-epoch00249.ckpt &

# python3 ../src/train.py experiment=alphaparticle_stacked trainer.devices=[0] 

# python3 ../src/eval.py experiment_eval=threebody2d ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241114-005933_5dagsn/checkpoints/latest-epoch00063.ckpt
# python3 ../src/eval.py experiment_eval=threebody2d ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241114-235331_bzbs5o/checkpoints/latest-epoch00636.ckpt &
# python3 ../src/eval.py experiment_eval=threebody output_dir_suffix=_epoch399 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241107-030411_vca0yr/checkpoints/epoch00955-val_loss1.255e-01.ckpt &

wait