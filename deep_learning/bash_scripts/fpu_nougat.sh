#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
# export DATA_DIR=$PROJECT_ROOT/data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5
# export DATA_DIR=$PROJECT_ROOT/data1/fpu/omega=300/version4/Dt=1e0
export DATA_DIR=$PROJECT_ROOT/data1/fpu/omega=50/version1/Dt=1e0

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=fpu trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 1., 1., 0., 0.]" &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 1., 1., 1., 1.]" &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] module/scheduler=none & 
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module/scheduler=1cycle &

python3 ../src/train.py experiment=fpu trainer.devices=[1] &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module/loss=mse &

# python3 ../src/train.py experiment=fpu trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240819-191819_m0b3jt/checkpoints/latest-epoch00085.ckpt &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] trainer.max_epochs=5000 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024907_zqq6f7/checkpoints/last.ckpt &

# python3 ../src/eval.py output_dir_suffix=_energybalanced_misfit ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240820-001828_pu02hz/checkpoints/epoch00499-val_loss1.195e-07.ckpt & 
# python3 ../src/eval.py output_dir_suffix=_energybalanced_integrator ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240820-002437_idu8xj/checkpoints/epoch00499-val_loss1.246e-07.ckpt & 
# python3 ../src/eval.py module._target_=models.BoostedSolutionMap ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240426-193755_y5vbsq/checkpoints/epoch999-val_loss3.288e-06.ckpt &
# python3 ../src/eval.py output_dir_suffix=_2000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024649_76cpvy/checkpoints/epoch1999-val_loss8.837e-07.ckpt &
# python3 ../src/eval.py output_dir_suffix=_5000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024907_zqq6f7/checkpoints/last.ckpt &
# python3 ../src/eval.py output_dir_suffix=_10000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240215-211116_uffl43/checkpoints/epoch9999-val_loss1.610e-07.ckpt &

wait