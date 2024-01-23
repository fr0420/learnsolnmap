#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5

python3 ../src/train.py experiment=fpu trainer.devices=[0] "callbacks.fixed_seq_weights.weights=[1., 0., 0., 0., 0.]" &
python3 ../src/train.py experiment=fpu trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 0., 0., 0.]" &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] module/scheduler=reduceonplateau module/optimizer=sgd module/loss=mse &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module/scheduler=reduceonplateau module/optimizer=sgd module/loss=mse module.network.h2h.n_linears_per_block=2 &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] "module.network.h2h.layer_sizes=[12, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 12]" &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] &

# python3 ../src/eval.py ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240116-115144_klq2ep/checkpoints/epoch999-val_loss3.504e-05.ckpt & 
# python3 ../src/eval.py ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240116-115145_k0nuaj/checkpoints/epoch999-val_loss1.952e-04.ckpt &
# python3 ../src/eval.py ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240114-200848_jy4rpb/checkpoints/epoch999-val_loss3.750e-05.ckpt &

wait