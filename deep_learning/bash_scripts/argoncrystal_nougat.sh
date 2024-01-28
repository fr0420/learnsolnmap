#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=/workspace/projects_rui/learnsolnmap/data/lj/rhmc-H0/dt2e-4_h5e-8/Nchains100_Njumps2000/sigma1e-2/Dt1e-4

# python3 ../src/train.py debug=runtime &

python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] "callbacks.fixed_seq_weights.weights=[1., 1., 0., 0., 0.]" &
# python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 0., 0., 0.]" resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240125-192836_xjcilo/checkpoints/last.ckpt &
# python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] "callbacks.fixed_seq_weights.weights=[1., 1., 1., 0., 0.]" resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240125-215740_c2pivv/checkpoints/last.ckpt &
python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[1., 1., 1., 0., 0.]" &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] module/scheduler=reduceonplateau module/optimizer=sgd module/loss=mse &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module/scheduler=reduceonplateau module/optimizer=sgd module/loss=mse module.network.h2h.n_linears_per_block=2 &

# python3 ../src/eval.py ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240125-161044_jvgwwx/checkpoints/epoch105-val_loss1.775e+03.ckpt & 

wait