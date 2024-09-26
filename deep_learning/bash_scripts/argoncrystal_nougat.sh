#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=/workspace/projects_rui/learnsolnmap/data/lj/rhmc-H0/dt2e-4_h5e-8/Nchains100_Njumps2000/sigma1e-2/Dt1e-3

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] module.network.h2h.hidden_dim=1000 &
# python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] module.network.h2h.n_hidden_layers=8 &

# python3 ../src/train.py experiment=argoncrystal trainer.devices=[0] "callbacks.fixed_seq_weights.weights=[0., 1., 1., 0., 0., 0.]" &
# python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] "callbacks.fixed_seq_weights.weights=[0., 1., 1., 1., 0., 0.]" &

# python3 ../src/train.py experiment=argoncrystal trainer.devices=[1]  &
# python3 ../src/train.py experiment=argoncrystal trainer.devices=[1] trainer.max_epochs=400 module.optimizer.lr=2.5e-5 init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240205-223702_dvzfsx/checkpoints/epoch948-val_loss3.126e-05.ckpt &

python3 ../src/eval.py output_dir_suffix=_15000epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240415-143825_v1cqnf/checkpoints/epoch14873-val_loss9.635e-04.ckpt &
# python3 ../src/eval.py output_dir_suffix=_3steps ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240324-195630_if1h3b/checkpoints/epoch4945-val_loss3.114e-04.ckpt &
# python3 ../src/eval.py output_dir_suffix=_4steps ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240308-124009_1glq6b/checkpoints/epoch4961-val_loss2.219e-03.ckpt &
# python3 ../src/eval.py output_dir_suffix=_5steps ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240306-115844_b5lchg/checkpoints/epoch912-val_loss9.051e-03.ckpt
wait