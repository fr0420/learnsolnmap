#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data1/nco/epsilon=1e-2/version23/Dt=5e0

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=nco trainer.devices=[1] &
# python3 ../src/train.py experiment=nco trainer.devices=[1] seed=300 &
# python3 ../src/train.py experiment=nco trainer.devices=[0] seed=100 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240708-161013_t2ig58/checkpoints/last.ckpt &
# python3 ../src/train.py experiment=nco trainer.devices=[0] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240712-004726_zusjqp/checkpoints/latest-epoch00049.ckpt &

# python3 ../src/eval.py save_test_predictions=true predict=false predict_nsteps=40000 output_dir_suffix=_data_v22_mse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version22/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240711-115119_h82d8k/checkpoints/latest-epoch00129.ckpt &
# python3 ../src/eval.py predict_nsteps=40000 output_dir_suffix=_data_v20_downsampled 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version20/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240712-124026_sprphh/checkpoints/epoch00923-val_loss2.183e-06.ckpt &

python3 ../src/eval.py save_test_predictions=true predict_nsteps=1000 output_dir_suffix=_data_v22_nmse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version22/Dt\=5e0' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240717-160541_6zk1bq/checkpoints/latest-epoch02186.ckpt & 
# python3 ../src/eval.py save_test_predictions=true predict_nsteps=1000 output_dir_suffix=_data_v20_downsampled_latest 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version20/Dt\=5e0' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240713-173330_m8d5dk/checkpoints/latest-epoch04329.ckpt & 
# python3 ../src/eval.py save_test_predictions=true predict_nsteps=1000 output_dir_suffix=_data_v22_downsampled_latest 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version22/Dt\=5e0' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240713-173353_7ogtwm/checkpoints/latest-epoch01574.ckpt & 
# python3 ../src/eval.py predict_nsteps=40000 output_dir_suffix=_data_v20_wmse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version20/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240711-004452_ahaiv3/checkpoints/latest-epoch00314.ckpt &
# python3 ../src/eval.py predict_nsteps=1000 output_dir_suffix=_data_v22_wmse_modmlp 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version22/Dt\=5e0' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240711-160856_e4zy9f/checkpoints/latest-epoch02073.ckpt & 
# python3 ../src/eval.py module._target_=models.BoostedSolutionMap ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240426-193755_y5vbsq/checkpoints/epoch999-val_loss3.288e-06.ckpt &

wait