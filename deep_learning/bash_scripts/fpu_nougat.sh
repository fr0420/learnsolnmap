#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
# export DATA_DIR=$PROJECT_ROOT/data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5
# export DATA_DIR=$PROJECT_ROOT/data1/fpu/omega=300/version4/Dt=1e0
export DATA_DIR=$PROJECT_ROOT/data1/fpu/omega=50/version2/Dt=1e0

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=fpu trainer.devices=[0] module/scheduler=none & 
# python3 ../src/train.py experiment=fpu trainer.devices=[1] module/scheduler=1cycle &

# python3 ../src/train.py experiment=fpu_fixedstep trainer.devices=[0] &

# python3 ../src/train.py experiment=fpu_stacked trainer.devices=[1] &
# python3 ../src/train.py experiment=fpu_stacked_tl trainer.devices=[0] &

# python3 ../src/train.py experiment=fpu_idenf_taylor trainer.devices=[0] &

# python3 ../src/train.py experiment=fpu_idenf_taylor_sf trainer.devices=[0] &
python3 ../src/train.py experiment=fpu_idenf_taylor_sf trainer.devices=[1] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250330-174512_nv964y/checkpoints/latest-epoch00249.ckpt &

# python3 ../src/train.py experiment=fpu_idenf trainer.devices=[1] &
# python3 ../src/train.py experiment=fpu_identityenforced trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241023-193612_chqucx/checkpoints/latest-epoch00312.ckpt &

# python3 ../src/train.py experiment=fpu_t0centered trainer.devices=[1] &
# python3 ../src/train.py experiment=fpu_t0centered trainer.devices=[1] seed=100 &
# python3 ../src/train.py experiment=fpu_t0centered trainer.devices=[0] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241130-131917_etjdzz/checkpoints/epoch06249-val_loss1.911e-07.ckpt &
# python3 ../src/train.py experiment=fpu_t0centered trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250202-165242_clxw3u/checkpoints/latest-epoch00122.ckpt &

# python3 ../src/train.py experiment=fpu trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240819-191819_m0b3jt/checkpoints/latest-epoch00085.ckpt &
# python3 ../src/train.py experiment=fpu trainer.devices=[1] trainer.max_epochs=5000 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024907_zqq6f7/checkpoints/last.ckpt &

# python3 ../src/eval.py experiment_eval=fpu ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241020-135554_xnywg1/checkpoints/latest-epoch00535.ckpt &

# python3 ../src/eval.py experiment_eval=fpu ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241017-220331_5lhjrs/checkpoints/latest-epoch00624.ckpt &
# python3 ../src/eval.py experiment_eval=fpu ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241024-021610_xyfvb2/checkpoints/latest-epoch00484.ckpt &
# python3 ../src/eval.py experiment_eval=fpu ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241017-224441_fk3q0x/checkpoints/latest-epoch00624.ckpt &
# python3 ../src/eval.py output_dir_suffix=_energybalanced_misfit ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240820-001828_pu02hz/checkpoints/epoch00499-val_loss1.195e-07.ckpt & 
# python3 ../src/eval.py output_dir_suffix=_energybalanced_integrator ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240820-002437_idu8xj/checkpoints/epoch00499-val_loss1.246e-07.ckpt & 
# python3 ../src/eval.py module._target_=models.BoostedSolutionMap ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240426-193755_y5vbsq/checkpoints/epoch999-val_loss3.288e-06.ckpt &
# python3 ../src/eval.py output_dir_suffix=_2000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024649_76cpvy/checkpoints/epoch1999-val_loss8.837e-07.ckpt &
# python3 ../src/eval.py output_dir_suffix=_5000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240213-024907_zqq6f7/checkpoints/last.ckpt &
# python3 ../src/eval.py output_dir_suffix=_10000_epochs ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240215-211116_uffl43/checkpoints/epoch9999-val_loss1.610e-07.ckpt &

wait