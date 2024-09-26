#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
# export DATA_DIR=$PROJECT_ROOT/data1/nco/epsilon=1e-2/version22/max_Dt=1e1
export DATA_DIR=$PROJECT_ROOT/data1/nco/epsilon=1e-2/version22/max_Dt=1e1_min_Dt=-1e1

# python3 ../src/train.py debug=runtime &

python3 ../src/train.py experiment=nco_var_dt trainer.devices=[1] &
# python3 ../src/train.py experiment=nco_var_dt trainer.devices=[0] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240923-150159_osiflw/checkpoints/latest-epoch00999.ckpt &

# python3 ../src/train.py experiment=nco trainer.devices=[0] &
# python3 ../src/train.py experiment=nco trainer.devices=[1] seed=300 &
# python3 ../src/train.py experiment=nco trainer.devices=[0] seed=100 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240708-161013_t2ig58/checkpoints/last.ckpt &
# python3 ../src/train.py experiment=nco trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240925-173053_avxvh0/checkpoints/latest-epoch00520.ckpt &

# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_bs1000_misfit_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240922-022957_rq6dtd/checkpoints/epoch00995-val_loss3.117e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T4_misfit_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240921-122231_9rrked/checkpoints/epoch00645-val_loss1.514e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T4_misfit_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240921-121701_vbirx4/checkpoints/epoch00256-val_loss7.255e-06.ckpt & 
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T6_misfit_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240921-122630_tpi1t6/checkpoints/epoch00170-val_loss1.451e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T10_480k_misfit ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240921-122805_piuowu/checkpoints/epoch00130-val_loss5.407e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=false output_dir_suffix=_T10_960k_misfit ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240922-173137_jvjc57/checkpoints/epoch00101-val_loss4.538e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T10_1920k_misfit ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240922-202025_gsst2x/checkpoints/epoch00051-val_loss4.195e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=false output_dir_suffix=_T10_960k_bs1000_misfit_testout ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240923-023313_u4uijc/checkpoints/epoch00654-val_loss1.705e-05.ckpt
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=false output_dir_suffix=_T10_misfit_testout  ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240912-164910_a8t2f3/checkpoints/epoch00997-val_loss1.046e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=false output_dir_suffix=_T10_480k_vv1e-1_temp ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240922-025044_ar5s7d/checkpoints/epoch00126-val_loss6.379e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T10_480k_vv5e-2 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240922-025357_5hrfah/checkpoints/epoch00133-val_loss5.225e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_T10_480k_net4m_misfit ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240922-022602_z5kus3/checkpoints/epoch00204-val_loss5.044e-05.ckpt
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_misfit_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240918-233403_89jc2q/checkpoints/epoch00999-val_loss6.618e-05.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_addvv_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240918-233721_3pjpn3/checkpoints/epoch00994-val_loss9.997e-07.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_addvv2_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240919-174046_np78fa/checkpoints/epoch00966-val_loss1.377e-06.ckpt &
# python3 ../src/eval.py experiment_eval=nco_var_dt save_test_predictions=true output_dir_suffix=_addvv3_testall ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240919-174603_ulpgwl/checkpoints/epoch00990-val_loss1.122e-05.ckpt &

# python3 ../src/eval.py module._target_=modules.solutionmap.SolutionMap save_test_predictions=false predict_nsteps=1000 output_dir_suffix=_gamma10 datamodule=default datamodule.train.sequence_len=2 datamodule.test.sequence_len=2 datamodule.batch_size=1000 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240912-164910_a8t2f3/checkpoints/epoch00997-val_loss1.046e-05.ckpt &
# python3 ../src/eval.py save_test_predictions=false predict_nsteps=1000 output_dir_suffix=_combined10-1000 datamodule=temp datamodule.supervised.train.downsample_factor=0.01 datamodule.supervised.batch_size=10 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240826-212752_kazi2z/checkpoints/epoch01996-val_loss6.801e-04.ckpt & 
# python3 ../src/eval.py predict_nsteps=40000 output_dir_suffix=_data_v20_wmse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version20/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240711-004452_ahaiv3/checkpoints/latest-epoch00314.ckpt &
# python3 ../src/eval.py predict_nsteps=40000 output_dir_suffix=_data_v22_nmse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version22/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240717-210241_g1k36u/checkpoints/latest-epoch00736.ckpt & 
# python3 ../src/eval.py module._target_=models.BoostedSolutionMap ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240426-193755_y5vbsq/checkpoints/epoch999-val_loss3.288e-06.ckpt &

wait