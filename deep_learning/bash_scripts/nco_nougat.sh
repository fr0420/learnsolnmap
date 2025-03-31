#!/bin/bash

# set wandb env variable to avoid the broken pipe error (see https://github.com/wandb/wandb/pull/3031)
export WANDB_START_METHOD="thread"


export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
# export DATA_DIR=$PROJECT_ROOT/data1/nco/epsilon=1e-2/version22/Dt=5e1
export DATA_DIR=$PROJECT_ROOT/data1/nco/epsilon=1e-2/version22/max_Dt=1e1_min_Dt=-1e1

# python3 ../src/train.py debug=runtime &

# python3 ../src/train.py experiment=nco_stacked trainer.devices=[1] &
# python3 ../src/train.py experiment=nco_stacked trainer.devices=[1] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241010-234145_6mh9io/checkpoints/latest-epoch01279.ckpt & 

# python3 ../src/train.py experiment=nco_t0centered trainer.devices=[0] &
# python3 ../src/train.py experiment=nco_t0centered_tl trainer.devices=[0] &

# python3 ../src/train.py experiment=nco_t0centered_taylor trainer.devices=[0] &
# python3 ../src/train.py experiment=nco_t0centered_taylor trainer.devices=[1] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250325-030342_fi4vle/checkpoints/epoch00162-val_loss3.479e-05.ckpt &

# python3 ../src/train.py experiment=nco_idenf trainer.devices=[0] &
# python3 ../src/train.py experiment=nco_idenf trainer.devices=[1] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241123-202635_6h3a99/checkpoints/latest-epoch00208.ckpt &
# python3 ../src/train.py experiment=nco_var_dt trainer.devices=[0] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learnsing/logs/train/runs/20240930-173426_oihr49/checkpoints/latest-epoch00019.ckpt &

# python3 ../src/train.py experiment=nco_idenf_taylor trainer.devices=[1] &
# python3 ../src/train.py experiment=nco_idenf_taylor trainer.devices=[0] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250323-234359_zk28ds/checkpoints/latest-epoch00417.ckpt &

# python3 ../src/train.py experiment=nco_idenf_taylor_sf trainer.devices=[0] &
python3 ../src/train.py experiment=nco_idenf_taylor_sf trainer.devices=[0] init_model_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250329-142604_mngtlq/checkpoints/latest-epoch00417.ckpt &


# python3 ../src/train.py experiment=nco_fixedstep trainer.devices=[0] &
# python3 ../src/train.py experiment=nco trainer.devices=[1] seed=300 &
# python3 ../src/train.py experiment=nco trainer.devices=[0] seed=100 resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240708-161013_t2ig58/checkpoints/last.ckpt &
# python3 ../src/train.py experiment=nco trainer.devices=[1] resume_from_ckpt=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240925-173053_avxvh0/checkpoints/latest-epoch00520.ckpt &

# python3 ../src/eval.py experiment_eval=nco save_test_predictions=false output_dir_suffix=_stacked ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241011-084415_7spxkv/checkpoints/epoch03991-val_loss1.838e-08.ckpt &
# python3 ../src/eval.py experiment_eval=nco save_test_predictions=false output_dir_suffix=_nonstacked ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241007-160548_olk1f5/checkpoints/latest-epoch06349.ckpt &

# python3 ../src/eval.py experiment_eval=nco save_test_predictions=false output_dir_suffix=_fixedstep ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241007-001223_q5844w/checkpoints/latest-epoch03657.ckpt &
# python3 ../src/eval.py experiment_eval=nco_t0centered save_test_predictions=false output_dir_suffix=_data_vv ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241007-033302_8jy3l5/checkpoints/latest-epoch02347.ckpt & 
# python3 ../src/eval.py experiment_eval=nco_t0centered save_test_predictions=false output_dir_suffix=_data_ruth3 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241010-141434_4kdnci/checkpoints/epoch01832-val_loss1.147e-04.ckpt & 
# python3 ../src/eval.py experiment_eval=nco_t0centered save_test_predictions=false output_dir_suffix=_data_em ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241010-020823_4xfdqs/checkpoints/epoch01947-val_loss1.203e-04.ckpt & 
# python3 ../src/eval.py experiment_eval=nco_t0centered save_test_predictions=false output_dir_suffix=_data_im ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241010-021306_ht1ngs/checkpoints/epoch02527-val_loss1.187e-04.ckpt & 
# python3 ../src/eval.py experiment_eval=nco_t0centered save_test_predictions=false output_dir_suffix=_data_rk4 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20241010-021855_gxf0zf/checkpoints/epoch02329-val_loss1.210e-04.ckpt & 

# python3 ../src/eval.py module._target_=modules.solutionmap.SolutionMap save_test_predictions=false predict_nsteps=1000 output_dir_suffix=_gamma10 datamodule=default datamodule.train.sequence_len=2 datamodule.test.sequence_len=2 datamodule.batch_size=1000 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240912-164910_a8t2f3/checkpoints/epoch00997-val_loss1.046e-05.ckpt &
# python3 ../src/eval.py save_test_predictions=false predict_nsteps=1000 output_dir_suffix=_combined10-1000 datamodule=temp datamodule.supervised.train.downsample_factor=0.01 datamodule.supervised.batch_size=10 ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240826-212752_kazi2z/checkpoints/epoch01996-val_loss6.801e-04.ckpt & 
# python3 ../src/eval.py predict_nsteps=40000 output_dir_suffix=_data_v20_wmse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version20/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240711-004452_ahaiv3/checkpoints/latest-epoch00314.ckpt &
# python3 ../src/eval.py predict_nsteps=40000 output_dir_suffix=_data_v22_nmse 'datamodule.train_dir=/workspace/projects_rui/learnsolnmap/data1/nco/epsilon\=1e-2/version22/Dt\=5e-2' ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240717-210241_g1k36u/checkpoints/latest-epoch00736.ckpt & 
# python3 ../src/eval.py module._target_=models.BoostedSolutionMap ckpt_path=/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20240426-193755_y5vbsq/checkpoints/epoch999-val_loss3.288e-06.ckpt &

wait