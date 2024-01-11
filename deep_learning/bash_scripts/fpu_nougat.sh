#!/bin/bash

PROJECT_DIR=/workspace/projects_rui/learnsolnmap
FPU_DATA_DIR1=$PROJECT_DIR/data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5

python3 ../src/train.py paths.data_dir=$FPU_DATA_DIR1 trainer.devices=[0] &
python3 ../src/train.py paths.data_dir=$FPU_DATA_DIR1 trainer.devices=[1] module.network.h2h.n_linears_per_block=2 &

wait