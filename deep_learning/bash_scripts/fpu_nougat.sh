#!/bin/bash

export PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
export DATA_DIR=$PROJECT_ROOT/data/fpu/omega300/rhmc-H0/css4_dt4e-1_h5e-6_Nchains100_Njumps2000_sigma1e-1/ma5_Dt1e0_h1.5259e-5

python3 ../src/train.py experiment=fpu trainer.devices=[0] module/scheduler=reduceonplateau &
python3 ../src/train.py experiment=fpu trainer.devices=[1] module.network.h2h.n_linears_per_block=2 module/scheduler=reduceonplateau &

# python3 ../src/train.py trainer.devices=[0] module/network=unet &
# python3 ../src/train.py trainer.devices=[1] module/network=unet module.network.h2h.use_bn=False &

wait