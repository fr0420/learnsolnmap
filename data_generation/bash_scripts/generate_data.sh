#!/bin/bash

PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
PROBLEM=fpu
DATA_DIR=$PROJECT_ROOT/data1/$PROBLEM/omega=100/version1
DIR_INPUTS=$DATA_DIR/inputs/
DIR_TARGETS=$DATA_DIR/Dt=1e0
# DIR_TARGETS=$DATA_DIR/inputs

# generate inputs 
# julia src/data_generation/generate_inputs.jl configs/$PROBLEM/inputs.toml --output_dir $DIR_INPUTS 

# generate targets
mkdir -p $DIR_TARGETS
cp $DIR_INPUTS/U0.csv $DIR_TARGETS
julia src/data_generation/generate_targets.jl configs/$PROBLEM/targets.toml --output_dir $DIR_TARGETS 
# julia src/data_generation/generate_var_dt_targets.jl configs/$PROBLEM/targets.toml --output_dir $DIR_TARGETS

# split into train/test sets
python3 deep_learning/src/split_data.py --data_dir $DIR_TARGETS --train_fraction 0.8