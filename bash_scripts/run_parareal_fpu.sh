#!/bin/bash

PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
PROBLEM=fpu
OUTPUT_DIR=$PROJECT_ROOT/out/$PROBLEM/omega=300/1/202503252227

# julia src/parareal/run_sequential.jl configs/$PROBLEM/sequential.toml --output_dir $OUTPUT_DIR/ref
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_plain.toml --output_dir $OUTPUT_DIR/plain1
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_plain_nn.toml --output_dir $OUTPUT_DIR/plain_nn1
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_procrustes.toml --output_dir $OUTPUT_DIR/procrustes_temp
julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_procrustes_nn.toml --output_dir $OUTPUT_DIR/procrustes_nn_temp
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_interpolative.toml --output_dir $OUTPUT_DIR/interpolative
