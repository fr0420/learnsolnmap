#!/bin/bash

PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
PROBLEM=argoncrystal
OUTPUT_DIR=$PROJECT_ROOT/out/$PROBLEM/202403181802

# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_sequential.toml --output_dir $OUTPUT_DIR/ref
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_plain.toml --output_dir $OUTPUT_DIR/plain
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_procrustes.toml --output_dir $OUTPUT_DIR/procrustes
julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_interpolative.toml --output_dir $OUTPUT_DIR/interpolative2