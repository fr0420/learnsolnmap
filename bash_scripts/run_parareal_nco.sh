#!/bin/bash

PROJECT_ROOT=/workspace/projects_rui/learnsolnmap
PROBLEM=nco
OUTPUT_DIR=$PROJECT_ROOT/out/$PROBLEM/eps=1e-2/const_energy_1/202504080025

# julia src/parareal/run_sequential.jl configs/$PROBLEM/sequential.toml --output_dir $OUTPUT_DIR/ref
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_sequential.toml --output_dir $OUTPUT_DIR/ref
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_plain.toml --output_dir $OUTPUT_DIR/plain_Dt_40
julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_plain_nn.toml --output_dir $OUTPUT_DIR/plain_phi_Dt_40
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_procrustes.toml --output_dir $OUTPUT_DIR/procrustes_Dt_40_emb2
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_procrustes_nn.toml --output_dir $OUTPUT_DIR/procrustes_phi_Dt_20
# julia src/parareal/run_parareal.jl configs/$PROBLEM/parareal_interpolative.toml --output_dir $OUTPUT_DIR/interpolative

