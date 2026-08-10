#!/bin/bash
set -e

# ==============================================================================
# Script: generate_data.sh
# Purpose: End-to-end synthetic dataset generation (inputs, targets, split)
#
# Positional Arguments:
#   $1 - Problem name (e.g. fpu, nco, 3body, alphaparticle; default: "fpu")
#   $2 - [Custom problem] Version subpath (default: "version1")
#   $3 - [Custom problem] Target Dt directory name (default: "Dt=1e0")
#
# Dry-Run Mode:
#   Set env variable DRY_RUN=1 or include 'dryrun' anywhere in the arguments.
#
# Examples:
#   ./data_generation/bash_scripts/generate_data.sh fpu
#   ./data_generation/bash_scripts/generate_data.sh nco dryrun
#   DRY_RUN=1 ./data_generation/bash_scripts/generate_data.sh custom_prob v1 Dt=2e0
# ==============================================================================

# Parse 'dryrun' flag from any argument without polluting positional $1, $2, $3
EXEC=""
NEW_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "dryrun" ] || [ "$arg" = "--dry-run" ]; then
    DRY_RUN=1
  else
    NEW_ARGS+=("$arg")
  fi
done
set -- "${NEW_ARGS[@]}"

if [ "$DRY_RUN" = "1" ]; then
    EXEC="echo [DRY-RUN]"
    echo "========================================================================"
    echo " DRY-RUN MODE ENABLED: Commands will be displayed without execution."
    echo "========================================================================"
fi

# Repository root (auto-detected)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Problem selection
PROBLEM=${1:-"fpu"}

# Problem-specific default presets
case "$PROBLEM" in
  fpu)
    VERSION_PATH="omega=100/version1"
    DT_DIR="Dt=1e0"
    ;;
  nco)
    VERSION_PATH="epsilon=1e-2/202510071037_rhmc-H0"
    DT_DIR="Dt=5e0"
    ;;
  3body|3body-2d)
    VERSION_PATH="version1"
    DT_DIR="Dt=10e0"
    ;;
  alphaparticle)
    VERSION_PATH="version1"
    DT_DIR="Dt=0.4"
    ;;
  *)
    VERSION_PATH="${2:-version1}"
    DT_DIR="${3:-Dt=1e0}"
    ;;
esac

# Derived directory paths
DATA_DIR="$PROJECT_ROOT/data/$PROBLEM/$VERSION_PATH"
DIR_INPUTS="$DATA_DIR/inputs"
DIR_TARGETS="$DATA_DIR/$DT_DIR"
CONFIG_DIR="$PROJECT_ROOT/data_generation/configs/$PROBLEM"

echo "========================================================================"
echo " Running Data Generation Pipeline"
echo " Problem:     $PROBLEM"
echo " Inputs Dir:  $DIR_INPUTS"
echo " Targets Dir: $DIR_TARGETS"
echo " Config Dir:  $CONFIG_DIR"
echo "========================================================================"

# 1. Generate inputs
$EXEC mkdir -p "$DIR_INPUTS"
$EXEC julia "$PROJECT_ROOT/data_generation/src/data_generation/generate_inputs.jl" \
      "$CONFIG_DIR/inputs.toml" --output_dir "$DIR_INPUTS"

# 2. Generate targets
$EXEC mkdir -p "$DIR_TARGETS"
if [ -z "$EXEC" ]; then
    cp "$DIR_INPUTS/U0.csv" "$DIR_TARGETS/" 2>/dev/null || true
else
    echo "[DRY-RUN] cp $DIR_INPUTS/U0.csv $DIR_TARGETS/"
fi
$EXEC julia "$PROJECT_ROOT/data_generation/src/data_generation/generate_targets.jl" \
      "$CONFIG_DIR/targets.toml" --output_dir "$DIR_TARGETS"

# 3. Split into train/test sets
$EXEC python3 "$PROJECT_ROOT/deep_learning/src/split_data.py" \
        --data_dir "$DIR_TARGETS" --train_fraction 0.8

echo "Done! Target location: $DIR_TARGETS"