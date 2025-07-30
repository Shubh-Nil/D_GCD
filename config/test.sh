#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TEST_PATH="$REPO_ROOT/scripts/test.py"

# Define dataset name (customizable by the user)
DATASET="PACS"  # Options: PACS, Office_Home, Domain_Net

if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dgcd || { echo "Conda env 'dgcd' not found. Please create it first."; exit 1; }
else
    echo "Conda not found. Please install Anaconda/Miniconda or adjust the script."
    exit 1
fi

# Optional: set threading environment vars
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Loop through checkpoints from 0 to 9
for checkpoint in {0..9}; do
    echo "Processing checkpoint $checkpoint..."

    python3 "$TEST_PATH" \
        --checkpoint "$checkpoint" \
        --dataset_name "$DATASET"

    echo "Testing completed for checkpoint $checkpoint."
done

echo "All checkpoints processed."
