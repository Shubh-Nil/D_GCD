#!/bin/bash

# Script to run the episodic-training process for Domain Generalization
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TRAIN_PATH="$REPO_ROOT/scripts/train.py"

# Activate the Python environment
# Modify this section according to how the environment is set up
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dgcd || { echo "Conda env 'dgcd' not found. Please create it first."; exit 1; }
else
    echo "Conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# List of domains in the PACS dataset
dataset="PACS"
declare -a domains=("art_painting" "cartoon" "photo" "sketch")

# List of domains in the Office Home dataset
# dataset="Office_Home"
# declare -a domains=("Art" "Clipart" "Real_world" "Product")

# List of domains in the Domain Net dataset
# dataset="Domain_Net"
# declare -a domains=("sketch" "painting" "clipart")

# Create logs directory if not exists
mkdir -p "$REPO_ROOT/logs/$dataset"

# Loop through each domain and execute the training script
for domain in "${domains[@]}"
do
    echo "Starting training for the domain: $domain"
    python "$TRAIN_PATH" \
        --global_epochs 10 \
        --episodes 6 \
        --task_epochs 8 \
        --task_lr 0.01 \
        --batch_size 128 \
        --alpha 0.7 \
        --n_views 2 \
        --image_size 224 \
        --dataset_name "$dataset" \
        --source_domain_name "$domain" \
        --transform "imagenet" \
        --device_id 0 \
        > "$REPO_ROOT/logs/${dataset}/DGCD_${domain}_log.out" 2>&1 
    echo "Training launched for domain: $domain"
done

wait  

echo "All domain trainings are completed."
