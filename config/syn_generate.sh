#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"

# Local variables for key arguments
DATASET="PACS"
PROMPT="Add Snow Background"
BACKGROUND_TYPE=$(echo "$PROMPT" | awk '{print $2}')

SOURCE_DIR="$REPO_ROOT/datasets/$DATASET"
TARGET_DIR="$REPO_ROOT/datasets_synthetic/$DATASET/${BACKGROUND_TYPE}_Background_Dataset"

# Check and create target directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
fi

# Run the Python script
python3 "$REPO_ROOT/scripts/syn_generate.py" \
  --source_dir "$SOURCE_DIR" \
  --target_dir "$TARGET_DIR" \
  --prompt "$PROMPT" \
  --model_id "timbrooks/instruct-pix2pix" \
  --device "cuda:0" \
  --steps 10 \
  --guidance_scale 1.0
