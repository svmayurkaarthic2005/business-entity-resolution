#!/bin/bash
# Exit on error
set -e

echo "Starting Kaggle training pipeline..."

# Run the full pipeline or train script
# Modify this line based on what exactly you want to run.
# For example, to run the training:
export PYTHONIOENCODING=utf-8
cd /kaggle/working/business-entity-resolution || cd .

# Assuming data is in /kaggle/input and cache/output is in /kaggle/working
echo "Running training script..."
python src/train.py

# Automatically save the progress and update the dataset
echo "Saving progress..."

# Create a dataset directory if it doesn't exist
mkdir -p /kaggle/working/ber-model-checkpoint

# Copy models, cache, output, and logs to the checkpoint folder
cp -r cache/models /kaggle/working/ber-model-checkpoint/ 2>/dev/null || true
cp cache/calib.pkl /kaggle/working/ber-model-checkpoint/ 2>/dev/null || true
cp cache/pred_TRAIN.parquet /kaggle/working/ber-model-checkpoint/ 2>/dev/null || true
cp -r output* /kaggle/working/ber-model-checkpoint/ 2>/dev/null || true

# Update Kaggle dataset using kaggle CLI
# Make sure you have initialized the dataset first via `kaggle datasets init -p /kaggle/working/ber-model-checkpoint`
# and populated dataset-metadata.json

if [ -f "/kaggle/working/ber-model-checkpoint/dataset-metadata.json" ]; then
    echo "Updating Kaggle dataset..."
    kaggle datasets version -p /kaggle/working/ber-model-checkpoint -m "Auto-update from training run $(date)" --dir-mode zip
else
    echo "Warning: dataset-metadata.json not found in /kaggle/working/ber-model-checkpoint."
    echo "Please run 'kaggle datasets init' in that folder and configure it first if you want auto-updates to a dataset."
    # If running in a Kaggle notebook, all files in /kaggle/working are auto-saved when the notebook finishes!
fi

echo "Training and saving completed!"
