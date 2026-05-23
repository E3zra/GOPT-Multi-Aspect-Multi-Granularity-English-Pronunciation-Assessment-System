#!/bin/bash
# GOPT Fine-tuning Script (Linux/Mac)
# Usage: ./run_finetune.sh

set -x

# Activate virtual environment (modify path if needed)
# source ../venv-gopt/bin/activate

# ============================================================
# Configuration
# ============================================================

# Learning rate (lower for fine-tuning)
lr=5e-4

# Model architecture (MUST match pretrained model)
depth=3
head=1
embed_dim=24

# Training parameters
batch_size=25
n_epochs=50
model=gopt
am=librispeech  # or paiia, paiib, custom

# Loss weights (adjust for different objectives)
# Default (1:1:1) for overall performance
# (2:1:1) for phoneme accuracy focus
# (1:1:2) for fluency/prosody focus
loss_w_phn=1.0
loss_w_word=1.0
loss_w_utt=1.0

# Data augmentation
noise=0.3  # Noise level (0-1), higher = more augmentation

# Pretrained model path
pretrained=../pretrained_models/gopt_librispeech/best_audio_model.pth

# Output directory
exp_dir=../exp/finetune-lr${lr}-${am}-noise${noise}

# ============================================================
# Advanced Options
# ============================================================

# Freezing strategies (uncomment to enable)
# freeze_blocks="--freeze_blocks"          # Freeze transformer, train only classifiers
# freeze_embeddings="--freeze_embeddings"  # Freeze position/phone embeddings
# progressive_unfreeze="--progressive_unfreeze"  # Gradually unfreeze layers

# Early stopping (0 to disable)
early_stopping=15

# Learning rate scheduler
lr_scheduler=multistep  # Options: multistep, cosine, plateau

# ============================================================
# Run Training
# ============================================================

mkdir -p $exp_dir

python ./traintest_finetune.py \
  --lr ${lr} \
  --exp-dir ${exp_dir} \
  --pretrained ${pretrained} \
  --goptdepth ${depth} \
  --goptheads ${head} \
  --batch_size ${batch_size} \
  --embed_dim ${embed_dim} \
  --n-epochs ${n_epochs} \
  --model ${model} \
  --am ${am} \
  --loss_w_phn ${loss_w_phn} \
  --loss_w_word ${loss_w_word} \
  --loss_w_utt ${loss_w_utt} \
  --noise ${noise} \
  --early_stopping ${early_stopping} \
  --lr_scheduler ${lr_scheduler} \
  ${freeze_blocks} \
  ${freeze_embeddings} \
  ${progressive_unfreeze}

echo "Fine-tuning completed!"
echo "Results saved to: $exp_dir"
echo "Best model: $exp_dir/models/best_audio_model.pth"
echo "Training log: $exp_dir/result.csv"

