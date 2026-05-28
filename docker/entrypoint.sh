#!/bin/bash
################################################################################
# GOPT Container Entrypoint Script
#
# This script runs automatically when the container starts.
# It sets up symbolic links for LibriSpeech models to ensure
# Kaldi can find them in the expected locations.
################################################################################

set -e

echo "=============================================================="
echo "GOPT Container Initialization"
echo "=============================================================="

# Setup Kaldi LibriSpeech model links
if [ -d "/workspace/models/librispeech" ]; then
    echo "[1/5] Setting up LibriSpeech model links..."

    cd /opt/kaldi/egs/librispeech/s5

    # Link exp and data directories
    ln -sf /workspace/models/librispeech/exp exp 2>/dev/null || true
    ln -sf /workspace/models/librispeech/data data 2>/dev/null || true

    # Create nnet3_cleaned directory and link tdnn_sp
    mkdir -p exp/nnet3_cleaned 2>/dev/null || true
    cd exp/nnet3_cleaned
    ln -sf ../chain_cleaned/tdnn_1d_sp tdnn_sp 2>/dev/null || true

    # Create lang link
    cd /opt/kaldi/egs/librispeech/s5/data
    mkdir -p . 2>/dev/null || true
    ln -sf lang_test_tgsmall lang 2>/dev/null || true

    echo "   ✓ Model links created successfully"

    # Verify links
    if [ -L "/opt/kaldi/egs/librispeech/s5/exp/nnet3_cleaned/tdnn_sp" ]; then
        echo "   ✓ tdnn_sp link verified"
    else
        echo "   ⚠ Warning: tdnn_sp link not created"
    fi

    if [ -L "/opt/kaldi/egs/librispeech/s5/data/lang" ]; then
        echo "   ✓ lang link verified"
    else
        echo "   ⚠ Warning: lang link not created"
    fi
else
    echo "[1/5] ⚠ LibriSpeech models not found at /workspace/models/librispeech/"
    echo "   Please run: docker exec gopt-pipeline bash /workspace/gopt/docker/setup_models.sh"
    echo "   Or manually download models to ./models/librispeech/"
fi

echo "[2/5] Checking GOPT pretrained model..."
if [ -f "/workspace/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth" ]; then
    echo "   ✓ GOPT pretrained model found"
else
    echo "   ⚠ Warning: GOPT pretrained model not found"
    echo "   Expected: /workspace/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth"
fi

echo "[3/5] Checking Python dependencies..."
python3 -c "import torch; import numpy; import kaldi_io" 2>/dev/null && \
    echo "   ✓ Core Python dependencies available" || \
    echo "   ⚠ Warning: Some Python dependencies may be missing"

echo "[4/5] Creating output directories..."
mkdir -p /workspace/audio_input 2>/dev/null || true
mkdir -p /workspace/audio_output 2>/dev/null || true
echo "   ✓ Output directories ready"

echo "[5/5] Deploying pipeline scripts..."
# Ensure models.py is available in all import locations
if [ -f "/workspace/gopt/models.py" ]; then
    cp /workspace/gopt/models.py /workspace/gopt/src/models.py 2>/dev/null || true
    cp /workspace/gopt/models.py /workspace/gopt/pretrained_models/models.py 2>/dev/null || true
    echo "   ✓ models.py deployed"
fi

# Ensure the automated pipeline script is in place
if [ -f "/workspace/scripts/run_pipeline_auto.sh" ]; then
    chmod +x /workspace/scripts/run_pipeline_auto.sh
    # Update process_custom_audio.sh to use the automated pipeline
    echo '#!/bin/bash' > /workspace/gopt/process_custom_audio.sh
    echo 'bash /workspace/scripts/run_pipeline_auto.sh "$@"' >> /workspace/gopt/process_custom_audio.sh
    chmod +x /workspace/gopt/process_custom_audio.sh
    echo "   ✓ Automated pipeline script deployed"
else
    echo "   ⚠ Warning: run_pipeline_auto.sh not found, using fallback"
fi

echo "=============================================================="
echo "Initialization complete! Container ready for use."
echo "=============================================================="
echo ""

# Execute the main command (keep container running or run specified command)
exec "$@"