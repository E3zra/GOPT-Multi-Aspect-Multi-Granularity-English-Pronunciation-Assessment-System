#!/bin/bash
################################################################################
# Model Setup Script for GOPT
#
# This script automatically sets up LibriSpeech ASR models by:
# 1. Checking if models exist in /opt/kaldi/egs/librispeech/s5/exp/
# 2. If not, checking if models are mounted at /workspace/models/librispeech/
# 3. If not, downloading them from OpenSLR
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo ""
echo "========================================================================"
echo "LibriSpeech ASR Model Setup"
echo "========================================================================"
echo ""

# Target directories
KALDI_EXP_DIR="/opt/kaldi/egs/librispeech/s5/exp"
MOUNTED_MODEL_DIR="/workspace/models/librispeech/exp"
MODEL_SUBDIR="nnet3_cleaned/tdnn_sp"

# Step 1: Check if models already exist in Kaldi directory
print_info "Step 1: Checking for models in Kaldi directory..."
if [ -d "$KALDI_EXP_DIR/$MODEL_SUBDIR" ]; then
    print_success "Models already exist at $KALDI_EXP_DIR/$MODEL_SUBDIR"
    print_info "No setup needed!"
    exit 0
fi

# Step 2: Check if models are mounted from host
print_info "Step 2: Checking for mounted models..."
if [ -d "$MOUNTED_MODEL_DIR/$MODEL_SUBDIR" ]; then
    print_success "Found models in mounted volume!"
    print_info "Creating directory structure..."
    
    mkdir -p "$KALDI_EXP_DIR"
    
    print_info "Creating symbolic link..."
    ln -sf "$MOUNTED_MODEL_DIR" "$KALDI_EXP_DIR"
    
    if [ -d "$KALDI_EXP_DIR/$MODEL_SUBDIR" ]; then
        print_success "Models successfully linked!"
        exit 0
    else
        print_error "Symbolic link created but model not accessible"
    fi
fi

# Step 3: Check if we're in the librispeech/s5 directory
print_info "Step 3: Setting up directory structure..."
cd /opt/kaldi/egs/librispeech/s5 2>/dev/null || {
    print_error "Kaldi librispeech recipe directory not found!"
    print_info "This script must be run inside a properly configured Kaldi container"
    exit 1
}

# Step 4: Download models if not found
print_warning "Models not found. Downloading from OpenSLR..."
print_info "This will download approximately 2GB of data"

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -c"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -O -C -"
else
    print_error "Neither wget nor curl is available!"
    exit 1
fi

# Download chain model
MODEL_URL="http://www.openslr.org/resources/11/0013_librispeech_v1_chain.tar.gz"
MODEL_FILE="0013_librispeech_v1_chain.tar.gz"

print_info "Downloading $MODEL_FILE..."
$DOWNLOAD_CMD "$MODEL_URL" || {
    print_error "Download failed!"
    print_info "You can manually download from: $MODEL_URL"
    print_info "Then extract to: $KALDI_EXP_DIR"
    exit 1
}

# Extract models
print_info "Extracting models..."
tar -xzf "$MODEL_FILE" || {
    print_error "Extraction failed!"
    exit 1
}

# Clean up
rm -f "$MODEL_FILE"

# Verify installation
if [ -d "$KALDI_EXP_DIR/$MODEL_SUBDIR" ]; then
    print_success "Models successfully installed!"
    
    # Optionally copy to mounted volume for persistence
    if [ -d "/workspace/models/librispeech" ]; then
        print_info "Copying models to mounted volume for persistence..."
        mkdir -p "/workspace/models/librispeech"
        cp -r "$KALDI_EXP_DIR" "/workspace/models/librispeech/" 2>/dev/null || {
            print_warning "Could not copy to mounted volume (may not have write permission)"
        }
    fi
    
    echo ""
    echo "========================================================================"
    print_success "Model setup complete!"
    echo "========================================================================"
    echo ""
else
    print_error "Verification failed - models not found after installation"
    exit 1
fi









