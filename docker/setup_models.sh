#!/bin/bash
################################################################################
# Download and Setup Librispeech ASR Models for GOPT
#
# This script downloads the required Librispeech models from Kaldi website
#
# Usage:
#   ./setup_models.sh [target_directory]
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

TARGET_DIR=${1:-"/workspace/models/librispeech"}
TEMP_DIR="/tmp/librispeech_download"

print_info "Setting up Librispeech models..."
print_info "Target directory: $TARGET_DIR"

mkdir -p "$TARGET_DIR"
mkdir -p "$TEMP_DIR"

print_info "Downloading models from https://kaldi-asr.org/models/m13"
print_warning "This may take several minutes depending on your internet connection..."

cd "$TEMP_DIR"

# Download the main model archive
print_info "Downloading 0013_librispeech_v1 model (chain)..."
if ! wget -c https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz; then
    print_warning "Download failed. Please manually download from:"
    print_warning "  https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz"
    print_warning "And place in: $TARGET_DIR"
    exit 1
fi

print_info "Downloading 0013_librispeech_v1 model (extractor)..."
if ! wget -c https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz; then
    print_warning "Download failed. Please manually download from:"
    print_warning "  https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz"
    print_warning "And place in: $TARGET_DIR"
    exit 1
fi

print_info "Downloading language model..."
if ! wget -c https://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz; then
    print_warning "Download failed. Please manually download from:"
    print_warning "  https://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz"
    print_warning "And place in: $TARGET_DIR"
    exit 1
fi

# Extract archives
print_info "Extracting models..."
tar -xzf 0013_librispeech_v1_chain.tar.gz -C "$TARGET_DIR"
tar -xzf 0013_librispeech_v1_extractor.tar.gz -C "$TARGET_DIR"
tar -xzf 0013_librispeech_v1_lm.tar.gz -C "$TARGET_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

print_success "Librispeech models successfully installed to: $TARGET_DIR"
print_info "Model structure:"
tree -L 2 "$TARGET_DIR" || ls -R "$TARGET_DIR"

print_success "Setup complete!"
print_info "You can now run the GOPT pipeline."

