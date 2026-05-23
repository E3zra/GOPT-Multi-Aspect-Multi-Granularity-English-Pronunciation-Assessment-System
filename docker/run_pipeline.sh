#!/bin/bash
################################################################################
# GOPT Full Pipeline Automation Script (Docker)
# 
# This script automates the entire GOPT pronunciation assessment pipeline:
#   1. Prepare Kaldi data directory
#   2. Extract GOP features using Kaldi
#   3. Convert features to sequence format
#   4. Run GOPT inference
#
# Usage:
#   ./run_pipeline.sh <audio_file> <transcript> [dataset_name]
#
# Example:
#   ./run_pipeline.sh /workspace/audio_input/test.wav "HELLO WORLD" my_audio
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GOPT_ROOT="/workspace/gopt"
KALDI_ROOT="/opt/kaldi"
KALDI_GOP_DIR="$KALDI_ROOT/egs/gop_speechocean762/s5"
LIBRISPEECH_MODEL_DIR="/workspace/models/librispeech"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found!"
        exit 1
    fi
    print_success "Python3 found: $(python3 --version)"
    
    # Check PyTorch
    if ! python3 -c "import torch" 2>/dev/null; then
        print_error "PyTorch not installed!"
        exit 1
    fi
    print_success "PyTorch found"
    
    # Check Kaldi
    if [ ! -d "$KALDI_ROOT" ]; then
        print_error "Kaldi not found at $KALDI_ROOT"
        exit 1
    fi
    print_success "Kaldi found at $KALDI_ROOT"
    
    # Check Librispeech models
    if [ ! -d "$LIBRISPEECH_MODEL_DIR" ]; then
        print_warning "Librispeech models not found at $LIBRISPEECH_MODEL_DIR"
        print_info "Please run: docker exec -it gopt-pipeline /workspace/scripts/setup_models.sh"
        print_info "Or manually place models in ./models/librispeech/"
    fi
    
    # Check pretrained GOPT model
    if [ ! -f "$GOPT_ROOT/pretrained_models/gopt_librispeech/best_audio_model.pth" ]; then
        print_error "GOPT pretrained model not found!"
        print_info "Please ensure pretrained_models directory is properly mounted"
        exit 1
    fi
    print_success "GOPT pretrained model found"
}

# Function to validate audio file
validate_audio() {
    local audio_file=$1
    
    print_info "Validating audio file: $audio_file"
    
    if [ ! -f "$audio_file" ]; then
        print_error "Audio file not found: $audio_file"
        exit 1
    fi
    
    # Check if it's a WAV file
    if [[ ! "$audio_file" =~ \.wav$ ]]; then
        print_warning "Audio file is not .wav format. Kaldi expects 16kHz mono WAV."
    fi
    
    print_success "Audio file validated"
}

# Step 1: Prepare Kaldi data directory
prepare_kaldi_data() {
    local audio_file=$1
    local transcript=$2
    local dataset_name=$3
    
    print_info "Step 1: Preparing Kaldi data directory for dataset: $dataset_name"
    
    local data_dir="$GOPT_ROOT/data/temp_custom_audio/$dataset_name"
    local train_dir="$data_dir/train"
    local test_dir="$data_dir/test"
    
    # Create directories
    mkdir -p "$train_dir" "$test_dir"
    
    # Generate IDs
    local utt_id="utt_001"
    local spk_id="spk_001"
    
    # Get absolute path
    audio_file=$(realpath "$audio_file")
    
    # Create Kaldi format files for test set
    echo "$utt_id $audio_file" > "$test_dir/wav.scp"
    echo "$utt_id ${transcript^^}" > "$test_dir/text"  # Convert to uppercase
    echo "$utt_id $spk_id" > "$test_dir/utt2spk"
    echo "$spk_id $utt_id" > "$test_dir/spk2utt"
    echo "$spk_id adult" > "$test_dir/spk2age"
    echo "$spk_id m" > "$test_dir/spk2gender"
    
    # Copy to train set (simplified)
    cp "$test_dir"/* "$train_dir/"
    
    print_success "Kaldi data directory prepared at: $data_dir"
    echo "$data_dir"
}

# Step 2: Run Kaldi GOP extraction
run_kaldi_gop() {
    local dataset_name=$1
    
    print_info "Step 2: Running Kaldi GOP feature extraction"
    
    if [ ! -d "$KALDI_GOP_DIR" ]; then
        print_error "Kaldi GOP recipe not found at: $KALDI_GOP_DIR"
        exit 1
    fi
    
    cd "$KALDI_GOP_DIR"
    
    # Link data directory
    local data_dir="$GOPT_ROOT/data/temp_custom_audio/$dataset_name"
    ln -sf "$data_dir" "$KALDI_GOP_DIR/data/$dataset_name"
    
    print_info "Running GOP extraction... This may take several minutes."
    print_info "Kaldi working directory: $KALDI_GOP_DIR"
    
    # Note: This requires run.sh to be properly configured
    # For Docker, we assume it's already configured or user will manually run it
    if [ -f "run.sh" ]; then
        print_warning "GOP extraction requires manual Kaldi run.sh execution"
        print_info "Please configure run.sh with:"
        print_info "  - librispeech_eg=$LIBRISPEECH_MODEL_DIR"
        print_info "  - stage=2"
        print_info "  - nj=1"
        print_info "  - data_dir=$dataset_name"
        print_info ""
        print_info "Then run: cd $KALDI_GOP_DIR && ./run.sh"
        return 1
    else
        print_error "run.sh not found in Kaldi GOP directory"
        return 1
    fi
}

# Step 3: Extract GOP features to CSV
extract_gop_features() {
    local dataset_name=$1
    
    print_info "Step 3: Extracting GOP features to CSV format"
    
    cd "$KALDI_GOP_DIR"
    
    # Check if GOP extraction completed
    if [ ! -f "exp/gop_test/gop.scp" ]; then
        print_error "GOP extraction not completed. gop.scp not found."
        exit 1
    fi
    
    # Run extraction script
    if [ -f "local/extract_gop_feats.py" ]; then
        python3 local/extract_gop_feats.py
        print_success "GOP features extracted"
        
        # Copy to GOPT data directory
        local target_dir="$GOPT_ROOT/data/raw_kaldi_gop/$dataset_name"
        mkdir -p "$target_dir"
        cp -r gopt_feats/* "$target_dir/"
        
        print_success "Features copied to: $target_dir"
    else
        print_error "extract_gop_feats.py not found in local/"
        exit 1
    fi
}

# Step 4: Convert to sequence format
convert_to_sequence() {
    local dataset_name=$1
    
    print_info "Step 4: Converting GOP features to sequence format"
    
    cd "$GOPT_ROOT/src/prep_data"
    
    # Create temporary modified version of gen_seq_data_phn.py
    python3 - <<EOF
import sys
sys.path.append('$GOPT_ROOT/src')
import numpy as np

def load_feat(path):
    file = np.loadtxt(path, delimiter=',')
    return file

def load_keys(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def load_label(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def gen_phn_dict(label):
    phn_dict = {}
    phn_idx = 0
    for i in range(label.shape[0]):
        if label[i] not in phn_dict:
            phn_dict[label[i]] = phn_idx
            phn_idx += 1
    return phn_dict

def process_feat_seq(feat, keys, labels, phn_dict):
    key_set = []
    for i in range(keys.shape[0]):
        cur_key = keys[i].split('.')[0]
        key_set.append(cur_key)
    
    feat_dim = feat.shape[1] - 1
    utt_cnt = len(list(set(key_set)))
    
    print(f'Processing {utt_cnt} utterances')
    
    seq_feat = np.zeros([utt_cnt, 50, feat_dim])
    seq_label = np.zeros([utt_cnt, 50, 2]) - 1
    
    prev_utt_id = keys[0].split('.')[0]
    row = 0
    
    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id
        
        seq_feat[row, cur_tok_id, :] = feat[i, 1:]
        seq_label[row, cur_tok_id, 0] = phn_dict[labels[i]]
    
    return seq_feat, seq_label

# Process data
dataset = "$dataset_name"
base_path = "$GOPT_ROOT/data/raw_kaldi_gop/" + dataset

print(f"Loading training data from {base_path}")
tr_feat = load_feat(base_path + '/tr_feats.csv')
tr_keys = load_keys(base_path + '/tr_keys_phn.csv')
tr_label = load_label(base_path + '/tr_labels_phn.csv')

phn_dict = gen_phn_dict(tr_label)
tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict)

output_dir = "$GOPT_ROOT/data/seq_data_" + dataset
import os
os.makedirs(output_dir, exist_ok=True)

np.save(output_dir + '/tr_feat.npy', tr_feat)
np.save(output_dir + '/tr_label_phn.npy', tr_label)

print(f"Loading test data from {base_path}")
te_feat = load_feat(base_path + '/te_feats.csv')
te_keys = load_keys(base_path + '/te_keys_phn.csv')
te_label = load_label(base_path + '/te_labels_phn.csv')

te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label, phn_dict)

np.save(output_dir + '/te_feat.npy', te_feat)
np.save(output_dir + '/te_label_phn.npy', te_label)

print(f"Sequence data saved to: {output_dir}")
print(f"Training shape: {tr_feat.shape}")
print(f"Test shape: {te_feat.shape}")
EOF
    
    print_success "Sequence data generated"
}

# Step 5: Run GOPT inference
run_gopt_inference() {
    local dataset_name=$1
    local output_file=$2
    
    print_info "Step 5: Running GOPT inference"
    
    cd "$GOPT_ROOT"
    
    python3 - <<EOF
import sys
sys.path.append('$GOPT_ROOT/src')
import torch
import numpy as np
from models import GOPT
import json

# Load model
print("Loading GOPT model...")
model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
model = torch.nn.DataParallel(model)
checkpoint = torch.load('pretrained_models/gopt_librispeech/best_audio_model.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=True)
model.eval()

# Load features
dataset = "$dataset_name"
feat_path = f"data/seq_data_{dataset}/te_feat.npy"
phn_path = f"data/seq_data_{dataset}/te_label_phn.npy"

print(f"Loading features from {feat_path}")
feat = np.load(feat_path)
phn = np.load(phn_path)

# Normalize
norm_mean, norm_std = 3.203, 4.045
feat_norm = (feat - norm_mean) / norm_std

# Inference
print("Running inference...")
with torch.no_grad():
    feat_tensor = torch.FloatTensor(feat_norm)
    phn_tensor = torch.FloatTensor(phn[:, :, 0])
    
    u1, u2, u3, u4, u5, p, w1, w2, w3 = model(feat_tensor, phn_tensor)

# Extract results
results = {
    'utterance_scores': {
        'accuracy': float(u1.squeeze().numpy()),
        'completeness': float(u2.squeeze().numpy()),
        'fluency': float(u3.squeeze().numpy()),
        'prosodic': float(u4.squeeze().numpy()),
        'total': float(u5.squeeze().numpy())
    },
    'phone_scores': p.squeeze().numpy().tolist(),
    'word_scores': {
        'accuracy': w1.squeeze().numpy().tolist(),
        'stress': w2.squeeze().numpy().tolist(),
        'total': w3.squeeze().numpy().tolist()
    }
}

# Save results
output_file = "$output_file"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\n" + "="*60)
print("PRONUNCIATION ASSESSMENT RESULTS")
print("="*60)
print("\nUtterance-Level Scores (0-2 scale):")
for aspect, score in results['utterance_scores'].items():
    print(f"  {aspect:15s}: {score:.3f}")

print("\nPhone-Level Scores (first 10 phones):")
for i, score in enumerate(results['phone_scores'][:10]):
    if score > 0:
        print(f"  Phone {i+1:2d}: {score:.3f}")

print("\n" + "="*60)
EOF
    
    print_success "Inference completed!"
}

# Main function
main() {
    if [ "$#" -lt 2 ]; then
        echo "Usage: $0 <audio_file> <transcript> [dataset_name]"
        echo ""
        echo "Example:"
        echo "  $0 /workspace/audio_input/test.wav \"HELLO WORLD\" my_test"
        echo ""
        echo "Arguments:"
        echo "  audio_file    - Path to WAV audio file (16kHz mono recommended)"
        echo "  transcript    - Text transcript (will be converted to uppercase)"
        echo "  dataset_name  - Optional custom dataset name (default: custom_audio)"
        exit 1
    fi
    
    local audio_file=$1
    local transcript=$2
    local dataset_name=${3:-custom_audio}
    local output_file="/workspace/audio_output/${dataset_name}_results.json"
    
    echo ""
    echo "========================================================================"
    echo "           GOPT Pronunciation Assessment Pipeline"
    echo "========================================================================"
    echo ""
    echo "Audio File:    $audio_file"
    echo "Transcript:    $transcript"
    echo "Dataset Name:  $dataset_name"
    echo "Output File:   $output_file"
    echo ""
    echo "========================================================================"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Validate audio
    validate_audio "$audio_file"
    
    # Run pipeline
    prepare_kaldi_data "$audio_file" "$transcript" "$dataset_name"
    
    print_warning "========================================================================"
    print_warning "MANUAL STEP REQUIRED: Kaldi GOP Extraction"
    print_warning "========================================================================"
    print_info ""
    print_info "Due to Kaldi's complexity, you need to manually run GOP extraction:"
    print_info ""
    print_info "1. Configure Kaldi run.sh:"
    print_info "   cd $KALDI_GOP_DIR"
    print_info "   nano run.sh  # or vim run.sh"
    print_info ""
    print_info "   Modify these lines (around line 38-42):"
    print_info "     librispeech_eg=$LIBRISPEECH_MODEL_DIR"
    print_info "     model=\$librispeech_eg/exp/chain_cleaned/tdnn_1d_sp"
    print_info "     ivector_extractor=\$librispeech_eg/exp/nnet3_cleaned/extractor"
    print_info "     lang=\$librispeech_eg/data/lang_test_tgsmall"
    print_info "     stage=2"
    print_info "     nj=1"
    print_info ""
    print_info "2. Run GOP extraction:"
    print_info "   ./run.sh"
    print_info ""
    print_info "3. After completion, run:"
    print_info "   /workspace/scripts/continue_pipeline.sh $dataset_name $output_file"
    print_info ""
    print_warning "========================================================================"
}

# Run main function
main "$@"

