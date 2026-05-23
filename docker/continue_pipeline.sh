#!/bin/bash
################################################################################
# Continue GOPT Pipeline After Kaldi GOP Extraction
#
# This script continues the pipeline after manual Kaldi GOP extraction
#
# Usage:
#   ./continue_pipeline.sh <dataset_name> <output_file>
################################################################################

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

GOPT_ROOT="/workspace/gopt"
KALDI_GOP_DIR="/opt/kaldi/egs/gop_speechocean762/s5"

dataset_name=$1
output_file=${2:-"/workspace/audio_output/${dataset_name}_results.json"}

if [ -z "$dataset_name" ]; then
    echo "Usage: $0 <dataset_name> [output_file]"
    exit 1
fi

print_info "Continuing pipeline for dataset: $dataset_name"

# Extract GOP features to CSV
print_info "Extracting GOP features..."
cd "$KALDI_GOP_DIR"
python3 local/extract_gop_feats.py

# Copy to GOPT
target_dir="$GOPT_ROOT/data/raw_kaldi_gop/$dataset_name"
mkdir -p "$target_dir"
cp -r gopt_feats/* "$target_dir/"
print_success "Features copied to: $target_dir"

# Convert to sequence format
print_info "Converting to sequence format..."
cd "$GOPT_ROOT"
output_dir="$GOPT_ROOT/data/seq_data_$dataset_name"
mkdir -p "$output_dir"

python3 - <<EOF
import sys
import numpy as np

def load_feat(path):
    return np.loadtxt(path, delimiter=',')

def load_keys(path):
    return np.loadtxt(path, delimiter=',', dtype=str)

def load_label(path):
    return np.loadtxt(path, delimiter=',', dtype=str)

def gen_phn_dict(label):
    phn_dict = {}
    phn_idx = 0
    for i in range(label.shape[0]):
        if label[i] not in phn_dict:
            phn_dict[label[i]] = phn_idx
            phn_idx += 1
    return phn_dict

def process_feat_seq(feat, keys, labels, phn_dict):
    key_set = [keys[i].split('.')[0] for i in range(keys.shape[0])]
    feat_dim = feat.shape[1] - 1
    utt_cnt = len(list(set(key_set)))
    
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

base_path = "$target_dir"
tr_feat = load_feat(base_path + '/tr_feats.csv')
tr_keys = load_keys(base_path + '/tr_keys_phn.csv')
tr_label = load_label(base_path + '/tr_labels_phn.csv')

phn_dict = gen_phn_dict(tr_label)
tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict)

np.save('$output_dir/tr_feat.npy', tr_feat)
np.save('$output_dir/tr_label_phn.npy', tr_label)

te_feat = load_feat(base_path + '/te_feats.csv')
te_keys = load_keys(base_path + '/te_keys_phn.csv')
te_label = load_label(base_path + '/te_labels_phn.csv')
te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label, phn_dict)

np.save('$output_dir/te_feat.npy', te_feat)
np.save('$output_dir/te_label_phn.npy', te_label)

print("Sequence data saved")
EOF

print_success "Sequence conversion completed"

# Run inference
print_info "Running GOPT inference..."
cd "$GOPT_ROOT"

python3 - <<EOF
import sys
sys.path.append('$GOPT_ROOT/src')
import torch
import numpy as np
from models import GOPT
import json

model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
model = torch.nn.DataParallel(model)
checkpoint = torch.load('pretrained_models/gopt_librispeech/best_audio_model.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=True)
model.eval()

feat = np.load('$output_dir/te_feat.npy')
phn = np.load('$output_dir/te_label_phn.npy')

norm_mean, norm_std = 3.203, 4.045
feat_norm = (feat - norm_mean) / norm_std

with torch.no_grad():
    feat_tensor = torch.FloatTensor(feat_norm)
    phn_tensor = torch.FloatTensor(phn[:, :, 0])
    u1, u2, u3, u4, u5, p, w1, w2, w3 = model(feat_tensor, phn_tensor)

results = {
    'utterance_scores': {
        'accuracy': float(u1.squeeze().numpy()),
        'completeness': float(u2.squeeze().numpy()),
        'fluency': float(u3.squeeze().numpy()),
        'prosodic': float(u4.squeeze().numpy()),
        'total': float(u5.squeeze().numpy())
    },
    'phone_scores': p.squeeze().numpy().tolist(),
}

with open('$output_file', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("PRONUNCIATION ASSESSMENT RESULTS")
print("="*60)
print("\nUtterance-Level Scores (0-2 scale):")
for aspect, score in results['utterance_scores'].items():
    print(f"  {aspect:15s}: {score:.3f}")
print("\n" + "="*60)
print(f"\nFull results saved to: $output_file")
EOF

print_success "Pipeline completed successfully!"
print_info "Results saved to: $output_file"

