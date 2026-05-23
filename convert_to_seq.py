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
        phone = label[i, 0] if label.ndim > 1 else label[i]
        if phone not in phn_dict:
            phn_dict[phone] = phn_idx
            phn_idx += 1
    return phn_dict

def process_feat_seq(feat, keys, labels, phn_dict):
    key_set = [keys[i].split('.')[0] for i in range(keys.shape[0])]
    feat_dim = feat.shape[1] - 1
    utt_cnt = len(list(set(key_set)))
    
    print(f'Processing {utt_cnt} utterances, feat_dim={feat_dim}')
    
    # Use fixed sequence length of 50 (model expects this)
    max_len = 50
    
    seq_feat = np.zeros([utt_cnt, max_len, feat_dim])
    seq_label = np.zeros([utt_cnt, max_len, 2]) - 1
    
    prev_utt_id = keys[0].split('.')[0]
    row = 0
    
    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id
        # Only process if within max_len
        if cur_tok_id < max_len:
            seq_feat[row, cur_tok_id, :] = feat[i, 1:]
            phone = labels[i, 0] if labels.ndim > 1 else labels[i]
            seq_label[row, cur_tok_id, 0] = phn_dict[phone]
        elif cur_tok_id == max_len:
            print(f'Warning: Utterance {cur_utt_id} has more than {max_len} tokens, truncating...')
    
    return seq_feat, seq_label

base_path = "/workspace/gopt/data/raw_kaldi_gop/test_audio_001"
output_dir = "/workspace/gopt/data/seq_data_test_audio_001"

import os
os.makedirs(output_dir, exist_ok=True)

print("Loading training data...")
tr_feat = load_feat(base_path + '/tr_feats.csv')
tr_keys = load_keys(base_path + '/tr_keys_phn.csv')
tr_label = load_label(base_path + '/tr_labels_phn.csv')

print(f"Training: {tr_feat.shape[0]} samples")

phn_dict = gen_phn_dict(tr_label)
print(f"Phone dictionary: {len(phn_dict)} phones")

tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict)

np.save(output_dir + '/tr_feat.npy', tr_feat)
np.save(output_dir + '/tr_label_phn.npy', tr_label)

print("\nLoading test data...")
te_feat = load_feat(base_path + '/te_feats.csv')
te_keys = load_keys(base_path + '/te_keys_phn.csv')
te_label = load_label(base_path + '/te_labels_phn.csv')

print(f"Test: {te_feat.shape[0]} samples")

te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label, phn_dict)

np.save(output_dir + '/te_feat.npy', te_feat)
np.save(output_dir + '/te_label_phn.npy', te_label)

print(f"\nSequence data saved to: {output_dir}")
print(f"Training shape: {tr_feat.shape}")
print(f"Test shape: {te_feat.shape}")

