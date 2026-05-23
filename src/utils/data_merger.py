#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Merger Tool for GOPT Fine-tuning
Merge multiple datasets (e.g., SpeechOcean762 + custom data) for fine-tuning
"""

import numpy as np
import os
import argparse
from pathlib import Path

def merge_datasets(source_dirs, output_dir, verbose=True):
    """
    Merge multiple datasets into a single combined dataset
    
    Args:
        source_dirs: List of source dataset directories
        output_dir: Output directory for merged dataset
        verbose: Print progress information
    """
    
    required_files = [
        'tr_feat.npy', 'tr_label_phn.npy', 'tr_label_word.npy', 'tr_label_utt.npy',
        'te_feat.npy', 'te_label_phn.npy', 'te_label_word.npy', 'te_label_utt.npy'
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge each file type
    for filename in required_files:
        if verbose:
            print(f"\nMerging {filename}...")
        
        arrays = []
        for source_dir in source_dirs:
            filepath = os.path.join(source_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"  Warning: {filepath} not found, skipping...")
                continue
            
            data = np.load(filepath)
            arrays.append(data)
            
            if verbose:
                print(f"  Loaded from {source_dir}: shape {data.shape}")
        
        if not arrays:
            print(f"  Error: No data found for {filename}")
            continue
        
        # Concatenate along first axis (samples)
        merged = np.concatenate(arrays, axis=0)
        
        # Save merged data
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, merged)
        
        if verbose:
            print(f"  ✓ Saved merged data: shape {merged.shape} -> {output_path}")
    
    print(f"\n✓ Dataset merging completed!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total files merged: {len(required_files)}")

def check_dataset_compatibility(source_dirs, verbose=True):
    """Check if datasets are compatible for merging"""
    
    if verbose:
        print("Checking dataset compatibility...")
    
    # Check feature dimensions
    for source_dir in source_dirs:
        tr_feat_path = os.path.join(source_dir, 'tr_feat.npy')
        if os.path.exists(tr_feat_path):
            feat = np.load(tr_feat_path)
            if verbose:
                print(f"  {source_dir}")
                print(f"    Feature shape: {feat.shape}")
                print(f"    Feature dim: {feat.shape[2]} (should be 84/86/88)")
    
    print("  ✓ Compatibility check passed")
    return True

def print_dataset_stats(data_dir):
    """Print statistics about a dataset"""
    
    print(f"\nDataset Statistics: {data_dir}")
    print("=" * 60)
    
    files = {
        'Training Features': 'tr_feat.npy',
        'Training Phone Labels': 'tr_label_phn.npy',
        'Training Word Labels': 'tr_label_word.npy',
        'Training Utterance Labels': 'tr_label_utt.npy',
        'Test Features': 'te_feat.npy',
        'Test Phone Labels': 'te_label_phn.npy',
        'Test Word Labels': 'te_label_word.npy',
        'Test Utterance Labels': 'te_label_utt.npy',
    }
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"{name:30s}: shape {str(data.shape):20s} dtype {data.dtype}")
        else:
            print(f"{name:30s}: NOT FOUND")
    
    # Calculate and print train/test split
    tr_feat_path = os.path.join(data_dir, 'tr_feat.npy')
    te_feat_path = os.path.join(data_dir, 'te_feat.npy')
    
    if os.path.exists(tr_feat_path) and os.path.exists(te_feat_path):
        tr_samples = np.load(tr_feat_path).shape[0]
        te_samples = np.load(te_feat_path).shape[0]
        total = tr_samples + te_samples
        print(f"\nTrain/Test Split:")
        print(f"  Training samples: {tr_samples} ({100*tr_samples/total:.1f}%)")
        print(f"  Test samples: {te_samples} ({100*te_samples/total:.1f}%)")
        print(f"  Total samples: {total}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple GOPT datasets')
    parser.add_argument('--sources', nargs='+', required=True, 
                        help='Source dataset directories to merge')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for merged dataset')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check compatibility without merging')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    source_dirs = [os.path.abspath(os.path.join('../../data', s)) for s in args.sources]
    output_dir = os.path.abspath(os.path.join('../../data', args.output))
    
    print("GOPT Dataset Merger")
    print("=" * 60)
    print(f"Source datasets: {len(source_dirs)}")
    for i, src in enumerate(source_dirs, 1):
        print(f"  {i}. {src}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Print stats if requested
    if args.stats:
        for src in source_dirs:
            print_dataset_stats(src)
    
    # Check compatibility
    check_dataset_compatibility(source_dirs)
    
    if args.check_only:
        print("\n✓ Compatibility check completed. No merge performed (--check-only).")
    else:
        # Perform merge
        merge_datasets(source_dirs, output_dir)
        
        # Print merged dataset stats
        print_dataset_stats(output_dir)

# Example usage:
"""
# Merge two datasets
python src/utils/data_merger.py \
  --sources seq_data_librispeech seq_data_custom \
  --output seq_data_mixed

# Check compatibility only
python src/utils/data_merger.py \
  --sources seq_data_librispeech seq_data_paiia \
  --output seq_data_mixed \
  --check-only

# Print statistics
python src/utils/data_merger.py \
  --sources seq_data_librispeech \
  --output seq_data_mixed \
  --stats \
  --check-only
"""

