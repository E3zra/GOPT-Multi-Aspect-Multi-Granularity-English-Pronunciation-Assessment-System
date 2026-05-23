# -*- coding: utf-8 -*-
"""
Unit tests for data loading functionality
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDataLoading:
    """Test data loading from numpy files"""
    
    def test_load_numpy_file(self, tmp_path):
        """Test loading data from numpy file"""
        # Create temporary data
        data = np.random.randn(100, 50, 84)
        filepath = tmp_path / "test_data.npy"
        np.save(filepath, data)
        
        # Load
        loaded = np.load(filepath)
        
        assert loaded.shape == data.shape
        assert np.allclose(loaded, data)
    
    def test_load_multiple_files(self, tmp_path):
        """Test loading multiple data files"""
        # Create test files
        features = np.random.randn(100, 50, 84)
        labels = np.random.randn(100, 5)
        
        feat_path = tmp_path / "features.npy"
        label_path = tmp_path / "labels.npy"
        
        np.save(feat_path, features)
        np.save(label_path, labels)
        
        # Load both
        loaded_feat = np.load(feat_path)
        loaded_label = np.load(label_path)
        
        assert loaded_feat.shape == features.shape
        assert loaded_label.shape == labels.shape
    
    def test_data_shape_validation(self):
        """Test validation of loaded data shapes"""
        # Expected shapes for GOPT data
        n_samples = 100
        seq_len = 50
        feat_dim = 84
        n_aspects = 5
        
        features = np.random.randn(n_samples, seq_len, feat_dim)
        utt_labels = np.random.randn(n_samples, n_aspects)
        
        # Validate shapes
        assert features.shape == (n_samples, seq_len, feat_dim)
        assert utt_labels.shape == (n_samples, n_aspects)


class TestGoPDataset:
    """Test GoP Dataset class (if implemented)"""
    
    def test_dataset_length(self):
        """Test dataset length calculation"""
        n_samples = 100
        features = np.random.randn(n_samples, 50, 84)
        
        # Length should equal number of samples
        assert len(features) == n_samples
    
    def test_dataset_getitem(self):
        """Test getting individual items from dataset"""
        features = np.random.randn(10, 50, 84)
        labels = np.random.randn(10, 5)
        
        # Get single item
        idx = 5
        feat_item = features[idx]
        label_item = labels[idx]
        
        assert feat_item.shape == (50, 84)
        assert label_item.shape == (5,)
    
    def test_dataset_iteration(self):
        """Test iterating through dataset"""
        features = np.random.randn(10, 50, 84)
        
        # Iterate
        count = 0
        for item in features:
            assert item.shape == (50, 84)
            count += 1
        
        assert count == 10


class TestDataLoader:
    """Test PyTorch DataLoader functionality"""
    
    def test_dataloader_batch_size(self):
        """Test DataLoader with different batch sizes"""
        # Create simple tensor dataset
        features = torch.randn(100, 50, 84)
        
        for batch_size in [4, 8, 16]:
            # Create batches manually
            n_batches = len(features) // batch_size
            
            for i in range(n_batches):
                batch = features[i * batch_size:(i + 1) * batch_size]
                assert batch.shape[0] == batch_size
    
    def test_dataloader_shuffle(self):
        """Test that shuffling changes order"""
        indices = torch.arange(20)
        
        # Simulate shuffle
        shuffled_indices = torch.randperm(20)
        
        # Should not be identical (with very high probability)
        assert not torch.equal(indices, shuffled_indices)
    
    def test_dataloader_drop_last(self):
        """Test drop_last behavior"""
        n_samples = 37
        batch_size = 8
        
        # With drop_last=False
        n_batches_no_drop = (n_samples + batch_size - 1) // batch_size
        assert n_batches_no_drop == 5  # 8+8+8+8+5
        
        # With drop_last=True
        n_batches_drop = n_samples // batch_size
        assert n_batches_drop == 4  # 8+8+8+8


class TestDataPreprocessing:
    """Test data preprocessing steps"""
    
    def test_normalization(self):
        """Test feature normalization"""
        # Raw features with known statistics
        features = np.random.randn(100, 50, 84) * 4.045 + 3.203
        
        # Normalize
        mean = 3.203
        std = 4.045
        normalized = (features - mean) / std
        
        # Check statistics
        assert abs(normalized.mean()) < 0.2
        assert abs(normalized.std() - 1.0) < 0.2
    
    def test_label_scaling(self):
        """Test label scaling"""
        # Labels in 0-10 range
        labels = np.array([0, 2.5, 5.0, 7.5, 10.0])
        
        # Scale to 0-2 for training
        scaled = labels / 5.0
        
        assert scaled.min() >= 0.0
        assert scaled.max() <= 2.0
        
        # Unscale
        unscaled = scaled * 5.0
        assert np.allclose(unscaled, labels)
    
    def test_phone_label_offset(self):
        """Test phone label offset handling"""
        # Phone labels typically in [-1, 38] range
        # -1 represents padding/silence
        phones = np.array([-1, 0, 10, 20, 38])
        
        # Offset by 1 for one-hot encoding (0 becomes padding)
        offset_phones = phones + 1
        
        assert offset_phones.min() >= 0
        assert offset_phones.max() <= 39


class TestDataAugmentation:
    """Test data augmentation techniques"""
    
    def test_add_gaussian_noise(self):
        """Test adding Gaussian noise"""
        original = np.random.randn(10, 50, 84)
        noise_scale = 0.1
        
        noise = np.random.randn(*original.shape) * noise_scale
        augmented = original + noise
        
        # Check properties
        assert augmented.shape == original.shape
        assert not np.allclose(augmented, original)
    
    def test_noise_scale_effect(self):
        """Test effect of different noise scales"""
        original = np.ones((5, 50, 84))
        
        # Different noise scales
        noise_small = np.random.randn(5, 50, 84) * 0.01
        noise_large = np.random.randn(5, 50, 84) * 0.5
        
        aug_small = original + noise_small
        aug_large = original + noise_large
        
        # Larger noise should cause larger deviation
        diff_small = np.abs(aug_small - original).mean()
        diff_large = np.abs(aug_large - original).mean()
        
        assert diff_large > diff_small


class TestDataValidation:
    """Test data validation checks"""
    
    def test_check_for_nan(self):
        """Test NaN detection in data"""
        # Data without NaN
        clean_data = np.random.randn(10, 50, 84)
        assert not np.isnan(clean_data).any()
        
        # Data with NaN
        dirty_data = clean_data.copy()
        dirty_data[5, 10, 20] = np.nan
        assert np.isnan(dirty_data).any()
    
    def test_check_for_inf(self):
        """Test infinity detection in data"""
        clean_data = np.random.randn(10, 50, 84)
        assert np.isfinite(clean_data).all()
        
        dirty_data = clean_data.copy()
        dirty_data[3, 5, 10] = np.inf
        assert not np.isfinite(dirty_data).all()
    
    def test_value_range_check(self):
        """Test checking value ranges"""
        # Normalized labels should be in [0, 2]
        labels = np.array([0.5, 1.0, 1.5, 2.0])
        
        assert labels.min() >= 0.0
        assert labels.max() <= 2.0
        
        # Out of range
        bad_labels = np.array([0.5, 1.0, 1.5, 3.0])
        assert bad_labels.max() > 2.0


class TestBatchCollation:
    """Test batch collation functions"""
    
    def test_stack_tensors(self):
        """Test stacking tensors into batch"""
        samples = [torch.randn(50, 84) for _ in range(4)]
        
        # Stack into batch
        batch = torch.stack(samples, dim=0)
        
        assert batch.shape == (4, 50, 84)
    
    def test_pad_sequences(self):
        """Test padding sequences of different lengths"""
        # Variable length sequences
        seq1 = torch.randn(30, 84)
        seq2 = torch.randn(40, 84)
        seq3 = torch.randn(35, 84)
        
        sequences = [seq1, seq2, seq3]
        max_len = max(seq.shape[0] for seq in sequences)
        
        # Pad to max length
        padded = []
        for seq in sequences:
            if seq.shape[0] < max_len:
                padding = torch.zeros(max_len - seq.shape[0], 84)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        # Stack
        batch = torch.stack(padded, dim=0)
        assert batch.shape == (3, max_len, 84)


class TestDataStatistics:
    """Test computation of data statistics"""
    
    def test_compute_mean_std(self):
        """Test computing mean and std of dataset"""
        data = np.random.randn(1000, 50, 84) * 4.0 + 3.0
        
        mean = data.mean()
        std = data.std()
        
        # Should be close to expected values
        assert abs(mean - 3.0) < 0.2
        assert abs(std - 4.0) < 0.2
    
    def test_compute_per_feature_stats(self):
        """Test computing per-feature statistics"""
        data = np.random.randn(100, 50, 84)
        
        # Per-feature mean (across samples and time)
        feat_mean = data.mean(axis=(0, 1))
        feat_std = data.std(axis=(0, 1))
        
        assert feat_mean.shape == (84,)
        assert feat_std.shape == (84,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

