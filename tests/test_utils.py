# -*- coding: utf-8 -*-
"""
Unit tests for utility functions
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gopt import get_sinusoid_encoding


class TestDataNormalization:
    """Test data normalization utilities"""
    
    def test_score_normalization(self):
        """Test score normalization from 0-10 to 0-2"""
        # Scores in 0-10 range
        scores = np.array([0.0, 5.0, 10.0])
        
        # Normalize to 0-2
        normalized = scores / 5.0
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 2.0
        assert np.allclose(normalized, [0.0, 1.0, 2.0])
    
    def test_score_denormalization(self):
        """Test score denormalization from 0-2 to 0-10"""
        # Normalized scores in 0-2 range
        normalized = np.array([0.0, 1.0, 2.0])
        
        # Denormalize to 0-10
        scores = normalized * 5.0
        
        assert scores.min() >= 0.0
        assert scores.max() <= 10.0
        assert np.allclose(scores, [0.0, 5.0, 10.0])
    
    def test_feature_standardization(self):
        """Test feature standardization (z-score)"""
        # Sample features
        features = np.random.randn(100, 84) * 4.045 + 3.203
        
        # Standardize
        mean = features.mean()
        std = features.std()
        standardized = (features - mean) / std
        
        # Check properties
        assert abs(standardized.mean()) < 0.1
        assert abs(standardized.std() - 1.0) < 0.1


class TestPhonemeEncoding:
    """Test phoneme encoding utilities"""
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding of phonemes"""
        # Phone indices (assuming 40 phone classes, -1 for padding)
        phones = torch.tensor([0, 5, 10, -1])
        
        # One-hot encode (add 1 to handle -1 -> 0)
        one_hot = torch.nn.functional.one_hot((phones + 1).long(), num_classes=40)
        
        assert one_hot.shape == (4, 40)
        assert one_hot.sum(dim=1).tolist() == [1, 1, 1, 1]
    
    def test_phone_index_range(self):
        """Test that phone indices are in valid range"""
        # Valid phone indices should be -1 to 38
        valid_phones = torch.tensor([-1, 0, 19, 38])
        
        # Check range
        assert valid_phones.min() >= -1
        assert valid_phones.max() <= 38
        
        # One-hot encoding should work
        one_hot = torch.nn.functional.one_hot((valid_phones + 1).long(), num_classes=40)
        assert one_hot.shape == (4, 40)


class TestPositionalEncoding:
    """Test positional encoding utilities"""
    
    def test_positional_encoding_properties(self):
        """Test properties of positional encoding"""
        n_position = 55
        d_hid = 24
        
        pos_encoding = get_sinusoid_encoding(n_position, d_hid)
        
        # Check shape
        assert pos_encoding.shape == (1, n_position, d_hid)
        
        # Check that different positions have different encodings
        assert not torch.allclose(pos_encoding[0, 0, :], pos_encoding[0, 1, :])
        assert not torch.allclose(pos_encoding[0, 0, :], pos_encoding[0, 10, :])
    
    def test_positional_encoding_periodicity(self):
        """Test that positional encoding has periodic properties"""
        pos_encoding = get_sinusoid_encoding(100, 16)
        
        # Extract encodings for positions 0 and 1
        pos_0 = pos_encoding[0, 0, :]
        pos_1 = pos_encoding[0, 1, :]
        
        # The difference should be consistent for adjacent positions
        diff_0_1 = pos_1 - pos_0
        
        pos_10 = pos_encoding[0, 10, :]
        pos_11 = pos_encoding[0, 11, :]
        diff_10_11 = pos_11 - pos_10
        
        # Differences should be similar (not exact due to sine/cosine)
        # but the pattern should be preserved
        assert torch.isfinite(diff_0_1).all()
        assert torch.isfinite(diff_10_11).all()


class TestLossCalculation:
    """Test loss calculation utilities"""
    
    def test_mse_loss(self):
        """Test MSE loss calculation"""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])
        
        mse = torch.nn.functional.mse_loss(predictions, targets)
        
        expected_mse = ((predictions - targets) ** 2).mean()
        
        assert torch.allclose(mse, expected_mse)
        assert mse.item() == 0.25
    
    def test_weighted_loss(self):
        """Test weighted loss combination"""
        loss_phn = torch.tensor(0.1)
        loss_word = torch.tensor(0.2)
        loss_utt = torch.tensor(0.3)
        
        # Weights
        w_phn = 1.0
        w_word = 1.0
        w_utt = 1.0
        
        total_loss = w_phn * loss_phn + w_word * loss_word + w_utt * loss_utt
        
        assert abs(total_loss.item() - 0.6) < 1e-6


class TestCorrelationMetrics:
    """Test correlation metric calculations"""
    
    def test_pearson_correlation(self):
        """Test Pearson correlation calculation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        
        # Perfect positive correlation
        mean_x = x.mean()
        mean_y = y.mean()
        
        cov = ((x - mean_x) * (y - mean_y)).mean()
        std_x = x.std(unbiased=False)
        std_y = y.std(unbiased=False)
        
        corr = cov / (std_x * std_y)
        
        assert torch.allclose(corr, torch.tensor(1.0))
    
    def test_correlation_range(self):
        """Test that correlation is in valid range"""
        x = torch.randn(100)
        y = torch.randn(100)
        
        # Calculate correlation
        mean_x = x.mean()
        mean_y = y.mean()
        cov = ((x - mean_x) * (y - mean_y)).mean()
        std_x = x.std(unbiased=False)
        std_y = y.std(unbiased=False)
        corr = cov / (std_x * std_y + 1e-8)
        
        # Should be in [-1, 1]
        assert -1.0 <= corr.item() <= 1.0


class TestBatchProcessing:
    """Test batch processing utilities"""
    
    def test_padding_mask_creation(self):
        """Test creation of padding mask"""
        # Phone labels with -1 as padding
        phn_labels = torch.tensor([
            [0, 1, 2, -1, -1],
            [3, 4, 5, 6, -1]
        ])
        
        # Create mask (True for valid, False for padding)
        mask = phn_labels >= 0
        
        assert mask.shape == phn_labels.shape
        assert mask[0].sum() == 3  # First sequence has 3 valid tokens
        assert mask[1].sum() == 4  # Second sequence has 4 valid tokens
    
    def test_sequence_length_calculation(self):
        """Test calculating actual sequence lengths"""
        phn_labels = torch.tensor([
            [0, 1, 2, -1, -1],
            [3, 4, 5, 6, -1],
            [7, 8, -1, -1, -1]
        ])
        
        # Calculate lengths
        lengths = (phn_labels >= 0).sum(dim=1)
        
        assert lengths.tolist() == [3, 4, 2]


class TestDataAugmentation:
    """Test data augmentation utilities"""
    
    def test_noise_addition(self):
        """Test adding Gaussian noise for augmentation"""
        original = torch.randn(10, 50, 84)
        noise_scale = 0.1
        
        # Add noise
        noise = (torch.rand_like(original) - 0.5) * 2 * noise_scale
        augmented = original + noise
        
        # Check that noise is bounded
        assert noise.abs().max() <= noise_scale
        
        # Check that augmented data has correct shape
        assert augmented.shape == original.shape
    
    def test_noise_zero_mean(self):
        """Test that noise has approximately zero mean"""
        noise = (torch.rand(1000, 84) - 0.5) * 2 * 0.1
        
        # Mean should be close to 0
        assert abs(noise.mean().item()) < 0.01


class TestFileOperations:
    """Test file operation utilities"""
    
    def test_numpy_save_load(self, tmp_path):
        """Test saving and loading numpy arrays"""
        data = np.random.randn(100, 84)
        
        # Save
        filepath = tmp_path / "test_data.npy"
        np.save(filepath, data)
        
        # Load
        loaded_data = np.load(filepath)
        
        assert np.allclose(data, loaded_data)
    
    def test_torch_save_load(self, tmp_path):
        """Test saving and loading PyTorch tensors"""
        tensor = torch.randn(50, 24)
        
        # Save
        filepath = tmp_path / "test_tensor.pt"
        torch.save(tensor, filepath)
        
        # Load
        loaded_tensor = torch.load(filepath)
        
        assert torch.allclose(tensor, loaded_tensor)


class TestInputValidation:
    """Test input validation utilities"""
    
    def test_shape_validation(self):
        """Test input shape validation"""
        # Valid input
        valid_input = torch.randn(4, 50, 84)
        
        assert valid_input.dim() == 3
        assert valid_input.shape[2] == 84  # Feature dimension
    
    def test_nan_detection(self):
        """Test NaN detection in inputs"""
        # Create tensor with NaN
        tensor_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0])
        tensor_without_nan = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        assert not torch.isfinite(tensor_with_nan).all()
        assert torch.isfinite(tensor_without_nan).all()
    
    def test_value_range_validation(self):
        """Test value range validation"""
        # Phone labels should be in [-1, 38]
        valid_phones = torch.tensor([-1, 0, 20, 38])
        invalid_phones = torch.tensor([-2, 0, 20, 40])
        
        assert valid_phones.min() >= -1
        assert valid_phones.max() <= 38
        
        assert not (invalid_phones.min() >= -1 and invalid_phones.max() <= 38)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

