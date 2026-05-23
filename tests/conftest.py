# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for GOPT tests
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def device():
    """Fixture to get the device (CPU or CUDA)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests"""
    return 4


@pytest.fixture
def seq_len():
    """Standard sequence length for tests"""
    return 50


@pytest.fixture
def feat_dim():
    """Standard feature dimension (Kaldi GOP features)"""
    return 84


@pytest.fixture
def embed_dim():
    """Standard embedding dimension"""
    return 24


@pytest.fixture
def num_heads():
    """Standard number of attention heads"""
    return 1


@pytest.fixture
def depth():
    """Standard model depth"""
    return 3


@pytest.fixture
def sample_input(batch_size, seq_len, feat_dim):
    """Create sample input tensor"""
    return torch.randn(batch_size, seq_len, feat_dim)


@pytest.fixture
def sample_phn_labels(batch_size, seq_len):
    """Create sample phone labels (range: -1 to 38, -1 = padding)"""
    # Random phone indices from 0 to 38, occasionally -1 for padding
    labels = torch.randint(0, 39, (batch_size, seq_len)).float()
    # Set some random positions to -1 (padding)
    mask = torch.rand(batch_size, seq_len) > 0.9
    labels[mask] = -1
    return labels


@pytest.fixture
def sample_utt_labels(batch_size):
    """Create sample utterance-level labels (5 aspects, normalized 0-2)"""
    # 5 aspects: accuracy, completeness, fluency, prosodic, total
    return torch.rand(batch_size, 5)


@pytest.fixture
def sample_word_labels(batch_size, seq_len):
    """Create sample word-level labels (3 aspects, normalized 0-2)"""
    # 3 aspects: accuracy, stress, total
    return torch.rand(batch_size, seq_len, 3)


@pytest.fixture
def gopt_config():
    """Default configuration for GOPT model"""
    return {
        'embed_dim': 24,
        'num_heads': 1,
        'depth': 3,
        'input_dim': 84
    }


@pytest.fixture
def lstm_config():
    """Default configuration for LSTM baseline"""
    return {
        'embed_dim': 24,
        'depth': 2,
        'input_dim': 84
    }

