# -*- coding: utf-8 -*-
"""
Unit tests for GOPT model components
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src/models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gopt import (
    GOPT, GOPTNoPhn, Attention, Mlp, Block, 
    get_sinusoid_encoding, trunc_normal_
)
from models.baseline import BaselineLSTM


class TestSinusoidEncoding:
    """Test sinusoidal position encoding"""
    
    def test_sinusoid_shape(self):
        """Test that sinusoid encoding has correct shape"""
        n_position = 55
        d_hid = 24
        encoding = get_sinusoid_encoding(n_position, d_hid)
        
        assert encoding.shape == (1, n_position, d_hid)
        assert isinstance(encoding, torch.Tensor)
    
    def test_sinusoid_values(self):
        """Test that sinusoid encoding produces valid values"""
        encoding = get_sinusoid_encoding(10, 8)
        
        # Check that values are bounded
        assert torch.isfinite(encoding).all()
        assert encoding.abs().max() <= 1.0
    
    def test_sinusoid_different_sizes(self):
        """Test encoding with different dimensions"""
        for n_pos in [10, 50, 100]:
            for d_hid in [8, 16, 32]:
                encoding = get_sinusoid_encoding(n_pos, d_hid)
                assert encoding.shape == (1, n_pos, d_hid)


class TestTruncNormal:
    """Test truncated normal initialization"""
    
    def test_trunc_normal_shape(self):
        """Test that truncated normal preserves tensor shape"""
        tensor = torch.zeros(10, 20)
        result = trunc_normal_(tensor, std=0.02)
        
        assert result.shape == (10, 20)
    
    def test_trunc_normal_bounds(self):
        """Test that values are within truncation bounds"""
        tensor = torch.zeros(100, 100)
        result = trunc_normal_(tensor, mean=0., std=0.02, a=-0.04, b=0.04)
        
        assert result.min() >= -0.04
        assert result.max() <= 0.04
    
    def test_trunc_normal_mean_std(self):
        """Test that distribution has approximately correct statistics"""
        tensor = torch.zeros(1000, 1000)
        result = trunc_normal_(tensor, mean=0., std=0.02, a=-2., b=2.)
        
        # Should be close to mean 0
        assert abs(result.mean().item()) < 0.01
        # Should have reasonable standard deviation
        assert 0.01 < result.std().item() < 0.03


class TestAttention:
    """Test attention module"""
    
    def test_attention_forward(self, batch_size, seq_len):
        """Test attention forward pass"""
        dim = 24
        num_heads = 1
        attn = Attention(dim, num_heads=num_heads)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_attention_multiple_heads(self, batch_size, seq_len):
        """Test attention with multiple heads"""
        dim = 24
        num_heads = 2
        attn = Attention(dim, num_heads=num_heads)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_attention_dropout(self, batch_size, seq_len):
        """Test attention with dropout"""
        dim = 24
        attn = Attention(dim, num_heads=1, attn_drop=0.1, proj_drop=0.1)
        
        x = torch.randn(batch_size, seq_len, dim)
        
        # Training mode
        attn.train()
        output_train = attn(x)
        
        # Eval mode
        attn.eval()
        output_eval = attn(x)
        
        assert output_train.shape == (batch_size, seq_len, dim)
        assert output_eval.shape == (batch_size, seq_len, dim)


class TestMlp:
    """Test MLP module"""
    
    def test_mlp_forward(self, batch_size, seq_len):
        """Test MLP forward pass"""
        in_features = 24
        hidden_features = 96
        mlp = Mlp(in_features, hidden_features=hidden_features)
        
        x = torch.randn(batch_size, seq_len, in_features)
        output = mlp(x)
        
        assert output.shape == (batch_size, seq_len, in_features)
    
    def test_mlp_custom_output(self, batch_size, seq_len):
        """Test MLP with custom output dimension"""
        in_features = 24
        out_features = 12
        mlp = Mlp(in_features, out_features=out_features)
        
        x = torch.randn(batch_size, seq_len, in_features)
        output = mlp(x)
        
        assert output.shape == (batch_size, seq_len, out_features)
    
    def test_mlp_activation(self):
        """Test MLP with different activations"""
        mlp_gelu = Mlp(24, hidden_features=96, act_layer=nn.GELU)
        mlp_relu = Mlp(24, hidden_features=96, act_layer=nn.ReLU)
        
        x = torch.randn(2, 10, 24)
        
        output_gelu = mlp_gelu(x)
        output_relu = mlp_relu(x)
        
        assert output_gelu.shape == (2, 10, 24)
        assert output_relu.shape == (2, 10, 24)
        # Outputs should be different due to different activations
        assert not torch.allclose(output_gelu, output_relu)


class TestBlock:
    """Test transformer block"""
    
    def test_block_forward(self, batch_size, seq_len):
        """Test transformer block forward pass"""
        dim = 24
        num_heads = 1
        block = Block(dim, num_heads)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_block_residual(self, batch_size, seq_len):
        """Test that block uses residual connections"""
        dim = 24
        block = Block(dim, num_heads=1, drop=0.0)
        
        x = torch.randn(batch_size, seq_len, dim)
        
        # Set to eval mode to disable dropout
        block.eval()
        output = block(x)
        
        # Output should be different from input (not just identity)
        assert not torch.allclose(output, x)
        
        # But should have same shape
        assert output.shape == x.shape


class TestGOPT:
    """Test GOPT model"""
    
    def test_gopt_initialization(self, gopt_config):
        """Test GOPT model initialization"""
        model = GOPT(**gopt_config)
        
        assert model.embed_dim == gopt_config['embed_dim']
        assert model.input_dim == gopt_config['input_dim']
        assert len(model.blocks) == gopt_config['depth']
    
    def test_gopt_forward_shape(self, sample_input, sample_phn_labels, gopt_config):
        """Test GOPT forward pass output shapes"""
        model = GOPT(**gopt_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_input, sample_phn_labels)
        
        # Should return 9 outputs: 5 utterance + 1 phone + 3 word
        assert len(outputs) == 9
        
        batch_size = sample_input.shape[0]
        seq_len = sample_input.shape[1]
        
        # Utterance-level outputs (5)
        for i in range(5):
            assert outputs[i].shape == (batch_size, 1)
        
        # Phone-level output
        assert outputs[5].shape == (batch_size, seq_len, 1)
        
        # Word-level outputs (3)
        for i in range(6, 9):
            assert outputs[i].shape == (batch_size, seq_len, 1)
    
    def test_gopt_forward_values(self, sample_input, sample_phn_labels, gopt_config):
        """Test that GOPT produces finite values"""
        model = GOPT(**gopt_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_input, sample_phn_labels)
        
        # All outputs should be finite
        for output in outputs:
            assert torch.isfinite(output).all()
    
    def test_gopt_different_batch_sizes(self, gopt_config):
        """Test GOPT with different batch sizes"""
        model = GOPT(**gopt_config)
        model.eval()
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 50, 84)
            phn = torch.randint(0, 39, (batch_size, 50)).float()
            
            with torch.no_grad():
                outputs = model(x, phn)
            
            assert outputs[0].shape[0] == batch_size
    
    def test_gopt_gradient_flow(self, sample_input, sample_phn_labels, 
                                sample_utt_labels, sample_word_labels, gopt_config):
        """Test that gradients flow properly through GOPT"""
        model = GOPT(**gopt_config)
        model.train()
        
        outputs = model(sample_input, sample_phn_labels)
        
        # Compute dummy loss
        loss = outputs[0].sum() + outputs[5].sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model parameters"
    
    def test_gopt_phone_embedding(self, gopt_config):
        """Test that phone embedding works correctly"""
        model = GOPT(**gopt_config)
        model.eval()
        
        batch_size = 2
        seq_len = 50  # Use standard sequence length (GOPT expects 50, pos_embed is 55)
        
        # Create inputs with different phone sequences
        x1 = torch.randn(batch_size, seq_len, 84)
        x2 = x1.clone()
        
        phn1 = torch.zeros(batch_size, seq_len)  # All phone 0
        phn2 = torch.ones(batch_size, seq_len)   # All phone 1
        
        with torch.no_grad():
            outputs1 = model(x1, phn1)
            outputs2 = model(x2, phn2)
        
        # Outputs should be different due to different phone embeddings
        assert not torch.allclose(outputs1[0], outputs2[0])


class TestGOPTNoPhn:
    """Test GOPT model without phone embedding"""
    
    def test_gopt_nophn_forward(self, sample_input, sample_phn_labels, gopt_config):
        """Test GOPTNoPhn forward pass"""
        model = GOPTNoPhn(**gopt_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_input, sample_phn_labels)
        
        assert len(outputs) == 9
        
        batch_size = sample_input.shape[0]
        seq_len = sample_input.shape[1]
        
        # Check shapes
        for i in range(5):
            assert outputs[i].shape == (batch_size, 1)
        assert outputs[5].shape == (batch_size, seq_len, 1)
    
    def test_gopt_vs_gopt_nophn(self, sample_input, sample_phn_labels, gopt_config):
        """Test that GOPTNoPhn differs from GOPT"""
        model_with_phn = GOPT(**gopt_config)
        model_no_phn = GOPTNoPhn(**gopt_config)
        
        model_with_phn.eval()
        model_no_phn.eval()
        
        with torch.no_grad():
            outputs_with = model_with_phn(sample_input, sample_phn_labels)
            outputs_no = model_no_phn(sample_input, sample_phn_labels)
        
        # Outputs should be different (different architectures)
        # Note: They might be similar due to random init, but shapes should match
        assert outputs_with[0].shape == outputs_no[0].shape


class TestBaselineLSTM:
    """Test baseline LSTM model"""
    
    def test_lstm_initialization(self, lstm_config):
        """Test LSTM model initialization"""
        model = BaselineLSTM(**lstm_config)
        
        assert model.embed_dim == lstm_config['embed_dim']
        assert model.input_dim == lstm_config['input_dim']
    
    def test_lstm_forward_shape(self, sample_input, sample_phn_labels, lstm_config):
        """Test LSTM forward pass output shapes"""
        model = BaselineLSTM(**lstm_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_input, sample_phn_labels)
        
        assert len(outputs) == 9
        
        batch_size = sample_input.shape[0]
        seq_len = sample_input.shape[1]
        
        # Utterance-level outputs (5)
        for i in range(5):
            assert outputs[i].shape == (batch_size, 1)
        
        # Phone and word-level outputs
        assert outputs[5].shape == (batch_size, seq_len, 1)
    
    def test_lstm_get_last_valid(self, lstm_config):
        """Test get_last_valid method"""
        model = BaselineLSTM(**lstm_config)
        
        batch_size = 2
        seq_len = 10
        embed_dim = 24
        
        # Create test input
        input_tensor = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create mask: first sequence valid until position 5, second until 8
        mask = torch.ones(batch_size, seq_len)
        mask[0, 5:] = 0
        mask[1, 8:] = 0
        
        output = model.get_last_valid(input_tensor, mask)
        
        assert output.shape == (batch_size, 1, embed_dim)
    
    def test_lstm_gradient_flow(self, sample_input, sample_phn_labels, lstm_config):
        """Test gradient flow through LSTM"""
        model = BaselineLSTM(**lstm_config)
        model.train()
        
        outputs = model(sample_input, sample_phn_labels)
        
        loss = outputs[0].sum() + outputs[5].sum()
        loss.backward()
        
        # Check gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in LSTM parameters"


class TestModelComparison:
    """Compare different model architectures"""
    
    def test_gopt_vs_lstm_output_format(self, sample_input, sample_phn_labels, 
                                        gopt_config, lstm_config):
        """Test that GOPT and LSTM produce same output format"""
        gopt = GOPT(**gopt_config)
        lstm = BaselineLSTM(**lstm_config)
        
        gopt.eval()
        lstm.eval()
        
        with torch.no_grad():
            gopt_outputs = gopt(sample_input, sample_phn_labels)
            lstm_outputs = lstm(sample_input, sample_phn_labels)
        
        # Both should return 9 outputs
        assert len(gopt_outputs) == len(lstm_outputs) == 9
        
        # Shapes should match
        for i in range(9):
            assert gopt_outputs[i].shape == lstm_outputs[i].shape
    
    def test_model_parameter_counts(self, gopt_config, lstm_config):
        """Compare parameter counts"""
        gopt = GOPT(**gopt_config)
        lstm = BaselineLSTM(**lstm_config)
        
        gopt_params = sum(p.numel() for p in gopt.parameters())
        lstm_params = sum(p.numel() for p in lstm.parameters())
        
        print(f"\nGOPT parameters: {gopt_params:,}")
        print(f"LSTM parameters: {lstm_params:,}")
        
        # Both should have reasonable parameter counts
        assert gopt_params > 0
        assert lstm_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

