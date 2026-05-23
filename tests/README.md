# GOPT Unit Tests

Comprehensive unit test suite for the GOPT (Goodness of Pronunciation with Transformer) project.

## Overview

This test suite provides comprehensive coverage of:
- **Model Components**: GOPT, BaselineLSTM, Attention, MLP, Transformer blocks
- **Utility Functions**: Positional encoding, normalization, correlation metrics
- **Data Loading**: NumPy file loading, data validation, preprocessing

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared pytest fixtures
├── test_models.py           # Model architecture tests
├── test_utils.py            # Utility function tests
├── test_data_loading.py     # Data loading and preprocessing tests
├── run_tests.py             # Test runner script
└── README.md                # This file
```

## Running Tests

### Option 1: Using pytest directly

```bash
# Activate virtual environment
.\venv-gopt-new\Scripts\Activate.ps1  # Windows
# or: source venv-gopt-new/bin/activate  # Linux/Mac

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test class
pytest tests/test_models.py::TestGOPT -v

# Run specific test
pytest tests/test_models.py::TestGOPT::test_gopt_forward_shape -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Option 2: Using the test runner script

```bash
python tests/run_tests.py
```

### Option 3: Run individual test files

```bash
python tests/test_models.py
python tests/test_utils.py
python tests/test_data_loading.py
```

## Test Categories

### 1. Model Tests (`test_models.py`)

**TestSinusoidEncoding**
- Sinusoidal position encoding shape and values
- Different encoding dimensions

**TestAttention**
- Attention mechanism forward pass
- Multiple attention heads
- Dropout behavior

**TestMlp**
- MLP forward pass
- Custom output dimensions
- Different activation functions

**TestBlock**
- Transformer block forward pass
- Residual connections

**TestGOPT**
- Model initialization
- Forward pass output shapes
- Gradient flow
- Phone embedding functionality
- Different batch sizes

**TestGOPTNoPhn**
- Model without phone embedding
- Comparison with standard GOPT

**TestBaselineLSTM**
- LSTM model initialization
- Forward pass
- get_last_valid method
- Gradient flow

**TestModelComparison**
- Output format compatibility
- Parameter counts

### 2. Utility Tests (`test_utils.py`)

**TestDataNormalization**
- Score normalization (0-10 to 0-2)
- Score denormalization
- Feature standardization (z-score)

**TestPhonemeEncoding**
- One-hot encoding of phonemes
- Phone index range validation

**TestPositionalEncoding**
- Positional encoding properties
- Periodicity

**TestLossCalculation**
- MSE loss
- Weighted loss combination

**TestCorrelationMetrics**
- Pearson correlation
- Correlation range validation

**TestBatchProcessing**
- Padding mask creation
- Sequence length calculation

**TestDataAugmentation**
- Noise addition
- Zero-mean noise

**TestFileOperations**
- NumPy save/load
- PyTorch save/load

**TestInputValidation**
- Shape validation
- NaN detection
- Value range validation

### 3. Data Loading Tests (`test_data_loading.py`)

**TestDataLoading**
- Loading NumPy files
- Multiple file loading
- Shape validation

**TestGoPDataset**
- Dataset length
- Item retrieval
- Iteration

**TestDataLoader**
- Batch size handling
- Shuffling
- Drop last batch

**TestDataPreprocessing**
- Feature normalization
- Label scaling
- Phone label offset

**TestDataAugmentation**
- Gaussian noise addition
- Noise scale effects

**TestDataValidation**
- NaN detection
- Infinity detection
- Value range checking

**TestBatchCollation**
- Tensor stacking
- Sequence padding

**TestDataStatistics**
- Mean and std computation
- Per-feature statistics

## Fixtures (from conftest.py)

The following fixtures are available in all tests:

- `device`: PyTorch device (CPU or CUDA)
- `batch_size`: Standard batch size (4)
- `seq_len`: Standard sequence length (50)
- `feat_dim`: Feature dimension (84)
- `embed_dim`: Embedding dimension (24)
- `num_heads`: Number of attention heads (1)
- `depth`: Model depth (3)
- `sample_input`: Random input tensor
- `sample_phn_labels`: Random phone labels
- `sample_utt_labels`: Random utterance labels
- `sample_word_labels`: Random word labels
- `gopt_config`: GOPT model configuration
- `lstm_config`: LSTM model configuration

## Requirements

The tests require the following packages (already in requirements.txt):
- pytest >= 7.0.0
- pytest-cov >= 3.0.0 (optional, for coverage)
- torch == 1.12.1
- numpy == 1.24.4

## Adding New Tests

### 1. Add test to existing file

```python
class TestNewFeature:
    """Test new feature"""
    
    def test_feature_works(self):
        """Test that feature works correctly"""
        result = new_feature()
        assert result is not None
```

### 2. Create new test file

```python
# tests/test_new_module.py
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from new_module import new_function

class TestNewModule:
    def test_new_function(self):
        assert new_function() == expected_value
```

## Test Coverage

To generate test coverage report:

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

Open `htmlcov/index.html` in browser to view detailed coverage.

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=src
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running from project root directory

### CUDA Errors

Tests will automatically use CPU if CUDA is not available. To force CPU:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/ -v
```

### Slow Tests

To run only fast tests (skip slow integration tests):

```bash
pytest tests/ -v -m "not slow"
```

## Contributing

When adding new functionality:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain test coverage above 80%
4. Document new tests in this README

## License

Same as parent project (see LICENSE file in project root).

