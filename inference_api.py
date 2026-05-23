#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOPT Inference API - Core Module

This module provides a clean Python API for GOPT pronunciation assessment.
It wraps the model loading, preprocessing, and inference logic into an easy-to-use interface.

Usage:
    from inference_api import GOPTInference
    
    # Initialize the inference engine
    gopt = GOPTInference(model_path='pretrained_models/gopt_librispeech/best_audio_model.pth')
    
    # Load preprocessed features
    results = gopt.predict_from_features(features, phonemes)
    
    # Or predict from dataset
    results = gopt.predict_from_dataset('librispeech', sample_indices=[0, 1, 2])
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import json
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import GOPT, GOPTNoPhn, BaselineLSTM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GOPTInference:
    """
    GOPT Inference API
    
    A high-level interface for GOPT pronunciation assessment model inference.
    """
    
    # Model architecture configurations
    MODEL_CONFIGS = {
        'librispeech': {
            'input_dim': 84,
            'embed_dim': 24,
            'num_heads': 1,
            'depth': 3,
            'norm_mean': 3.203,
            'norm_std': 4.045,
            'dataset_dir': 'seq_data_librispeech'
        },
        'paiia': {
            'input_dim': 86,
            'embed_dim': 24,
            'num_heads': 1,
            'depth': 3,
            'norm_mean': -0.652,
            'norm_std': 9.737,
            'dataset_dir': 'seq_data_paiia'
        },
        'paiib': {
            'input_dim': 88,
            'embed_dim': 24,
            'num_heads': 1,
            'depth': 3,
            'norm_mean': -0.516,
            'norm_std': 9.247,
            'dataset_dir': 'seq_data_paiib'
        }
    }
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'gopt',
        dataset_name: str = 'librispeech',
        device: Optional[str] = None,
        data_dir: str = 'data'
    ):
        """
        Initialize the GOPT inference engine.
        
        Args:
            model_path: Path to the pretrained model (.pth file)
            model_type: Type of model ('gopt', 'gopt_nophn', 'lstm')
            dataset_name: Dataset used for training ('librispeech', 'paiia', 'paiib')
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            data_dir: Base directory containing the data
        """
        self.model_path = model_path
        self.model_type = model_type
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        
        # Validate dataset name
        if self.dataset_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(self.MODEL_CONFIGS.keys())}")
        
        # Get configuration
        self.config = self.MODEL_CONFIGS[self.dataset_name]
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        logger.info(f"Model loaded successfully from {model_path}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the model from checkpoint."""
        # Create model architecture
        if self.model_type == 'gopt':
            model = GOPT(
                embed_dim=self.config['embed_dim'],
                num_heads=self.config['num_heads'],
                depth=self.config['depth'],
                input_dim=self.config['input_dim']
            )
        elif self.model_type == 'gopt_nophn':
            model = GOPTNoPhn(
                embed_dim=self.config['embed_dim'],
                num_heads=self.config['num_heads'],
                depth=self.config['depth'],
                input_dim=self.config['input_dim']
            )
        elif self.model_type == 'lstm':
            model = BaselineLSTM(
                embed_dim=self.config['embed_dim'],
                depth=self.config['depth'],
                input_dim=self.config['input_dim']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Wrap with DataParallel (model was trained with DataParallel)
        model = torch.nn.DataParallel(model)
        
        # Load checkpoint
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint, strict=True)
        
        # Set to evaluation mode
        model.eval()
        model.to(self.device)
        
        return model
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using dataset-specific mean and std.
        
        Args:
            features: Input features of shape [batch_size, seq_len, feat_dim]
            
        Returns:
            Normalized features
        """
        norm_mean = self.config['norm_mean']
        norm_std = self.config['norm_std']
        
        # Create normalized features
        norm_features = np.zeros_like(features)
        
        # Normalize only valid tokens (non-zero)
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                if features[i, j, 0] != 0:
                    norm_features[i, j, :] = (features[i, j, :] - norm_mean) / norm_std
                else:
                    break
        
        return norm_features
    
    def predict(
        self,
        features: np.ndarray,
        phonemes: np.ndarray,
        normalize: bool = False
    ) -> Dict:
        """
        Run inference on preprocessed features.
        
        Args:
            features: Input features of shape [batch_size, seq_len, feat_dim]
            phonemes: Phoneme labels of shape [batch_size, seq_len]
            normalize: Whether to normalize features (set False if already normalized)
            
        Returns:
            Dictionary containing predictions:
                - utterance_scores: Dict of utterance-level scores (accuracy, completeness, etc.)
                - phone_scores: Array of phone-level scores
                - word_scores: Dict of word-level scores
        """
        # Normalize features if requested
        if normalize:
            features = self._normalize_features(features)
        
        # Convert to tensors
        feat_tensor = torch.FloatTensor(features).to(self.device)
        phn_tensor = torch.LongTensor(phonemes).to(self.device)
        
        # Run inference
        with torch.no_grad():
            u1, u2, u3, u4, u5, p, w1, w2, w3 = self.model(feat_tensor, phn_tensor)
        
        # Move to CPU and convert to numpy
        u1 = u1.cpu().numpy()
        u2 = u2.cpu().numpy()
        u3 = u3.cpu().numpy()
        u4 = u4.cpu().numpy()
        u5 = u5.cpu().numpy()
        p = p.cpu().numpy()
        w1 = w1.cpu().numpy()
        w2 = w2.cpu().numpy()
        w3 = w3.cpu().numpy()
        
        # Format results
        results = {
            'utterance_scores': {
                'accuracy': u1,      # Shape: [batch_size, 1]
                'completeness': u2,
                'fluency': u3,
                'prosodic': u4,
                'total': u5
            },
            'phone_scores': p,       # Shape: [batch_size, seq_len, 1]
            'word_scores': {
                'accuracy': w1,      # Shape: [batch_size, seq_len, 1]
                'stress': w2,
                'total': w3
            }
        }
        
        return results
    
    def predict_single(
        self,
        features: np.ndarray,
        phonemes: np.ndarray,
        normalize: bool = False,
        denormalize_scores: bool = True
    ) -> Dict:
        """
        Run inference on a single sample.
        
        Args:
            features: Input features of shape [seq_len, feat_dim]
            phonemes: Phoneme labels of shape [seq_len]
            normalize: Whether to normalize features
            denormalize_scores: Whether to denormalize scores to 0-10 range
            
        Returns:
            Dictionary containing predictions for a single sample
        """
        # Add batch dimension
        features = features[np.newaxis, :]  # [1, seq_len, feat_dim]
        phonemes = phonemes[np.newaxis, :]  # [1, seq_len]
        
        # Run prediction
        results = self.predict(features, phonemes, normalize=normalize)
        
        # Extract single sample results
        single_results = {
            'utterance_scores': {},
            'phone_scores': results['phone_scores'][0].squeeze(),  # [seq_len]
            'word_scores': {}
        }
        
        # Extract utterance scores
        for key, value in results['utterance_scores'].items():
            score = value[0, 0]
            if denormalize_scores:
                score = score * 5  # Convert from 0-2 to 0-10 range
            single_results['utterance_scores'][key] = float(score)
        
        # Extract word scores
        for key, value in results['word_scores'].items():
            scores = value[0].squeeze()  # [seq_len]
            if denormalize_scores:
                scores = scores * 5
            single_results['word_scores'][key] = scores
        
        return single_results
    
    def load_dataset(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a dataset from disk.
        
        Args:
            dataset_name: Name of dataset (default: use self.dataset_name)
            split: 'train' or 'test'
            
        Returns:
            Tuple of (features, phn_labels, utt_labels, word_labels)
        """
        if dataset_name is None:
            dataset_name = self.dataset_name
        
        config = self.MODEL_CONFIGS[dataset_name]
        dataset_dir = self.data_dir / config['dataset_dir']
        
        prefix = 'tr' if split == 'train' else 'te'
        
        feat = np.load(dataset_dir / f'{prefix}_feat.npy')
        phn_label = np.load(dataset_dir / f'{prefix}_label_phn.npy')
        utt_label = np.load(dataset_dir / f'{prefix}_label_utt.npy')
        word_label = np.load(dataset_dir / f'{prefix}_label_word.npy')
        
        logger.info(f"Loaded {split} dataset: {dataset_name}")
        logger.info(f"  Features: {feat.shape}")
        logger.info(f"  Phoneme labels: {phn_label.shape}")
        logger.info(f"  Utterance labels: {utt_label.shape}")
        logger.info(f"  Word labels: {word_label.shape}")
        
        return feat, phn_label, utt_label, word_label
    
    def predict_from_dataset(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        sample_indices: Optional[List[int]] = None,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Run inference on samples from a dataset.
        
        Args:
            dataset_name: Name of dataset (default: use self.dataset_name)
            split: 'train' or 'test'
            sample_indices: List of sample indices to predict (default: all)
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries, one per sample
        """
        # Load dataset
        feat, phn_label, utt_label, word_label = self.load_dataset(dataset_name, split)
        
        # Select samples
        if sample_indices is not None:
            feat = feat[sample_indices]
            phn_label = phn_label[sample_indices]
            utt_label = utt_label[sample_indices]
            word_label = word_label[sample_indices]
        
        # Run predictions in batches
        all_results = []
        num_samples = len(feat)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_feat = feat[i:end_idx]
            batch_phn = phn_label[i:end_idx, :, 0]  # Extract phoneme IDs
            
            # Run prediction
            batch_results = self.predict(batch_feat, batch_phn, normalize=True)
            
            # Convert to list of individual results
            for j in range(end_idx - i):
                sample_result = {
                    'sample_index': i + j if sample_indices is None else sample_indices[i + j],
                    'utterance_scores': {},
                    'phone_scores': batch_results['phone_scores'][j].squeeze(),
                    'word_scores': {},
                    'ground_truth': {
                        'utterance_scores': utt_label[i + j],
                        'phone_scores': phn_label[i + j, :, 1],  # Phone scores
                        'word_scores': word_label[i + j, :, 0:3]
                    }
                }
                
                # Extract utterance scores (denormalize to 0-10 range)
                for k, key in enumerate(['accuracy', 'completeness', 'fluency', 'prosodic', 'total']):
                    pred_score = batch_results['utterance_scores'][key][j, 0] * 5
                    true_score = utt_label[i + j, k]
                    sample_result['utterance_scores'][key] = {
                        'predicted': float(pred_score),
                        'ground_truth': float(true_score),
                        'error': float(abs(pred_score - true_score))
                    }
                
                # Extract word scores
                for k, key in enumerate(['accuracy', 'stress', 'total']):
                    sample_result['word_scores'][key] = {
                        'predicted': batch_results['word_scores'][key][j].squeeze() * 5,
                        'ground_truth': word_label[i + j, :, k]
                    }
                
                all_results.append(sample_result)
        
        logger.info(f"Predicted {len(all_results)} samples")
        return all_results
    
    def evaluate_dataset(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test'
    ) -> Dict:
        """
        Evaluate the model on an entire dataset.
        
        Args:
            dataset_name: Name of dataset (default: use self.dataset_name)
            split: 'train' or 'test'
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.predict_from_dataset(dataset_name, split)
        
        # Calculate metrics
        metrics = {
            'num_samples': len(results),
            'utterance_metrics': {},
            'phone_metrics': {},
            'word_metrics': {}
        }
        
        # Utterance-level metrics
        for aspect in ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']:
            errors = [r['utterance_scores'][aspect]['error'] for r in results]
            predictions = [r['utterance_scores'][aspect]['predicted'] for r in results]
            ground_truth = [r['utterance_scores'][aspect]['ground_truth'] for r in results]
            
            mae = np.mean(errors)
            mse = np.mean(np.array(errors) ** 2)
            rmse = np.sqrt(mse)
            pcc = np.corrcoef(predictions, ground_truth)[0, 1] if len(predictions) > 1 else 0.0
            
            metrics['utterance_metrics'][aspect] = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'pcc': float(pcc)
            }
        
        logger.info(f"Evaluation complete on {len(results)} samples")
        return metrics
    
    def format_results_for_display(self, results: Dict, sample_index: int = 0) -> str:
        """
        Format prediction results for human-readable display.
        
        Args:
            results: Results dictionary from predict_from_dataset
            sample_index: Index in results list to display
            
        Returns:
            Formatted string
        """
        if isinstance(results, list):
            result = results[sample_index]
        else:
            result = results
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"GOPT Pronunciation Assessment Results")
        output.append(f"{'='*60}")
        
        if 'sample_index' in result:
            output.append(f"Sample Index: {result['sample_index']}")
        
        output.append(f"\nUtterance-level Scores (0-10 scale):")
        output.append(f"{'-'*60}")
        for aspect, scores in result['utterance_scores'].items():
            if isinstance(scores, dict):
                output.append(f"  {aspect.capitalize():15s}: {scores['predicted']:5.2f} "
                            f"(Ground Truth: {scores['ground_truth']:5.2f}, "
                            f"Error: {scores['error']:5.2f})")
            else:
                output.append(f"  {aspect.capitalize():15s}: {scores:5.2f}")
        
        output.append(f"\nPhone-level Scores (first 10 valid phones):")
        output.append(f"{'-'*60}")
        phone_scores = result['phone_scores']
        if 'ground_truth' in result:
            gt_phone_scores = result['ground_truth']['phone_scores']
            count = 0
            for i, score in enumerate(phone_scores):
                if gt_phone_scores[i] >= 0 and count < 10:  # Valid phone
                    output.append(f"  Phone {i+1:2d}: Predicted={score:.2f}, "
                                f"Ground Truth={gt_phone_scores[i]:.2f}")
                    count += 1
        else:
            for i, score in enumerate(phone_scores[:10]):
                if score != 0:
                    output.append(f"  Phone {i+1:2d}: {score:.2f}")
        
        output.append(f"{'='*60}\n")
        return '\n'.join(output)
    
    def save_results(self, results: Union[List[Dict], Dict], output_path: str):
        """
        Save prediction results to a JSON file.
        
        Args:
            results: Results dictionary or list of dictionaries
            output_path: Path to output JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Demo usage of the inference API."""
    print("GOPT Inference API - Demo")
    print("=" * 60)
    
    # Initialize inference engine
    model_path = 'pretrained_models/gopt_librispeech/best_audio_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please download the pretrained model first.")
        return
    
    gopt = GOPTInference(
        model_path=model_path,
        model_type='gopt',
        dataset_name='librispeech'
    )
    
    # Predict on a few samples from test set
    print("\nRunning predictions on 5 test samples...")
    results = gopt.predict_from_dataset(
        dataset_name='librispeech',
        split='test',
        sample_indices=[0, 1, 2, 3, 4]
    )
    
    # Display first result
    print(gopt.format_results_for_display(results, sample_index=0))
    
    # Calculate average metrics
    print("\nAverage Metrics across 5 samples:")
    print("-" * 60)
    for aspect in ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']:
        avg_error = np.mean([r['utterance_scores'][aspect]['error'] for r in results])
        print(f"  {aspect.capitalize():15s}: MAE = {avg_error:.3f}")
    
    # Save results
    output_path = 'inference_results.json'
    gopt.save_results(results, output_path)
    print(f"\nResults saved to {output_path}")
    
    # Evaluate on full test set
    print("\nEvaluating on full test set...")
    metrics = gopt.evaluate_dataset('librispeech', 'test')
    
    print("\nFull Test Set Evaluation:")
    print("-" * 60)
    print(f"Number of samples: {metrics['num_samples']}")
    print("\nUtterance-level Metrics:")
    for aspect, scores in metrics['utterance_metrics'].items():
        print(f"  {aspect.capitalize():15s}: MAE={scores['mae']:.3f}, "
              f"PCC={scores['pcc']:.3f}")


if __name__ == '__main__':
    main()

