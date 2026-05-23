#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOPT Inference - Command Line Interface

A user-friendly command-line interface for running GOPT pronunciation assessment.

Usage Examples:

1. Predict on test dataset samples:
   python inference_cli.py --samples 0 1 2 3 4

2. Evaluate on full test set:
   python inference_cli.py --evaluate

3. Use different model:
   python inference_cli.py --model pretrained_models/gopt_paiia/best_audio_model.pth \\
                          --dataset paiia --samples 0 1 2

4. Save results to file:
   python inference_cli.py --samples 0 1 2 --output results.json

5. Run in batch mode (all test samples):
   python inference_cli.py --batch --batch-size 64
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from inference_api import GOPTInference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GOPT Pronunciation Assessment - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on specific samples
  python inference_cli.py --samples 0 1 2 3 4
  
  # Evaluate on full test set
  python inference_cli.py --evaluate
  
  # Use different dataset
  python inference_cli.py --dataset paiia --samples 0 1 2
  
  # Save results to file
  python inference_cli.py --samples 0 1 2 --output results.json
  
  # Batch processing
  python inference_cli.py --batch --batch-size 64
        """
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='pretrained_models/gopt_librispeech/best_audio_model.pth',
        help='Path to pretrained model file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='gopt',
        choices=['gopt', 'gopt_nophn', 'lstm'],
        help='Type of model architecture'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='librispeech',
        choices=['librispeech', 'paiia', 'paiib'],
        help='Dataset used for training the model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: auto-detect)'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base directory containing datasets'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'test'],
        help='Dataset split to use'
    )
    
    # Inference mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--samples',
        type=int,
        nargs='+',
        metavar='IDX',
        help='Specific sample indices to predict on'
    )
    mode_group.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model on full dataset'
    )
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Run inference on all samples (batch mode)'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path to save results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed results for each sample'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output (only save to file)'
    )
    
    return parser.parse_args()


def print_summary_statistics(results):
    """Print summary statistics from results."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nNumber of samples: {len(results)}")
    
    # Calculate metrics for each aspect
    print("\nUtterance-level Scores:")
    print("-"*70)
    print(f"{'Aspect':<15} {'Mean Pred':>10} {'Mean GT':>10} {'MAE':>10} {'RMSE':>10}")
    print("-"*70)
    
    for aspect in ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']:
        predictions = [r['utterance_scores'][aspect]['predicted'] for r in results]
        ground_truth = [r['utterance_scores'][aspect]['ground_truth'] for r in results]
        errors = [r['utterance_scores'][aspect]['error'] for r in results]
        
        mean_pred = np.mean(predictions)
        mean_gt = np.mean(ground_truth)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        
        print(f"{aspect:<15} {mean_pred:>10.3f} {mean_gt:>10.3f} {mae:>10.3f} {rmse:>10.3f}")
    
    # Calculate correlation
    print("\nCorrelation (Pearson's r):")
    print("-"*70)
    for aspect in ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']:
        predictions = [r['utterance_scores'][aspect]['predicted'] for r in results]
        ground_truth = [r['utterance_scores'][aspect]['ground_truth'] for r in results]
        
        if len(predictions) > 1:
            pcc = np.corrcoef(predictions, ground_truth)[0, 1]
            print(f"  {aspect:<15} r = {pcc:.4f}")
    
    print("="*70)


def print_detailed_results(results, max_samples=None):
    """Print detailed results for each sample."""
    if max_samples is None:
        max_samples = len(results)
    
    for i, result in enumerate(results[:max_samples]):
        print("\n" + "="*70)
        print(f"Sample {result.get('sample_index', i)}")
        print("="*70)
        
        print("\nUtterance-level Scores (0-10 scale):")
        print("-"*70)
        print(f"{'Aspect':<15} {'Predicted':>10} {'Ground Truth':>12} {'Error':>10}")
        print("-"*70)
        
        for aspect, scores in result['utterance_scores'].items():
            print(f"{aspect:<15} {scores['predicted']:>10.2f} "
                  f"{scores['ground_truth']:>12.2f} {scores['error']:>10.2f}")
        
        # Print phone scores (first 5 valid)
        print("\nPhone-level Scores (first 5 valid phones):")
        print("-"*70)
        phone_scores = result['phone_scores']
        gt_phone_scores = result['ground_truth']['phone_scores']
        
        count = 0
        for j, (pred, gt) in enumerate(zip(phone_scores, gt_phone_scores)):
            if gt >= 0 and count < 5:  # Valid phone
                print(f"  Phone {j+1:2d}: Predicted={pred:.2f}, Ground Truth={gt:.2f}")
                count += 1


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Validate model file exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}", file=sys.stderr)
        print(f"\nAvailable pretrained models:", file=sys.stderr)
        models_dir = Path('pretrained_models')
        if models_dir.exists():
            for model_path in models_dir.glob('*/best_audio_model.pth'):
                print(f"  - {model_path}", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print("="*70)
        print("GOPT Pronunciation Assessment - Command Line Interface")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Model: {args.model}")
        print(f"  Model Type: {args.model_type}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Split: {args.split}")
        print(f"  Device: {args.device or 'auto'}")
    
    try:
        # Initialize inference engine
        if not args.quiet:
            print("\n📦 Loading model...")
        
        gopt = GOPTInference(
            model_path=args.model,
            model_type=args.model_type,
            dataset_name=args.dataset,
            device=args.device,
            data_dir=args.data_dir
        )
        
        if not args.quiet:
            print("✅ Model loaded successfully")
        
        # Run inference based on mode
        if args.evaluate:
            # Evaluation mode
            if not args.quiet:
                print("\n📊 Evaluating on full dataset...")
            
            metrics = gopt.evaluate_dataset(args.dataset, args.split)
            
            if not args.quiet:
                print("\n" + "="*70)
                print("EVALUATION RESULTS")
                print("="*70)
                print(f"\nDataset: {args.dataset} ({args.split} split)")
                print(f"Number of samples: {metrics['num_samples']}")
                
                print("\nUtterance-level Metrics:")
                print("-"*70)
                print(f"{'Aspect':<15} {'MAE':>10} {'MSE':>10} {'RMSE':>10} {'PCC':>10}")
                print("-"*70)
                
                for aspect, scores in metrics['utterance_metrics'].items():
                    print(f"{aspect:<15} {scores['mae']:>10.3f} {scores['mse']:>10.3f} "
                          f"{scores['rmse']:>10.3f} {scores['pcc']:>10.4f}")
                print("="*70)
            
            # Save metrics if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n💾 Metrics saved to {args.output}")
        
        else:
            # Prediction mode
            if args.samples is not None:
                sample_indices = args.samples
                if not args.quiet:
                    print(f"\n🔮 Running predictions on {len(sample_indices)} samples: {sample_indices}")
            elif args.batch:
                sample_indices = None
                if not args.quiet:
                    print(f"\n🔮 Running predictions on all samples (batch mode, batch_size={args.batch_size})")
            else:
                # Default: predict on first 5 samples
                sample_indices = [0, 1, 2, 3, 4]
                if not args.quiet:
                    print(f"\n🔮 Running predictions on first 5 samples (default)")
            
            results = gopt.predict_from_dataset(
                dataset_name=args.dataset,
                split=args.split,
                sample_indices=sample_indices,
                batch_size=args.batch_size
            )
            
            if not args.quiet:
                print(f"✅ Predictions complete ({len(results)} samples)")
                
                # Print results
                if args.verbose:
                    print_detailed_results(results)
                else:
                    print_summary_statistics(results)
            
            # Save results if output specified
            if args.output:
                gopt.save_results(results, args.output)
                if not args.quiet:
                    print(f"\n💾 Results saved to {args.output}")
        
        if not args.quiet:
            print("\n✨ Done!")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print("\nPlease ensure the data files are present in the data directory.", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

