#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOPT Inference - REST API Server

A FastAPI-based REST API server for GOPT pronunciation assessment.
Provides web service endpoints for model inference.

Usage:
    # Start the server
    python inference_server.py
    
    # Or with custom configuration
    python inference_server.py --host 0.0.0.0 --port 8000 --model-path path/to/model.pth
    
    # Access API documentation
    http://localhost:8000/docs

API Endpoints:
    GET  /              - Server information
    GET  /health        - Health check
    POST /predict       - Run prediction on provided features
    POST /predict/sample - Run prediction on dataset sample
    POST /evaluate      - Evaluate model on dataset

Installation:
    pip install fastapi uvicorn pydantic
"""

import os
import sys
import argparse
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Body
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Error: FastAPI dependencies not installed.")
    print("Please install: pip install fastapi uvicorn pydantic")
    sys.exit(1)

from inference_api import GOPTInference

# ==================== Pydantic Models ====================

class ServerInfo(BaseModel):
    """Server information."""
    name: str = "GOPT Pronunciation Assessment API"
    version: str = "1.0.0"
    status: str = "running"
    model_loaded: bool
    model_path: str
    dataset: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: List[List[List[float]]] = Field(
        ...,
        description="Input features of shape [batch_size, seq_len, feat_dim]"
    )
    phonemes: List[List[int]] = Field(
        ...,
        description="Phoneme labels of shape [batch_size, seq_len]"
    )
    normalize: bool = Field(
        False,
        description="Whether to normalize features (set False if already normalized)"
    )
    denormalize_scores: bool = Field(
        True,
        description="Whether to denormalize scores to 0-10 range"
    )


class PredictSampleRequest(BaseModel):
    """Request model for predicting on dataset samples."""
    dataset: Optional[str] = Field(
        None,
        description="Dataset name (librispeech, paiia, paiib). Default: use server's dataset"
    )
    split: str = Field(
        "test",
        description="Dataset split (train or test)"
    )
    sample_indices: List[int] = Field(
        ...,
        description="List of sample indices to predict on"
    )
    batch_size: int = Field(
        32,
        description="Batch size for inference"
    )


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""
    dataset: Optional[str] = Field(
        None,
        description="Dataset name (librispeech, paiia, paiib). Default: use server's dataset"
    )
    split: str = Field(
        "test",
        description="Dataset split (train or test)"
    )


class UtteranceScore(BaseModel):
    """Utterance-level score."""
    predicted: float
    ground_truth: Optional[float] = None
    error: Optional[float] = None


class PredictResponse(BaseModel):
    """Response model for prediction."""
    success: bool
    num_samples: int
    utterance_scores: Dict[str, Union[float, Dict[str, float]]]
    phone_scores: Optional[List[float]] = None
    word_scores: Optional[Dict] = None


class EvaluateResponse(BaseModel):
    """Response model for evaluation."""
    success: bool
    num_samples: int
    utterance_metrics: Dict[str, Dict[str, float]]


# ==================== FastAPI Application ====================

class GOPTServer:
    """GOPT Inference Server."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'gopt',
        dataset_name: str = 'librispeech',
        device: Optional[str] = None,
        data_dir: str = 'data'
    ):
        """Initialize the server."""
        self.model_path = model_path
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.device = device
        self.data_dir = data_dir
        
        # Initialize inference engine
        print(f"Loading GOPT model from {model_path}...")
        try:
            self.inference = GOPTInference(
                model_path=model_path,
                model_type=model_type,
                dataset_name=dataset_name,
                device=device,
                data_dir=data_dir
            )
            self.model_loaded = True
            print(f"[OK] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            self.model_loaded = False
            self.inference = None
        
        # Create FastAPI app
        self.app = FastAPI(
            title="GOPT Pronunciation Assessment API",
            description="REST API for GOPT-based pronunciation assessment",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/", response_model=ServerInfo)
        async def root():
            """Get server information."""
            return ServerInfo(
                model_loaded=self.model_loaded,
                model_path=self.model_path,
                dataset=self.dataset_name
            )
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(model_loaded=self.model_loaded)
        
        @self.app.post("/predict")
        async def predict(request: PredictRequest):
            """
            Run prediction on provided features.
            
            This endpoint accepts preprocessed features and phoneme labels,
            and returns pronunciation assessment scores.
            """
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Convert to numpy arrays
                features = np.array(request.features, dtype=np.float32)
                phonemes = np.array(request.phonemes, dtype=np.int64)
                
                # Validate shapes
                if features.ndim != 3:
                    raise ValueError(f"Features must be 3D (batch_size, seq_len, feat_dim), got shape {features.shape}")
                if phonemes.ndim != 2:
                    raise ValueError(f"Phonemes must be 2D (batch_size, seq_len), got shape {phonemes.shape}")
                if features.shape[0] != phonemes.shape[0]:
                    raise ValueError(f"Batch size mismatch: features={features.shape[0]}, phonemes={phonemes.shape[0]}")
                
                # Run prediction
                results = self.inference.predict(
                    features=features,
                    phonemes=phonemes,
                    normalize=request.normalize
                )
                
                # Format response
                response = {
                    'success': True,
                    'num_samples': features.shape[0],
                    'utterance_scores': {},
                    'phone_scores': results['phone_scores'].tolist(),
                    'word_scores': {
                        k: v.tolist() for k, v in results['word_scores'].items()
                    }
                }
                
                # Extract utterance scores
                for key, value in results['utterance_scores'].items():
                    if request.denormalize_scores:
                        response['utterance_scores'][key] = (value * 5).tolist()
                    else:
                        response['utterance_scores'][key] = value.tolist()
                
                return JSONResponse(content=response)
            
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/predict/sample")
        async def predict_sample(request: PredictSampleRequest):
            """
            Run prediction on samples from a dataset.
            
            This endpoint loads samples from the specified dataset and runs prediction.
            """
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                results = self.inference.predict_from_dataset(
                    dataset_name=request.dataset,
                    split=request.split,
                    sample_indices=request.sample_indices,
                    batch_size=request.batch_size
                )
                
                return JSONResponse(content={
                    'success': True,
                    'num_samples': len(results),
                    'results': results
                })
            
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/evaluate", response_model=EvaluateResponse)
        async def evaluate(request: EvaluateRequest):
            """
            Evaluate model on a dataset.
            
            This endpoint evaluates the model on the specified dataset and returns
            comprehensive metrics.
            """
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                metrics = self.inference.evaluate_dataset(
                    dataset_name=request.dataset,
                    split=request.split
                )
                
                return EvaluateResponse(
                    success=True,
                    num_samples=metrics['num_samples'],
                    utterance_metrics=metrics['utterance_metrics']
                )
            
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server."""
        print(f"\n{'='*70}")
        print("GOPT Inference Server")
        print(f"{'='*70}")
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Model loaded: {self.model_loaded}")
        print(f"\nStarting server at http://{host}:{port}")
        print(f"API documentation: http://{host}:{port}/docs")
        print(f"{'='*70}\n")
        
        uvicorn.run(self.app, host=host, port=port)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GOPT Inference REST API Server'
    )
    
    parser.add_argument(
        '--model-path',
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
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base directory containing datasets'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate model file
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Error: Model file not found: {args.model_path}", file=sys.stderr)
        print(f"\nAvailable pretrained models:", file=sys.stderr)
        models_dir = Path('pretrained_models')
        if models_dir.exists():
            for model_path in models_dir.glob('*/best_audio_model.pth'):
                print(f"  - {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create and run server
    server = GOPTServer(
        model_path=args.model_path,
        model_type=args.model_type,
        dataset_name=args.dataset,
        device=args.device,
        data_dir=args.data_dir
    )
    
    server.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()

