#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper ASR Service for GOPT Web Interface

Provides automatic speech recognition using OpenAI Whisper model.
Designed for server-side transcription of audio files.

Usage:
    from whisper_asr import WhisperASR
    
    asr = WhisperASR(model_size='base')
    result = asr.transcribe('audio.wav')
    print(result['text'])
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class WhisperASR:
    """
    Whisper Automatic Speech Recognition Service
    
    Provides speech-to-text transcription using OpenAI Whisper model.
    Optimized for English pronunciation assessment use case.
    """
    
    def __init__(self, model_size: str = 'base', device: Optional[str] = None):
        """
        Initialize Whisper ASR service.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       Default 'base' provides good balance of speed and accuracy (~150MB)
            device: Device to run model on ('cpu', 'cuda'). Auto-detect if None.
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._model_loaded = False
        
        logger.info(f"Initializing WhisperASR with model size: {model_size}")
    
    def _load_model(self):
        """
        Load Whisper model (lazy loading).
        Model is loaded on first transcribe() call to save memory.
        """
        if self._model_loaded:
            return
        
        try:
            import whisper
            
            logger.info(f"Loading Whisper {self.model_size} model... (this may take a moment)")
            start_time = time.time()
            
            # Load model with specified device
            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )
            
            load_time = time.time() - start_time
            self._model_loaded = True
            
            logger.info(f"Whisper model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model device: {self.model.device}")
            
        except ImportError:
            logger.error("Whisper package not installed. Please install: pip install openai-whisper")
            raise RuntimeError("Whisper not installed. Install with: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = 'en',
        task: str = 'transcribe',
        **kwargs
    ) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)
            language: Language code (default 'en' for English)
            task: 'transcribe' or 'translate' (default 'transcribe')
            **kwargs: Additional Whisper transcribe parameters
        
        Returns:
            Dictionary containing:
                - text: Transcribed text (uppercase, no punctuation)
                - raw_text: Original transcription with punctuation
                - language: Detected language
                - duration: Audio duration in seconds
                - processing_time: Time taken for transcription
                - segments: Detailed segment information (if available)
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        # Validate audio file
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio: {audio_path.name}")
        start_time = time.time()
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                fp16=False,  # Use FP32 for CPU compatibility
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            # Extract transcription text
            raw_text = result.get('text', '').strip()
            
            # Process text for GOPT (uppercase, remove punctuation)
            processed_text = self._process_text_for_gopt(raw_text)
            
            # Prepare response
            response = {
                'text': processed_text,
                'raw_text': raw_text,
                'language': result.get('language', language),
                'duration': result.get('duration', 0.0),
                'processing_time': processing_time,
                'segments': result.get('segments', [])
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            logger.info(f"Raw text: {raw_text}")
            logger.info(f"Processed text: {processed_text}")
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _process_text_for_gopt(self, text: str) -> str:
        """
        Process transcribed text for GOPT evaluation.
        
        Converts to uppercase and removes punctuation.
        Handles common contractions.
        
        Args:
            text: Raw transcription text
        
        Returns:
            Processed text (uppercase, no punctuation)
        """
        import re
        import string
        
        # Convert to uppercase
        text = text.upper()
        
        # Handle common contractions (expand them)
        contractions = {
            "I'M": "I AM",
            "YOU'RE": "YOU ARE",
            "HE'S": "HE IS",
            "SHE'S": "SHE IS",
            "IT'S": "IT IS",
            "WE'RE": "WE ARE",
            "THEY'RE": "THEY ARE",
            "ISN'T": "IS NOT",
            "AREN'T": "ARE NOT",
            "WASN'T": "WAS NOT",
            "WEREN'T": "WERE NOT",
            "DON'T": "DO NOT",
            "DOESN'T": "DOES NOT",
            "DIDN'T": "DID NOT",
            "WON'T": "WILL NOT",
            "CAN'T": "CAN NOT",
            "CANNOT": "CAN NOT",
            "I'VE": "I HAVE",
            "YOU'VE": "YOU HAVE",
            "WE'VE": "WE HAVE",
            "THEY'VE": "THEY HAVE",
            "I'LL": "I WILL",
            "YOU'LL": "YOU WILL",
            "HE'LL": "HE WILL",
            "SHE'LL": "SHE WILL",
            "WE'LL": "WE WILL",
            "THEY'LL": "THEY WILL",
        }
        
        # Replace contractions
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace (collapse multiple spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_available(self) -> bool:
        """
        Check if Whisper ASR is available and ready to use.
        
        Returns:
            True if Whisper can be imported and model can be loaded
        """
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'model_loaded': self._model_loaded,
            'device': str(self.model.device) if self._model_loaded else 'not loaded',
            'available': self.is_available()
        }


# Singleton instance for reuse across requests
_whisper_instance: Optional[WhisperASR] = None


def get_whisper_instance(model_size: str = 'base') -> WhisperASR:
    """
    Get or create singleton Whisper ASR instance.
    
    This ensures the model is loaded only once and reused across requests,
    significantly improving performance.
    
    Args:
        model_size: Whisper model size (default 'base')
    
    Returns:
        WhisperASR instance
    """
    global _whisper_instance
    
    if _whisper_instance is None:
        _whisper_instance = WhisperASR(model_size=model_size)
        logger.info("Created new WhisperASR singleton instance")
    
    return _whisper_instance


# Main function for CLI testing
def main():
    """
    Command-line interface for testing Whisper ASR.
    
    Usage:
        python whisper_asr.py <audio_file> [model_size]
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python whisper_asr.py <audio_file> [model_size]")
        print("Example: python whisper_asr.py audio.wav base")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else 'base'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create ASR instance
    print(f"\n{'='*60}")
    print(f"Whisper ASR Test")
    print(f"{'='*60}")
    print(f"Audio file: {audio_file}")
    print(f"Model size: {model_size}")
    print(f"{'='*60}\n")
    
    asr = WhisperASR(model_size=model_size)
    
    # Check availability
    if not asr.is_available():
        print("ERROR: Whisper is not available. Please install: pip install openai-whisper")
        sys.exit(1)
    
    # Transcribe
    try:
        result = asr.transcribe(audio_file)
        
        print(f"\n{'='*60}")
        print(f"Transcription Results")
        print(f"{'='*60}")
        print(f"Raw text:       {result['raw_text']}")
        print(f"Processed text: {result['text']}")
        print(f"Language:       {result['language']}")
        print(f"Duration:       {result['duration']:.2f}s")
        print(f"Processing:     {result['processing_time']:.2f}s")
        print(f"{'='*60}\n")
        
        # Show model info
        model_info = asr.get_model_info()
        print(f"Model info: {model_info}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

