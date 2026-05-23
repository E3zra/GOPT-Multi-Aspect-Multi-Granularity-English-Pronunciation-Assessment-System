#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOPT Web Interface Server

A FastAPI-based web server for uploading audio files and running pronunciation assessment
through Docker container integration.

Usage:
    python web_server.py --host 0.0.0.0 --port 8080
"""

import os
import sys
import json
import uuid
import asyncio
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import aiofiles
    import uvicorn
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install fastapi uvicorn aiofiles python-multipart")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for G2P helper and segmentation modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Try to import G2P helper for OOV detection
try:
    from utils.g2p_helper import G2PConverter
    G2P_AVAILABLE = True
    logger.info("G2P helper loaded successfully for OOV word detection")
except ImportError:
    G2P_AVAILABLE = False
    logger.warning("G2P helper not available. OOV detection will be limited.")

# Try to import segmentation modules for long audio support
try:
    from text_segmenter import TextSegmenter
    from audio_segmenter import AudioSegmenter
    from result_merger import ResultMerger
    SEGMENTATION_AVAILABLE = True
    logger.info("Segmentation modules loaded successfully for long audio support")
except ImportError as e:
    SEGMENTATION_AVAILABLE = False
    logger.warning(f"Segmentation modules not available: {e}")

# Try to import Whisper ASR for automatic speech recognition
try:
    from whisper_asr import WhisperASR, get_whisper_instance
    WHISPER_AVAILABLE = True
    logger.info("Whisper ASR loaded successfully for automatic speech recognition")
except ImportError as e:
    WHISPER_AVAILABLE = False
    logger.warning(f"Whisper ASR not available: {e}")

# Configuration
AUDIO_INPUT_DIR = Path("audio_input")
AUDIO_OUTPUT_DIR = Path("audio_output")
STATIC_DIR = Path("static")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
DOCKER_CONTAINER = "gopt-pipeline"
PROCESS_SCRIPT = "/workspace/gopt/process_custom_audio.sh"

# Ensure directories exist
AUDIO_INPUT_DIR.mkdir(exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Task storage (in-memory, for simple implementation)
# In production, use Redis or database
tasks: Dict[str, Dict] = {}

# Create FastAPI app
app = FastAPI(
    title="GOPT Web Interface",
    description="Web interface for GOPT pronunciation assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_docker_container():
    """Check if Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={DOCKER_CONTAINER}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return DOCKER_CONTAINER in result.stdout
    except Exception as e:
        logger.error(f"Error checking Docker container: {e}")
        return False


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {e}")
        return False


def load_librispeech_lexicon() -> set:
    """
    Load LibriSpeech lexicon words from the container.
    
    Returns:
        Set of uppercase words in the lexicon
    """
    try:
        # Try to load lexicon from Docker container
        result = subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "cat", 
             "/opt/kaldi/egs/gop_speechocean762/s5/librispeech-lexicon.txt"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            words = set()
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if parts:
                    words.add(parts[0].upper())
            logger.info(f"Loaded {len(words)} words from LibriSpeech lexicon")
            return words
        else:
            logger.warning("Failed to load LibriSpeech lexicon from container")
            return set()
    except Exception as e:
        logger.error(f"Error loading lexicon: {e}")
        return set()


def check_oov_words(transcript: str) -> Tuple[List[str], List[Dict]]:
    """
    Check transcript for out-of-vocabulary (OOV) words.
    
    Args:
        transcript: Input transcript text
        
    Returns:
        Tuple of (oov_words list, oov_details list with word info and G2P phonemes)
    """
    # Load lexicon (cache this in production)
    lexicon = load_librispeech_lexicon()
    
    if not lexicon:
        logger.warning("Lexicon not available, cannot check for OOV words")
        return [], []
    
    # Split transcript into words
    words = transcript.upper().strip().split()
    
    # Find OOV words
    oov_words = []
    oov_details = []
    
    for word in words:
        if word not in lexicon:
            oov_words.append(word)
            
            # Try to generate phonemes with G2P
            phonemes = []
            if G2P_AVAILABLE:
                try:
                    g2p = G2PConverter()
                    phonemes = g2p.convert(word)
                except Exception as e:
                    logger.error(f"G2P conversion failed for {word}: {e}")
            
            oov_details.append({
                "word": word,
                "g2p_available": G2P_AVAILABLE,
                "phonemes": phonemes if phonemes else None,
                "phonemes_text": " ".join(phonemes) if phonemes else "N/A"
            })
    
    return oov_words, oov_details


def get_audio_info(filepath: Path) -> Optional[Dict]:
    """
    Get audio file information using ffprobe.
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Dictionary with audio info or None if failed
    """
    try:
        # Copy file to container for analysis
        container_path = f"/workspace/audio_input/{filepath.name}"
        
        result = subprocess.run(
            [
                "docker", "exec", DOCKER_CONTAINER,
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_name,sample_rate,channels",
                "-of", "json",
                container_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            logger.warning(f"ffprobe failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return None


async def convert_audio(input_path: Path, output_path: Path, task_id: str) -> tuple:
    """
    Convert audio to 16kHz mono WAV format using FFmpeg in Docker container.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        task_id: Task ID for progress tracking
        
    Returns:
        (success: bool, message: str)
    """
    try:
        # Update task status
        if task_id in tasks:
            tasks[task_id]["stage"] = "Converting audio format..."
        
        # Paths in container
        container_input = f"/workspace/audio_input/{input_path.name}"
        container_output = f"/workspace/audio_input/{output_path.name}"
        
        logger.info(f"Converting audio: {input_path.name} -> {output_path.name}")
        
        # Run FFmpeg conversion in Docker container
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", DOCKER_CONTAINER,
            "ffmpeg",
            "-i", container_input,
            "-ar", "16000",           # Sample rate: 16kHz
            "-ac", "1",               # Channels: mono
            "-acodec", "pcm_s16le",   # Codec: PCM 16-bit
            "-y",                     # Overwrite output
            container_output,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Audio conversion successful: {output_path.name}")
            if task_id in tasks:
                tasks[task_id]["stage"] = "Audio converted successfully"
            return (True, "Conversion successful")
        else:
            error_msg = stderr.decode('utf-8', errors='replace')
            logger.error(f"FFmpeg conversion failed: {error_msg[:500]}")
            return (False, f"FFmpeg conversion failed. Please check audio file format.")
            
    except Exception as e:
        logger.error(f"Error during audio conversion: {e}")
        return (False, f"Conversion error: {str(e)}")


async def save_upload_file(upload_file: UploadFile, destination: Path) -> int:
    """Save uploaded file to destination and return file size."""
    file_size = 0
    async with aiofiles.open(destination, 'wb') as out_file:
        while content := await upload_file.read(1024 * 1024):  # Read 1MB at a time
            file_size += len(content)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB")
            await out_file.write(content)
    return file_size


async def process_single_segment(
    audio_filename: str,
    transcript: str,
    segment_task_id: str,
    segment_index: Optional[int] = None
) -> Optional[dict]:
    """
    Process a single audio segment.
    
    Args:
        audio_filename: Audio file name in audio_input/
        transcript: Transcript text
        segment_task_id: Task ID for this segment
        segment_index: Optional segment index for logging
        
    Returns:
        Result dictionary or None if failed
    """
    try:
        audio_path = f"/workspace/audio_input/{audio_filename}"
        
        command = [
            "docker", "exec", DOCKER_CONTAINER,
            "bash", PROCESS_SCRIPT,
            audio_path,
            transcript.upper(),
            segment_task_id
        ]
        
        logger.info(f"Processing segment {segment_index if segment_index else 'single'}: {transcript[:50]}...")
        
        # Run the process
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Read result file
            result_file = AUDIO_OUTPUT_DIR / f"{segment_task_id}_results.json"
            if result_file.exists():
                async with aiofiles.open(result_file, 'r') as f:
                    content = await f.read()
                    result = json.loads(content)
                return result
            else:
                logger.error(f"Result file not found for segment {segment_task_id}")
                return None
        else:
            logger.error(f"Segment processing failed: {stderr.decode('utf-8', errors='replace')[:500]}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing segment: {e}")
        return None


async def run_docker_process(task_id: str, audio_filename: str, transcript: str):
    """Run the processing script in Docker container asynchronously with auto-segmentation support."""
    try:
        # Update task status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["stage"] = "Analyzing transcript length..."
        
        # Check if segmentation is needed
        needs_segmentation = False
        text_segments = None
        
        if SEGMENTATION_AVAILABLE:
            try:
                segmenter = TextSegmenter(max_phonemes=45)
                text_segments = segmenter.segment_transcript(transcript)
                
                total_phonemes = sum(seg.phoneme_count for seg in text_segments)
                tasks[task_id]["total_words"] = sum(seg.word_count for seg in text_segments)
                tasks[task_id]["estimated_phonemes"] = total_phonemes
                
                if len(text_segments) > 1:
                    needs_segmentation = True
                    tasks[task_id]["num_segments"] = len(text_segments)
                    tasks[task_id]["stage"] = f"Long transcript detected, splitting into {len(text_segments)} segments..."
                    logger.info(f"Task {task_id}: Segmentation needed - {len(text_segments)} segments")
                else:
                    tasks[task_id]["stage"] = "Transcript fits in single segment..."
                    
            except Exception as e:
                logger.error(f"Error in segmentation detection: {e}")
                needs_segmentation = False
        
        # Process based on segmentation need
        if needs_segmentation and text_segments and len(text_segments) > 1:
            # Multi-segment processing
            await run_multi_segment_process(task_id, audio_filename, text_segments)
        else:
            # Single-segment processing (original flow)
            await run_single_segment_process(task_id, audio_filename, transcript)
            
    except Exception as e:
        logger.error(f"Error in Docker process for task {task_id}: {e}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["stage"] = "Error occurred"


async def run_single_segment_process(task_id: str, audio_filename: str, transcript: str):
    """Original single-segment processing flow."""
    try:
        tasks[task_id]["stage"] = "Running GOP extraction..."
        
        audio_path = f"/workspace/audio_input/{audio_filename}"
        
        command = [
            "docker", "exec", DOCKER_CONTAINER,
            "bash", PROCESS_SCRIPT,
            audio_path,
            transcript.upper(),
            task_id
        ]
        
        logger.info(f"Running command: {' '.join(command)}")
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        tasks[task_id]["process"] = process
        stdout, stderr = await process.communicate()
        
        tasks[task_id]["stdout"] = stdout.decode('utf-8', errors='replace')
        tasks[task_id]["stderr"] = stderr.decode('utf-8', errors='replace')
        tasks[task_id]["return_code"] = process.returncode
        
        if process.returncode == 0:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["stage"] = "Processing completed successfully"
            
            result_file = AUDIO_OUTPUT_DIR / f"{task_id}_results.json"
            if result_file.exists():
                tasks[task_id]["result_available"] = True
            else:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = "Result file not generated"
        else:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = f"Process failed with return code {process.returncode}"
            tasks[task_id]["stage"] = "Processing failed"
            
    except Exception as e:
        logger.error(f"Error in single segment process: {e}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)


async def run_multi_segment_process(task_id: str, audio_filename: str, text_segments: List):
    """Multi-segment processing with automatic audio splitting and result merging."""
    try:
        audio_path = AUDIO_INPUT_DIR / audio_filename
        
        # Step 1: Split audio
        tasks[task_id]["stage"] = f"Splitting audio into {len(text_segments)} segments..."
        logger.info(f"Task {task_id}: Splitting audio...")
        
        audio_segmenter = AudioSegmenter()
        
        try:
            audio_segments = audio_segmenter.split_audio_by_text_ratio(
                audio_path, text_segments
            )
            tasks[task_id]["stage"] = f"Audio split into {len(audio_segments)} segments"
            logger.info(f"Task {task_id}: Audio split successful")
        except Exception as e:
            logger.error(f"Audio splitting failed: {e}")
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = f"Audio splitting failed: {str(e)}"
            return
        
        # Step 2: Process each segment
        segment_results = []
        total_segments = len(audio_segments)
        
        for i, (audio_seg, text_seg) in enumerate(zip(audio_segments, text_segments)):
            segment_task_id = f"{task_id}_seg{i}"
            tasks[task_id]["stage"] = f"Processing segment {i+1}/{total_segments}..."
            tasks[task_id]["current_segment"] = i + 1
            
            logger.info(f"Task {task_id}: Processing segment {i+1}/{total_segments}")
            
            result = await process_single_segment(
                audio_seg.path.name,
                text_seg.text,
                segment_task_id,
                segment_index=i+1
            )
            
            if result is None:
                logger.error(f"Segment {i+1} processing failed")
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = f"Segment {i+1} processing failed"
                # Cleanup
                audio_segmenter.cleanup_segments(audio_segments)
                return
            
            segment_results.append(result)
            logger.info(f"Task {task_id}: Segment {i+1} completed successfully")
        
        # Step 3: Merge results
        tasks[task_id]["stage"] = "Merging segment results..."
        logger.info(f"Task {task_id}: Merging {len(segment_results)} segment results")
        
        merger = ResultMerger()
        merged_result = merger.merge_results(segment_results, text_segments)
        
        # Save merged result
        result_file = AUDIO_OUTPUT_DIR / f"{task_id}_results.json"
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(json.dumps(merged_result, indent=2, ensure_ascii=False))
        
        # Cleanup temporary segment files
        audio_segmenter.cleanup_segments(audio_segments)
        
        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["stage"] = f"Processing completed ({total_segments} segments merged)"
        tasks[task_id]["result_available"] = True
        
        logger.info(f"Task {task_id}: Multi-segment processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in multi-segment process: {e}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return JSONResponse({
            "message": "GOPT Web Interface",
            "status": "running",
            "error": "Frontend not found. Please create static/index.html"
        })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    docker_status = check_docker_container()
    return {
        "status": "healthy",
        "docker_container": DOCKER_CONTAINER,
        "docker_running": docker_status,
        "g2p_available": G2P_AVAILABLE,
        "segmentation_available": SEGMENTATION_AVAILABLE,
        "whisper_available": WHISPER_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper ASR.
    
    Args:
        file: Audio file to transcribe
    
    Returns:
        JSON with transcribed text and metadata
    """
    # Check if Whisper is available
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Whisper ASR service is not available. Please install: pip install openai-whisper"
        )
    
    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Generate temporary filename
    temp_id = uuid.uuid4().hex[:8]
    temp_filename = f"transcribe_{temp_id}{file_extension}"
    temp_path = AUDIO_INPUT_DIR / temp_filename
    
    try:
        # Save uploaded file temporarily
        file_size = await save_upload_file(file, temp_path)
        logger.info(f"Saved audio for transcription: {temp_filename} ({file_size} bytes)")
        
        # Get Whisper instance
        whisper = get_whisper_instance(model_size='base')
        
        # Transcribe
        logger.info(f"Starting transcription of {temp_filename}...")
        result = whisper.transcribe(temp_path, language='en')
        
        # Return result
        response = {
            "success": True,
            "text": result['text'],
            "raw_text": result['raw_text'],
            "language": result['language'],
            "duration": result['duration'],
            "processing_time": result['processing_time'],
            "filename": file.filename
        }
        
        logger.info(f"Transcription successful: {result['text']}")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.info(f"Cleaned up temp file: {temp_filename}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@app.post("/check-transcript")
async def check_transcript(transcript: str = Form(...)):
    """
    Check transcript for out-of-vocabulary words before processing.
    
    Args:
        transcript: Text transcription to check
        
    Returns:
        Information about OOV words and whether G2P will be used
    """
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required")
    
    # Check Docker container
    if not check_docker_container():
        raise HTTPException(
            status_code=503,
            detail=f"Docker container '{DOCKER_CONTAINER}' is not running"
        )
    
    # Check for OOV words
    oov_words, oov_details = check_oov_words(transcript)
    
    total_words = len(transcript.upper().strip().split())
    
    return JSONResponse({
        "success": True,
        "total_words": total_words,
        "oov_count": len(oov_words),
        "oov_words": oov_words,
        "oov_details": oov_details,
        "g2p_available": G2P_AVAILABLE,
        "message": (
            "All words in lexicon" if len(oov_words) == 0 
            else f"{len(oov_words)} OOV word(s) will use G2P" if G2P_AVAILABLE
            else f"{len(oov_words)} OOV word(s) may be skipped (G2P not available)"
        )
    })


@app.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    transcript: str = Form(default="")
):
    """
    Upload audio file and transcript for processing.
    Supports multiple audio formats: WAV, MP3, M4A, FLAC, OGG, AAC, WMA
    Automatically converts to 16kHz mono WAV if needed.
    
    Args:
        file: Audio file (WAV, MP3, M4A, FLAC, OGG, AAC, WMA)
        transcript: Text transcription (optional - will use Whisper ASR if empty)
    
    Returns:
        task_id: Unique task identifier for tracking
    """
    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Check if transcript is empty - will use ASR if needed
    needs_asr = not transcript or not transcript.strip()
    
    # If no transcript and Whisper not available, error
    if needs_asr and not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="Transcript is required (Whisper ASR not available). Please provide text manually."
        )
    
    # Check Docker container
    if not check_docker_container():
        raise HTTPException(
            status_code=503,
            detail=f"Docker container '{DOCKER_CONTAINER}' is not running. Please start it first."
        )
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        raise HTTPException(
            status_code=503,
            detail="Audio conversion service (FFmpeg) is not available in Docker container"
        )
    
    # Generate unique task ID
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    
    # Save original file with its extension
    original_filename = f"{task_id}_original{file_extension}"
    original_path = AUDIO_INPUT_DIR / original_filename
    
    # Final WAV file path
    final_audio_filename = f"{task_id}.wav"
    final_audio_path = AUDIO_INPUT_DIR / final_audio_filename
    
    try:
        # Save uploaded file
        file_size = await save_upload_file(file, original_path)
        
        logger.info(f"Task {task_id}: Uploaded {file.filename} ({file_size} bytes, {file_extension})")
        
        # If transcript is empty, use Whisper ASR to transcribe
        if needs_asr:
            logger.info(f"Task {task_id}: No transcript provided, using Whisper ASR...")
            
            try:
                # Get Whisper instance
                whisper = get_whisper_instance(model_size='base')
                
                # Transcribe the audio
                asr_result = whisper.transcribe(original_path, language='en')
                transcript = asr_result['text']
                
                logger.info(f"Task {task_id}: ASR transcription successful: {transcript}")
                logger.info(f"Task {task_id}: ASR processing time: {asr_result['processing_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"Task {task_id}: ASR failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Automatic transcription failed: {str(e)}. Please provide transcript manually."
                )
        
        # Create initial task record
        tasks[task_id] = {
            "task_id": task_id,
            "status": "uploaded",
            "stage": "File uploaded, checking format...",
            "filename": file.filename,
            "audio_filename": final_audio_filename,
            "transcript": transcript,
            "file_size": file_size,
            "original_format": file_extension,
            "created_at": datetime.now().isoformat(),
            "result_available": False,
            "used_asr": needs_asr
        }
        
        # Determine if conversion is needed
        needs_conversion = False
        conversion_reason = ""
        
        if file_extension != '.wav':
            needs_conversion = True
            conversion_reason = f"Non-WAV format ({file_extension})"
        else:
            # Check if WAV file meets requirements (16kHz, mono)
            tasks[task_id]["stage"] = "Analyzing WAV file properties..."
            audio_info = get_audio_info(original_path)
            
            if audio_info and 'streams' in audio_info and len(audio_info['streams']) > 0:
                stream = audio_info['streams'][0]
                sample_rate = int(stream.get('sample_rate', 0))
                channels = int(stream.get('channels', 0))
                
                logger.info(f"Task {task_id}: WAV properties - {sample_rate}Hz, {channels} channel(s)")
                
                if sample_rate != 16000 or channels != 1:
                    needs_conversion = True
                    conversion_reason = f"{sample_rate}Hz, {channels}ch (need 16kHz mono)"
            else:
                # If we can't get info, assume conversion is needed
                needs_conversion = True
                conversion_reason = "Unable to verify WAV properties"
        
        # Convert if needed
        if needs_conversion:
            logger.info(f"Task {task_id}: Conversion needed - {conversion_reason}")
            tasks[task_id]["stage"] = f"Converting audio ({conversion_reason})..."
            tasks[task_id]["conversion_applied"] = True
            
            success, message = await convert_audio(original_path, final_audio_path, task_id)
            
            if not success:
                # Clean up files
                original_path.unlink(missing_ok=True)
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = message
                raise HTTPException(status_code=400, detail=f"Audio conversion failed: {message}")
            
            logger.info(f"Task {task_id}: Conversion successful")
            tasks[task_id]["stage"] = "Audio converted to 16kHz mono WAV"
            
            # Optionally delete original file to save space
            # original_path.unlink(missing_ok=True)
        else:
            # No conversion needed, just rename
            logger.info(f"Task {task_id}: No conversion needed, WAV already in correct format")
            original_path.rename(final_audio_path)
            tasks[task_id]["stage"] = "Audio format verified (16kHz mono)"
            tasks[task_id]["conversion_applied"] = False
        
        # Check for OOV words
        oov_words, oov_details = check_oov_words(transcript)
        
        # Store OOV information in task
        tasks[task_id]["oov_count"] = len(oov_words)
        tasks[task_id]["oov_words"] = oov_words
        tasks[task_id]["oov_details"] = oov_details
        
        # Update task status
        tasks[task_id]["status"] = "ready"
        tasks[task_id]["stage"] = "Ready for processing"
        
        logger.info(f"Task {task_id} created and ready for processing")
        if oov_words:
            logger.info(f"Task {task_id}: {len(oov_words)} OOV word(s) detected: {', '.join(oov_words)}")
        
        # Start processing in background
        asyncio.create_task(run_docker_process(task_id, final_audio_filename, transcript))
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": "File uploaded successfully. Processing started.",
            "filename": file.filename,
            "file_size": file_size,
            "original_format": file_extension,
            "conversion_applied": needs_conversion,
            "conversion_reason": conversion_reason if needs_conversion else None,
            "oov_count": len(oov_words),
            "oov_words": oov_words,
            "oov_info": {
                "count": len(oov_words),
                "words": oov_words,
                "g2p_available": G2P_AVAILABLE,
                "message": (
                    "All words in lexicon" if len(oov_words) == 0
                    else f"{len(oov_words)} OOV word(s) will use G2P" if G2P_AVAILABLE
                    else f"Warning: {len(oov_words)} OOV word(s) may be skipped"
                )
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        # Clean up files if they were created
        original_path.unlink(missing_ok=True)
        final_audio_path.unlink(missing_ok=True)
        
        if task_id in tasks:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)
        
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Get processing status for a task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status information
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "stage": task.get("stage", ""),
        "filename": task.get("filename", ""),
        "transcript": task.get("transcript", ""),
        "created_at": task.get("created_at", ""),
        "result_available": task.get("result_available", False),
        "error": task.get("error", None)
    }
    
    # Add segmentation info if available
    if "num_segments" in task:
        response["segmentation"] = {
            "num_segments": task.get("num_segments"),
            "current_segment": task.get("current_segment"),
            "total_words": task.get("total_words"),
            "estimated_phonemes": task.get("estimated_phonemes")
        }
    
    return JSONResponse(response)


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Get processing results for a completed task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Assessment results
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task['status']}"
        )
    
    # Read result file
    result_file = AUDIO_OUTPUT_DIR / f"{task_id}_results.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        async with aiofiles.open(result_file, 'r') as f:
            content = await f.read()
            result_data = json.loads(content)
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "filename": task.get("filename", ""),
            "transcript": task.get("transcript", ""),
            "results": result_data
        })
        
    except Exception as e:
        logger.error(f"Error reading result file for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")


@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    Download result JSON file.
    
    Args:
        task_id: Task identifier
        
    Returns:
        JSON file download
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result_file = AUDIO_OUTPUT_DIR / f"{task_id}_results.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_file,
        media_type="application/json",
        filename=f"{task_id}_results.json"
    )


@app.get("/logs/{task_id}")
async def get_logs(task_id: str):
    """
    Get processing logs for a task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Processing logs
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return JSONResponse({
        "task_id": task_id,
        "stdout": task.get("stdout", ""),
        "stderr": task.get("stderr", ""),
        "return_code": task.get("return_code", None)
    })


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task and its associated files.
    
    Args:
        task_id: Task identifier
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    # Delete audio file
    audio_file = AUDIO_INPUT_DIR / task.get("audio_filename", "")
    if audio_file.exists():
        audio_file.unlink()
    
    # Delete result file
    result_file = AUDIO_OUTPUT_DIR / f"{task_id}_results.json"
    if result_file.exists():
        result_file.unlink()
    
    # Remove from tasks
    del tasks[task_id]
    
    return JSONResponse({
        "success": True,
        "message": f"Task {task_id} deleted"
    })


# Mount static files (must be after routes to avoid conflicts)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GOPT Web Interface Server'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind the server to (default: 8080)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("GOPT Web Interface Server")
    print("="*70)
    print(f"Server URL: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"Docker Container: {DOCKER_CONTAINER}")
    print(f"Docker Running: {check_docker_container()}")
    print("="*70 + "\n")
    
    if not check_docker_container():
        print(f"[WARNING] Docker container '{DOCKER_CONTAINER}' is not running!")
        print(f"          Start it with: docker start {DOCKER_CONTAINER}")
        print()
    
    uvicorn.run(
        "web_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()

