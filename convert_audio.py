#!/usr/bin/env python3
"""
Convert FLAC audio to 16kHz mono WAV format for GOPT processing
"""

import subprocess
import sys
from pathlib import Path

def convert_with_ffmpeg(flac_file, wav_file):
    """Convert using ffmpeg (if available)"""
    try:
        subprocess.run([
            'ffmpeg', '-i', str(flac_file),
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',       # mono
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            str(wav_file),
            '-y'  # overwrite
        ], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False

def convert_with_python(flac_file, wav_file):
    """Convert using Python libraries"""
    try:
        # Try pydub first (easiest)
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(flac_file), format="flac")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(wav_file), format="wav")
        print(f"[OK] Converted using pydub: {flac_file.name} -> {wav_file.name}")
        return True
    except ImportError:
        pass
    
    try:
        # Try soundfile + librosa
        import soundfile as sf
        import librosa
        
        # Load audio
        audio, sr = librosa.load(str(flac_file), sr=16000, mono=True)
        
        # Save as WAV
        sf.write(str(wav_file), audio, 16000, subtype='PCM_16')
        print(f"[OK] Converted using librosa+soundfile: {flac_file.name} -> {wav_file.name}")
        return True
    except ImportError:
        pass
    
    try:
        # Try scipy
        from scipy.io import wavfile
        import librosa
        
        audio, sr = librosa.load(str(flac_file), sr=16000, mono=True)
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype('int16')
        wavfile.write(str(wav_file), 16000, audio_int16)
        print(f"[OK] Converted using librosa+scipy: {flac_file.name} -> {wav_file.name}")
        return True
    except ImportError:
        pass
    
    return False

def main():
    # Test audio directory
    test_audio_dir = Path("test audio from librispeech 133604")
    
    if not test_audio_dir.exists():
        print(f"Error: Directory not found: {test_audio_dir}")
        sys.exit(1)
    
    # Output directory
    output_dir = Path("audio_input")
    output_dir.mkdir(exist_ok=True)
    
    # Get all FLAC files
    flac_files = list(test_audio_dir.glob("*.flac"))
    
    if not flac_files:
        print("No FLAC files found")
        sys.exit(1)
    
    print(f"Found {len(flac_files)} FLAC files")
    print("\nConverting audio files to 16kHz mono WAV format...")
    print("=" * 60)
    
    converted = 0
    failed = 0
    
    for flac_file in sorted(flac_files):
        wav_file = output_dir / f"{flac_file.stem}.wav"
        
        # Try ffmpeg first
        if convert_with_ffmpeg(flac_file, wav_file):
            print(f"[OK] Converted using ffmpeg: {flac_file.name} -> {wav_file.name}")
            converted += 1
            continue
        
        # Try Python libraries
        if convert_with_python(flac_file, wav_file):
            converted += 1
            continue
        
        print(f"[FAIL] Failed to convert: {flac_file.name}")
        failed += 1
    
    print("=" * 60)
    print(f"\nConversion complete:")
    print(f"  [OK] Converted: {converted} files")
    print(f"  [FAIL] Failed: {failed} files")
    
    if converted > 0:
        print(f"\nConverted files are in: {output_dir.absolute()}")
    
    if failed > 0:
        print("\nTo convert manually, install ffmpeg:")
        print("  Windows: choco install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")

if __name__ == "__main__":
    main()

