#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and setup LibriSpeech ASR models for GOPT
"""

import subprocess
import sys
import os

def run_cmd(cmd, description):
    """Run command and return success status."""
    print(f"\n[INFO] {description}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for download
        )
        if result.returncode == 0:
            print(f"[OK] Success")
            if result.stdout.strip():
                # Print first and last few lines for large output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 20:
                    print('\n'.join(lines[:5]))
                    print(f"... ({len(lines)-10} lines omitted)")
                    print('\n'.join(lines[-5:]))
                else:
                    print(result.stdout.strip())
            return True
        else:
            print(f"[ERROR] Failed (exit code: {result.returncode})")
            if result.stderr.strip():
                print(result.stderr.strip()[:500])
            return False
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False

print("="*70)
print("LibriSpeech ASR Models Setup")
print("="*70)

print("""
This script will download and setup LibriSpeech ASR models (~2GB).
These models are required for GOP (Goodness of Pronunciation) extraction.

Models will be downloaded to:
  ./models/librispeech/
  
and copied to container:
  /workspace/models/librispeech/
""")

# Step 1: Check if models directory exists
print("\n[Step 1] Checking models directory...")
os.makedirs("models/librispeech", exist_ok=True)
print("[OK] Directory ready: models/librispeech/")

# Step 2: Check if models already exist
print("\n[Step 2] Checking for existing models...")
check_cmd = 'docker exec gopt-pipeline test -d /opt/kaldi/egs/librispeech/s5/exp/nnet3_cleaned/tdnn_sp'
if run_cmd(check_cmd, "Checking models in container"):
    print("\n[INFO] Models already exist! No download needed.")
    print("[OK] Setup complete!")
    sys.exit(0)

# Step 3: Check if models are in host models directory
print("\n[Step 3] Checking host models directory...")
if os.path.exists("models/librispeech/exp/nnet3_cleaned/tdnn_sp"):
    print("[INFO] Models found in host directory")
    print("[INFO] Copying to container...")
    # Copy will happen via volume mount
    run_cmd(
        'docker exec gopt-pipeline bash -c "mkdir -p /opt/kaldi/egs/librispeech/s5/exp && cp -r /workspace/models/librispeech/exp /opt/kaldi/egs/librispeech/s5/"',
        "Copying models to Kaldi directory"
    )
    sys.exit(0)

# Step 4: Download models
print("\n[Step 4] Downloading LibriSpeech ASR models...")
print("[INFO] This will download ~2GB of data. This may take 10-30 minutes.")
print("[INFO] Download URL: http://www.openslr.org/resources/11/")

# Auto-proceed with download in automation mode
print("\n[INFO] Auto-proceeding with download...")

# Download using container
download_cmd = '''docker exec gopt-pipeline bash -c "
cd /opt/kaldi/egs/librispeech/s5
# Download chain model
wget -c http://www.openslr.org/resources/11/0013_librispeech_v1_chain.tar.gz
tar -xzf 0013_librispeech_v1_chain.tar.gz
rm 0013_librispeech_v1_chain.tar.gz
echo 'Models downloaded and extracted'
"'''

if not run_cmd(download_cmd, "Downloading and extracting models"):
    print("\n[ERROR] Download failed!")
    print("\nAlternative: Manual download")
    print("1. Download: http://www.openslr.org/resources/11/0013_librispeech_v1_chain.tar.gz")
    print("2. Extract to: models/librispeech/")
    print("3. Copy to container:")
    print("   docker cp models/librispeech gopt-pipeline:/opt/kaldi/egs/librispeech/s5/")
    sys.exit(1)

# Step 5: Verify installation
print("\n[Step 5] Verifying installation...")
if run_cmd(check_cmd, "Verifying models"):
    print("\n" + "="*70)
    print("[OK] LibriSpeech ASR models successfully installed!")
    print("="*70)
    print("\nYou can now:")
    print("1. Upload your audio file again")
    print("2. Processing should complete successfully")
else:
    print("\n[ERROR] Verification failed")
    print("Models may not be in the correct location")
    sys.exit(1)

