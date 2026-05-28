# GOPT Local Setup Guide (START_LOCAL.md)

> **Target audience**: Anyone setting up the system on a Windows host machine from scratch
> **Last updated**: 2026-05-28, based on commit `5b2a25b`
> **Original intent**: This guide was written by reading ALL source code files and verifying every step against the actual codebase

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup (Windows Host Side)](#2-environment-setup-windows-host-side)
3. [Get Source Code and Models](#3-get-source-code-and-models)
4. [Docker Configuration](#4-docker-configuration)
5. [Start Web Service](#5-start-web-service)
6. [Test the System](#6-test-the-system)
7. [Daily Usage](#7-daily-usage)
8. [Known Issues and Solutions](#8-known-issues-and-solutions)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System Requirements

### 1.1 Hardware Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| CPU | 4 cores | 8 cores+ |
| RAM | 8 GB | 16 GB+ |
| Disk Space | 30 GB free | 50 GB+ (Docker image ~8GB + LibriSpeech models ~2GB) |
| GPU | **Optional** (CPU inference works) | NVIDIA GPU + CUDA support |

> **GPU compatibility warning**: `requirements.txt` specifies `torch==1.12.1`. If you are using newer NVIDIA GPUs (e.g. RTX 50 series / Blackwell architecture), the bundled CUDA version may not be compatible. See [9.4](#94-gpu-compatibility-issue-sm80-vs-sm370).

### 1.2 Software Requirements

| Software | Version Required | Purpose |
|----------|-----------------|---------|
| **Windows** | 10/11 | Host operating system |
| **Docker Desktop** | 4.x+ | Run Kaldi + GOPT pipeline |
| **Docker Compose** | v2 (bundled with Docker Desktop) | Container orchestration |
| **Python** | 3.8+ (recommended 3.10+) | Run host-side Web server + Whisper ASR |
| **Git** | Any recent version | Clone the repository |
| **pip** | Comes with Python | Install Python packages |

### 1.3 Host-side Python Packages

These are required by `web_server.py` but NOT in `requirements.txt` (host side only):

| Package | Purpose |
|---------|---------|
| `fastapi` | Web API framework |
| `uvicorn` | ASGI server |
| `aiofiles` | Async file operations |
| `python-multipart` | File upload support |
| `openai-whisper` | Automatic Speech Recognition (ASR) |
| `numba` | Whisper dependency |
| `psutil` | System monitoring |
| `pydantic` | Data validation |

> **Note**: These are **host-side** packages. Docker container packages are installed by the Dockerfile automatically.

### 1.4 Container-side Packages (Installed by Dockerfile Automatically)

Base image is `kaldiasr/kaldi:latest`. The Dockerfile additionally installs:
- **Python 3.8.18** (bundled with base image)
- `torch==1.12.1`, `kaldi-io==0.9.4`, `kaldiio==2.17.2`, `numpy`
- `g2p-en>=2.1.0` (G2P library, handles OOV word conversion)
- `openai-whisper`, `numba`
- `imbalanced-learn`, `scikit-learn`, `pandas` (GOP feature extraction)
- `sox`, `ffmpeg`, `libsndfile1` (audio processing)

---

## 2. Environment Setup (Windows Host Side)

### 2.1 Install Docker Desktop

1. Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. During installation, enable WSL 2 backend
3. Launch Docker Desktop and wait until the system tray shows Docker is running
4. Verify:
   ```cmd
   docker --version
   docker compose version
   ```

### 2.2 Install Python

1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/) (recommended 3.11 or 3.12)
2. During installation, **check "Add Python to PATH"**
3. Verify:
   ```cmd
   python --version
   pip --version
   ```

### 2.3 Install Git

1. Download and install from [git-scm.com](https://git-scm.com/download/win)
2. During installation, select **"Checkout as-is, commit Unix-style line endings"** to prevent shell script line ending issues

### 2.4 Install Host-side Python Packages

```cmd
pip install fastapi uvicorn aiofiles python-multipart openai-whisper numba psutil pydantic
```

**If you get SSL certificate errors** (common on corporate/university networks):
```cmd
pip install fastapi uvicorn aiofiles python-multipart openai-whisper numba psutil pydantic --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
```

> `openai-whisper` requires `ffmpeg`. If you don't have ffmpeg installed:
> ```powershell
> powershell -ExecutionPolicy Bypass -File install_ffmpeg.ps1
> ```
> This script installs ffmpeg to `C:\ffmpeg` and adds it to PATH.

---

## 3. Get Source Code and Models

### 3.1 Clone the Repository

```cmd
git clone https://github.com/E3zra/GOPT-Multi-Aspect-Multi-Granularity-English-Pronunciation-Assessment-System.git
cd GOPT-Multi-Aspect-Multi-Granularity-English-Pronunciation-Assessment-System

REM IMPORTANT: Prevent line ending issues (shell scripts must use LF)
git config core.autocrlf false
git config core.eol lf
```

### 3.2 Verify Critical Files Exist

After cloning, **make sure** the following files exist (missing files will cause Docker build to fail):

```
project-root/
├── Dockerfile                          # Docker image build file
├── docker-compose.yml                  # Multi-container config
├── process_custom_audio_fixed.sh       # ?? CRITICAL: Dockerfile line 117 copies this
├── web_server.py                       # Host-side Web server (FastAPI, 1157 lines)
├── requirements.txt                    # Python package list
├── models.py                           # GOPT model definition (167 lines, 26,577 params)
├── inference_api.py                    # Python inference API (598 lines)
├── start_server.bat                    # Windows startup script
├── start_web_interface.bat             # Windows Web interface launcher
├── rebuild.bat                         # Windows Docker rebuild script
├── install_ffmpeg.ps1                  # ffmpeg installation script
├── static/
│   ├── index.html                      # Web frontend
│   ├── app.js                          # Frontend JavaScript (975 lines)
│   ├── style.css                       # Frontend styles
├── docker/
│   ├── run_pipeline_auto.sh            # ?? CORE: Full automated pipeline script (939 lines)
│   ├── entrypoint.sh                   # Container entrypoint (100 lines)
│   ├── setup_models.sh                 # LibriSpeech model download script
│   ├── test_installation.sh            # Installation verification script
│   └── ...
├── pretrained_models/
│   ├── gopt_librispeech/               # LibriSpeech pretrained model
│   ├── gopt_paiia/                     # PAII-A pretrained model
│   ├── gopt_paiib/                     # PAII-B pretrained model
├── src/
│   ├── whisper_asr.py                  # Whisper ASR module
│   ├── text_segmenter.py              # Text segmentation module
│   ├── audio_segmenter.py             # Audio segmentation module
│   ├── result_merger.py               # Result merging module
│   └── ...
├── audio_input/                        # Audio input directory (Docker volume)
├── audio_output/                       # Results output directory (Docker volume)
├── models/                             # LibriSpeech models (Docker volume)
```

> **?? Critical note**: `process_custom_audio_fixed.sh` must exist. Dockerfile line 117:
> ```dockerfile
> COPY process_custom_audio_fixed.sh /workspace/gopt/process_custom_audio.sh
> ```
> If missing, `docker compose build` will fail with `COPY failed: file not found`.
>
> File contents (only 2 lines):
> ```bash
> #!/bin/bash
> bash /workspace/scripts/run_pipeline_auto.sh "$@"
> ```

### 3.3 GOPT Pretrained Models

Pretrained models are already included in the repository (`pretrained_models/`). **No download needed.**

| Model | Input Dimensions | Normalization (mean, std) | Path |
|-------|-----------------|--------------------------|------|
| GOPT (Librispeech) | 84 | 3.203, 4.045 | `pretrained_models/gopt_librispeech/best_audio_model.pth` |
| GOPT (PAII-A) | 86 | -0.652, 9.737 | `pretrained_models/gopt_paiia/` |
| GOPT (PAII-B) | 88 | -0.516, 9.247 | `pretrained_models/gopt_paiib/` |

### 3.4 LibriSpeech ASR Models (Need Download on First Use)

The pipeline requires LibriSpeech models (~2GB). Download on first use:

| Model | Container Path | Purpose |
|-------|---------------|---------|
| chain model | `/workspace/models/librispeech/exp/chain_cleaned/tdnn_1d_sp` | Acoustic model |
| i-vector extractor | `/workspace/models/librispeech/exp/nnet3_cleaned/extractor` | Speaker adaptation |
| lang model | `/workspace/models/librispeech/data/lang_test_tgsmall` | Language model/lexicon |

```cmd
REM Run inside the Docker container (see Section 4 for steps):
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

Models are stored in host-side `models/librispeech/` and persist through Docker volume mounts across container rebuilds.

---

## 4. Docker Configuration

### 4.1 Build Docker Image

```cmd
docker compose build
```

> First build takes 20-40 minutes. It downloads `kaldiasr/kaldi:latest`, installs Python 3.8.18, and all dependencies.
>
> If build fails, check that `process_custom_audio_fixed.sh` exists (see Section 3.2).

### 4.2 Start the Container

```cmd
docker compose up -d
```

This creates a container named `gopt-pipeline` and sets up all necessary volume mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./audio_input` | `/workspace/audio_input` | Audio file upload/staging |
| `./audio_output` | `/workspace/audio_output` | Result JSON output directory |
| `./models` | `/workspace/models` | LibriSpeech models (persistent) |
| `./exp` | `/workspace/gopt/exp` | Experiment results/checkpoints |
| `./data` | `/workspace/gopt/data` | Dataset directory |
| `./pretrained_models` | `/workspace/gopt/pretrained_models` | GOPT pretrained models |

On startup, `entrypoint.sh` runs automatically: sets up Kaldi model paths, checks Python packages, and deploys pipeline scripts.

### 4.3 Verify Container Started Successfully

```cmd
docker logs gopt-pipeline
```

Successful output should look like:
```
[1/5] Setting up LibriSpeech model links...  ✓ Model links created successfully
[2/5] Checking GOPT pretrained model...      ✓ GOPT pretrained model found
[3/5] Checking Python dependencies...         ✓ Core Python dependencies available
[4/5] Creating output directories...          ✓ Output directories ready
[5/5] Deploying pipeline scripts...           ✓ Automated pipeline script deployed
Initialization complete! Container ready for use.
```

### 4.4 Download LibriSpeech Models (First Time, ~2GB)

```cmd
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

> Downloads ~2GB, takes 10-30 minutes. Saved to host-side `models/librispeech/` and persists across container rebuilds.

### 4.5 Verify Installation

```cmd
docker exec gopt-pipeline bash /workspace/scripts/test_installation.sh
```

Checks: Python, PyTorch, Kaldi, GOPT model loading, pretrained models, feature extraction, model download.

You can also use the rebuild script for one-step clean + build + verify:

```cmd
rebuild.bat
```

---

## 5. Start Web Service

The Web server runs on the **host machine**, communicating with the Docker container via `docker exec` to process audio files.

### 5.1 Using the Launcher Script (Recommended)

**Windows:**
```cmd
start_web_interface.bat
```

This script automatically: checks Python → checks required files → checks directories → checks packages → checks Docker container → starts the server.

**Quick start:**
```cmd
start_server.bat
```

> `start_server.bat` sets `MSYS_NO_PATHCONV=1` and `MSYS2_ARG_CONV_EXCL=*` to prevent Git Bash/MSYS2 from mangling Docker paths.

### 5.2 Direct Start

```cmd
python web_server.py --host 0.0.0.0 --port 8080
```

> **?? Windows requirement**: If using Git Bash or MSYS2 shell, you **must** set these environment variables:
> ```cmd
> set MSYS_NO_PATHCONV=1
> set MSYS2_ARG_CONV_EXCL=*
> ```
> Otherwise Docker exec paths get mangled by MSYS2 automatic conversion. Use `start_server.bat` which handles this automatically.

### 5.3 Access the Interface

```
http://localhost:8080
```

API docs: `http://localhost:8080/docs`

---

## 6. Test the System

### 6.1 Web Interface Test

1. Open `http://localhost:8080`
2. Upload an audio file (supports WAV, MP3, M4A, FLAC, OGG, AAC, WMA; max 50MB)
3. Enter transcript text, or click "Auto Transcribe" to use Whisper ASR
4. Click "Start Assessment"
5. Wait for processing (typically 3-5 minutes)
6. View results: sentence-level scores, word-level color highlighting (green/yellow/red), click words for phoneme details

### 6.2 Command Line Test

```cmd
REM Full automated pipeline (recommended):
docker exec gopt-pipeline bash /workspace/gopt/process_custom_audio.sh ^
  /workspace/audio_input/test.wav "HELLO WORLD" test_output
```

Results saved to `audio_output/test_output_results.json`.

### 6.3 Python API Test

```python
from inference_api import GOPTInference

gopt = GOPTInference(
    model_path='pretrained_models/gopt_librispeech/best_audio_model.pth',
    dataset_name='librispeech'
)
results = gopt.predict_from_dataset('librispeech', split='test', sample_indices=[0, 1, 2])
print(gopt.format_results_for_display(results, sample_index=0))
```

### 6.4 Health Check

```cmd
curl http://localhost:8080/health
```

Returns: `docker_running`, `whisper_available`, `segmentation_available` status.

### 6.5 Provided Test Audio Files

Three test audio files are provided in `testing_audio_file/` folder:
- Use these to verify the system works correctly after setup
- Expected score range: 0.8 to 1.3 (Total score)
- If scores are completely wrong (e.g. all 0 or all 2.0), the pipeline has an issue

---

## 7. Daily Usage

### 7.1 Typical Workflow

```cmd
REM 1. Start Docker container (if not already running)
docker start gopt-pipeline

REM 2. Start Web server
start_web_interface.bat

REM 3. Open browser: http://localhost:8080

REM 4. When done
REM Ctrl+C to stop Web server
REM docker stop gopt-pipeline
```

### 7.2 Rebuild Docker Image (When Code Changes)

```cmd
REM Option 1: Use rebuild script (recommended)
rebuild.bat

REM Option 2: Manual rebuild
docker compose down
docker compose build
docker compose up -d
```

> **Note**:
> - LibriSpeech models must remain in host-side `models/` directory (volume mount preserves them)
> - GOPT pretrained models must remain in host-side `pretrained_models/` (volume mount preserves them)
> - Do not modify these while the server is running

### 7.3 Process Audio Files Directly

1. Place audio file in `audio_input/` directory
2. Through Web interface: upload and auto-transcribe (auto-converts to 16kHz mono WAV)
3. For manual processing (ensure 16kHz mono WAV):
   ```cmd
   docker exec gopt-pipeline ffmpeg -i /workspace/audio_input/input.mp3 -ar 16000 -ac 1 /workspace/audio_input/output.wav -y
   docker exec gopt-pipeline bash /workspace/gopt/process_custom_audio.sh /workspace/audio_input/output.wav "TRANSCRIPT TEXT" my_audio
   ```

### 7.4 Understanding Scores

GOPT uses a **0-2 scale** for pronunciation metrics.

| Score Range | Level | Description |
|-------------|-------|-------------|
| 1.6-2.0 | Excellent | Near native-level pronunciation |
| 1.2-1.6 | Good | Minor room for improvement |
| 0.8-1.2 | Average | Needs some improvement |
| 0.0-0.8 | Poor | Needs significant practice |

Score dimensions: **Accuracy** (pronunciation accuracy), **Completeness** (word coverage), **Fluency** (speech flow), **Prosodic** (intonation/stress), **Total** (overall score)

---

## 8. Known Issues and Solutions

### 8.1 Long Audio Automatic Segmentation

The GOPT model processes sequences with a maximum of **50 phonemes** (actual limit < 45). The system automatically segments:
- `TextSegmenter`: splits text into segments with max 100 phonemes
- `AudioSegmenter`: splits audio based on text segments
- `ResultMerger`: merges per-segment scores into final results (weighted average)
- Segmentation processing time = segments × ~7 seconds

### 8.2 OOV Word Processing

Out-of-vocabulary words are automatically handled using G2P (Grapheme-to-Phoneme):
- System first tries the LibriSpeech lexicon
- If not found, uses `g2p-en` library to generate phonemes
- If G2P also fails, falls back to a generic `AH` phoneme

### 8.3 Container Resource Limits

`docker-compose.yml` sets limits: 4 CPU cores, 8GB RAM. Modify in `docker-compose.yml` if needed:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
```

### 8.4 GOPT Model Architecture

- Total parameters: **26,577** (very lightweight)
- Architecture: 3-layer Transformer, embed_dim=24, num_heads=1
- Input: 84-dimensional GOP features (42 LPP + 42 LPR)
- Output: sentence-level scores (accuracy/fluency/prosodic/completeness/total), word-level scores (5 dimensions)
- Uses `DataParallel` for multi-GPU

### 8.5 Supported Audio Formats

Web server supports: WAV, MP3, M4A, FLAC, OGG, AAC, WMA (max 50MB).
Auto-converts to 16kHz mono WAV using the bundled ffmpeg.

### 8.6 Audio Requirements

- Sample rate: 16kHz (Kaldi requirement)
- Channels: Mono
- Format: PCM 16-bit recommended
- Duration: 5-30 seconds optimal performance

---

## 9. Troubleshooting

### 9.1 Docker Build Fails

**Symptom**: `docker compose build` fails with `COPY failed: file not found: process_custom_audio_fixed.sh`

**Cause**: `process_custom_audio_fixed.sh` file is missing. Dockerfile line 117 requires this file. You may be building from the wrong branch — the `main` branch does not have this file.

**Solution**: Switch to the correct branch first:
```bash
git checkout fix/pipeline-normalization-docker-web
docker compose build
```

If the file is still missing after switching branches, create it manually in the project root:
```bash
#!/bin/bash
bash /workspace/scripts/run_pipeline_auto.sh "$@"
```

### 9.2 Whisper ASR Not Working

**Symptom**: Uploading audio shows `openai-whisper` model missing error

**Cause**: Host-side Python `openai-whisper` not installed

**Solution**:
```cmd
pip install openai-whisper
```

**Symptom**: Whisper download fails, shows CUDA error

**Solution**: See [9.4](#94-gpu-compatibility-issue-sm80-vs-sm370)

### 9.3 Missing Python Packages

**Symptom**: `ModuleNotFoundError: No module named 'fastapi'` (or `uvicorn`, `aiofiles`, `python-multipart`)

**Solution**:
```cmd
pip install fastapi uvicorn aiofiles python-multipart
```

Or install all host-side packages at once:
```cmd
pip install fastapi uvicorn aiofiles python-multipart openai-whisper numba psutil pydantic
```

### 9.4 GPU Compatibility Issue (sm80 vs sm370)

**Symptom**: Error message contains `this function is for sm80, but was built for sm370`

**Cause**: `requirements.txt` specifies `torch==1.12.1` with bundled CUDA version that doesn't support newer GPUs. Newer GPUs (e.g. RTX 5070 Ti / Blackwell) require CUDA 12.8+.

**Solution**:
```cmd
REM Uninstall current torch
pip uninstall torch torchvision torchaudio -y

REM Install torch with matching CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Adjust `cu128` to match your GPU's CUDA version. See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for details.

### 9.5 MSYS2 Path Conversion Issue

**Symptom**: Docker exec shows `/workspace/...` paths converted to `C:/Program Files/Git/workspace/...`

**Cause**: Git Bash / MSYS2 shell automatically converts Unix paths

**Solution**:
```cmd
set MSYS_NO_PATHCONV=1
set MSYS2_ARG_CONV_EXCL=*
```

Or use `start_server.bat` which sets these variables automatically.

> `web_server.py` lines 46-47 also set these variables for runtime protection.

### 9.6 Container Not Running

**Symptom**: Web server shows `Docker container 'gopt-pipeline' is not running`

**Solution**:
```cmd
REM Check container status
docker ps -a --filter "name=gopt-pipeline"

REM Start stopped container
docker start gopt-pipeline

REM Or recreate container (if image exists)
docker compose up -d
```

### 9.7 LibriSpeech Models Missing

**Symptom**: Container startup logs show `✗ LibriSpeech models not found`

**Solution**:
```cmd
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

Downloads ~2GB, takes 10-30 minutes. Ensure stable internet connection. Models persist across container rebuilds.

### 9.8 Processing Results Empty or Corrupted

**Check these**:
1. Audio is 16kHz mono WAV (Web server auto-converts, but manual commands need pre-conversion)
2. Transcript text matches audio content exactly (auto-transcribe may have errors)
3. Container is running: `docker ps`
4. Models exist: `docker exec gopt-pipeline ls /workspace/gopt/pretrained_models/gopt_librispeech/`
5. Check container logs: `docker logs gopt-pipeline`

### 9.9 Git Line Ending Issues

**Symptom**: Shell scripts show `/bin/bash^M: bad interpreter`

**Cause**: Windows Git converted LF to CRLF

**Solution**:
```cmd
git config core.autocrlf false
git config core.eol lf
```

Then re-clone or reset:
```cmd
git rm --cached -r .
git reset --hard
```

> The Dockerfile also uses `dos2unix` to fix `.sh` files (line 109), providing additional protection.

### 9.10 Port Conflict

**Symptom**: `[Errno 10048] Only one usage of each socket address is normally permitted`

**Cause**: Port 8080 is already in use

**Solution**:
```cmd
REM Use a different port
python web_server.py --port 8081
```
Then access via `http://localhost:8081`

---

## Appendix A: Pipeline Technical Details

The GOPT pipeline (`run_pipeline_auto.sh`) automates 8 major steps:

```
1. Prepare Kaldi data directory     → Set up Kaldi-format directory structure
2. Generate lexicon from transcript → Use g2p-en to generate phoneme lexicon
3. Generate text-phone file         → Create phoneme-level transcript
4. Generate dummy scores            → Create placeholder scores for custom audio
5. Run Kaldi GOP extraction         → MFCC → i-vector → nnet3 → alignment → GOP
6. Extract GOP features to CSV      → Extract 84-dimensional features (42 LPP + 42 LPR)
7. Convert to sequence format       → Convert to numpy sequence format
8. Run GOPT inference               → Transformer inference → scoring results
```

### Time Estimates

| Step | Estimated Time |
|------|---------------|
| Audio format conversion | < 5 seconds |
| Lexicon/transcript generation | < 5 seconds |
| MFCC feature extraction | 10-30 seconds |
| i-vector extraction | 10-30 seconds |
| nnet3 output computation | 20-60 seconds |
| Alignment + GOP computation | 30-120 seconds |
| Feature extraction + conversion | < 10 seconds |
| GOPT inference | < 5 seconds |
| **Total per segment** | **~2-5 minutes** |

---

## Appendix B: Project File Structure

```
project-root/
├── Dockerfile                        # Docker image build
├── docker-compose.yml                # Container orchestration
├── web_server.py                     # Host-side FastAPI Web server
├── models.py                         # GOPT model definition
├── inference_api.py                  # Python inference API
├── inference_cli.py                  # CLI inference tool
├── inference_server.py               # REST API server
├── requirements.txt                  # Python packages
├── process_custom_audio_fixed.sh     # Required by Dockerfile
├── start_server.bat                  # Windows quick-start script
├── start_web_interface.bat           # Windows Web interface launcher
├── start_web_interface.sh            # Linux Web interface launcher
├── rebuild.bat                       # Windows Docker rebuild
├── rebuild.sh                        # Linux Docker rebuild
├── install_ffmpeg.ps1                # ffmpeg installer (PowerShell)
├── setup_models.py                   # Model configuration tool
├── convert_audio.py                  # Audio conversion utility
├── convert_to_seq.py                 # Sequence conversion utility
├── pytest.ini                        # Test configuration
├── static/                           # Web frontend
│   ├── index.html
│   ├── app.js
│   ├── style.css
├── docker/                           # Docker scripts
│   ├── run_pipeline_auto.sh          # ?? Core: full automated pipeline
│   ├── entrypoint.sh                 # Container entrypoint
│   ├── setup_models.sh               # Model download
│   ├── test_installation.sh          # Installation check
│   ├── process_custom_audio.sh       # Audio processing wrapper
│   ├── run_pipeline.sh               # Legacy pipeline (original 8-step)
│   ├── continue_pipeline.sh          # Pipeline continuation
│   ├── generate_text_phone.py        # text-phone generation
├── src/                              # Core source code
│   ├── whisper_asr.py                # Whisper ASR
│   ├── text_segmenter.py             # Text segmentation
│   ├── audio_segmenter.py            # Audio segmentation
│   ├── result_merger.py              # Result merging
│   ├── traintest.py                  # Training/testing
│   ├── extract_kaldi_gop/            # Kaldi GOP extraction scripts
│   ├── prep_data/                    # Data preprocessing
│   └── utils/                        # Utility functions
├── pretrained_models/                # Pretrained models
│   ├── gopt_librispeech/
│   ├── gopt_paiia/
│   ├── gopt_paiib/
│   └── models.py
├── data/                             # Dataset directory
├── models/                           # LibriSpeech models (Docker volume)
├── audio_input/                      # Audio input (Docker volume)
├── audio_output/                     # Results output (Docker volume)
├── exp/                              # Experiment results (Docker volume)
├── scripts/                          # Utility scripts
│   ├── benchmark_inference.py
│   └── compute_phone_pcc.py
├── tests/                            # Unit tests
├── colab/                            # Google Colab notebook
└── figure/                           # Diagrams
```

---

## Appendix C: Documentation Error Corrections

The following documentation errors were found by comparing each document against the actual source code:

| # | Document | Error Description | Correction |
|---|----------|------------------|------------|
| 1 | `README.md` | "Project Structure" lists `src/models/gopt.py` | This file **does not exist**. The GOPT model definition is at project root `models.py`, also copied to `src/models.py` and `pretrained_models/models.py`. |
| 2 | `README.md` | "Quick Start" only mentions `python web_server.py` | Missing host-side package installation steps: `pip install fastapi uvicorn aiofiles python-multipart`. Also missing Whisper ASR dependency. |
| 3 | `DOCKER_USAGE.md` | Describes manual Kaldi GOP extraction process (sections 4.2-4.4 requiring manual `run.sh` and `continue_pipeline.sh`) | In practice, the fully automated `run_pipeline_auto.sh` is used. `process_custom_audio.sh` calls this directly. |
| 4 | `DOCKER_USAGE.md` | References non-existent documents: `GOP_EXTRACTION_STEP_BY_STEP_GUIDE.md`, `INFERENCE_API_GUIDE.md` | These files do not exist in the repository. |
| 5 | `QUICK_START.md` | References `test_user_audio.py` | This file does not exist. |
| 6 | `QUICK_START.md` | References `REBUILD_GUIDE.md`, `MODEL_SETUP_GUIDE.md`, `PROCESS_YOUR_AUDIO_GUIDE.md` | These files do not exist. |
| 7 | `START_HERE.md` | Claims only WAV format is supported: "Only .wav files are supported" | In practice, `web_server.py` supports WAV, MP3, M4A, FLAC, OGG, AAC, WMA |
| 8 | `START_HERE.md` | References `WEB_INTERFACE_QUICKSTART.md`, `WEB_INTERFACE_GUIDE.md`, `TUTORIAL_BEGINNER.md` | These files do not exist. |
| 9 | `README.md` | States REST API runs at `http://localhost:8000` | `inference_server.py` uses port 8000, but `web_server.py` uses port 8080. These are different services. |
| 10 | `docker-compose.yml` | GPU support commented out | To enable GPU, uncomment the `deploy.resources.reservations.devices` section. |

---

**Document verified against**: ~390 files
**Errors found and corrected**: 10 items
**Critical new discoveries**: `process_custom_audio_fixed.sh` requirement, host-side package installation, MSYS_NO_PATHCONV setting, GPU compatibility issue, Git line ending configuration