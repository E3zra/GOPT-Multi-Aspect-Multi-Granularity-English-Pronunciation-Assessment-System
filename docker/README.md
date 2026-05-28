# GOPT Docker Scripts

This directory contains helper scripts for running GOPT in Docker.

## Scripts

### 0. `run_pipeline_auto.sh` (RECOMMENDED)
Fully automated pipeline script (939 lines). Handles all 8 steps end-to-end with no manual intervention.

**Usage:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline_auto.sh \
  <audio_file> <transcript> [dataset_name]
```

**Example:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline_auto.sh \
  /workspace/audio_input/test.wav \
  "HELLO WORLD" \
  my_test
```

**What it does (fully automated):**
1. Prepares Kaldi data directory
2. Generates lexicon and phone transcripts
3. Extracts MFCC features
4. Extracts i-vectors
5. Computes nnet3 output
6. Runs Kaldi GOP extraction
7. Extracts GOP features to CSV
8. Converts to sequence format and runs GOPT inference

### 1. `run_pipeline.sh` (LEGACY)
Original pipeline script — requires manual Kaldi GOP extraction step.

**Usage:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh \
  <audio_file> <transcript> [dataset_name]
```

**Example:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh \
  /workspace/audio_input/test.wav \
  "HELLO WORLD" \
  my_test
```

**What it does:**
1. Prepares Kaldi data directory
2. Guides through Kaldi GOP extraction (manual step)
3. Waits for user to complete Kaldi processing

### 2. `continue_pipeline.sh`
Continues pipeline after Kaldi GOP extraction completes.

**Usage:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/continue_pipeline.sh \
  <dataset_name> [output_file]
```

**Example:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/continue_pipeline.sh \
  my_test \
  /workspace/audio_output/my_test_results.json
```

**What it does:**
1. Extracts GOP features to CSV
2. Converts features to sequence format
3. Runs GOPT inference
4. Saves results to JSON

### 3. `setup_models.sh`
Downloads and sets up Librispeech ASR models.

**Usage:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/setup_models.sh [target_dir]
```

**Example:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/setup_models.sh
```

**What it does:**
1. Downloads Librispeech models from kaldi-asr.org
2. Extracts to `/workspace/models/librispeech/`
3. Verifies installation

**Note:** Downloads ~2GB of data. May take 10-30 minutes.

### 4. `test_installation.sh`
Tests if all GOPT components are properly installed.

**Usage:**
```bash
docker exec -it gopt-pipeline /workspace/scripts/test_installation.sh
```

**What it tests:**
- Python installation
- Python packages (torch, numpy, kaldiio)
- Kaldi installation
- GOPT code and models
- Directory structure
- Model loading

**Expected output:**
```
========================================================================
           GOPT Docker Installation Test
========================================================================

[INFO] Testing Python installation...
[✓] Python installed
[✓] torch installed
[✓] numpy installed
[✓] kaldiio installed
...
[✓] Installation test completed!
```

## Quick Start

### First Time Setup

```bash
# 1. Build container
docker-compose build

# 2. Start container
docker-compose up -d

# 3. Test installation
docker exec -it gopt-pipeline /workspace/scripts/test_installation.sh

# 4. Download Librispeech models (optional, can be manual)
docker exec -it gopt-pipeline /workspace/scripts/setup_models.sh
```

### Process Audio

```bash
# 1. Place audio in audio_input/
cp your_audio.wav audio_input/

# 2. Run pipeline
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh \
  /workspace/audio_input/your_audio.wav \
  "YOUR TEXT TRANSCRIPT" \
  my_audio

# 3. Manually complete Kaldi GOP extraction
docker exec -it gopt-pipeline bash
cd /opt/kaldi/egs/gop_speechocean762/s5
# Edit run.sh and run ./run.sh

# 4. Continue pipeline
docker exec -it gopt-pipeline /workspace/scripts/continue_pipeline.sh \
  my_audio \
  /workspace/audio_output/my_audio_results.json

# 5. View results
cat audio_output/my_audio_results.json
```

## Detailed Documentation

For comprehensive documentation, see:
- **[DOCKER_USAGE.md](../DOCKER_USAGE.md)** - Complete usage guide
- **[DOCKER_USAGE.md](../DOCKER_USAGE.md)** - Full Docker guide (consolidated)

## Troubleshooting

### Scripts are not executable

```bash
docker exec -it gopt-pipeline bash
chmod +x /workspace/scripts/*.sh
```

### Kaldi GOP extraction fails

Check logs:
```bash
docker exec -it gopt-pipeline bash
cd /opt/kaldi/egs/gop_speechocean762/s5
cat exp/gop_test/log/*.log
```

### Model not found

Verify model exists:
```bash
docker exec -it gopt-pipeline ls -la /workspace/gopt/pretrained_models/gopt_librispeech/
```

Should contain `best_audio_model.pth`.

### Out of memory

Increase Docker memory:
- Docker Desktop: Settings > Resources > Memory > 8GB+

## Notes

- All scripts use colored output for better visibility
- Scripts check prerequisites before running
- Error messages include helpful debugging information
- Paths use `/workspace/` convention for Docker environment

## Support

For issues:
1. Check script output for error messages
2. View container logs: `docker-compose logs gopt`
3. Test installation: `test_installation.sh`
4. See troubleshooting in [DOCKER_USAGE.md](../DOCKER_USAGE.md)

