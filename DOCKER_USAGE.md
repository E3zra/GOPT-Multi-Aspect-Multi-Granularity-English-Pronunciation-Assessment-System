# GOPT Docker Guide

This document supersedes the previous quick-start, walkthrough, and Kaldi notes. It describes how to run the full Kaldi → GOP → GOPT pipeline in a single container.

---

## 1. Why Docker?

- Pre-built Kaldi environment (no local compilation).
- Reproducible Python stack (PyTorch 1.12.1 + project dependencies).
- Shared volumes for audio input, model data, and outputs.
- Works on Windows, Linux, and macOS.

---

## 2. Prerequisites

- Docker Engine 20.10+ and Docker Compose 1.29+.
- ~20 GB free disk space (base image + LibriSpeech models + outputs).
- Optional NVIDIA GPU (requires NVIDIA Container Toolkit).

Verify installation:

```bash
docker --version
docker-compose --version
```

---

## 3. Initial Setup

```bash
cd path/to/gopt

# 1. Build image (15–20 min first time)
docker-compose build

# 2. Start container (runs as gopt-pipeline)
docker-compose up -d

# 3. Download required LibriSpeech models (~2 GB)
docker exec -it gopt-pipeline /workspace/scripts/setup_models.sh

# 4. Verify everything
docker exec -it gopt-pipeline /workspace/scripts/test_installation.sh
```

> **Windows 11 hosts:** Open PowerShell (or Windows Terminal), `cd D:\Projects\gopt_1104\gopt`, and run the commands above while Docker Desktop is running. The new syntax `docker compose` works interchangeably with `docker-compose` on Windows 11. After cloning, run `git config core.autocrlf false` and `git config core.eol lf` in the repository once so that shell scripts retain Linux line endings.

Directory layout after setup:

```
audio_input/          # drop 16 kHz mono WAV files here
audio_output/         # JSON results produced here
models/librispeech/   # ASR models downloaded by setup_models.sh
docker/               # helper scripts mounted to /workspace/scripts
```

---

## 4. Processing Audio

### 4.1 Convert / copy audio

Audio must be 16 kHz, mono WAV (Kaldi requirement). You can convert inside the container:

```bash
docker exec -it gopt-pipeline bash
cd /workspace/gopt
ffmpeg -i "test audio from librispeech 133604/1188-133604-0006.flac" \
       -ar 16000 -ac 1 audio_input/test_001.wav -y
exit
```

### 4.2 Run the orchestrated pipeline

```bash
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh \
  /workspace/audio_input/test_001.wav \
  "THEN HE COMES TO THE BEAK OF IT" \
  test_001
```

What happens:

1. Validates audio + transcript; prepares Kaldi data dir.
2. Prompts you to run Kaldi GOP extraction (manual step below).
3. After GOP completion, `continue_pipeline.sh` converts features and runs inference.

### 4.3 Manual Kaldi GOP extraction

```bash
docker exec -it gopt-pipeline bash
cd /opt/kaldi/egs/gop_speechocean762/s5

# (one time) edit run.sh to point to the downloaded Librispeech models
nano run.sh
# set librispeech_eg=/workspace/models/librispeech
# stage=2, nj=1, data_dir=<dataset name from run_pipeline>

./run.sh
exit
```

### 4.4 Finish the pipeline

```bash
docker exec -it gopt-pipeline /workspace/scripts/continue_pipeline.sh \
  test_001 \
  /workspace/audio_output/test_001_results.json
```

Results appear under `audio_output/` on the host machine.

---

## 5. Batch Processing Pattern

Example PowerShell / Bash loop:

```bash
for wav in audio_input/*.wav; do
  name=$(basename "$wav" .wav)
  transcript=$(python scripts/get_transcript.py "$name")  # implement to match your data

  docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh \
    "/workspace/audio_input/${name}.wav" \
    "$transcript" \
    "$name"

  # wait for Kaldi run.sh to finish, then continue
  docker exec -it gopt-pipeline /workspace/scripts/continue_pipeline.sh \
    "$name" \
    "/workspace/audio_output/${name}_results.json"
done
```

Tips:

- Keep transcripts uppercase and punctuation-free.
- For large batches, run multiple Kaldi jobs by increasing `nj` in `run.sh` (ensure sufficient CPU/RAM).
- Use `docker cp` to move large result sets out of the container if needed.

---

## 6. GPU Support (Optional)

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Uncomment the GPU section in `docker-compose.yml` (or add `deploy.resources.reservations.devices`).
3. Rebuild and restart: `docker-compose down && docker-compose up -d`.

The inference step automatically picks up CUDA via PyTorch; Kaldi GOP stays CPU-bound.

---

## 7. Troubleshooting

| Symptom | Resolution |
|---------|------------|
| `docker-compose build` fails | Ensure Docker Desktop is running, free up disk, retry with `--no-cache`. |
| `setup_models.sh` interrupted | Re-run the script; it skips completed downloads. |
| Kaldi GOP logs missing | Check `/opt/kaldi/egs/gop_speechocean762/s5/exp/*/log/` inside the container. |
| Transcript mismatch errors | Ensure transcripts are uppercase and match the spoken text exactly. |
| `imblearn` / `sklearn` import errors | Rebuild image (`docker-compose build`) to pick up bundled dependencies. |
| Need to inspect environment | `docker exec -it gopt-pipeline bash` gives an interactive shell. |

Log helpers:

```bash
docker-compose logs -f gopt                # Container output
docker exec gopt-pipeline ls audio_output  # Check generated files
```

Reset everything:

```bash
docker-compose down -v
docker system prune -af
```

---

## 8. Reference

- `docker/README.md` – script descriptions.
- `GOP_EXTRACTION_STEP_BY_STEP_GUIDE.md` – deeper dive into Kaldi workflow.
- `INFERENCE_API_GUIDE.md` – using the API once GOP features exist.

---

**Last updated:** November 5, 2025

