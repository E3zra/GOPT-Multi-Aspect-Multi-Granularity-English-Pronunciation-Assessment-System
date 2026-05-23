# GOPT: Multi-Aspect Multi-Granularity English Pronunciation Assessment System

A Transformer-based system for evaluating non-native English pronunciation. Given an audio recording and its transcript, it scores pronunciation across **multiple aspects** (accuracy, fluency, prosody, completeness) at **multiple granularities** (phoneme, word, utterance).

> **This is an enhanced fork** of the [original GOPT project](https://github.com/YuanGongND/gopt) by Yuan Gong et al. (MIT & PAII), originally published at ICASSP 2022. This fork adds a web interface, Docker deployment, Whisper ASR integration, automatic OOV handling, and more.

<p align="center"><img src="figure/gopt_poster.png" alt="GOPT Architecture" width="800"/></p>

---

## Table of Contents

- [What's New in This Fork](#whats-new-in-this-fork)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface](#1-web-interface)
  - [Docker Pipeline](#2-docker-pipeline-recommended)
  - [Python API](#3-python-api)
  - [Command Line](#4-command-line)
  - [REST API](#5-rest-api)
- [Training](#training)
- [Pretrained Models](#pretrained-models)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contributors](#contributors)
- [License](#license)

---

## What's New in This Fork

| Feature | Description |
|---------|-------------|
| **Web Interface** | Browser-based UI with drag-and-drop audio upload, real-time processing, color-coded word-level feedback |
| **Docker Deployment** | Pre-configured container with Kaldi + all dependencies. No manual compilation needed |
| **Whisper ASR Integration** | Automatic speech-to-text transcription via OpenAI Whisper, no need to type transcripts manually |
| **Automatic OOV Handling** | Out-of-vocabulary words are automatically converted to phonemes using G2P (grapheme-to-phoneme) |
| **Long Audio Segmentation** | Automatically splits long recordings into processable segments |
| **REST API** | HTTP API for integration with other applications |
| **Multiple Audio Formats** | Supports WAV, MP3, M4A, FLAC, OGG and more (auto-converted to 16kHz mono WAV) |

---

## Quick Start

The easiest way to get started is Docker:

```bash
# 1. Build and start the container
docker-compose build
docker-compose up -d

# 2. Start the web interface
python web_server.py

# 3. Open http://localhost:8080 in your browser
```

See [DOCKER_USAGE.md](DOCKER_USAGE.md) for detailed instructions.

---

## Usage

### 1. Web Interface

The most user-friendly option. Upload audio, enter (or auto-transcribe) text, and get visual feedback.

```bash
# Make sure Docker container is running
docker-compose up -d

# Start the web server
python web_server.py
# Open http://localhost:8080
```

Results include:
- Sentence-level scores (accuracy, completeness, fluency, prosody, total)
- Word-level scores with color highlighting (green/yellow/red)
- Phoneme-level detail on click
- JSON export

### 2. Docker Pipeline (Recommended)

Process audio files through the command line with Docker:

```bash
# Start container
docker-compose up -d

# Process an audio file
docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh \
  /workspace/audio_input/your_audio.wav \
  "THE TEXT TRANSCRIPT" \
  output_name
```

Results are saved to `audio_output/` as JSON.

### 3. Python API

```python
from inference_api import GOPTInference

gopt = GOPTInference(model_path='pretrained_models/gopt_librispeech/best_audio_model.pth')
results = gopt.predict_from_dataset('librispeech', split='test', sample_indices=[0, 1, 2])
```

### 4. Command Line

```bash
python inference_cli.py --samples 0 1 2 3 4
```

### 5. REST API

```bash
python inference_server.py
# API documentation at http://localhost:8000/docs
```

---

## Training

### Option A: Google Colab (Easiest)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/gopt/blob/master/colab/GOPT_GPU.ipynb)

No GPU or Kaldi installation needed. Just run the notebook.

### Option B: Local Training

**Step 0. Setup environment:**

```bash
python3 -m venv venv-gopt
source venv-gopt/bin/activate   # Linux/Mac
# or: venv-gopt\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Step 1. Get the data:**

Download the preprocessed GOP features (Kaldi-free option):
- [Dropbox](https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1)
- [Tencent Weiyun](https://share.weiyun.com/vJCAXjFY)

Or extract them yourself from the [SpeechOcean762](https://www.openslr.org/101) dataset using Kaldi (see [data/README.md](data/README.md)).

**Step 2. Convert to sequences:**

```bash
cd src/prep_data
python gen_seq_data_phn.py
python gen_seq_data_word.py
python gen_seq_data_utt.py
```

**Step 3. Train:**

```bash
cd src
./run.sh           # Local
# sbatch run.sh    # SLURM cluster
```

Results and best model are saved to the `exp_dir` specified in `src/run.sh`.

---

## Pretrained Models

Three pretrained models are included in `pretrained_models/`:

| Model               | Phn PCC | Word Total PCC | Utt Acc PCC | Utt Flu PCC | Utt Pros PCC | Utt Total PCC |
|---------------------|:-------:|:--------------:|:-----------:|:-----------:|:------------:|:-------------:|
| GOPT (Librispeech)  |  0.616  |     0.552      |    0.718    |    0.756    |     0.764    |     0.743     |
| GOPT (PAII-A)       |  0.679  |     0.606      |    0.727    |    0.692    |     0.695    |     0.731     |
| GOPT (PAII-B)       |  0.664  |     0.602      |    0.722    |    0.721    |     0.723    |     0.740     |

PCC = Pearson Correlation Coefficient (higher is better). Best-single-run results on SpeechOcean762.

---

## Project Structure

```
gopt/
├── src/                    # Core source code
│   ├── models/gopt.py      # GOPT Transformer model
│   ├── traintest.py        # Training and evaluation
│   ├── whisper_asr.py      # Whisper ASR integration
│   ├── extract_kaldi_gop/  # Kaldi GOP feature extraction
│   ├── prep_data/          # Data preprocessing
│   └── utils/              # G2P helper, data utilities
├── pretrained_models/      # Pretrained model weights (3 models)
├── data/                   # Training/test datasets
├── docker/                 # Docker scripts
├── static/                 # Web frontend (HTML/CSS/JS)
├── tests/                  # Unit tests
├── scripts/                # Benchmark and utility scripts
├── colab/                  # Google Colab notebook
├── inference_api.py        # Python API
├── inference_cli.py        # CLI interface
├── inference_server.py     # REST API server
├── web_server.py           # Web interface server
├── docker-compose.yml      # Docker Compose config
├── Dockerfile
└── requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE).

The SpeechOcean762 dataset is licensed under CC BY 4.0 and can be downloaded from [OpenSLR](https://www.openslr.org/101).

---

## 中文说明

### 项目简介

这是一个**英语发音自动评估系统**，基于 ICASSP 2022 论文 [GOPT](https://arxiv.org/abs/2205.03432) 开发。

**功能**：输入一段英语音频和对应文本，系统自动给出发音评分：

- **评分维度**：准确度、流利度、韵律、完整度
- **评分粒度**：音素级（每个音）、单词级（每个词）、句子级（整句话）

### 本 Fork 新增功能

| 功能 | 说明 |
|------|------|
| **网页界面** | 浏览器上传音频，自动评分，可视化反馈（颜色标注好/中/差） |
| **Docker 一键部署** | 无需手动安装 Kaldi，`docker-compose up` 即可使用 |
| **Whisper 语音识别** | 自动将音频转为文本，无需手动输入 |
| **OOV 自动处理** | 遇到词典外的生词自动用 G2P 转音素 |
| **长音频分段** | 自动将长录音切分为可处理的片段 |
| **多格式支持** | WAV、MP3、M4A、FLAC 等自动转换 |

### 快速开始

```bash
# 1. 启动 Docker 容器
docker-compose build && docker-compose up -d

# 2. 启动网页服务
python web_server.py

# 3. 浏览器打开 http://localhost:8080
```

### 致谢

本项目的核心算法来自 [YuanGongND/gopt](https://github.com/YuanGongND/gopt)（MIT & PAII 团队）。本 Fork 在其基础上增加了 Web 界面、Docker 部署、Whisper ASR 等功能。
