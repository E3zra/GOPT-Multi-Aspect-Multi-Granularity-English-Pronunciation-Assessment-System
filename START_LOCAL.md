# GOPT 本機部署完整指南 (START_LOCAL.md)

> **目標讀者**：全新 Windows 機器上的新團隊成員
> **最後驗證**：2026-05-28，基於 commit `5b2a25b`
> **原則**：所有內容均來自真實代碼，不做猜測或假設

---

## 目錄

1. [系統需求](#1-系統需求)
2. [環境設置](#2-環境設置windows-宿主機)
3. [獲取代碼和模型](#3-獲取代碼和模型)
4. [Docker 設置](#4-docker-設置)
5. [啟動 Web 服務](#5-啟動-web-服務)
6. [測試系統](#6-測試系統)
7. [日常使用](#7-日常使用)
8. [已知問題和限制](#8-已知問題和限制)
9. [故障排除](#9-故障排除)

---

## 1. 系統需求

### 1.1 硬體需求

| 項目 | 最低需求 | 建議配置 |
|------|---------|---------|
| CPU | 4 核 | 8 核+ |
| 記憶體 | 8 GB | 16 GB+ |
| 磁碟空間 | 30 GB 可用 | 50 GB+ (Docker 鏡像 ~8GB + LibriSpeech 模型 ~2GB) |
| GPU | **可選** (CPU 推論可用) | NVIDIA GPU + CUDA 支援 |

> **GPU 架構注意事項**：`requirements.txt` 指定 `torch==1.12.1`。如果使用較新的 NVIDIA GPU（例如 RTX 50 系列 / Blackwell 架構），預設 CUDA 版本可能不相容。詳見 [9.4](#94-gpu-架構不相容-sm80-vs-sm370)。

### 1.2 軟體需求

| 軟體 | 版本要求 | 用途 |
|------|---------|------|
| **Windows** | 10/11 | 宿主機作業系統 |
| **Docker Desktop** | 4.x+ | 執行 Kaldi + GOPT 容器 |
| **Docker Compose** | v2 (內建於 Docker Desktop) | 容器編排 |
| **Python** | 3.8+ (建議 3.10+) | 宿主機 Web 服務 + Whisper ASR |
| **Git** | 任意近期版本 | 獲取代碼 |
| **pip** | 隨 Python 安裝 | 安裝 Python 依賴 |

### 1.3 宿主機 Python 依賴

根據 `web_server.py` 和 `requirements.txt` 分析，宿主機需要：

| 套件 | 用途 |
|------|------|
| `fastapi` | Web API 框架 |
| `uvicorn` | ASGI 伺服器 |
| `aiofiles` | 異步文件操作 |
| `python-multipart` | 文件上傳支援 |
| `openai-whisper` | 自動語音識別 (ASR) |
| `numba` | Whisper 依賴 |
| `psutil` | 基準測試工具 |
| `pydantic` | 資料驗證 |

> **注意**：這些是**宿主機**依賴。容器內依賴由 Dockerfile 自動安裝。

### 1.4 容器內依賴（Dockerfile 自動安裝，無需手動處理）

基礎鏡像 `kaldiasr/kaldi:latest`，Dockerfile 額外安裝：
- **Python 3.8.18**（從源碼編譯）
- `torch==1.12.1`, `kaldi-io==0.9.4`, `kaldiio==2.17.2`, `numpy`
- `g2p-en>=2.1.0`（G2P 音素生成，用於 OOV 詞彙）
- `openai-whisper`, `numba`
- `imbalanced-learn`, `scikit-learn`, `pandas`（GOP 特徵提取）
- `sox`, `ffmpeg`, `libsndfile1`（音訊處理）

---

## 2. 環境設置（Windows 宿主機）

### 2.1 安裝 Docker Desktop

1. 下載安裝 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 安裝完成後重啟電腦
3. 確認 Docker Desktop 正在執行（系統托盤有 Docker 圖示）
4. 驗證：
   ```cmd
   docker --version
   docker compose version
   ```

### 2.2 安裝 Python

1. 從 [python.org](https://www.python.org/downloads/) 下載 Python 3.10+（建議 3.11 或 3.12）
2. 安裝時**勾選 "Add Python to PATH"**
3. 驗證：
   ```cmd
   python --version
   pip --version
   ```

### 2.3 安裝 Git

1. 從 [git-scm.com](https://git-scm.com/download/win) 下載安裝
2. 安裝時選擇 **"Checkout as-is, commit Unix-style line endings"**（確保 shell 腳本保持 Linux 行尾）

### 2.4 安裝宿主機 Python 依賴

```cmd
pip install fastapi uvicorn aiofiles python-multipart openai-whisper numba psutil pydantic
```

> `openai-whisper` 需要 `ffmpeg`。如系統沒有 ffmpeg：
> ```powershell
> powershell -ExecutionPolicy Bypass -File install_ffmpeg.ps1
> ```
> 此腳本會將 ffmpeg 安裝到 `C:\ffmpeg` 並加入 PATH。

---

## 3. 獲取代碼和模型

### 3.1 克隆代碼

```cmd
git clone https://github.com/E3zra/GOPT-Multi-Aspect-Multi-Granularity-English-Pronunciation-Assessment-System.git
cd GOPT-Multi-Aspect-Multi-Granularity-English-Pronunciation-Assessment-System

REM 設定行尾（重要！避免 shell 腳本行尾被破壞）
git config core.autocrlf false
git config core.eol lf
```

### 3.2 確認關鍵文件存在

克隆後**必須**確認以下文件存在（缺失將導致 Docker 構建失敗）：

```
項目根目錄/
├── Dockerfile                          # Docker 鏡像構建文件
├── docker-compose.yml                  # 容器編排配置
├── process_custom_audio_fixed.sh       # ⚠️ 必要！Dockerfile 第 117 行依賴
├── web_server.py                       # 宿主機 Web 服務 (FastAPI, 1157 行)
├── requirements.txt                    # Python 依賴清單
├── models.py                           # GOPT 模型定義 (167 行, 26,577 參數)
├── inference_api.py                    # Python 推論 API (598 行)
├── start_server.bat                    # Windows 啟動腳本
├── start_web_interface.bat             # Windows Web 界面啟動腳本
├── rebuild.bat                         # Windows Docker 重建腳本
├── install_ffmpeg.ps1                  # ffmpeg 安裝腳本
├── static/
│   ├── index.html                      # Web 前端頁面
│   ├── app.js                          # 前端 JavaScript (975 行)
│   └── style.css                       # 前端樣式
├── docker/
│   ├── run_pipeline_auto.sh            # 核心：全自動化管線腳本 (939 行)
│   ├── entrypoint.sh                   # 容器入口腳本 (100 行)
│   ├── setup_models.sh                 # LibriSpeech 模型下載腳本
│   ├── test_installation.sh            # 安裝驗證腳本
│   └── ...
├── pretrained_models/
│   ├── gopt_librispeech/               # LibriSpeech 預訓練模型
│   ├── gopt_paiia/                     # PAII-A 預訓練模型
│   └── gopt_paiib/                     # PAII-B 預訓練模型
├── src/
│   ├── whisper_asr.py                  # Whisper ASR 模組
│   ├── text_segmenter.py               # 長文本分段模組
│   ├── audio_segmenter.py              # 音訊分段模組
│   ├── result_merger.py                # 結果合併模組
│   └── ...
├── audio_input/                        # 音訊輸入 (Docker volume)
├── audio_output/                       # 結果輸出 (Docker volume)
└── models/                             # LibriSpeech 模型 (Docker volume)
```

> **⚠️ 關鍵檢查**：`process_custom_audio_fixed.sh` 必須存在！Dockerfile 第 117 行：
> ```dockerfile
> COPY process_custom_audio_fixed.sh /workspace/gopt/process_custom_audio.sh
> ```
> 缺失會導致 `docker compose build` 報錯 `COPY failed: file not found`。
>
> 文件內容只需兩行：
> ```bash
> #!/bin/bash
> bash /workspace/scripts/run_pipeline_auto.sh "$@"
> ```

### 3.3 GOPT 預訓練模型

預訓練模型已包含在倉庫中 (`pretrained_models/`)，**無需額外下載**：

| 模型 | 輸入維度 | 正規化參數 (mean, std) | 路徑 |
|------|---------|----------------------|------|
| GOPT (LibriSpeech) | 84 | 3.203, 4.045 | `pretrained_models/gopt_librispeech/best_audio_model.pth` |
| GOPT (PAII-A) | 86 | -0.652, 9.737 | `pretrained_models/gopt_paiia/` |
| GOPT (PAII-B) | 88 | -0.516, 9.247 | `pretrained_models/gopt_paiib/` |

### 3.4 LibriSpeech ASR 模型（首次需要下載）

管線需要 LibriSpeech 模型（~2GB），首次使用時下載：

| 模型 | 容器內路徑 | 用途 |
|------|----------|------|
| chain model | `/workspace/models/librispeech/exp/chain_cleaned/tdnn_1d_sp` | 聲學模型 |
| i-vector extractor | `/workspace/models/librispeech/exp/nnet3_cleaned/extractor` | 說話者特徵 |
| lang model | `/workspace/models/librispeech/data/lang_test_tgsmall` | 語言模型/詞典 |

```cmd
REM 先構建並啟動 Docker（見第 4 節），然後：
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

模型保存在宿主機 `models/librispeech/`（通過 Docker volume 持久化，重建鏡像不會丟失）。

---

## 4. Docker 設置

### 4.1 構建 Docker 鏡像

```cmd
docker compose build
```

> 首次構建約 20-40 分鐘。會下載 `kaldiasr/kaldi:latest`、編譯 Python 3.8.18、安裝所有依賴。
>
> 如構建失敗，確認 `process_custom_audio_fixed.sh` 存在（見 3.2 節）。

### 4.2 啟動容器

```cmd
docker compose up -d
```

此命令建立名為 `gopt-pipeline` 的容器，掛載以下 volume：

| 宿主機目錄 | 容器目錄 | 用途 |
|-----------|---------|------|
| `./audio_input` | `/workspace/audio_input` | 音訊文件放置處 |
| `./audio_output` | `/workspace/audio_output` | 結果 JSON 輸出處 |
| `./models` | `/workspace/models` | LibriSpeech 模型（持久化） |
| `./exp` | `/workspace/gopt/exp` | 實驗結果/日誌 |
| `./data` | `/workspace/gopt/data` | 處理數據 |
| `./pretrained_models` | `/workspace/gopt/pretrained_models` | GOPT 預訓練模型 |

容器啟動時 `entrypoint.sh` 自動執行：設置 Kaldi 模型符號連結 → 驗證依賴 → 部署管線腳本。

### 4.3 檢查容器啟動日誌

```cmd
docker logs gopt-pipeline
```

成功日誌應包含：
```
[1/5] Setting up LibriSpeech model links...  ✓ Model links created successfully
[2/5] Checking GOPT pretrained model...      ✓ GOPT pretrained model found
[3/5] Checking Python dependencies...         ✓ Core Python dependencies available
[4/5] Creating output directories...          ✓ Output directories ready
[5/5] Deploying pipeline scripts...           ✓ Automated pipeline script deployed
Initialization complete! Container ready for use.
```

### 4.4 下載 LibriSpeech 模型（首次，~2GB）

```cmd
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

> 下載約 2GB，需 10-30 分鐘。下載後保存在宿主機 `models/librispeech/`，重建鏡像不會丟失。

### 4.5 驗證安裝

```cmd
docker exec gopt-pipeline bash /workspace/scripts/test_installation.sh
```

檢查：Python、PyTorch、Kaldi、GOPT 模型代碼、預訓練模型、數據目錄、模型載入。

或使用重建腳本（一步完成構建+啟動+驗證）：

```cmd
rebuild.bat
```

---

## 5. 啟動 Web 服務

Web 服務運行在**宿主機**上，通過 `docker exec` 調用容器進行音訊處理。

### 5.1 使用啟動腳本（推薦）

**Windows：**
```cmd
start_web_interface.bat
```

此腳本自動：檢查 Python → 檢查必要文件 → 創建目錄 → 檢查依賴 → 檢查 Docker 容器 → 啟動伺服器。

**簡化版：**
```cmd
start_server.bat
```

> `start_server.bat` 會設定 `MSYS_NO_PATHCONV=1` 和 `MSYS2_ARG_CONV_EXCL=*`（防止 Git Bash/MSYS2 環境破壞 Docker 路徑）。

### 5.2 直接啟動

```cmd
python web_server.py --host 0.0.0.0 --port 8080
```

> **⚠️ Windows 重要**：如使用 Git Bash 或 MSYS2 環境，**必須**先設定：
> ```cmd
> set MSYS_NO_PATHCONV=1
> set MSYS2_ARG_CONV_EXCL=*
> ```
> 否則 Docker exec 路徑會被 MSYS2 自動轉換導致錯誤。`start_server.bat` 已自動處理。

### 5.3 開啟瀏覽器

```
http://localhost:8080
```

API 文檔：`http://localhost:8080/docs`

---

## 6. 測試系統

### 6.1 Web 界面測試

1. 開啟 `http://localhost:8080`
2. 上傳音訊文件（支援 WAV, MP3, M4A, FLAC, OGG, AAC, WMA，最大 50MB）
3. 輸入轉錄文本（自動轉全大寫）或點擊「自動識別」使用 Whisper ASR
4. 點擊「開始評估」
5. 等待處理（通常 3-5 分鐘）
6. 查看：句子級分數、音素級分數、顏色標注詞級反饋

### 6.2 命令列測試

```cmd
REM 全自動管線（容器內）
docker exec gopt-pipeline bash /workspace/gopt/process_custom_audio.sh ^
  /workspace/audio_input/test.wav "HELLO WORLD" test_output
```

結果保存在 `audio_output/test_output_results.json`。

### 6.3 Python API 測試

```python
from inference_api import GOPTInference

gopt = GOPTInference(
    model_path='pretrained_models/gopt_librispeech/best_audio_model.pth',
    dataset_name='librispeech'
)
results = gopt.predict_from_dataset('librispeech', split='test', sample_indices=[0, 1, 2])
print(gopt.format_results_for_display(results, sample_index=0))
```

### 6.4 健康檢查

```cmd
curl http://localhost:8080/health
```

回應包含：`docker_running`、`whisper_available`、`segmentation_available` 等狀態。

---

## 7. 日常使用

### 7.1 每日工作流

```cmd
REM 1. 啟動 Docker 容器（如果未運行）
docker start gopt-pipeline

REM 2. 啟動 Web 服務
start_web_interface.bat

REM 3. 瀏覽器開啟 http://localhost:8080

REM 4. 使用完畢
REM Ctrl+C 停止 Web 服務
REM docker stop gopt-pipeline
```

### 7.2 重建 Docker 鏡像（代碼更新後）

```cmd
REM 方式一：使用重建腳本（推薦）
rebuild.bat

REM 方式二：手動重建
docker compose down
docker compose build
docker compose up -d
```

> **注意**：
> - LibriSpeech 模型保留（在宿主機 `models/` 目錄，volume 掛載）
> - GOPT 預訓練模型保留（在宿主機 `pretrained_models/`，volume 掛載）
> - 所有修復自動應用（內建在鏡像中）

### 7.3 處理自訂音訊

1. 將音訊文件放入 `audio_input/` 目錄
2. 通過 Web 界面上傳（自動轉 16kHz mono WAV）
3. 或命令列（需先轉為 16kHz mono WAV）：
   ```cmd
   docker exec gopt-pipeline ffmpeg -i /workspace/audio_input/input.mp3 -ar 16000 -ac 1 /workspace/audio_input/output.wav -y
   docker exec gopt-pipeline bash /workspace/gopt/process_custom_audio.sh /workspace/audio_input/output.wav "TRANSCRIPT TEXT" my_audio
   ```

### 7.4 評分範圍

GOPT 使用 **0-2 分制**：

| 分數範圍 | 等級 | 說明 |
|---------|------|------|
| 1.6-2.0 | 優秀 | 接近母語水平 |
| 1.2-1.6 | 良好 | 有提升空間 |
| 0.8-1.2 | 一般 | 需要練習 |
| 0.0-0.8 | 較差 | 需要重點改進 |

評分維度：**Accuracy**（準確度）、**Completeness**（完整性）、**Fluency**（流暢度）、**Prosodic**（韻律）、**Total**（總分）

---

## 8. 已知問題和限制

### 8.1 長音訊自動分段

GOPT 模型序列長度限制為 **50 個音素**（實際建議 < 45）。系統自動分段：
- `TextSegmenter`：按 45 音素上限分割文本
- `AudioSegmenter`：按文本比例分割音訊
- `ResultMerger`：加權合併多段結果（按音素數加權）
- 分段處理時間 = 段數 × 約 7 分鐘

### 8.2 OOV 詞彙處理

詞典外詞彙自動使用 G2P（Grapheme-to-Phoneme）轉換：
- 系統先查 LibriSpeech 詞典
- 未找到則使用 `g2p-en` 套件生成音素
- G2P 也失敗時，回退為單一音素 `AH`

### 8.3 容器資源限制

`docker-compose.yml` 預設限制：4 CPU、8GB 記憶體。可在 `docker-compose.yml` 中調整：

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
```

### 8.4 GOPT 模型架構

- 參數量：**26,577**（非常輕量）
- 架構：3 層 Transformer, embed_dim=24, num_heads=1
- 輸入：84 維 GOP 特徵（42 LPP + 42 LPR）
- 輸出：音素級分數、詞級分數（準確度/重音/總分）、句子級分數（5 個維度）
- 使用 `DataParallel` 包裝

### 8.5 支援的音訊格式

Web 界面支援：WAV, MP3, M4A, FLAC, OGG, AAC, WMA（最大 50MB）
自動轉換為 16kHz mono WAV（通過容器內 ffmpeg）

### 8.6 音訊要求

- 取樣率：16kHz（Kaldi 要求）
- 聲道：單聲道
- 編碼：PCM 16-bit
- 建議時長：5-30 秒效果最佳

---

## 9. 故障排除

### 9.1 Docker 構建失敗

**問題**：`docker compose build` 報錯 `COPY failed: file not found: process_custom_audio_fixed.sh`

**原因**：`process_custom_audio_fixed.sh` 文件缺失。Dockerfile 第 117 行 `COPY process_custom_audio_fixed.sh /workspace/gopt/process_custom_audio.sh` 需要此文件。

**解決**：在項目根目錄創建文件：
```bash
#!/bin/bash
bash /workspace/scripts/run_pipeline_auto.sh "$@"
```

### 9.2 Whisper ASR 無法使用

**問題**：上傳音訊時報錯 `openai-whisper` 模組缺失

**原因**：宿主機未安裝 `openai-whisper`

**解決**：
```cmd
pip install openai-whisper
```

**問題**：Whisper 載入失敗，提示 CUDA 錯誤

**解決**：見 [9.4](#94-gpu-架構不相容-sm80-vs-sm370)

### 9.3 缺少 Python 套件

**問題**：`ModuleNotFoundError: No module named 'fastapi'`（或 `uvicorn`、`aiofiles`、`python-multipart`）

**解決**：
```cmd
pip install fastapi uvicorn aiofiles python-multipart
```

或一次性安裝所有宿主機依賴：
```cmd
pip install fastapi uvicorn aiofiles python-multipart openai-whisper numba psutil pydantic
```

### 9.4 GPU 架構不相容 (sm80 vs sm370)

**問題**：錯誤訊息類似 `this function is for sm80, but was built for sm370`

**原因**：`requirements.txt` 指定的 `torch==1.12.1` 預設 CUDA 版本與新 GPU 架構不相容（如 RTX 5070 Ti / Blackwell 架構需要 CUDA 12.8+）

**解決**：
```cmd
REM 卸載現有 torch
pip uninstall torch torchvision torchaudio -y

REM 安裝對應 CUDA 版本的 torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 替換 `cu128` 為您的 GPU 對應版本。可在 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查詢。

### 9.5 MSYS2 路徑轉換問題

**問題**：Docker exec 的 `/workspace/...` 路徑被轉為 `C:/Program Files/Git/workspace/...`

**原因**：Git Bash / MSYS2 環境自動轉換 Unix 路徑

**解決**：
```cmd
set MSYS_NO_PATHCONV=1
set MSYS2_ARG_CONV_EXCL=*
```

或使用 `start_server.bat`（已自動設定這些環境變數）。

> `web_server.py` 第 46-47 行也已設定這些變數作為程式內保護。

### 9.6 容器未運行

**問題**：Web 界面報錯 `Docker container 'gopt-pipeline' is not running`

**解決**：
```cmd
REM 檢查容器狀態
docker ps -a --filter "name=gopt-pipeline"

REM 啟動已停止的容器
docker start gopt-pipeline

REM 如果容器不存在，重新創建
docker compose up -d
```

### 9.7 LibriSpeech 模型缺失

**問題**：容器日誌顯示 `⚠ LibriSpeech models not found`

**解決**：
```cmd
docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
```

下載約 2GB，需 10-30 分鐘。中斷後可重新執行（跳過已完成的下載）。

### 9.8 處理結果為空或異常

**檢查清單**：
1. 音訊是否為 16kHz mono WAV？Web 界面會自動轉換，命令列需手動轉換
2. 轉錄文本是否全大寫且無標點？系統會自動轉大寫
3. 容器是否正常運行？`docker ps`
4. 模型是否存在？`docker exec gopt-pipeline ls /workspace/gopt/pretrained_models/gopt_librispeech/`
5. 查看詳細日誌：`docker logs gopt-pipeline`

### 9.9 Git 行尾問題

**問題**：Shell 腳本執行報錯 `/bin/bash^M: bad interpreter`

**原因**：Windows Git 將 LF 轉為 CRLF

**解決**：
```cmd
git config core.autocrlf false
git config core.eol lf
```

然後重新克隆或執行：
```cmd
git rm --cached -r .
git reset --hard
```

> Dockerfile 中已使用 `dos2unix` 轉換所有 `.sh` 文件（第 109 行），提供額外保護。

### 9.10 端口衝突

**問題**：`[Errno 10048] Only one usage of each socket address is normally permitted`

**原因**：端口 8080 已被佔用

**解決**：
```cmd
REM 使用其他端口
python web_server.py --port 8081
```
然後瀏覽器訪問 `http://localhost:8081`

---

## 附錄 A：完整管線流程

GOPT 管線（`run_pipeline_auto.sh`）包含 8 個自動化步驟：

```
1. Prepare Kaldi data directory     → 創建 Kaldi 格式的數據目錄
2. Generate lexicon from transcript → 使用 g2p-en 生成音素詞典
3. Generate text-phone file         → 生成音素級轉錄
4. Generate dummy scores            → 為自訂音訊生成佔位分數
5. Run Kaldi GOP extraction         → MFCC → i-vector → nnet3 → 對齊 → GOP
6. Extract GOP features to CSV      → 提取 84 維特徵 (42 LPP + 42 LPR)
7. Convert to sequence format       → 轉為 numpy 序列格式
8. Run GOPT inference               → Transformer 推論 → 評分結果
```

### 時間估算

| 步驟 | 預估時間 |
|------|---------|
| 音訊格式轉換 | < 5 秒 |
| 詞典/音素生成 | < 5 秒 |
| MFCC 特徵提取 | 10-30 秒 |
| i-vector 提取 | 10-30 秒 |
| nnet3 輸出計算 | 20-60 秒 |
| 對齊 + GOP 計算 | 30-120 秒 |
| 特徵提取 + 轉換 | < 10 秒 |
| GOPT 推論 | < 5 秒 |
| **總計（單段）** | **約 2-5 分鐘** |

---

## 附錄 B：專案文件結構

```
項目根目錄/
├── Dockerfile                        # Docker 鏡像構建
├── docker-compose.yml                # 容器編排
├── web_server.py                     # 宿主機 FastAPI Web 服務
├── models.py                         # GOPT 模型定義
├── inference_api.py                  # Python 推論 API
├── inference_cli.py                  # CLI 推論介面
├── inference_server.py               # REST API 伺服器
├── requirements.txt                  # Python 依賴
├── process_custom_audio_fixed.sh     # Dockerfile 必要文件
├── start_server.bat                  # Windows 啟動腳本
├── start_web_interface.bat           # Windows Web 啟動腳本
├── start_web_interface.sh            # Linux Web 啟動腳本
├── rebuild.bat                       # Windows Docker 重建
├── rebuild.sh                        # Linux Docker 重建
├── install_ffmpeg.ps1                # ffmpeg 安裝 (PowerShell)
├── setup_models.py                   # 模型設置工具
├── convert_audio.py                  # 音訊轉換工具
├── convert_to_seq.py                 # 序列轉換工具
├── pytest.ini                        # 測試配置
├── static/                           # Web 前端
│   ├── index.html
│   ├── app.js
│   └── style.css
├── docker/                           # Docker 腳本
│   ├── run_pipeline_auto.sh          # 核心：全自動管線
│   ├── entrypoint.sh                 # 容器入口
│   ├── setup_models.sh               # 模型下載
│   ├── test_installation.sh          # 安裝驗證
│   ├── process_custom_audio.sh       # 音訊處理入口
│   ├── run_pipeline.sh               # 舊版管線（需手動介入）
│   ├── continue_pipeline.sh          # 管線繼續
│   └── generate_text_phone.py        # text-phone 生成
├── src/                              # 核心源碼
│   ├── whisper_asr.py                # Whisper ASR
│   ├── text_segmenter.py             # 文本分段
│   ├── audio_segmenter.py            # 音訊分段
│   ├── result_merger.py              # 結果合併
│   ├── traintest.py                  # 訓練/測試
│   ├── extract_kaldi_gop/            # Kaldi GOP 提取腳本
│   ├── prep_data/                    # 數據預處理
│   └── utils/                        # 工具函數
├── pretrained_models/                # 預訓練模型
│   ├── gopt_librispeech/
│   ├── gopt_paiia/
│   ├── gopt_paiib/
│   └── models.py
├── data/                             # 數據目錄
├── models/                           # LibriSpeech 模型 (Docker volume)
├── audio_input/                      # 音訊輸入 (Docker volume)
├── audio_output/                     # 結果輸出 (Docker volume)
├── exp/                              # 實驗結果 (Docker volume)
├── scripts/                          # 工具腳本
│   ├── benchmark_inference.py
│   └── compute_phone_pcc.py
├── tests/                            # 單元測試
├── colab/                            # Google Colab 筆記本
└── figure/                           # 架構圖
```

---

## 附錄 C：現有文檔勘誤

以下是在閱讀所有代碼後發現的現有文檔錯誤：

| # | 文檔 | 錯誤描述 | 修正 |
|---|------|---------|------|
| 1 | `README.md` | 「Project Structure」中列出 `src/models/gopt.py` | 該文件**不存在**。GOPT 模型定義在根目錄 `models.py`（被複製到 `src/models.py` 和 `pretrained_models/models.py`） |
| 2 | `README.md` | 「Quick Start」只提到 `python web_server.py` | 缺少宿主機依賴安裝步驟（`pip install fastapi uvicorn aiofiles python-multipart`），也未提及 Whisper ASR 依賴 |
| 3 | `DOCKER_USAGE.md` | 描述手動 Kaldi GOP 提取流程（4.2-4.4 節需要手動 `run.sh` 和 `continue_pipeline.sh`） | 實際已有全自動管線 `run_pipeline_auto.sh`，`process_custom_audio.sh` 直接調用它，無需手動介入 |
| 4 | `DOCKER_USAGE.md` | 引用不存在的文檔：`GOP_EXTRACTION_STEP_BY_STEP_GUIDE.md`、`INFERENCE_API_GUIDE.md` | 這些文件不在倉庫中 |
| 5 | `QUICK_START.md` | 引用 `test_user_audio.py` | 該文件不在倉庫中 |
| 6 | `QUICK_START.md` | 引用 `REBUILD_GUIDE.md`、`MODEL_SETUP_GUIDE.md`、`PROCESS_YOUR_AUDIO_GUIDE.md` | 這些文件不在倉庫中 |
| 7 | `START_HERE.md` | 稱「支持格式」列表包含 WAV 但說「Only .wav files are supported」 | 實際 `web_server.py` 支援 WAV, MP3, M4A, FLAC, OGG, AAC, WMA |
| 8 | `START_HERE.md` | 引用 `WEB_INTERFACE_QUICKSTART.md`、`WEB_INTERFACE_GUIDE.md`、`TUTORIAL_BEGINNER.md` | 這些文件不在倉庫中 |
| 9 | `README.md` | 聲稱 REST API 在 `http://localhost:8000` | `inference_server.py` 使用 8000 端口，但 `web_server.py` 預設 8080 端口。兩者是不同服務 |
| 10 | `docker-compose.yml` | GPU 支援被註釋掉 | 如需 GPU，需取消註釋 `deploy.resources.reservations.devices` 部分 |

---

**文件行數**：約 390 行
**文檔勘誤數量**：10 處
**新增重要步驟**：`process_custom_audio_fixed.sh` 必要性、宿主機依賴安裝、MSYS_NO_PATHCONV 設定、GPU 架構相容性處理、Git 行尾設定