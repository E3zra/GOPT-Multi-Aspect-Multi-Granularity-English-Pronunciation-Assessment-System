# Pull Request Summary: fix/pipeline-normalization-docker-web

**Branch**: `fix/pipeline-normalization-docker-web`
**Base**: `17ba7f2` (Clean fork: web interface, Docker, Whisper ASR, OOV handling)
**Commits**: 12 (5 fix, 2 feat, 3 docs, 1 test, 1 chore)
**Files Changed**: 22 unique files, ~3,300 lines added, ~472,300 lines removed (mostly large data files excluded)

---

## What This Branch Does

The GOPT pronunciation assessment system had several critical bugs that prevented it from working end-to-end on custom audio. The pipeline produced `word_scores` in a dict-of-arrays format (`{accuracy: [], stress: [], total: []}`), but both the Python result merger (`result_merger.py`) and the frontend JavaScript (`app.js`) only handled a list-of-dicts format, causing the entire word-level scoring display to break. Additionally, the Docker container could not find `models.py` because `inference_api.py` imported it from multiple locations that weren't populated. On Windows, MSYS2/Git Bash automatically converted Docker paths like `/workspace/...` to `C:/Program Files/Git/workspace/...`, breaking all `docker exec` calls. This branch fixes all these issues, adds a fully automated pipeline script (replacing manual 8-step process), provides comprehensive setup documentation, and includes test audio files for verification.

---

## Test Results

| Audio File | Expected Range | Status |
|-----------|---------------|--------|
| `better.mp3` | Total: 1.2 - 1.3 | Verified on dev machine |
| `normal.mp3` | Total: 1.0 - 1.1 | Verified on dev machine |
| `low.mp3` | Total: 0.8 - 0.9 | Verified on dev machine |

Transcript: "HI I AM GORDON MY ESSAY IS ABOUT EARLY MORNING CLASS AND HOW THEY AFFECT STUDENTS PERFORMANCE AND HEALTHY"

Verified on: 2 Windows machines (original dev machine + clean test machine)

---

## Changes Made (12 commits)

### Bug Fixes (Critical)

#### 1. `00910fb` — fix: Fix result_merger to support two word_scores data formats
- **File changed**: `src/result_merger.py` (+28/-13)
- **Bug**: `_merge_word_scores()` only accepted `list-of-dicts` format (e.g., `[{"word_id": 0, "accuracy": 1.66}]`), but the GOPT pipeline actually produces `dict-of-arrays` format (e.g., `{"accuracy": [1.66], "stress": [1.5], "total": [1.58]}`). This caused long audio segment merging to fail silently.
- **Fix**: Added `isinstance(word_scores, dict)` check to handle both formats. For dict-of-arrays, extends merged lists directly. For list-of-dicts, extracts values from each dict.
- **Why it matters**: Without this fix, multi-segment audio results (long recordings) would produce empty or incorrect word scores.

#### 2. `bc3f3c0` — fix: Frontend app.js supports both dict-of-arrays and list-of-dicts word_scores formats
- **File changed**: `static/app.js` (+48/-9)
- **Bug**: `displayWordScores()` and `calculateFeedback()` only handled list-of-dicts format. When receiving dict-of-arrays from the pipeline, the word score display section was empty or threw JavaScript errors.
- **Fix**: Added format detection logic — if `wordScores` is an array, use directly; if object, convert `{accuracy: [], stress: [], total: []}` to array of objects. Added null-safe property access (`wordData.total || 0`).
- **Why it matters**: Users could not see word-level pronunciation feedback in the web interface, which is the main visual output of the system.

#### 3. `f19e726` — fix: Docker ensures models.py is deployed to all Python import paths
- **Files changed**: `Dockerfile` (+4), `docker/entrypoint.sh` (+38/-15)
- **Bug**: `inference_api.py` imports `models` from multiple locations (`src/`, `pretrained_models/`, root). Only the root `models.py` existed in the container, causing `ModuleNotFoundError` in some code paths.
- **Fix**: Dockerfile adds `RUN cp models.py src/models.py && cp models.py pretrained_models/models.py`. Entrypoint.sh adds step 5/5 to copy models.py at runtime (survives container restarts without rebuild).
- **Why it matters**: GOPT inference failed with import errors on some code paths, making the pipeline unusable.

#### 4. `22bf2de` — fix: Increase max_phonemes 45→100 and fix Windows MSYS path conversion
- **File changed**: `web_server.py` (+6/-1)
- **Bug 1**: `TextSegmenter(max_phonemes=45)` was too aggressive — short sentences were unnecessarily split into multiple segments, adding processing overhead and potentially affecting score accuracy.
- **Bug 2**: No MSYS2 path protection. On Windows with Git Bash, `docker exec` paths like `/workspace/audio_input/test.wav` were converted to `C:/Program Files/Git/workspace/audio_input/test.wav`, causing all pipeline commands to fail.
- **Fix**: Changed `max_phonemes=100`. Added `os.environ["MSYS_NO_PATHCONV"] = "1"` and `os.environ["MSYS2_ARG_CONV_EXCL"] = "*"` at module level.
- **Why it matters**: Short sentences no longer waste time on unnecessary segmentation. Windows users no longer need to manually set environment variables.

#### 5. `2ec5308` — fix: Correct model path in test script, update docs with SSL fix and branch warning, fix ffmpeg script encoding
- **Files changed**: `docker/test_installation.sh` (+8/-3), `START_LOCAL.md` (+15/-2), `install_ffmpeg.ps1` (rewritten)
- **Bug 1**: `test_installation.sh` checked for `/workspace/gopt/src/models/gopt.py` which doesn't exist. The actual model file is at `/workspace/gopt/models.py`. This caused `[✗] GOPT model code not found` on every run even though the model loaded correctly.
- **Bug 2**: `install_ffmpeg.ps1` contained garbled Chinese characters causing PowerShell syntax errors on new machines.
- **Fix**: Updated path check to `/workspace/gopt/models.py` with fallback to `/workspace/gopt/src/models.py`. Rewrote install_ffmpeg.ps1 entirely in English. Added SSL certificate error workaround and branch warning to START_LOCAL.md.
- **Why it matters**: New machine setup was blocked by misleading test failures and broken PowerShell script.

### New Features

#### 6. `ec30a34` — feat: Add fully automated GOPT pipeline script and supporting files
- **Files changed**: 8 files, +1,589 lines
  - `docker/run_pipeline_auto.sh` (939 lines) — Core automated pipeline
  - `models.py`, `docker/models.py`, `pretrained_models/models.py` — GOPT model definitions at all import paths
  - `process_custom_audio_fixed.sh` — Required by Dockerfile line 117
  - `docker/process_custom_audio.sh` — Wrapper script
  - `docker/generate_text_phone.py` (132 lines) — text-phone transcript generator
  - `docker/extract_models.py` (13 lines) — Model extraction helper
- **What was added**: `run_pipeline_auto.sh` automates the entire 8-step GOPT pipeline that previously required manual execution: prepare data → generate lexicon → text-phone → dummy scores → Kaldi GOP → extract CSV → convert to sequence → GOPT inference.
- **Why it was needed**: The original workflow required running 8+ separate commands manually with exact parameters. One wrong step would fail silently. The automated script handles errors, provides progress output, and can be called with a single command.

#### 7. `edb2fbe` — feat: Add start_server.bat Windows startup script
- **File changed**: `start_server.bat` (+4 lines)
- **What was added**: A 4-line batch file that sets `MSYS_NO_PATHCONV=1` and `MSYS2_ARG_CONV_EXCL=*` before starting `web_server.py`.
- **Why it was needed**: Windows users kept hitting the MSYS2 path conversion bug. Instead of documenting the workaround, this script makes it automatic.

### Documentation

#### 8. `3bc116f` — docs: Add START_LOCAL.md complete local setup guide (747 lines)
- **File changed**: `START_LOCAL.md` (+747 lines)
- **What was documented**: A comprehensive, code-verified setup guide with 9 sections (System Requirements, Environment Setup, Source Code & Models, Docker Configuration, Web Service, Testing, Daily Usage, Known Issues, Troubleshooting) and 3 appendices (Pipeline Technical Details, Project File Structure, Documentation Error Corrections).
- **What errors were corrected** (10 errors found across existing documentation):

  | # | Document | Error | Correction |
  |---|----------|-------|------------|
  | 1 | `README.md` | Lists `src/models/gopt.py` in Project Structure | File doesn't exist; model is at `models.py` |
  | 2 | `README.md` | Quick Start only says `python web_server.py` | Missing host-side package install (`pip install fastapi uvicorn aiofiles python-multipart`) and Whisper ASR dependency |
  | 3 | `DOCKER_USAGE.md` | Describes manual Kaldi GOP extraction (sections 4.2-4.4) | Fully automated `run_pipeline_auto.sh` replaces this |
  | 4 | `DOCKER_USAGE.md` | References `GOP_EXTRACTION_STEP_BY_STEP_GUIDE.md`, `INFERENCE_API_GUIDE.md` | Files don't exist |
  | 5 | `QUICK_START.md` | References `test_user_audio.py` | File doesn't exist |
  | 6 | `QUICK_START.md` | References `REBUILD_GUIDE.md`, `MODEL_SETUP_GUIDE.md`, `PROCESS_YOUR_AUDIO_GUIDE.md` | Files don't exist |
  | 7 | `START_HERE.md` | Claims "Only .wav files are supported" | Code supports WAV, MP3, M4A, FLAC, OGG, AAC, WMA |
  | 8 | `START_HERE.md` | References `WEB_INTERFACE_QUICKSTART.md`, `WEB_INTERFACE_GUIDE.md`, `TUTORIAL_BEGINNER.md` | Files don't exist |
  | 9 | `README.md` | REST API at `http://localhost:8000` | `inference_server.py` uses 8000, `web_server.py` uses 8080 (different services) |
  | 10 | `docker-compose.yml` | GPU support commented out | Must uncomment `deploy.resources.reservations.devices` section |

#### 9. `344b648` — docs: Add START_LOCAL.md notice at top of README
- **File changed**: `README.md` (+52/-42)
- **What was documented**: Added a warning box at the top of README.md directing users to START_LOCAL.md for setup, noting known issues in README.

#### 10. `0d945f3` — docs: Translate all Chinese text to English across README.md and START_LOCAL.md
- **Files changed**: `README.md` (+83/-), `START_LOCAL.md` (rewritten 738→390 lines)
- **What was documented**: Fixed garbled Unicode tree characters in README.md Project Structure section. Replaced garbled Chinese section (lines 234-270) with English "About This Project" section. Rewrote entire START_LOCAL.md from corrupted Chinese to clean English.

### Maintenance

#### 11. `89c44a0` — chore: Add data/temp_custom_audio/ to .gitignore
- **File changed**: `.gitignore` (+1 line)
- **What was done**: Excluded runtime temp directory containing task processing intermediate files from version control.

### Testing

#### 12. `2d27ea9` — test: Add three sample MP3 files for system verification
- **Files changed**: `testing_audio_file/` (+4 files)
  - `better.mp3` (190 KB) — Higher quality pronunciation
  - `normal.mp3` (223 KB) — Average pronunciation
  - `low.mp3` (311 KB) — Lower quality pronunciation
  - `text.txt` — Shared transcript text
- **What was added**: Three test audio recordings with known expected score ranges for system verification after setup.

---

## Files Changed Overview

| File | Change Type | Description |
|------|-------------|-------------|
| `src/result_merger.py` | Modified | Fixed `_merge_word_scores()` to handle dict-of-arrays format |
| `static/app.js` | Modified | Fixed `displayWordScores()` and `calculateFeedback()` for dual formats |
| `Dockerfile` | Modified | Added `models.py` copy to `src/` and `pretrained_models/` |
| `docker/entrypoint.sh` | Modified | Added step 5/5 for runtime models.py deployment + pipeline script setup |
| `web_server.py` | Modified | Increased max_phonemes 45→100, added MSYS_NO_PATHCONV env vars |
| `docker/test_installation.sh` | Fixed | Corrected model path check from `src/models/gopt.py` to `models.py` |
| `install_ffmpeg.ps1` | Rewritten | Translated all garbled Chinese to English, same logic preserved |
| `docker/run_pipeline_auto.sh` | **New** (939 lines) | Full automated GOPT pipeline script |
| `models.py` | **New** | GOPT Transformer model definition (root copy) |
| `docker/models.py` | **New** | GOPT model definition (Docker copy) |
| `pretrained_models/models.py` | **New** | GOPT model definition (pretrained_models copy) |
| `process_custom_audio_fixed.sh` | **New** | Required by Dockerfile line 117 |
| `docker/process_custom_audio.sh` | **New** | Wrapper for run_pipeline_auto.sh |
| `docker/generate_text_phone.py` | **New** (132 lines) | Text-phone transcript generator |
| `docker/extract_models.py` | **New** | Model extraction helper |
| `start_server.bat` | **New** | Windows startup with MSYS fix |
| `START_LOCAL.md` | **New** (rewritten) | Complete setup guide in English |
| `README.md` | Modified | Added START_LOCAL.md notice, fixed encoding, translated Chinese |
| `.gitignore` | Modified | Added `data/temp_custom_audio/` |
| `testing_audio_file/better.mp3` | **New** (190 KB) | Test audio |
| `testing_audio_file/normal.mp3` | **New** (223 KB) | Test audio |
| `testing_audio_file/low.mp3` | **New** (311 KB) | Test audio |
| `testing_audio_file/text.txt` | **New** | Shared transcript for test audio |

---

## Known Issues Still Present

- [✗] `test_installation.sh` `[✗] GOPT model code not found` — **FIXED in this branch** (commit `2ec5308`). The check now correctly finds `/workspace/gopt/models.py`.
- [!] `README.md` still contains some outdated content (references to `src/models/gopt.py`, incorrect port numbers) — documented in START_LOCAL.md Appendix C.
- [!] `DOCKER_USAGE.md` and `QUICK_START.md` still contain references to non-existent files — documented but not fixed (out of scope).
- [!] `main` branch has not been tested with these fixes — all testing was done on `fix/pipeline-normalization-docker-web`.

---

## Recommendation for Teammates

### Should this be merged to main?

**Pros:**
1. **Critical bug fixes**: Without commits `00910fb`, `bc3f3c0`, and `f19e726`, the system cannot correctly display word-level scores or run GOPT inference. These are not cosmetic issues — the core scoring output is broken without them.
2. **Windows compatibility**: The MSYS2 path conversion fix (`22bf2de`) is essential for any Windows user. Without it, all `docker exec` commands fail silently.
3. **Automated pipeline**: `run_pipeline_auto.sh` replaces a manual 8-step process that was error-prone and undocumented. This is the primary way to process custom audio.
4. **Verified setup guide**: `START_LOCAL.md` was written by reading all 20+ source files and has been used to set up the system on a clean machine.
5. **Test files included**: Three MP3 files with expected score ranges make it easy to verify the system works.

**Cons:**
1. **`main` branch not tested**: These changes have only been tested on `fix/pipeline-normalization-docker-web`. Merging should be followed by testing on `main`.
2. **Documentation debt**: `DOCKER_USAGE.md`, `QUICK_START.md`, and `START_HERE.md` still contain outdated references. START_LOCAL.md documents these but doesn't fix them.
3. **Large commit for pipeline script**: `ec30a34` adds 1,589 lines in one commit. The `run_pipeline_auto.sh` (939 lines) is the largest new file and should be reviewed carefully.
4. **Three copies of models.py**: The model definition exists at root, `src/`, and `pretrained_models/`. This is a workaround for import path issues but creates maintenance overhead (changes must be synced to all three).

**Verdict**: Recommend merge. The bug fixes are critical for the system to function. The documentation improvements and automated pipeline significantly improve usability. The cons are manageable and should be addressed in follow-up PRs.

---

## How to Test This Branch

```bash
# 1. Fetch and checkout the branch
git fetch origin
git checkout fix/pipeline-normalization-docker-web

# 2. Follow START_LOCAL.md for setup
# Key steps:
#    a. Install Docker Desktop, Python 3.10+, Git
#    b. pip install fastapi uvicorn aiofiles python-multipart openai-whisper numba psutil pydantic
#    c. docker compose build
#    d. docker compose up -d
#    e. docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh  # ~2GB, first time only
#    f. python web_server.py

# 3. Verify installation
docker exec gopt-pipeline bash /workspace/scripts/test_installation.sh

# 4. Test with provided audio files
# Upload testing_audio_file/better.mp3 with transcript from testing_audio_file/text.txt
# Expected: Total score 1.2 - 1.3

# 5. Test command line pipeline
docker exec gopt-pipeline bash /workspace/gopt/process_custom_audio.sh \
  /workspace/audio_input/better.mp3 \
  "HI I AM GORDON MY ESSAY IS ABOUT EARLY MORNING CLASS AND HOW THEY AFFECT STUDENTS PERFORMANCE AND HEALTHY" \
  test_better
```

---

**Generated**: 2026-05-29
**Author**: pl-MC023
**Total Commits**: 12 (on top of base commit `17ba7f2`)