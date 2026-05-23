@echo off
REM ################################################################################
REM GOPT Docker 镜像重建脚本 (Windows)
REM 
REM 用途: 自动化重建流程，包含验证和测试
REM 使用: rebuild.bat
REM ################################################################################

echo ==============================================================
echo GOPT Docker Image Rebuild Script
echo ==============================================================
echo.

REM Step 1: Check if Docker is running
echo [INFO] Step 1: Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker first.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is running
echo.

REM Step 2: Stop existing container
echo [INFO] Step 2: Stopping existing container...
docker-compose down 2>nul
echo [SUCCESS] Container stopped
echo.

REM Step 3: Check if models exist
echo [INFO] Step 3: Checking models...
if exist "models\librispeech" (
    dir /b "models\librispeech" | findstr "^" >nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] LibriSpeech models found (~2GB^)
        set MODEL_EXISTS=true
    ) else (
        echo [WARNING] LibriSpeech models directory is empty
        set MODEL_EXISTS=false
    )
) else (
    echo [WARNING] LibriSpeech models not found
    set MODEL_EXISTS=false
)
echo.

REM Step 4: Rebuild image
echo [INFO] Step 4: Rebuilding Docker image...
echo [WARNING] This may take 20-40 minutes on first build...
echo.

docker-compose build
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Image build failed
    pause
    exit /b 1
)
echo.
echo [SUCCESS] Image rebuilt successfully
echo.

REM Step 5: Start container
echo [INFO] Step 5: Starting container...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container
    pause
    exit /b 1
)
echo [SUCCESS] Container started
echo.

REM Wait for container to initialize
echo [INFO] Waiting for container initialization (10 seconds^)...
timeout /t 10 /nobreak >nul
echo.

REM Step 6: Check container logs
echo [INFO] Step 6: Checking initialization logs...
echo ------------------------------------------------------------
docker logs gopt-pipeline 2>&1 | more +0
echo ------------------------------------------------------------
echo.

REM Step 7: Verify setup
echo [INFO] Step 7: Verifying setup...

REM Check if container is running
docker ps | findstr "gopt-pipeline" >nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Container is running
) else (
    echo [ERROR] Container is not running
    pause
    exit /b 1
)

REM Check Python
docker exec gopt-pipeline python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Python is available
) else (
    echo [ERROR] Python is not available
)

REM Check PyTorch
docker exec gopt-pipeline python3 -c "import torch" 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] PyTorch is available
) else (
    echo [ERROR] PyTorch is not available
)

REM Check processing script
docker exec gopt-pipeline test -f /workspace/gopt/process_custom_audio.sh 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Processing script is in place
) else (
    echo [ERROR] Processing script is missing
)

REM Check model links
docker exec gopt-pipeline test -L /opt/kaldi/egs/librispeech/s5/exp/nnet3_cleaned/tdnn_sp 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Model links are created
) else (
    echo [WARNING] Model links not found (may need model download^)
)

echo.

REM Step 8: Provide next steps
echo ==============================================================
echo Rebuild Complete!
echo ==============================================================
echo.

if "%MODEL_EXISTS%"=="true" (
    echo [SUCCESS] System is ready to use!
    echo.
    echo Next steps:
    echo   1. Test with your audio: python test_user_audio.py
    echo   2. Or use web interface: python web_server.py --port 8080
) else (
    echo [WARNING] Models need to be downloaded
    echo.
    echo Next steps:
    echo   1. Download models (~2GB, 10-30 minutes^):
    echo      docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh
    echo.
    echo   2. Restart container after download:
    echo      docker-compose restart
    echo.
    echo   3. Then test: python test_user_audio.py
)

echo.
echo For detailed logs:
echo   docker logs gopt-pipeline
echo.
echo For help:
echo   See QUICK_START.md or REBUILD_GUIDE.md
echo.
echo ==============================================================
echo.
pause

