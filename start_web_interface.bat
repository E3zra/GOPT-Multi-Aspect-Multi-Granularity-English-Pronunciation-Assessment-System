@echo off
REM ============================================================================
REM GOPT Web Interface - Quick Start Script (Windows)
REM
REM This script checks prerequisites and starts the web interface server.
REM ============================================================================

echo.
echo ========================================================================
echo            GOPT Web Interface - Quick Start
echo ========================================================================
echo.

REM Check if Python is available
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)
echo [SUCCESS] Python is available

REM Check if required files exist
echo [INFO] Checking required files...
if not exist "web_server.py" (
    echo [ERROR] Required file not found: web_server.py
    pause
    exit /b 1
)
if not exist "static\index.html" (
    echo [ERROR] Required file not found: static\index.html
    pause
    exit /b 1
)
echo [SUCCESS] All required files found

REM Check if directories exist
echo [INFO] Checking directories...
if not exist "audio_input" mkdir audio_input
if not exist "audio_output" mkdir audio_output
if not exist "static" mkdir static
echo [SUCCESS] Directories ready

REM Check Python dependencies
echo [INFO] Checking Python dependencies...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [WARNING] Missing Python dependencies
    echo [INFO] Installing dependencies...
    pip install fastapi uvicorn aiofiles python-multipart
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [SUCCESS] Dependencies installed
) else (
    echo [SUCCESS] All Python dependencies installed
)

REM Check Docker container
echo [INFO] Checking Docker container...
docker ps --filter "name=gopt-pipeline" --format "{{.Names}}" | findstr /C:"gopt-pipeline" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Docker container 'gopt-pipeline' is not running
    echo The web interface will work, but audio processing will fail
    echo Start the container with: docker start gopt-pipeline
) else (
    echo [SUCCESS] Docker container 'gopt-pipeline' is running
)

REM Get port (default 8080)
set PORT=8080
if not "%1"=="" set PORT=%1

echo.
echo ========================================================================
echo Starting GOPT Web Interface Server
echo ========================================================================
echo.
echo [INFO] Server will be available at: http://localhost:%PORT%
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start the server
python web_server.py --host 0.0.0.0 --port %PORT%

pause

