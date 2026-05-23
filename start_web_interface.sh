#!/bin/bash
################################################################################
# GOPT Web Interface - Quick Start Script
#
# This script checks prerequisites and starts the web interface server.
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo ""
echo "========================================================================"
echo "           GOPT Web Interface - Quick Start"
echo "========================================================================"
echo ""

# Check if Python is available
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_success "Python 3 is available: $(python3 --version)"

# Check if required files exist
print_info "Checking required files..."
REQUIRED_FILES=("web_server.py" "static/index.html" "static/style.css" "static/app.js")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file not found: $file"
        exit 1
    fi
done
print_success "All required files found"

# Check if directories exist
print_info "Checking directories..."
mkdir -p audio_input audio_output static
print_success "Directories ready"

# Check Python dependencies
print_info "Checking Python dependencies..."
MISSING_DEPS=()

python3 -c "import fastapi" 2>/dev/null || MISSING_DEPS+=("fastapi")
python3 -c "import uvicorn" 2>/dev/null || MISSING_DEPS+=("uvicorn")
python3 -c "import aiofiles" 2>/dev/null || MISSING_DEPS+=("aiofiles")

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_warning "Missing Python dependencies: ${MISSING_DEPS[*]}"
    print_info "Installing dependencies..."
    pip install fastapi uvicorn aiofiles python-multipart
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed"
    else
        print_error "Failed to install dependencies"
        print_info "Please run: pip install -r requirements.txt"
        exit 1
    fi
else
    print_success "All Python dependencies installed"
fi

# Check Docker container
print_info "Checking Docker container..."
if command -v docker &> /dev/null; then
    if docker ps --filter "name=gopt-pipeline" --format "{{.Names}}" | grep -q "gopt-pipeline"; then
        print_success "Docker container 'gopt-pipeline' is running"
    else
        print_warning "Docker container 'gopt-pipeline' is not running"
        print_info "The web interface will work, but audio processing will fail"
        print_info "Start the container with: docker start gopt-pipeline"
    fi
else
    print_warning "Docker is not installed or not in PATH"
fi

# Get port
PORT=${1:-8080}

echo ""
echo "========================================================================"
echo "Starting GOPT Web Interface Server"
echo "========================================================================"
echo ""
print_info "Server will be available at: http://localhost:$PORT"
print_info "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 web_server.py --host 0.0.0.0 --port $PORT

