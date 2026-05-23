#!/bin/bash
################################################################################
# GOPT Docker 镜像重建脚本
# 
# 用途: 自动化重建流程，包含验证和测试
# 使用: bash rebuild.sh
################################################################################

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo "=============================================================="
echo "GOPT Docker Image Rebuild Script"
echo "=============================================================="
echo ""

# Step 1: Check if Docker is running
echo_info "Step 1: Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo_error "Docker is not running. Please start Docker first."
    exit 1
fi
echo_success "Docker is running"
echo ""

# Step 2: Stop existing container
echo_info "Step 2: Stopping existing container..."
docker-compose down 2>/dev/null || true
echo_success "Container stopped"
echo ""

# Step 3: Check if models exist
echo_info "Step 3: Checking models..."
if [ -d "models/librispeech" ] && [ "$(ls -A models/librispeech)" ]; then
    echo_success "LibriSpeech models found (~2GB)"
    MODEL_EXISTS=true
else
    echo_warning "LibriSpeech models not found"
    echo_info "Models will need to be downloaded after rebuild (~2GB, 10-30 minutes)"
    MODEL_EXISTS=false
fi
echo ""

# Step 4: Rebuild image
echo_info "Step 4: Rebuilding Docker image..."
echo_warning "This may take 20-40 minutes on first build..."
echo ""

if docker-compose build; then
    echo ""
    echo_success "Image rebuilt successfully"
else
    echo ""
    echo_error "Image build failed"
    exit 1
fi
echo ""

# Step 5: Start container
echo_info "Step 5: Starting container..."
if docker-compose up -d; then
    echo_success "Container started"
else
    echo_error "Failed to start container"
    exit 1
fi
echo ""

# Wait for container to initialize
echo_info "Waiting for container initialization (10 seconds)..."
sleep 10
echo ""

# Step 6: Check container logs
echo_info "Step 6: Checking initialization logs..."
echo "------------------------------------------------------------"
docker logs gopt-pipeline 2>&1 | head -30
echo "------------------------------------------------------------"
echo ""

# Step 7: Verify setup
echo_info "Step 7: Verifying setup..."

# Check if container is running
if docker ps | grep -q gopt-pipeline; then
    echo_success "✓ Container is running"
else
    echo_error "✗ Container is not running"
    exit 1
fi

# Check Python
if docker exec gopt-pipeline python3 --version > /dev/null 2>&1; then
    echo_success "✓ Python is available"
else
    echo_error "✗ Python is not available"
fi

# Check PyTorch
if docker exec gopt-pipeline python3 -c "import torch" 2>/dev/null; then
    echo_success "✓ PyTorch is available"
else
    echo_error "✗ PyTorch is not available"
fi

# Check processing script
if docker exec gopt-pipeline test -f /workspace/gopt/process_custom_audio.sh; then
    echo_success "✓ Processing script is in place"
else
    echo_error "✗ Processing script is missing"
fi

# Check model links
if docker exec gopt-pipeline test -L /opt/kaldi/egs/librispeech/s5/exp/nnet3_cleaned/tdnn_sp; then
    echo_success "✓ Model links are created"
else
    echo_warning "⚠ Model links not found (may need model download)"
fi

echo ""

# Step 8: Provide next steps
echo "=============================================================="
echo "Rebuild Complete!"
echo "=============================================================="
echo ""

if [ "$MODEL_EXISTS" = true ]; then
    echo_success "✅ System is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Test with your audio: python test_user_audio.py"
    echo "  2. Or use web interface: python web_server.py --port 8080"
else
    echo_warning "⚠ Models need to be downloaded"
    echo ""
    echo "Next steps:"
    echo "  1. Download models (~2GB, 10-30 minutes):"
    echo "     docker exec gopt-pipeline bash /workspace/scripts/setup_models.sh"
    echo ""
    echo "  2. Restart container after download:"
    echo "     docker-compose restart"
    echo ""
    echo "  3. Then test: python test_user_audio.py"
fi

echo ""
echo "For detailed logs:"
echo "  docker logs gopt-pipeline"
echo ""
echo "For help:"
echo "  See QUICK_START.md or REBUILD_GUIDE.md"
echo ""
echo "=============================================================="

