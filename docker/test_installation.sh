#!/bin/bash
################################################################################
# Test GOPT Docker Installation
#
# This script tests if all components are properly installed
################################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

echo "========================================================================"
echo "           GOPT Docker Installation Test"
echo "========================================================================"
echo ""

# Test Python
print_info "Testing Python installation..."
if python3 --version; then
    print_success "Python installed"
else
    print_error "Python not found"
    exit 1
fi

# Test Python packages
print_info "Testing Python packages..."

packages=("torch" "numpy" "kaldiio" "kaldi_io")
for pkg in "${packages[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        print_success "$pkg installed"
    else
        print_error "$pkg not installed"
    fi
done

# Test Kaldi
print_info "Testing Kaldi..."
if [ -d "/opt/kaldi" ]; then
    print_success "Kaldi directory found"
    if [ -d "/opt/kaldi/egs/gop_speechocean762/s5" ]; then
        print_success "Kaldi GOP recipe found"
    else
        print_error "Kaldi GOP recipe not found"
    fi
else
    print_error "Kaldi not found at /opt/kaldi"
fi

# Test GOPT
print_info "Testing GOPT installation..."
if [ -d "/workspace/gopt" ]; then
    print_success "GOPT directory found"
    
    if [ -f "/workspace/gopt/models.py" ]; then
        print_success "GOPT model code found (models.py)"
    elif [ -f "/workspace/gopt/src/models.py" ]; then
        print_success "GOPT model code found (src/models.py)"
    else
        print_error "GOPT model code not found (checked /workspace/gopt/models.py)"
    fi
    
    if [ -f "/workspace/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth" ]; then
        print_success "Pretrained model found"
    else
        print_error "Pretrained model not found"
    fi
else
    print_error "GOPT directory not found"
fi

# Test directories
print_info "Testing data directories..."
directories=(
    "/workspace/audio_input"
    "/workspace/audio_output"
    "/workspace/models"
    "/workspace/gopt/data"
    "/workspace/gopt/exp"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        print_success "$dir exists"
    else
        print_error "$dir not found"
        mkdir -p "$dir" && print_info "Created $dir"
    fi
done

# Test GOPT model loading
print_info "Testing GOPT model loading..."
python3 - <<'EOF'
import sys
sys.path.append('/workspace/gopt/src')
try:
    from models import GOPT
    import torch
    
    model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    print("[✓] GOPT model can be instantiated")
    
    # Try loading pretrained model
    try:
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('/workspace/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth', 
                                map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print("[✓] Pretrained model can be loaded")
    except Exception as e:
        print(f"[✗] Failed to load pretrained model: {e}")
        
except Exception as e:
    print(f"[✗] Failed to import or instantiate GOPT: {e}")
    sys.exit(1)
EOF

echo ""
echo "========================================================================"
print_success "Installation test completed!"
echo "========================================================================"
echo ""
print_info "Next steps:"
echo "  1. Place audio files in ./audio_input/"
echo "  2. Ensure Librispeech models are in ./models/librispeech/"
echo "  3. Run: docker exec -it gopt-pipeline /workspace/scripts/run_pipeline.sh"
echo ""

