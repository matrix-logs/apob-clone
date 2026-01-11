#!/bin/bash
# Vast.ai RTX 5090 Setup Script for APOB Clone
# Run this script after SSH-ing into your Vast.ai instance

set -e

echo "=========================================="
echo "  APOB Clone - Vast.ai Setup Script"
echo "  Target: RTX 5090 (32GB VRAM)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running on Vast.ai or similar cloud GPU
check_gpu() {
    log_info "Checking GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is this a GPU instance?"
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    log_info "GPU check passed!"
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    apt-get update -qq
    apt-get install -y -qq git wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 \
        build-essential python3-pip python3-venv
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."

    # Create virtual environment
    python3 -m venv /workspace/venv
    source /workspace/venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install PyTorch with CUDA 12.8 support (for RTX 5090)
    log_info "Installing PyTorch with CUDA 12.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    # Verify CUDA is available
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
}

# Install ComfyUI
install_comfyui() {
    log_info "Installing ComfyUI..."
    cd /workspace

    if [ ! -d "ComfyUI" ]; then
        git clone https://github.com/comfyanonymous/ComfyUI.git
    fi

    cd ComfyUI
    pip install -r requirements.txt

    # Install ComfyUI Manager
    cd custom_nodes
    if [ ! -d "ComfyUI-Manager" ]; then
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    fi

    log_info "ComfyUI installed successfully!"
}

# Install required custom nodes for ComfyUI
install_custom_nodes() {
    log_info "Installing ComfyUI custom nodes..."
    cd /workspace/ComfyUI/custom_nodes

    # Flux support
    if [ ! -d "ComfyUI-Flux" ]; then
        log_info "Installing ComfyUI-Flux..."
        git clone https://github.com/kijai/ComfyUI-FluxTrainer.git || true
    fi

    # PuLID for consistent character
    if [ ! -d "PuLID-ComfyUI" ]; then
        log_info "Installing PuLID-ComfyUI..."
        git clone https://github.com/cubiq/PuLID_ComfyUI.git || true
    fi

    # Wan 2.2 video generation
    if [ ! -d "ComfyUI-WAN" ]; then
        log_info "Installing ComfyUI-WAN..."
        git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git || true
    fi

    # ReActor face swap
    if [ ! -d "comfyui-reactor-node" ]; then
        log_info "Installing ComfyUI-ReActor..."
        git clone https://github.com/Gourieff/comfyui-reactor-node.git
        cd comfyui-reactor-node
        pip install -r requirements.txt
        cd ..
    fi

    # SadTalker / Audio2Face
    if [ ! -d "ComfyUI-LivePortrait" ]; then
        log_info "Installing ComfyUI-LivePortrait..."
        git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git || true
    fi

    # Video helper suite
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        log_info "Installing ComfyUI-VideoHelperSuite..."
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    fi

    # Install all node dependencies
    for dir in */; do
        if [ -f "${dir}requirements.txt" ]; then
            log_info "Installing dependencies for ${dir}..."
            pip install -r "${dir}requirements.txt" || true
        fi
    done

    log_info "Custom nodes installed!"
}

# Install FastAPI and other Python dependencies
install_api_deps() {
    log_info "Installing FastAPI and API dependencies..."
    pip install \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        aiofiles \
        websockets \
        httpx \
        pillow \
        numpy \
        pydantic \
        redis \
        celery \
        huggingface-hub
}

# Create directories for models
create_model_dirs() {
    log_info "Creating model directories..."
    mkdir -p /workspace/ComfyUI/models/checkpoints
    mkdir -p /workspace/ComfyUI/models/clip
    mkdir -p /workspace/ComfyUI/models/vae
    mkdir -p /workspace/ComfyUI/models/loras
    mkdir -p /workspace/ComfyUI/models/controlnet
    mkdir -p /workspace/ComfyUI/models/pulid
    mkdir -p /workspace/ComfyUI/models/insightface
    mkdir -p /workspace/ComfyUI/models/sadtalker
    mkdir -p /workspace/ComfyUI/models/wan
    mkdir -p /workspace/outputs
    mkdir -p /workspace/inputs
}

# Create startup script
create_startup_script() {
    log_info "Creating startup scripts..."

    cat > /workspace/start_comfyui.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header
EOF
    chmod +x /workspace/start_comfyui.sh

    cat > /workspace/start_api.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF
    chmod +x /workspace/start_api.sh

    cat > /workspace/start_all.sh << 'EOF'
#!/bin/bash
echo "Starting ComfyUI..."
/workspace/start_comfyui.sh &
sleep 10
echo "Starting API server..."
/workspace/start_api.sh &
echo "All services started!"
echo "ComfyUI: http://localhost:8188"
echo "API: http://localhost:8000"
wait
EOF
    chmod +x /workspace/start_all.sh
}

# Main execution
main() {
    check_gpu
    update_system
    setup_python
    install_comfyui
    install_custom_nodes
    install_api_deps
    create_model_dirs
    create_startup_script

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run: bash /workspace/scripts/download_models.sh"
    echo "  2. Copy your API files to /workspace/api/"
    echo "  3. Copy your workflows to /workspace/ComfyUI/user/"
    echo "  4. Start services: bash /workspace/start_all.sh"
    echo ""
    echo "Ports:"
    echo "  - ComfyUI: 8188"
    echo "  - FastAPI: 8000"
    echo ""
}

main "$@"
