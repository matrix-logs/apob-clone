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

    # KJNodes (required for many workflows)
    if [ ! -d "ComfyUI-KJNodes" ]; then
        log_info "Installing ComfyUI-KJNodes..."
        git clone https://github.com/kijai/ComfyUI-KJNodes.git
    fi

    # Wan 2.2 Animate preprocessing nodes
    if [ ! -d "ComfyUI-WanAnimatePreprocess" ]; then
        log_info "Installing ComfyUI-WanAnimatePreprocess..."
        git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
    fi

    # Segment Anything 2 (SAM2) for character segmentation
    if [ ! -d "ComfyUI-segment-anything-2" ]; then
        log_info "Installing ComfyUI-segment-anything-2..."
        git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
    fi

    # Install SageAttention for faster attention
    log_info "Installing SageAttention..."
    pip install sageattention || true

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
    mkdir -p /workspace/ComfyUI/models/diffusion_models/Wan2.2-Animate-14B
    mkdir -p /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B
    mkdir -p /workspace/ComfyUI/models/text_encoders
    mkdir -p /workspace/ComfyUI/models/sam2
    mkdir -p /workspace/ComfyUI/models/onnx
    mkdir -p /workspace/outputs
    mkdir -p /workspace/inputs
}

# Download Wan 2.2 Animate models
download_animate_models() {
    log_info "Downloading Wan 2.2 Animate models..."
    cd /workspace/ComfyUI/models

    # Wan 2.2 Animate 14B model (FP8 quantized, ~14GB)
    log_info "Downloading Wan2_2-Animate-14B (FP8)..."
    huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled \
        Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
        --local-dir diffusion_models/Wan2.2-Animate-14B || true

    # VAE
    log_info "Downloading VAE..."
    huggingface-cli download Kijai/WanVideo_comfy \
        Wan2_1_VAE_bf16.safetensors \
        --local-dir vae || true

    # Text Encoder
    log_info "Downloading Text Encoder..."
    huggingface-cli download Kijai/WanVideo_comfy \
        umt5-xxl-enc-bf16.safetensors \
        --local-dir text_encoders || true

    # CLIP Vision
    log_info "Downloading CLIP Vision..."
    huggingface-cli download Kijai/WanVideo_comfy \
        clip_vision_h.safetensors \
        --local-dir clip || true

    # LoRAs for Animate
    log_info "Downloading LoRAs..."
    mkdir -p loras/WanVideo/Lightx2v
    huggingface-cli download Kijai/WanVideo_comfy \
        Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
        --local-dir loras/WanVideo || true

    huggingface-cli download Kijai/WanVideo_comfy \
        LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors \
        --local-dir loras/WanVideo || true

    # Detection models (ONNX)
    log_info "Downloading detection models..."
    mkdir -p onnx
    wget -nc https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx \
        -O onnx/yolov10m.onnx || true

    wget -nc https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx \
        -O onnx/vitpose-l-wholebody.onnx || true

    # SAM2 model
    log_info "Downloading SAM2 model..."
    huggingface-cli download facebook/sam2.1-hiera-base-plus \
        sam2.1_hiera_base_plus.safetensors \
        --local-dir sam2 || true

    log_info "Animate models download complete!"
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
    echo "  1. Download Wan 2.2 Animate models:"
    echo "     bash /workspace/scripts/setup_vastai.sh download_animate"
    echo ""
    echo "  2. Copy your API files to /workspace/api/"
    echo "  3. Copy your workflows to /workspace/ComfyUI/user/"
    echo "  4. Start services: bash /workspace/start_all.sh"
    echo ""
    echo "Ports:"
    echo "  - ComfyUI: 8188"
    echo "  - FastAPI: 8000"
    echo ""
}

# Handle command-line arguments
case "${1:-}" in
    download_animate)
        download_animate_models
        ;;
    *)
        main "$@"
        ;;
esac
