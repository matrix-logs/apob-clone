#!/bin/bash
# =============================================================================
# Vast.ai Setup Script for APOB Clone
# =============================================================================
# This script sets up a complete ComfyUI environment for AI video/image generation
# including Wan 2.2 Animate, I2V, PuLID-Flux, ReActor, and LivePortrait.
#
# Target Hardware: RTX 5090 (32GB VRAM) or RTX 4090 (24GB VRAM)
# CUDA Version: 12.8+
#
# Usage:
#   bash setup_vastai.sh              # Full setup (no model downloads)
#   bash setup_vastai.sh download_i2v      # Download I2V models
#   bash setup_vastai.sh download_animate  # Download Animate models
#   bash setup_vastai.sh download_all      # Download all models
#   bash setup_vastai.sh fix_paths         # Fix model paths for workflows
# =============================================================================

set -e

echo "=========================================="
echo "  APOB Clone - Vast.ai Setup Script"
echo "  Version: 2.0 (January 2026)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# =============================================================================
# SYSTEM CHECKS
# =============================================================================

check_gpu() {
    log_section "GPU Check"
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is this a GPU instance?"
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    log_info "GPU check passed!"
}

# =============================================================================
# SYSTEM SETUP
# =============================================================================

update_system() {
    log_section "System Packages"
    apt-get update -qq
    apt-get install -y -qq git git-lfs wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 \
        build-essential python3-pip python3-venv
}

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

setup_python() {
    log_section "Python Environment"

    # Use existing venv or create new one
    if [ -d "/workspace/venv" ]; then
        log_info "Using existing virtual environment..."
    else
        log_info "Creating virtual environment..."
        python3 -m venv /workspace/venv
    fi

    source /workspace/venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install PyTorch with CUDA 12.8 support (for RTX 5090)
    log_info "Installing PyTorch with CUDA 12.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    # Verify CUDA is available
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
}

# =============================================================================
# COMFYUI INSTALLATION
# =============================================================================

install_comfyui() {
    log_section "ComfyUI Installation"
    cd /workspace

    if [ ! -d "ComfyUI" ]; then
        git clone https://github.com/comfyanonymous/ComfyUI.git
    fi

    cd ComfyUI
    pip install -r requirements.txt

    # Create user workflow directory
    mkdir -p /workspace/ComfyUI/user/default/workflows

    log_info "ComfyUI installed successfully!"
}

# =============================================================================
# CUSTOM NODES INSTALLATION
# =============================================================================

install_custom_nodes() {
    log_section "Custom Nodes Installation"
    cd /workspace/ComfyUI/custom_nodes

    # ComfyUI Manager (for easy node management)
    if [ ! -d "ComfyUI-Manager" ]; then
        log_info "Installing ComfyUI-Manager..."
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    fi

    # PuLID-Flux for consistent character (face identity)
    if [ ! -d "ComfyUI-PuLID-Flux" ]; then
        log_info "Installing ComfyUI-PuLID-Flux..."
        git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git
    fi

    # Original PuLID nodes (for SDXL)
    if [ ! -d "PuLID_ComfyUI" ]; then
        log_info "Installing PuLID_ComfyUI..."
        git clone https://github.com/cubiq/PuLID_ComfyUI.git || true
    fi

    # Wan Video Wrapper (I2V and Animate support)
    if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
        log_info "Installing ComfyUI-WanVideoWrapper..."
        git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
    fi

    # Wan 2.2 Animate Preprocessing (pose detection, face extraction)
    # CRITICAL: Required for Wan 2.2 Animate workflow
    if [ ! -d "ComfyUI-WanAnimatePreprocess" ]; then
        log_info "Installing ComfyUI-WanAnimatePreprocess..."
        git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
    fi

    # Segment Anything 2 (SAM2) for character segmentation
    # CRITICAL: Required for Wan 2.2 Animate workflow
    if [ ! -d "ComfyUI-segment-anything-2" ]; then
        log_info "Installing ComfyUI-segment-anything-2..."
        git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
    fi

    # Video Helper Suite (video I/O)
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        log_info "Installing ComfyUI-VideoHelperSuite..."
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    fi

    # KJNodes (utility nodes, required for many workflows)
    if [ ! -d "ComfyUI-KJNodes" ]; then
        log_info "Installing ComfyUI-KJNodes..."
        git clone https://github.com/kijai/ComfyUI-KJNodes.git
    fi

    # LivePortrait (face animation)
    if [ ! -d "ComfyUI-LivePortraitKJ" ]; then
        log_info "Installing ComfyUI-LivePortraitKJ..."
        git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git || true
    fi

    # ReActor (face swap)
    if [ ! -d "comfyui-reactor-node" ]; then
        log_info "Installing comfyui-reactor-node..."
        git clone https://github.com/Gourieff/comfyui-reactor-node.git
        cd comfyui-reactor-node
        pip install -r requirements.txt || true
        cd ..
    fi

    # Install SageAttention for faster attention
    log_info "Installing SageAttention..."
    pip install sageattention || true

    # Install all node dependencies
    log_info "Installing custom node dependencies..."
    for dir in */; do
        if [ -f "${dir}requirements.txt" ]; then
            log_info "Installing dependencies for ${dir}..."
            pip install -r "${dir}requirements.txt" || true
        fi
    done

    log_info "Custom nodes installed!"
}

# =============================================================================
# APPLY PATCHES AND FIXES
# =============================================================================

apply_patches() {
    log_section "Applying Patches"

    # PuLID-Flux compatibility fix for newer ComfyUI versions
    PULID_FILE="/workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py"
    if [ -f "$PULID_FILE" ]; then
        if ! grep -q "transformer_options={}" "$PULID_FILE"; then
            log_info "Applying PuLID-Flux compatibility patch..."
            sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' "$PULID_FILE"
            log_info "PuLID-Flux patch applied!"
        else
            log_info "PuLID-Flux patch already applied."
        fi
    fi
}

# =============================================================================
# API DEPENDENCIES
# =============================================================================

install_api_deps() {
    log_section "API Dependencies"
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
        huggingface-hub \
        onnxruntime \
        onnxruntime-gpu
}

# =============================================================================
# MODEL DIRECTORIES
# =============================================================================

create_model_dirs() {
    log_section "Model Directories"

    # Standard ComfyUI model directories
    mkdir -p /workspace/ComfyUI/models/checkpoints
    mkdir -p /workspace/ComfyUI/models/clip
    mkdir -p /workspace/ComfyUI/models/vae
    mkdir -p /workspace/ComfyUI/models/loras
    mkdir -p /workspace/ComfyUI/models/controlnet
    mkdir -p /workspace/ComfyUI/models/unet
    mkdir -p /workspace/ComfyUI/models/pulid
    mkdir -p /workspace/ComfyUI/models/insightface/models/antelopev2
    mkdir -p /workspace/ComfyUI/models/facerestore_models
    mkdir -p /workspace/ComfyUI/models/text_encoders
    mkdir -p /workspace/ComfyUI/models/sam2

    # CRITICAL: Detection models must be in models/detection/ for WanAnimatePreprocess
    # NOT in models/onnx/ - this is a common mistake!
    mkdir -p /workspace/ComfyUI/models/detection

    # Wan model directories
    mkdir -p /workspace/ComfyUI/models/diffusion_models/Wan2.2-Animate-14B
    mkdir -p /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B

    # LoRA directories with proper structure
    mkdir -p /workspace/ComfyUI/models/loras/WanVideo/Lightx2v
    mkdir -p /workspace/ComfyUI/models/loras/WanVideo/LoRAs/Wan22_relight

    # Workflow and output directories
    mkdir -p /workspace/ComfyUI/user/default/workflows
    mkdir -p /workspace/outputs
    mkdir -p /workspace/inputs

    log_info "Model directories created!"
}

# =============================================================================
# MODEL DOWNLOADS - WAN 2.2 I2V
# =============================================================================

download_i2v_models() {
    log_section "Downloading Wan 2.2 I2V Models"
    cd /workspace/ComfyUI/models

    # Wan 2.2 I2V model (full precision, ~28GB per transformer file)
    log_info "Downloading Wan2.2-I2V-A14B model..."
    huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-480P \
        --local-dir diffusion_models/Wan2.2-I2V-A14B || true

    # Create symlinks for common paths
    log_info "Creating symlinks..."
    if [ -f "diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth" ]; then
        ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
            /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth || true
    fi

    log_info "I2V models download complete!"
}

# =============================================================================
# MODEL DOWNLOADS - WAN 2.2 ANIMATE
# =============================================================================

download_animate_models() {
    log_section "Downloading Wan 2.2 Animate Models"
    cd /workspace/ComfyUI/models

    # Wan 2.2 Animate 14B model (FP8 quantized, ~18GB)
    log_info "Downloading Wan2_2-Animate-14B (FP8)..."
    huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled \
        Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
        --local-dir diffusion_models/Wan2.2-Animate-14B || true

    # VAE
    log_info "Downloading VAE..."
    huggingface-cli download Kijai/WanVideo_comfy \
        Wan2_1_VAE_bf16.safetensors \
        --local-dir vae || true

    # Text Encoder (UMT5-XXL)
    log_info "Downloading Text Encoder (UMT5-XXL)..."
    huggingface-cli download Kijai/WanVideo_comfy \
        umt5-xxl-enc-bf16.safetensors \
        --local-dir text_encoders || true

    # CLIP Vision H
    log_info "Downloading CLIP Vision H..."
    huggingface-cli download h94/IP-Adapter \
        models/image_encoder/model.safetensors \
        --local-dir clip_temp || true

    # Move CLIP Vision to correct location
    if [ -f "clip_temp/models/image_encoder/model.safetensors" ]; then
        mv clip_temp/models/image_encoder/model.safetensors clip/clip_vision_h.safetensors || true
        rm -rf clip_temp
    fi

    # LoRAs for Animate
    log_info "Downloading LoRAs..."
    huggingface-cli download Kijai/WanVideo_comfy \
        Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
        --local-dir loras/WanVideo || true

    huggingface-cli download Kijai/WanVideo_comfy \
        LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors \
        --local-dir loras/WanVideo || true

    # CRITICAL: Detection models go in models/detection/ NOT models/onnx/
    # WanAnimatePreprocess node looks in models/detection/ by default
    log_info "Downloading detection models (YOLOv10, ViTPose)..."
    wget -nc https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx \
        -O detection/yolov10m.onnx || true

    wget -nc https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx \
        -O detection/vitpose-l-wholebody.onnx || true

    # SAM2 model for segmentation
    log_info "Downloading SAM2 model..."
    huggingface-cli download facebook/sam2.1-hiera-base-plus \
        sam2.1_hiera_base_plus.safetensors \
        --local-dir sam2 || true

    log_info "Animate models download complete!"
}

# =============================================================================
# MODEL DOWNLOADS - FLUX
# =============================================================================

download_flux_models() {
    log_section "Downloading Flux Models"
    cd /workspace/ComfyUI/models

    # Flux Dev (~23GB)
    log_info "Downloading Flux Dev model..."
    huggingface-cli download black-forest-labs/FLUX.1-dev \
        flux1-dev.safetensors \
        --local-dir unet || true

    # Flux VAE
    log_info "Downloading Flux VAE..."
    huggingface-cli download black-forest-labs/FLUX.1-schnell \
        ae.safetensors \
        --local-dir vae || true

    # CLIP encoders
    log_info "Downloading CLIP encoders..."
    huggingface-cli download comfyanonymous/flux_text_encoders \
        clip_l.safetensors \
        --local-dir clip || true

    huggingface-cli download comfyanonymous/flux_text_encoders \
        t5xxl_fp16.safetensors \
        --local-dir clip || true

    log_info "Flux models download complete!"
}

# =============================================================================
# MODEL DOWNLOADS - PULID-FLUX
# =============================================================================

download_pulid_models() {
    log_section "Downloading PuLID-Flux Models"
    cd /workspace/ComfyUI/models

    # PuLID-Flux model
    log_info "Downloading PuLID-Flux model..."
    huggingface-cli download guozinan/PuLID \
        pulid_flux_v0.9.1.safetensors \
        --local-dir pulid || true

    # InsightFace AntelopeV2 (face detection)
    log_info "Downloading InsightFace AntelopeV2..."
    cd /workspace/ComfyUI/models/insightface/models/antelopev2
    wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx || true
    wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx || true
    wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx || true
    wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx || true
    wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx || true

    log_info "PuLID models download complete!"
}

# =============================================================================
# MODEL DOWNLOADS - FACE SWAP
# =============================================================================

download_faceswap_models() {
    log_section "Downloading Face Swap Models"
    cd /workspace/ComfyUI/models

    # InsightFace inswapper
    log_info "Downloading inswapper_128..."
    wget -nc https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
        -O insightface/inswapper_128.onnx || true

    # CodeFormer (face restoration)
    log_info "Downloading CodeFormer..."
    wget -nc https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth \
        -O facerestore_models/codeformer.pth || true

    log_info "Face swap models download complete!"
}

# =============================================================================
# FIX MODEL PATHS
# =============================================================================

fix_model_paths() {
    log_section "Fixing Model Paths"
    cd /workspace/ComfyUI/models

    # Move ONNX files from onnx/ to detection/ if they exist in wrong location
    if [ -d "onnx" ] && [ -f "onnx/yolov10m.onnx" ]; then
        log_info "Moving ONNX models from onnx/ to detection/..."
        mkdir -p detection
        mv onnx/*.onnx detection/ 2>/dev/null || true
    fi

    # Create symlinks for common path variations
    log_info "Creating symlinks for model paths..."

    # VAE symlink
    if [ -f "diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth" ] && [ ! -f "vae/Wan2.1_VAE.pth" ]; then
        ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
            vae/Wan2.1_VAE.pth || true
    fi

    # Text encoder symlink
    if [ -f "diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth" ] && [ ! -f "text_encoders/models_t5_umt5-xxl-enc-bf16.pth" ]; then
        ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
            text_encoders/models_t5_umt5-xxl-enc-bf16.pth || true
    fi

    log_info "Model paths fixed!"
}

# =============================================================================
# STARTUP SCRIPTS
# =============================================================================

create_startup_scripts() {
    log_section "Startup Scripts"

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

    log_info "Startup scripts created!"
}

# =============================================================================
# VERIFICATION
# =============================================================================

verify_installation() {
    log_section "Verification"

    echo ""
    echo "Checking custom nodes..."
    ls -d /workspace/ComfyUI/custom_nodes/*/ 2>/dev/null | head -20

    echo ""
    echo "Checking model directories..."
    for dir in vae clip detection text_encoders sam2 pulid; do
        if [ -d "/workspace/ComfyUI/models/$dir" ]; then
            count=$(ls /workspace/ComfyUI/models/$dir 2>/dev/null | wc -l)
            echo "  $dir: $count files"
        fi
    done

    echo ""
    echo "Checking diffusion models..."
    ls -la /workspace/ComfyUI/models/diffusion_models/ 2>/dev/null || echo "  No diffusion models yet"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    check_gpu
    update_system
    setup_python
    install_comfyui
    install_custom_nodes
    apply_patches
    install_api_deps
    create_model_dirs
    create_startup_scripts
    verify_installation

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Download models (choose one or more):"
    echo "     bash setup_vastai.sh download_flux      # Flux Dev (~35GB)"
    echo "     bash setup_vastai.sh download_pulid     # PuLID-Flux (~1.5GB)"
    echo "     bash setup_vastai.sh download_i2v       # Wan 2.2 I2V (~28GB)"
    echo "     bash setup_vastai.sh download_animate   # Wan 2.2 Animate (~35GB)"
    echo "     bash setup_vastai.sh download_faceswap  # Face Swap (~1GB)"
    echo "     bash setup_vastai.sh download_all       # Everything (~100GB)"
    echo ""
    echo "  2. Upload workflows to /workspace/ComfyUI/user/default/workflows/"
    echo ""
    echo "  3. Start services:"
    echo "     bash /workspace/start_comfyui.sh"
    echo ""
    echo "  4. Access ComfyUI via SSH port forwarding:"
    echo "     ssh -p <PORT> root@<IP> -L 8188:localhost:8188"
    echo "     Then open: http://localhost:8188"
    echo ""
}

download_all_models() {
    download_flux_models
    download_pulid_models
    download_i2v_models
    download_animate_models
    download_faceswap_models
    fix_model_paths
}

# Handle command-line arguments
case "${1:-}" in
    download_i2v)
        download_i2v_models
        fix_model_paths
        ;;
    download_animate)
        download_animate_models
        fix_model_paths
        ;;
    download_flux)
        download_flux_models
        ;;
    download_pulid)
        download_pulid_models
        ;;
    download_faceswap)
        download_faceswap_models
        ;;
    download_all)
        download_all_models
        ;;
    fix_paths)
        fix_model_paths
        ;;
    verify)
        verify_installation
        ;;
    *)
        main "$@"
        ;;
esac
