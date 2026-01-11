#!/bin/bash
# Model Download Script for APOB Clone
# Downloads all required models from HuggingFace and other sources

set -e

echo "=========================================="
echo "  APOB Clone - Model Download Script"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_download() { echo -e "${CYAN}[DOWNLOAD]${NC} $1"; }

# Activate virtual environment
source /workspace/venv/bin/activate 2>/dev/null || true

# Install huggingface-cli if not present
pip install -q huggingface-hub

# Base directories
COMFY_DIR="/workspace/ComfyUI"
MODELS_DIR="${COMFY_DIR}/models"

# Function to download with resume support
download_hf() {
    local repo=$1
    local dest=$2
    local filename=${3:-""}

    log_download "Downloading $repo to $dest..."

    if [ -n "$filename" ]; then
        huggingface-cli download "$repo" "$filename" --local-dir "$dest" --local-dir-use-symlinks False
    else
        huggingface-cli download "$repo" --local-dir "$dest" --local-dir-use-symlinks False
    fi
}

# Function to download from URL
download_url() {
    local url=$1
    local dest=$2
    local filename=$3

    log_download "Downloading $filename..."
    mkdir -p "$dest"
    wget -c -q --show-progress -O "${dest}/${filename}" "$url" || curl -L -C - -o "${dest}/${filename}" "$url"
}

# ============================================
# 1. FLUX Models (Portrait Generation)
# ============================================
download_flux() {
    log_info "=== Downloading Flux Models ==="

    # Flux Dev checkpoint
    log_download "Flux.1 Dev (main checkpoint)..."
    download_hf "black-forest-labs/FLUX.1-dev" "${MODELS_DIR}/unet" "flux1-dev.safetensors"

    # Flux VAE
    log_download "Flux VAE..."
    download_hf "black-forest-labs/FLUX.1-dev" "${MODELS_DIR}/vae" "ae.safetensors"

    # Flux text encoders (CLIP)
    log_download "Flux CLIP encoders..."
    mkdir -p "${MODELS_DIR}/clip"
    download_hf "comfyanonymous/flux_text_encoders" "${MODELS_DIR}/clip" "clip_l.safetensors"
    download_hf "comfyanonymous/flux_text_encoders" "${MODELS_DIR}/clip" "t5xxl_fp16.safetensors"

    log_info "Flux models downloaded!"
}

# ============================================
# 2. PuLID Models (Consistent Character)
# ============================================
download_pulid() {
    log_info "=== Downloading PuLID Models ==="

    mkdir -p "${MODELS_DIR}/pulid"

    # PuLID Flux model
    log_download "PuLID Flux model..."
    download_hf "guozinan/PuLID" "${MODELS_DIR}/pulid" "pulid_flux_v0.9.1.safetensors"

    # EVA CLIP for face encoding
    log_download "EVA CLIP..."
    download_hf "QuanSun/EVA-CLIP" "${MODELS_DIR}/clip" "EVA02_CLIP_L_336_psz14_s6B.pt"

    log_info "PuLID models downloaded!"
}

# ============================================
# 3. Wan 2.2 Models (Image to Video)
# ============================================
download_wan() {
    log_info "=== Downloading Wan 2.2 Models ==="

    mkdir -p "${MODELS_DIR}/wan"

    # Wan 2.2 Image-to-Video model
    log_download "Wan 2.2 I2V model (this is large, ~28GB)..."
    download_hf "Wan-AI/Wan2.2-I2V-A14B-480P" "${MODELS_DIR}/wan/Wan2.2-I2V-A14B-480P"

    # Alternative: smaller 5B model for faster inference
    log_download "Wan 2.1 5B model (faster, smaller)..."
    download_hf "Wan-AI/Wan2.1-I2V-5B-480P" "${MODELS_DIR}/wan/Wan2.1-I2V-5B-480P"

    log_info "Wan models downloaded!"
}

# ============================================
# 4. SadTalker / LivePortrait Models (Lip Sync)
# ============================================
download_lipsync() {
    log_info "=== Downloading Lip Sync Models ==="

    mkdir -p "${MODELS_DIR}/liveportrait"

    # LivePortrait models
    log_download "LivePortrait models..."
    download_hf "Kijai/LivePortrait_safetensors" "${MODELS_DIR}/liveportrait"

    # SadTalker models (alternative)
    mkdir -p "${MODELS_DIR}/sadtalker"
    log_download "SadTalker checkpoints..."
    download_hf "vinthony/SadTalker" "${MODELS_DIR}/sadtalker"

    log_info "Lip sync models downloaded!"
}

# ============================================
# 5. InsightFace / ReActor Models (Face Swap)
# ============================================
download_faceswap() {
    log_info "=== Downloading Face Swap Models ==="

    mkdir -p "${MODELS_DIR}/insightface"
    mkdir -p "${MODELS_DIR}/facerestore_models"

    # InsightFace inswapper model
    log_download "InsightFace inswapper..."
    download_url "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
        "${MODELS_DIR}/insightface" "inswapper_128.onnx"

    # Buffalo_l face analysis model
    log_download "Buffalo_L face analysis..."
    mkdir -p "${MODELS_DIR}/insightface/models/buffalo_l"
    download_hf "DIAMONIK7777/insightface" "${MODELS_DIR}/insightface/models"

    # CodeFormer for face restoration
    log_download "CodeFormer..."
    download_url "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" \
        "${MODELS_DIR}/facerestore_models" "codeformer.pth"

    # GFPGAN for face enhancement
    log_download "GFPGAN..."
    download_url "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
        "${MODELS_DIR}/facerestore_models" "GFPGANv1.4.pth"

    log_info "Face swap models downloaded!"
}

# ============================================
# 6. ControlNet Models (Pose Control)
# ============================================
download_controlnet() {
    log_info "=== Downloading ControlNet Models ==="

    mkdir -p "${MODELS_DIR}/controlnet"

    # OpenPose for Flux
    log_download "ControlNet OpenPose..."
    download_hf "InstantX/FLUX.1-dev-Controlnet-Union" "${MODELS_DIR}/controlnet"

    log_info "ControlNet models downloaded!"
}

# ============================================
# Main Download Menu
# ============================================
show_menu() {
    echo ""
    echo "Select models to download:"
    echo "  1) All models (recommended, ~80GB total)"
    echo "  2) Flux only (portraits)"
    echo "  3) Flux + PuLID (portraits + consistent character)"
    echo "  4) Wan 2.2 only (image to video)"
    echo "  5) Lip sync models only"
    echo "  6) Face swap models only"
    echo "  7) Exit"
    echo ""
    read -p "Enter choice [1-7]: " choice
}

download_all() {
    download_flux
    download_pulid
    download_wan
    download_lipsync
    download_faceswap
    download_controlnet
}

# Main
main() {
    if [ "$1" == "--all" ]; then
        download_all
    elif [ "$1" == "--flux" ]; then
        download_flux
    elif [ "$1" == "--pulid" ]; then
        download_flux
        download_pulid
    elif [ "$1" == "--wan" ]; then
        download_wan
    elif [ "$1" == "--lipsync" ]; then
        download_lipsync
    elif [ "$1" == "--faceswap" ]; then
        download_faceswap
    else
        show_menu
        case $choice in
            1) download_all ;;
            2) download_flux ;;
            3) download_flux; download_pulid ;;
            4) download_wan ;;
            5) download_lipsync ;;
            6) download_faceswap ;;
            7) exit 0 ;;
            *) log_warn "Invalid choice"; exit 1 ;;
        esac
    fi

    echo ""
    echo "=========================================="
    echo "  Download Complete!"
    echo "=========================================="
    echo ""
    echo "Models are stored in: ${MODELS_DIR}"
    echo ""
    echo "Disk usage:"
    du -sh ${MODELS_DIR}/* 2>/dev/null || true
    echo ""
}

main "$@"
