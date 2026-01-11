#!/bin/bash
# APOB Clone - Vast.ai Setup Script
# Run this on a fresh Vast.ai instance with ComfyUI template
set -e

echo "============================================"
echo "  APOB Clone - Vast.ai Setup Script"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root"
    exit 1
fi

# Check if ComfyUI exists
if [ ! -d "/workspace/ComfyUI" ]; then
    print_error "ComfyUI not found at /workspace/ComfyUI"
    print_status "Please use a Vast.ai template with ComfyUI pre-installed"
    exit 1
fi

print_status "ComfyUI found at /workspace/ComfyUI"

# ============================================
# Step 1: Install Python Dependencies
# ============================================
print_status "Installing Python dependencies..."
pip install -q facexlib pykalman accelerate timm gguf ftfy xformers diffusers insightface onnxruntime onnxruntime-gpu sageattention color-matcher librosa

# ============================================
# Step 2: Install Custom Nodes
# ============================================
print_status "Installing custom nodes..."
cd /workspace/ComfyUI/custom_nodes

# ComfyUI Manager
if [ ! -d "ComfyUI-Manager" ]; then
    print_status "Cloning ComfyUI-Manager..."
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
fi

# PuLID-Flux
if [ ! -d "ComfyUI-PuLID-Flux" ]; then
    print_status "Cloning ComfyUI-PuLID-Flux..."
    git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git
fi

# Wan Video Wrapper
if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
    print_status "Cloning ComfyUI-WanVideoWrapper..."
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
fi

# Video Helper Suite
if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
    print_status "Cloning ComfyUI-VideoHelperSuite..."
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
fi

# LivePortrait
if [ ! -d "ComfyUI-LivePortraitKJ" ]; then
    print_status "Cloning ComfyUI-LivePortraitKJ..."
    git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git
fi

# ReActor
if [ ! -d "comfyui-reactor-node" ]; then
    print_status "Cloning comfyui-reactor-node..."
    git clone https://github.com/Gourieff/comfyui-reactor-node.git
    cd comfyui-reactor-node
    python install.py
    cd ..
fi

# KJNodes (for Wan workflow utility nodes)
if [ ! -d "ComfyUI-KJNodes" ]; then
    print_status "Cloning ComfyUI-KJNodes..."
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
fi

# comfyui-various (additional utility nodes)
if [ ! -d "comfyui-various" ]; then
    print_status "Cloning comfyui-various..."
    git clone https://github.com/jamesWalker55/comfyui-various.git
fi

# ============================================
# Step 3: Apply PuLID-Flux Patch
# ============================================
print_status "Applying PuLID-Flux compatibility patch..."
PULID_FILE="/workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py"
if grep -q "transformer_options={}" "$PULID_FILE"; then
    print_status "Patch already applied"
else
    sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' "$PULID_FILE"
    print_status "Patch applied successfully"
fi

# ============================================
# Step 4: Create Directories
# ============================================
print_status "Creating model directories..."
mkdir -p /workspace/ComfyUI/models/pulid
mkdir -p /workspace/ComfyUI/models/text_encoders
mkdir -p /workspace/ComfyUI/models/insightface/models/antelopev2
mkdir -p /workspace/ComfyUI/models/facerestore_models
mkdir -p /workspace/ComfyUI/models/loras/WanVideo/Lightx2v
mkdir -p /workspace/ComfyUI/user/default/workflows
mkdir -p /workspace/outputs

# ============================================
# Step 5: Download InsightFace Models
# ============================================
print_status "Downloading InsightFace AntelopeV2 models..."
cd /workspace/ComfyUI/models/insightface/models/antelopev2

for model in 1k3d68.onnx 2d106det.onnx genderage.onnx glintr100.onnx scrfd_10g_bnkps.onnx; do
    if [ ! -f "$model" ]; then
        print_status "Downloading $model..."
        wget -q "https://huggingface.co/MonsterMMORPG/tools/resolve/main/$model"
    fi
done

# Download inswapper for ReActor
cd /workspace/ComfyUI/models/insightface
if [ ! -f "inswapper_128.onnx" ]; then
    print_status "Downloading inswapper_128.onnx..."
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
fi

# Download CodeFormer
cd /workspace/ComfyUI/models/facerestore_models
if [ ! -f "codeformer.pth" ]; then
    print_status "Downloading codeformer.pth..."
    wget -q "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
fi

# ============================================
# Step 6: Download Main Models (Interactive)
# ============================================
echo ""
print_warning "The following models need to be downloaded manually (large files):"
echo ""
echo "  1. Flux Dev (23GB):"
echo "     huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir /workspace/ComfyUI/models/unet"
echo ""
echo "  2. Flux VAE (320MB):"
echo "     huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir /workspace/ComfyUI/models/vae"
echo ""
echo "  3. Flux CLIP (9.6GB total):"
echo "     huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir /workspace/ComfyUI/models/clip"
echo "     huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir /workspace/ComfyUI/models/clip"
echo ""
echo "  4. PuLID-Flux (1.1GB):"
echo "     huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir /workspace/ComfyUI/models/pulid"
echo ""
echo "  5. Wan 2.2 I2V FP8 (28GB) - RECOMMENDED:"
echo "     hf download Kijai/WanVideo_comfy_fp8_scaled I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors --local-dir /workspace/ComfyUI/models/diffusion_models"
echo "     hf download Kijai/WanVideo_comfy_fp8_scaled I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors --local-dir /workspace/ComfyUI/models/diffusion_models"
echo ""
echo "  6. Wan 2.2 I2V Full (118GB) - OPTIONAL (for best quality):"
echo "     huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B"
echo ""
echo "  7. Lightx2v LoRA (738MB) - For faster Wan inference:"
echo "     hf download Kijai/WanVideo_comfy Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors --local-dir /workspace/ComfyUI/models/loras/WanVideo"
echo ""

read -p "Do you want to download the main models now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Downloading Flux Dev..."
    huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir /workspace/ComfyUI/models/unet

    print_status "Downloading Flux VAE..."
    huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir /workspace/ComfyUI/models/vae

    print_status "Downloading Flux CLIP..."
    huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir /workspace/ComfyUI/models/clip
    huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir /workspace/ComfyUI/models/clip

    print_status "Downloading PuLID-Flux..."
    huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir /workspace/ComfyUI/models/pulid

    read -p "Download Wan 2.2 I2V FP8 models (28GB, recommended)? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Downloading Wan 2.2 I2V FP8 HIGH model..."
        hf download Kijai/WanVideo_comfy_fp8_scaled I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors --local-dir /workspace/ComfyUI/models/diffusion_models
        print_status "Downloading Wan 2.2 I2V FP8 LOW model..."
        hf download Kijai/WanVideo_comfy_fp8_scaled I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors --local-dir /workspace/ComfyUI/models/diffusion_models
        print_status "Downloading Lightx2v LoRA..."
        hf download Kijai/WanVideo_comfy Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors --local-dir /workspace/ComfyUI/models/loras/WanVideo
    fi
fi

# ============================================
# Step 7: Create Symlinks
# ============================================
print_status "Creating symlinks..."

# Only create symlinks if Wan model exists
if [ -f "/workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth" ]; then
    ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth
    print_status "Created Wan VAE symlink"
fi

if [ -f "/workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth /workspace/ComfyUI/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
    print_status "Created Wan T5 encoder symlink"
fi

# ============================================
# Step 8: Copy Workflows
# ============================================
print_status "Copying workflows..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/workflows" ]; then
    cp -r "$SCRIPT_DIR/workflows/"*.json /workspace/ComfyUI/user/default/workflows/
    print_status "Workflows copied to ComfyUI"
else
    print_warning "Workflows folder not found at $SCRIPT_DIR/workflows"
    print_warning "Please manually copy workflows to /workspace/ComfyUI/user/default/workflows/"
fi

# ============================================
# Complete
# ============================================
echo ""
echo "============================================"
print_status "Setup complete!"
echo "============================================"
echo ""
echo "To start ComfyUI:"
echo "  cd /workspace/ComfyUI"
echo "  python3 main.py --listen 0.0.0.0 --port 8188"
echo ""
echo "Or run in background:"
echo "  nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &"
echo ""
echo "Access via SSH port forwarding:"
echo "  ssh -p <PORT> root@<IP> -L 8188:localhost:8188"
echo "  Then open: http://localhost:8188"
echo ""
