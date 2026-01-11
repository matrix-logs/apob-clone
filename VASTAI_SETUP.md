# Complete Vast.ai Setup Guide - APOB Clone

This guide documents everything needed to set up the APOB.ai clone from scratch on a new Vast.ai instance.

## Instance Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 4090 (24GB) | RTX 5090 (32GB) |
| VRAM | 24GB | 32GB |
| Disk Space | 300GB | 500GB |
| RAM | 32GB | 64GB+ |
| CUDA | 12.1+ | 12.8+ |
| PyTorch | 2.1+ | 2.9+ |

**Recommended Vast.ai Template**: ComfyUI template or PyTorch 2.x template

---

## Step 1: Initial Setup

```bash
# SSH into your instance
ssh -p <PORT> root@<IP_ADDRESS>

# Update system
apt-get update && apt-get upgrade -y

# Install required system packages
apt-get install -y git git-lfs ffmpeg libgl1-mesa-glx libglib2.0-0
```

---

## Step 2: Install ComfyUI (if not using ComfyUI template)

```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

---

## Step 3: Install Python Dependencies

```bash
pip install facexlib pykalman accelerate timm gguf ftfy xformers diffusers insightface onnxruntime onnxruntime-gpu
```

---

## Step 4: Install Custom Nodes

```bash
cd /workspace/ComfyUI/custom_nodes

# ComfyUI Manager (for easy node management)
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# PuLID-Flux (face identity preservation)
git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git

# Wan Video Wrapper (image to video)
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# Video Helper Suite (video I/O)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# LivePortrait (face animation)
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git

# ReActor (face swap)
git clone https://github.com/Gourieff/comfyui-reactor-node.git
cd comfyui-reactor-node && python install.py && cd ..

# Original PuLID nodes (optional, for SDXL)
git clone https://github.com/cubiq/PuLID_ComfyUI.git
git clone https://github.com/cubiq/pulid_comfyui.git
```

---

## Step 5: Download Models

### 5.1 Flux Dev (23GB)
```bash
cd /workspace/ComfyUI/models/unet
huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir .
```

### 5.2 Flux VAE and CLIP
```bash
# VAE
cd /workspace/ComfyUI/models/vae
huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir .

# CLIP
cd /workspace/ComfyUI/models/clip
huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir .
huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir .
```

### 5.3 PuLID-Flux Model (1.1GB)
```bash
cd /workspace/ComfyUI/models/pulid
huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir .
```

### 5.4 InsightFace AntelopeV2 (for face detection)
```bash
mkdir -p /workspace/ComfyUI/models/insightface/models/antelopev2
cd /workspace/ComfyUI/models/insightface/models/antelopev2
# Download from: https://huggingface.co/MonsterMMORPG/tools/tree/main
# Files needed: 1k3d68.onnx, 2d106det.onnx, genderage.onnx, glintr100.onnx, scrfd_10g_bnkps.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx
```

### 5.5 ReActor Models (face swap)
```bash
# InsightFace inswapper
mkdir -p /workspace/ComfyUI/models/insightface
cd /workspace/ComfyUI/models/insightface
wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

# CodeFormer (face restoration)
mkdir -p /workspace/ComfyUI/models/facerestore_models
cd /workspace/ComfyUI/models/facerestore_models
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
```

### 5.6 Wan 2.2 I2V Model (118GB) - Large download!
```bash
cd /workspace/ComfyUI/models/diffusion_models
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir Wan2.2-I2V-A14B
```

### 5.7 LivePortrait Models
```bash
mkdir -p /workspace/ComfyUI/models/liveportrait
cd /workspace/ComfyUI/models/liveportrait
# Models auto-download on first use, or manually from:
# https://huggingface.co/Kijai/LivePortrait_safetensors
```

---

## Step 6: Create Required Symlinks

```bash
# Wan VAE symlink (WanVideoVAELoader looks in /models/vae/)
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth

# Wan T5 encoder symlink (LoadWanVideoT5TextEncoder looks in /models/text_encoders/)
mkdir -p /workspace/ComfyUI/models/text_encoders
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth /workspace/ComfyUI/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
```

---

## Step 7: Apply Code Patches

### 7.1 PuLID-Flux Compatibility Fix
The PuLID-Flux node needs patching for newer ComfyUI versions:

```bash
# Add missing parameters to forward_orig function
sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' /workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py
```

**Verify the fix:**
```bash
grep -A12 '^def forward_orig' /workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py
```

Should show:
```python
def forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask=None,
) -> Tensor:
```

---

## Step 8: Create Workflow Directory

```bash
mkdir -p /workspace/ComfyUI/user/default/workflows
```

---

## Step 9: Start Services

### Start ComfyUI
```bash
cd /workspace/ComfyUI
nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
```

### Start FastAPI Backend (optional)
```bash
cd /workspace/myclone/ugc
nohup python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /workspace/api.log 2>&1 &
```

---

## Step 10: Access Services

From your **local machine**, set up SSH port forwarding:

```bash
# ComfyUI
ssh -p <PORT> root@<IP_ADDRESS> -L 8188:localhost:8188

# FastAPI (separate terminal)
ssh -p <PORT> root@<IP_ADDRESS> -L 8000:localhost:8000
```

Then open:
- ComfyUI: http://localhost:8188
- FastAPI: http://localhost:8000/docs

---

## Complete Automated Setup Script

Save this as `setup.sh` and run it:

```bash
#!/bin/bash
set -e

echo "=== APOB Clone Setup Script ==="

# Install dependencies
echo "Installing Python dependencies..."
pip install facexlib pykalman accelerate timm gguf ftfy xformers diffusers insightface onnxruntime onnxruntime-gpu

# Install custom nodes
echo "Installing custom nodes..."
cd /workspace/ComfyUI/custom_nodes

git clone https://github.com/ltdrdata/ComfyUI-Manager.git 2>/dev/null || true
git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git 2>/dev/null || true
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git 2>/dev/null || true
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 2>/dev/null || true
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git 2>/dev/null || true

if [ ! -d "comfyui-reactor-node" ]; then
    git clone https://github.com/Gourieff/comfyui-reactor-node.git
    cd comfyui-reactor-node && python install.py && cd ..
fi

# Apply PuLID-Flux patch
echo "Applying PuLID-Flux compatibility patch..."
sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' /workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py

# Create model directories
echo "Creating model directories..."
mkdir -p /workspace/ComfyUI/models/pulid
mkdir -p /workspace/ComfyUI/models/text_encoders
mkdir -p /workspace/ComfyUI/models/insightface/models/antelopev2
mkdir -p /workspace/ComfyUI/models/facerestore_models
mkdir -p /workspace/ComfyUI/user/default/workflows

# Download models (uncomment as needed)
echo "Downloading models..."

# Flux Dev
# huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir /workspace/ComfyUI/models/unet

# Flux VAE
# huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir /workspace/ComfyUI/models/vae

# Flux CLIP
# huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir /workspace/ComfyUI/models/clip
# huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir /workspace/ComfyUI/models/clip

# PuLID-Flux
# huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir /workspace/ComfyUI/models/pulid

# InsightFace AntelopeV2
cd /workspace/ComfyUI/models/insightface/models/antelopev2
wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx 2>/dev/null || true
wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx 2>/dev/null || true
wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx 2>/dev/null || true
wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx 2>/dev/null || true
wget -nc https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx 2>/dev/null || true

# Wan 2.2 (LARGE - 118GB, uncomment if needed)
# huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B

# Create symlinks
echo "Creating symlinks..."
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth 2>/dev/null || true
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth /workspace/ComfyUI/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth 2>/dev/null || true

echo "=== Setup Complete ==="
echo "Start ComfyUI with: cd /workspace/ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188"
```

---

## Model Summary

| Model | Size | Purpose |
|-------|------|---------|
| Flux Dev | 23GB | Base image generation |
| Flux VAE (ae.safetensors) | 320MB | Image encoding/decoding |
| CLIP L | 235MB | Text encoding |
| T5XXL FP16 | 9.4GB | Text encoding |
| PuLID-Flux | 1.1GB | Face identity preservation |
| InsightFace AntelopeV2 | ~200MB | Face detection |
| InsightFace inswapper | 529MB | Face swapping |
| CodeFormer | 360MB | Face restoration |
| Wan 2.2 I2V | 118GB | Image to video |
| LivePortrait | ~500MB | Face animation |

**Total Disk Space Required**: ~250GB (minimum), ~400GB (recommended)

---

## Troubleshooting

### ComfyUI won't start
```bash
# Check logs
tail -100 /workspace/comfyui.log

# Check if process is running
pgrep -a python3
```

### Model not found errors
- Verify model paths match workflow JSON
- Check symlinks are working: `ls -la /workspace/ComfyUI/models/vae/`

### PuLID errors about attn_mask
- Apply the patch from Step 7.1
- Restart ComfyUI

### Out of memory
- Use `offload_device` option in model loaders
- Enable VAE tiling for large images
- Close other GPU processes

### Custom node import errors
```bash
# Install missing dependencies
pip install <missing_package>

# Restart ComfyUI
pkill -f 'python3 main.py'
cd /workspace/ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188
```

---

## Workflows

After setup, import these workflows from `/workspace/ComfyUI/user/default/workflows/`:

| Workflow | File | Description |
|----------|------|-------------|
| Flux Portrait | `flux_portrait.json` | Text to portrait |
| PuLID-Flux Character | `pulid_flux_character.json` | Consistent face identity |
| Wan 2.2 I2V | `wan22_i2v.json` | Image to video |
| LivePortrait | `liveportrait_animate.json` | Face animation |
| ReActor Face Swap | `reactor_faceswap.json` | Face swap (images) |
| ReActor Video | `reactor_video_faceswap.json` | Face swap (videos) |

---

## Quick Start Commands

```bash
# Start ComfyUI
cd /workspace/ComfyUI && nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &

# Check status
pgrep -a python3 | grep main.py

# View logs
tail -f /workspace/comfyui.log

# Kill ComfyUI
pkill -f 'python3 main.py'
```

---

## Sources

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [Flux Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Wan 2.2 I2V](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [PuLID-Flux Model](https://huggingface.co/guozinan/PuLID)
