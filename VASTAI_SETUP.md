# Complete Vast.ai Setup Guide - APOB Clone

This guide documents everything needed to set up the APOB.ai clone from scratch on a new Vast.ai instance, including lessons learned and fixes discovered during deployment.

---

## Table of Contents

1. [Instance Requirements](#instance-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Setup Steps](#detailed-setup-steps)
4. [Model Downloads](#model-downloads)
5. [Wan 2.2 Animate vs I2V](#wan-22-animate-vs-i2v)
6. [Workflow Reference](#workflow-reference)
7. [Common Issues & Fixes](#common-issues--fixes)

---

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

## Quick Start

```bash
# SSH into your instance
ssh -p <PORT> root@<IP_ADDRESS>

# Clone the repository
git clone https://github.com/matrix-logs/myclone.git
cd myclone/ugc

# Run setup script
chmod +x scripts/setup_vastai.sh
bash scripts/setup_vastai.sh

# Download models (choose what you need)
bash scripts/setup_vastai.sh download_animate  # Wan 2.2 Animate (~35GB)
bash scripts/setup_vastai.sh download_flux     # Flux Dev (~35GB)
bash scripts/setup_vastai.sh download_pulid    # PuLID-Flux (~1.5GB)

# Start ComfyUI
bash /workspace/start_comfyui.sh
```

---

## Detailed Setup Steps

### Step 1: Initial System Setup

```bash
# SSH into your instance
ssh -p <PORT> root@<IP_ADDRESS>

# Update system
apt-get update && apt-get upgrade -y
apt-get install -y git git-lfs ffmpeg libgl1-mesa-glx libglib2.0-0
```

### Step 2: Install ComfyUI (if not using ComfyUI template)

```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
mkdir -p user/default/workflows
```

### Step 3: Install Custom Nodes

**Critical nodes for Wan 2.2 Animate:**
```bash
cd /workspace/ComfyUI/custom_nodes

# ComfyUI Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Wan Video Wrapper (REQUIRED - core Wan support)
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# Wan Animate Preprocess (REQUIRED - pose/face detection)
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git

# Segment Anything 2 (REQUIRED - character segmentation)
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git

# KJNodes (REQUIRED - utility nodes)
git clone https://github.com/kijai/ComfyUI-KJNodes.git

# Video Helper Suite (REQUIRED - video I/O)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# PuLID-Flux (for consistent character identity)
git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git

# LivePortrait (face animation)
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git

# ReActor (face swap)
git clone https://github.com/Gourieff/comfyui-reactor-node.git
cd comfyui-reactor-node && python install.py && cd ..
```

### Step 4: Install Python Dependencies

```bash
pip install sageattention onnxruntime onnxruntime-gpu facexlib pykalman accelerate
pip install timm gguf ftfy xformers diffusers insightface huggingface-hub

# Install node requirements
for dir in */; do
    if [ -f "${dir}requirements.txt" ]; then
        pip install -r "${dir}requirements.txt" || true
    fi
done
```

### Step 5: Create Model Directories

**IMPORTANT:** Detection models must go in `models/detection/`, NOT `models/onnx/`!

```bash
cd /workspace/ComfyUI/models
mkdir -p checkpoints clip vae loras unet pulid text_encoders sam2
mkdir -p detection  # CRITICAL: for YOLOv10 and ViTPose
mkdir -p diffusion_models/Wan2.2-Animate-14B
mkdir -p diffusion_models/Wan2.2-I2V-A14B
mkdir -p loras/WanVideo/Lightx2v
mkdir -p loras/WanVideo/LoRAs/Wan22_relight
mkdir -p insightface/models/antelopev2
mkdir -p facerestore_models
```

---

## Model Downloads

### Wan 2.2 Animate (Character Swap Workflow)

```bash
cd /workspace/ComfyUI/models

# Main model (FP8 quantized, ~18GB)
huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled \
    Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
    --local-dir diffusion_models/Wan2.2-Animate-14B

# VAE
huggingface-cli download Kijai/WanVideo_comfy \
    Wan2_1_VAE_bf16.safetensors --local-dir vae

# Text Encoder
huggingface-cli download Kijai/WanVideo_comfy \
    umt5-xxl-enc-bf16.safetensors --local-dir text_encoders

# CLIP Vision (from IP-Adapter repo)
huggingface-cli download h94/IP-Adapter models/image_encoder/model.safetensors \
    --local-dir clip_temp
mv clip_temp/models/image_encoder/model.safetensors clip/clip_vision_h.safetensors
rm -rf clip_temp

# LoRAs
huggingface-cli download Kijai/WanVideo_comfy \
    Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
    --local-dir loras/WanVideo

huggingface-cli download Kijai/WanVideo_comfy \
    LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors \
    --local-dir loras/WanVideo

# Detection models (MUST be in detection/, NOT onnx/)
wget https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx \
    -O detection/yolov10m.onnx

wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx \
    -O detection/vitpose-l-wholebody.onnx

# SAM2 for segmentation
huggingface-cli download facebook/sam2.1-hiera-base-plus \
    sam2.1_hiera_base_plus.safetensors --local-dir sam2
```

### Wan 2.2 I2V (Image to Video)

```bash
# Full model (~28GB per file, ~118GB total)
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-480P \
    --local-dir diffusion_models/Wan2.2-I2V-A14B

# Create symlinks
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
    /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth
```

### Flux + PuLID

```bash
# Flux Dev (~23GB)
huggingface-cli download black-forest-labs/FLUX.1-dev \
    flux1-dev.safetensors --local-dir unet

# Flux VAE + CLIP
huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir vae
huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir clip
huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir clip

# PuLID-Flux
huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir pulid

# InsightFace AntelopeV2
cd insightface/models/antelopev2
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx
```

---

## Wan 2.2 Animate vs I2V

These are **different models** for **different purposes**:

| Feature | I2V | Animate |
|---------|-----|---------|
| **Purpose** | Animate a still image with AI-generated motion | Replace character in video with another face |
| **Model** | `Wan2.2-I2V-A14B` (~28GB) | `Wan2.2-Animate-14B` (~18GB FP8) |
| **Input** | Single image + text prompt | Reference image + driving video |
| **Motion Source** | AI-generated from text description | Extracted from performer video (pose/face) |
| **Key Nodes** | 7-8 basic nodes | 12+ specialized nodes |
| **Extra Models** | None | YOLOv10, ViTPose, SAM2, CLIP Vision |
| **Custom Nodes** | WanVideoWrapper | WanVideoWrapper + WanAnimatePreprocess + SAM2 |

### Animate Workflow Pipeline

```
1. LoadVideo (driving video with performer)
   ↓
2. PoseAndFaceDetection (YOLOv10 + ViTPose)
   ├── Extract skeleton keypoints
   ├── Detect faces
   └── Create bounding boxes
   ↓
3. Sam2Segmentation
   └── Generate character mask
   ↓
4. WanVideoAnimateEmbeds
   ├── Reference character image
   ├── Detected poses/faces
   ├── Background frames
   └── Combined embeddings
   ↓
5. WanAnimateSampler (with LightX2V LoRA)
   └── Generate new character video
   ↓
6. ReLight LoRA (optional)
   └── Match original lighting
   ↓
7. VHS_VideoCombine
   └── Save output
```

---

## Workflow Reference

### Available Workflows

| Workflow | File | Purpose |
|----------|------|---------|
| **Wan 2.2 Animate** | `wan22_animate_faceswap.json` | Character swap in videos |
| **Wan 2.2 I2V** | `wan22_i2v_character.json` | Animate still images |
| **PuLID-Flux** | `pulid_flux_character.json` | Consistent face identity |
| **Flux Portrait** | `flux_portrait.json` | AI portrait generation |
| **ReActor** | `reactor_faceswap.json` | Face swap in images |
| **LivePortrait** | `liveportrait_animate.json` | Face animation |

### Uploading Workflows

```bash
# From your local machine
scp -P <PORT> workflows/*.json root@<IP>:/workspace/ComfyUI/user/default/workflows/
```

---

## Common Issues & Fixes

### 1. Workflow won't load - "Invalid uuid"

**Fix:** Workflow JSON needs proper UUID format:
```json
{
  "id": "a1b2c3d4-e5f6-4890-abcd-ef1234567890",
  ...
}
```

### 2. Workflow won't load - "Required at nodes[x].properties"

**Fix:** Add `"properties": {}` to every node missing it.

### 3. Detection models not found

**Fix:** Models must be in `models/detection/`, NOT `models/onnx/`:
```bash
mkdir -p /workspace/ComfyUI/models/detection
mv /workspace/ComfyUI/models/onnx/*.onnx /workspace/ComfyUI/models/detection/
```

### 4. CLIP Vision not found

**Fix:** Download from IP-Adapter repo:
```bash
huggingface-cli download h94/IP-Adapter models/image_encoder/model.safetensors \
    --local-dir /workspace/ComfyUI/models/clip_temp
mv /workspace/ComfyUI/models/clip_temp/models/image_encoder/model.safetensors \
    /workspace/ComfyUI/models/clip/clip_vision_h.safetensors
```

### 5. Model path mismatch

**Fix:** Check actual paths on server and update workflow:
```bash
ls /workspace/ComfyUI/models/diffusion_models/
ls /workspace/ComfyUI/models/detection/
```

### 6. PuLID-Flux compatibility error

**Fix:** Apply patch:
```bash
sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' \
    /workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py
```

---

## Start Services

### Start ComfyUI

```bash
cd /workspace/ComfyUI
nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
```

### Access via SSH Port Forwarding

From your **local machine**:
```bash
ssh -p <PORT> root@<IP_ADDRESS> -L 8188:localhost:8188
```

Then open: **http://localhost:8188**

---

## Model Summary

| Model | Size | Purpose |
|-------|------|---------|
| Wan2.2-Animate-14B (FP8) | 18GB | Character swap |
| Wan2.2-I2V-A14B | 118GB | Image to video |
| Flux Dev | 23GB | Image generation |
| PuLID-Flux | 1.1GB | Face identity |
| UMT5-XXL | 9.4GB | Text encoding |
| CLIP Vision H | 2.4GB | Image encoding |
| SAM2 | 300MB | Segmentation |
| YOLOv10 | 59MB | Person detection |
| ViTPose-L | 1.2GB | Pose estimation |

**Total for Animate workflow:** ~35GB
**Total for everything:** ~200GB

---

## Quick Commands Reference

```bash
# Start ComfyUI
cd /workspace/ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188

# Check GPU
nvidia-smi

# Check logs
tail -f /workspace/comfyui.log

# Check models
ls /workspace/ComfyUI/models/diffusion_models/
ls /workspace/ComfyUI/models/detection/

# Kill ComfyUI
pkill -f 'python3 main.py'

# Fix model paths
bash /path/to/setup_vastai.sh fix_paths
```

---

## Resources

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess)
- [Kijai's WanVideo Models](https://huggingface.co/Kijai/WanVideo_comfy)
- [Wan 2.2 Animate Documentation](https://docs.comfy.org/tutorials/video/wan/wan2-2-animate)
