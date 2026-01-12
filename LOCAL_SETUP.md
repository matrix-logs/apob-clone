# Local Setup Guide - APOB Clone

This guide covers setting up the APOB.ai clone on your local Windows/Linux machine with an NVIDIA GPU.

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 3090 (24GB) | RTX 4090/5090 (24-32GB) |
| VRAM | 16GB | 24GB+ |
| RAM | 32GB | 64GB |
| Storage | 200GB SSD | 500GB NVMe |
| OS | Windows 10/11 or Ubuntu 22.04+ | Ubuntu 22.04 |
| Python | 3.10+ | 3.11 |
| CUDA | 12.1+ | 12.8 |

---

## Windows Setup

### Step 1: Install Prerequisites

1. **Install Python 3.11** from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Install CUDA Toolkit 12.x** from [NVIDIA](https://developer.nvidia.com/cuda-downloads)

3. **Install Git** from [git-scm.com](https://git-scm.com/download/win)

4. **Install FFmpeg:**
   ```powershell
   # Using winget
   winget install FFmpeg

   # Or download from https://ffmpeg.org/download.html
   ```

### Step 2: Install ComfyUI

```powershell
# Create workspace
mkdir C:\ComfyUI
cd C:\ComfyUI

# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ComfyUI requirements
pip install -r requirements.txt
```

### Step 3: Install Custom Nodes

```powershell
cd C:\ComfyUI\ComfyUI\custom_nodes

# Essential nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git

# Install dependencies
pip install sageattention onnxruntime-gpu facexlib pykalman accelerate
pip install timm gguf ftfy xformers diffusers insightface huggingface-hub
```

### Step 4: Create Model Directories

```powershell
cd C:\ComfyUI\ComfyUI\models

# Create directories
mkdir checkpoints, clip, vae, loras, unet, pulid, text_encoders, sam2, detection
mkdir diffusion_models\Wan2.2-Animate-14B
mkdir diffusion_models\Wan2.2-I2V-A14B
mkdir loras\WanVideo\Lightx2v
mkdir loras\WanVideo\LoRAs\Wan22_relight
mkdir insightface\models\antelopev2
mkdir facerestore_models

# Create workflow directory
mkdir ..\user\default\workflows
```

### Step 5: Download Models

See the [Model Downloads](#model-downloads) section below.

### Step 6: Start ComfyUI

```powershell
cd C:\ComfyUI\ComfyUI
.\venv\Scripts\Activate.ps1
python main.py
```

Open **http://localhost:8188** in your browser.

---

## Linux Setup

### Step 1: Install Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y git git-lfs ffmpeg libgl1-mesa-glx libglib2.0-0 \
    build-essential python3-pip python3-venv wget curl

# Install CUDA (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### Step 2: Install ComfyUI

```bash
# Create workspace
mkdir -p ~/ComfyUI
cd ~/ComfyUI

# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ComfyUI requirements
pip install -r requirements.txt
```

### Step 3: Install Custom Nodes

```bash
cd ~/ComfyUI/ComfyUI/custom_nodes

# Essential nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git

# ReActor (face swap)
git clone https://github.com/Gourieff/comfyui-reactor-node.git
cd comfyui-reactor-node && python install.py && cd ..

# Install Python dependencies
pip install sageattention onnxruntime-gpu facexlib pykalman accelerate
pip install timm gguf ftfy xformers diffusers insightface huggingface-hub

# Install node requirements
for dir in */; do
    if [ -f "${dir}requirements.txt" ]; then
        pip install -r "${dir}requirements.txt" || true
    fi
done
```

### Step 4: Create Model Directories

```bash
cd ~/ComfyUI/ComfyUI/models

mkdir -p checkpoints clip vae loras unet pulid text_encoders sam2
mkdir -p detection  # CRITICAL: for YOLOv10 and ViTPose
mkdir -p diffusion_models/Wan2.2-Animate-14B
mkdir -p diffusion_models/Wan2.2-I2V-A14B
mkdir -p loras/WanVideo/Lightx2v
mkdir -p loras/WanVideo/LoRAs/Wan22_relight
mkdir -p insightface/models/antelopev2
mkdir -p facerestore_models

# Workflow directory
mkdir -p ../user/default/workflows
```

### Step 5: Start ComfyUI

```bash
cd ~/ComfyUI/ComfyUI
source venv/bin/activate
python main.py
```

Open **http://localhost:8188** in your browser.

---

## Model Downloads

### Using Hugging Face CLI

```bash
# Login first (optional, for private models)
huggingface-cli login

# Navigate to models directory
cd ~/ComfyUI/ComfyUI/models  # Linux
# or
cd C:\ComfyUI\ComfyUI\models  # Windows (PowerShell)
```

### Wan 2.2 Animate Models (~35GB total)

```bash
# Main model (FP8, ~18GB)
huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled \
    Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
    --local-dir diffusion_models/Wan2.2-Animate-14B

# VAE
huggingface-cli download Kijai/WanVideo_comfy \
    Wan2_1_VAE_bf16.safetensors --local-dir vae

# Text Encoder (~9.4GB)
huggingface-cli download Kijai/WanVideo_comfy \
    umt5-xxl-enc-bf16.safetensors --local-dir text_encoders

# CLIP Vision (~2.4GB)
huggingface-cli download h94/IP-Adapter \
    models/image_encoder/model.safetensors --local-dir clip_temp

# Move CLIP Vision
mv clip_temp/models/image_encoder/model.safetensors clip/clip_vision_h.safetensors
rm -rf clip_temp

# LoRAs
huggingface-cli download Kijai/WanVideo_comfy \
    Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
    --local-dir loras/WanVideo

huggingface-cli download Kijai/WanVideo_comfy \
    LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors \
    --local-dir loras/WanVideo

# Detection models (MUST be in detection/)
wget https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx \
    -O detection/yolov10m.onnx

wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx \
    -O detection/vitpose-l-wholebody.onnx

# SAM2
huggingface-cli download facebook/sam2.1-hiera-base-plus \
    sam2.1_hiera_base_plus.safetensors --local-dir sam2
```

### Flux + PuLID Models (~35GB total)

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
cd ../../..
```

---

## Applying Patches

### PuLID-Flux Compatibility Fix

```bash
# Linux
sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' \
    custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py

# Windows (PowerShell)
$file = "custom_nodes\ComfyUI-PuLID-Flux\pulidflux.py"
$content = Get-Content $file -Raw
$content = $content -replace '    control=None,$', "    control=None,`n    transformer_options={},`n    attn_mask=None,"
Set-Content $file $content
```

---

## Loading Workflows

1. Copy workflows to: `ComfyUI/user/default/workflows/`
2. Open ComfyUI at http://localhost:8188
3. Click "Workflows" in the left panel
4. Select the workflow you want to use

---

## Performance Tips

### VRAM Management

For 24GB cards (RTX 4090):
- Use FP8 quantized models when available
- Enable block swapping in model loaders
- Generate at 480p instead of 720p
- Reduce frame count for initial tests

For 16GB cards (RTX 3090):
- Use FP8 models exclusively
- Enable aggressive CPU offloading
- Generate at 480p with fewer frames
- Consider using Wan 2.1 5B instead of 14B

### Speed Optimization

1. **Enable SageAttention:**
   ```bash
   pip install sageattention
   ```
   Set attention mode to `sageattn` in model loader nodes.

2. **Use LightX2V LoRA:** Reduces required steps from 50 to 5-10.

3. **Enable torch.compile:** Use `WanVideoTorchCompileSettings` node.

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

### Quick Checks

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi

# Check model files
ls models/diffusion_models/
ls models/detection/

# Check ComfyUI logs
# Look at terminal output when running ComfyUI
```

---

## Directory Structure

After setup, your directory should look like:

```
ComfyUI/
├── custom_nodes/
│   ├── ComfyUI-Manager/
│   ├── ComfyUI-WanVideoWrapper/
│   ├── ComfyUI-WanAnimatePreprocess/
│   ├── ComfyUI-segment-anything-2/
│   ├── ComfyUI-KJNodes/
│   ├── ComfyUI-VideoHelperSuite/
│   ├── ComfyUI-PuLID-Flux/
│   └── ...
├── models/
│   ├── checkpoints/
│   ├── clip/
│   │   ├── clip_vision_h.safetensors
│   │   ├── clip_l.safetensors
│   │   └── t5xxl_fp16.safetensors
│   ├── detection/              # CRITICAL: ONNX models here
│   │   ├── yolov10m.onnx
│   │   └── vitpose-l-wholebody.onnx
│   ├── diffusion_models/
│   │   └── Wan2.2-Animate-14B/
│   ├── loras/
│   │   └── WanVideo/
│   ├── pulid/
│   ├── sam2/
│   ├── text_encoders/
│   ├── unet/
│   └── vae/
├── user/
│   └── default/
│       └── workflows/
│           ├── wan22_animate_faceswap.json
│           └── ...
└── venv/
```
