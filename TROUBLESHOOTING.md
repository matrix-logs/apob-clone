# Troubleshooting Guide

This guide documents common issues encountered during setup and workflow execution, along with their solutions.

---

## Table of Contents

1. [Workflow Loading Errors](#workflow-loading-errors)
2. [Model Path Errors](#model-path-errors)
3. [Custom Node Issues](#custom-node-issues)
4. [Memory and Performance](#memory-and-performance)
5. [Connection Issues](#connection-issues)

---

## Workflow Loading Errors

### "Invalid workflow against zod schema: Invalid uuid at 'id'"

**Cause:** ComfyUI workflow JSON requires a valid UUID format for the `id` field.

**Solution:** Change the workflow `id` from a string like `"my-workflow"` to a valid UUID:
```json
{
  "id": "a1b2c3d4-e5f6-4890-abcd-ef1234567890",
  ...
}
```

### "Required at 'nodes[x].properties'"

**Cause:** Each node in a ComfyUI workflow must have a `properties` field.

**Solution:** Add `"properties": {}` to every node that's missing it:
```json
{
  "id": 137,
  "type": "GetNode",
  "pos": [2441, -1173],
  "size": [210, 60],
  "flags": {"collapsed": true},
  "order": 0,
  "mode": 0,
  "outputs": [...],
  "properties": {},  // <-- Add this line
  "title": "Get_face_images",
  "widgets_values": ["face_images"]
}
```

### "Value not in list" for model selectors

**Cause:** The model path in the workflow doesn't match the actual path on the server.

**Solution:**
1. SSH into the server and list the actual model files:
   ```bash
   ls -la /workspace/ComfyUI/models/diffusion_models/
   ls -la /workspace/ComfyUI/models/vae/
   ls -la /workspace/ComfyUI/models/detection/
   ```
2. Update the workflow JSON to use the exact paths found on the server.

---

## Model Path Errors

### Detection models (ONNX) not found

**Cause:** WanAnimatePreprocess nodes look for ONNX models in `models/detection/`, not `models/onnx/`.

**Solution:** Move ONNX files to the correct directory:
```bash
mkdir -p /workspace/ComfyUI/models/detection
mv /workspace/ComfyUI/models/onnx/*.onnx /workspace/ComfyUI/models/detection/
```

### VAE not found

**Cause:** Workflow expects VAE in `models/vae/` but it's in `models/diffusion_models/Wan2.2-I2V-A14B/`.

**Solution:** Create a symlink:
```bash
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth \
    /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth
```

### Text encoder not found

**Cause:** Workflow expects text encoder in `models/text_encoders/`.

**Solution:** Create a symlink or download directly:
```bash
# Option 1: Symlink from I2V download
ln -sf /workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    /workspace/ComfyUI/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth

# Option 2: Download from Kijai's repo
huggingface-cli download Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors \
    --local-dir /workspace/ComfyUI/models/text_encoders
```

### CLIP Vision H not found

**Cause:** The CLIP Vision model wasn't downloaded or is in the wrong location.

**Solution:** Download from IP-Adapter repo:
```bash
cd /workspace/ComfyUI/models
huggingface-cli download h94/IP-Adapter models/image_encoder/model.safetensors \
    --local-dir clip_temp
mv clip_temp/models/image_encoder/model.safetensors clip/clip_vision_h.safetensors
rm -rf clip_temp
```

---

## Custom Node Issues

### Missing node type errors

**Cause:** Required custom nodes are not installed.

**Wan 2.2 Animate requires:**
```bash
cd /workspace/ComfyUI/custom_nodes

# WanVideoWrapper (core Wan support)
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# WanAnimatePreprocess (pose detection, face extraction)
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git

# Segment Anything 2 (character segmentation)
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git

# KJNodes (utility nodes)
git clone https://github.com/kijai/ComfyUI-KJNodes.git

# VideoHelperSuite (video I/O)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
```

### PuLID-Flux "unexpected keyword argument 'attn_mask'"

**Cause:** PuLID-Flux node is incompatible with newer ComfyUI versions.

**Solution:** Apply the compatibility patch:
```bash
sed -i 's/    control=None,$/    control=None,\n    transformer_options={},\n    attn_mask=None,/' \
    /workspace/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py
```

### Node dependencies missing

**Cause:** Custom node Python dependencies weren't installed.

**Solution:**
```bash
cd /workspace/ComfyUI/custom_nodes

# Install for specific node
pip install -r ComfyUI-WanVideoWrapper/requirements.txt
pip install -r comfyui-reactor-node/requirements.txt

# Or install all
for dir in */; do
    if [ -f "${dir}requirements.txt" ]; then
        pip install -r "${dir}requirements.txt" || true
    fi
done
```

---

## Memory and Performance

### Out of VRAM

**Symptoms:** CUDA out of memory errors during generation.

**Solutions:**

1. **Enable block swapping** in model loader nodes:
   - Set `offload_device` to enable CPU offloading

2. **Reduce resolution/frames:**
   - Use 480p instead of 720p
   - Generate fewer frames at a time

3. **Use quantized models:**
   - FP8 models use ~50% less VRAM than BF16
   - Example: `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors`

4. **Enable SageAttention:**
   ```bash
   pip install sageattention
   ```
   Then set attention mode to `sageattn` in model loader.

5. **Close other GPU processes:**
   ```bash
   nvidia-smi  # Check what's using VRAM
   ```

### Slow generation

**Solutions:**

1. **Use LightX2V LoRA** for faster step distillation:
   - Add `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`
   - Reduces required steps from 50 to 5-10

2. **Enable torch.compile:**
   - Use `WanVideoTorchCompileSettings` node
   - Set backend to `inductor`

3. **Use FP8 models** - faster inference than BF16

---

## Connection Issues

### Can't access ComfyUI in browser

**Cause:** ComfyUI runs on localhost only, not exposed to internet.

**Solution:** Use SSH port forwarding:
```bash
# On your LOCAL machine (not the server)
ssh -p <PORT> root@<IP_ADDRESS> -L 8188:localhost:8188

# Then open in browser
http://localhost:8188
```

### SSH connection refused

**Solutions:**

1. **Wait a few seconds** - Vast.ai auth can be slow
2. **Check SSH key is correct:**
   ```bash
   ssh-add -l  # List loaded keys
   ```
3. **Verify instance is running** in Vast.ai dashboard

### ComfyUI not starting

**Diagnosis:**
```bash
# Check if process is running
pgrep -a python3 | grep main.py

# Check logs
tail -100 /workspace/comfyui.log
```

**Common fixes:**
```bash
# Kill existing process
pkill -f 'python3 main.py'

# Restart
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188
```

---

## Model Download Issues

### huggingface-cli fails

**Solutions:**

1. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```

2. **Use direct wget for smaller files:**
   ```bash
   wget https://huggingface.co/<repo>/resolve/main/<file> -O <destination>
   ```

3. **Check disk space:**
   ```bash
   df -h /workspace
   ```

### Download interrupted

**Solution:** Use `--resume-download` flag:
```bash
huggingface-cli download <repo> --local-dir <dir> --resume-download
```

---

## Workflow-Specific Issues

### Wan 2.2 Animate: No faces detected

**Causes:**
- Input image quality too low
- Face too small or at extreme angle
- Detection models not loaded

**Solutions:**
1. Use clear, front-facing reference images
2. Ensure face is at least 64x64 pixels
3. Verify YOLOv10 and ViTPose models are in `models/detection/`

### Wan 2.2 Animate: Poor motion quality

**Solutions:**
1. Use driving videos with clear, smooth motion
2. Increase frame count for smoother interpolation
3. Add the relight LoRA for better lighting integration

### ReActor: Face swap quality issues

**Solutions:**
1. Ensure CodeFormer is installed for face restoration
2. Use higher `face_restore_visibility` values
3. Match lighting between source and target images

---

## Quick Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Check disk space
df -h /workspace

# Check Python packages
pip list | grep -E "(torch|comfy|wan)"

# Check custom nodes
ls /workspace/ComfyUI/custom_nodes/

# Check models
ls /workspace/ComfyUI/models/diffusion_models/
ls /workspace/ComfyUI/models/detection/
ls /workspace/ComfyUI/models/vae/

# Check logs
tail -50 /workspace/comfyui.log

# Restart ComfyUI
pkill -f 'python3 main.py'
cd /workspace/ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188
```

---

## Getting Help

1. **ComfyUI Discord:** https://discord.gg/comfyui
2. **Kijai's nodes:** https://github.com/kijai (check Issues)
3. **Wan AI:** https://github.com/Wan-Video/Wan2.2

When reporting issues, include:
- ComfyUI version
- Custom node versions (`git log -1` in node directory)
- Full error message from logs
- GPU model and VRAM
- Model paths and sizes
