# APOB Clone - Setup & Usage Instructions

## Workflows Available

| Workflow | File | Description |
|----------|------|-------------|
| **Flux Portrait** | `flux_portrait.json` | AI portrait generation from text |
| **PuLID-Flux Character** | `pulid_flux_character.json` | Consistent face identity with Flux (recommended) |
| **Wan 2.2 I2V** | `wan22_i2v.json` | Image to video animation |
| **LivePortrait** | `liveportrait_animate.json` | Face animation from driving video |
| **ReActor Face Swap** | `reactor_faceswap.json` | Swap faces in images |
| **ReActor Video** | `reactor_video_faceswap.json` | Swap faces in videos |
| **Wan 2.2 Animate** | `wan22_animate_faceswap.json` | Swap characters in videos using driving video motion |
| **Wan 2.2 I2V** | `wan22_i2v_character.json` | Animate still images with AI-generated motion |

## Quick Reference

**Instance Details:**
- SSH: `ssh -p 55622 root@69.176.92.113`
- GitHub Repo: https://github.com/matrix-logs/myclone

---

## Accessing Services

### ComfyUI (Port 8188)
ComfyUI is not directly accessible - you must use SSH port forwarding:

```bash
# Run this on your LOCAL machine (not the server)
ssh -p 55622 root@69.176.92.113 -L 8188:localhost:8188
```

Then open: **http://localhost:8188**

### FastAPI Backend (Port 8000)
```bash
ssh -p 55622 root@69.176.92.113 -L 8000:localhost:8000
```

Then open: **http://localhost:8000/docs** for API documentation

---

## Starting Services

### Start ComfyUI
```bash
ssh -p 55622 root@69.176.92.113
cd /workspace/ComfyUI
nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
```

### Start FastAPI Backend
```bash
ssh -p 55622 root@69.176.92.113
cd /workspace/myclone/ugc
nohup python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /workspace/api.log 2>&1 &
```

### Start Both Services (One Command)
```bash
ssh -p 55622 root@69.176.92.113 "cd /workspace/ComfyUI && nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 & cd /workspace/myclone/ugc && nohup python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /workspace/api.log 2>&1 &"
```

---

## Using ComfyUI Workflows

### Loading Workflows
1. Open ComfyUI at http://localhost:8188
2. Click the **Workflows** button in the left panel
3. Browse the available workflows:
   - `flux_portrait.json` - AI portrait generation
   - `pulid_flux_character.json` - Consistent character identity (PuLID-Flux)
   - `wan22_i2v.json` - Image to video
   - `wan22_animate_character.json` - Animate PuLID characters
   - `liveportrait_animate.json` - Face animation
   - `reactor_faceswap.json` - Face swap (images)
   - `reactor_video_faceswap.json` - Face swap (videos)

**Workflow Location**: `/workspace/ComfyUI/user/default/workflows/`

### Portrait Generation (Flux Dev)
1. Load `flux_portrait.json`
2. Find the **CLIP Text Encode** node
3. Enter your prompt (e.g., "professional headshot of a young woman, blonde hair, blue eyes")
4. Click **Queue Prompt** to generate

### Consistent Character (PuLID-Flux)
1. Load `pulid_flux_character.json`
2. Upload a reference face image to the **Load Image** node
3. Enter a prompt in the **CLIP Text Encode** node describing the scene
4. Adjust **weight** in ApplyPulidFlux (0.6-1.0, higher = more face similarity)
5. Click **Queue Prompt** to generate

#### PuLID-Flux Workflow Details

**Nodes Used:**
| Node | Purpose |
|------|---------|
| `UNETLoader` | Loads Flux Dev model (`flux1-dev.safetensors`) |
| `DualCLIPLoader` | Loads CLIP L + T5XXL text encoders |
| `VAELoader` | Loads Flux VAE (`ae.safetensors`) |
| `PulidFluxModelLoader` | Loads PuLID-Flux model (`pulid_flux_v0.9.1.safetensors`) |
| `PulidFluxInsightFaceLoader` | Face detection (use CUDA provider) |
| `PulidFluxEvaClipLoader` | EVA CLIP for face encoding |
| `LoadImage` | Reference face image input |
| `CLIPTextEncode` | Text prompt for generation |
| `FluxGuidance` | CFG scale (default: 3.5) |
| `ApplyPulidFlux` | Applies face identity to model |
| `EmptySD3LatentImage` | Creates latent (768x1024 for portraits) |
| `SamplerCustomAdvanced` | Sampling with BasicGuider |
| `VAEDecode` | Decodes latent to image |

**Key Parameters:**
- **weight** (ApplyPulidFlux): 0.6-1.0 - Higher = stronger face similarity
- **start_at** (ApplyPulidFlux): 0.0 - When to start applying identity
- **end_at** (ApplyPulidFlux): 1.0 - When to stop applying identity
- **guidance** (FluxGuidance): 3.5 - CFG scale for prompt adherence
- **steps** (BasicScheduler): 20 - Sampling steps
- **denoise** (BasicScheduler): 1.0 - Full generation

**Tips:**
- Use high-quality, front-facing reference photos for best results
- The face should be clearly visible and well-lit
- Adjust `weight` lower (0.6-0.8) if the result looks too similar to reference
- For different poses/angles, describe them in the prompt
- Works best with portrait-oriented images (768x1024 or similar)

**Custom Node Source:** [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)

### Image to Video (Wan 2.2 I2V)
1. Load `wan22_i2v_character.json`
2. Upload source image
3. Set duration (4-8 seconds)
4. Set motion strength (0.3-0.7)
5. Enter text prompt describing desired motion
6. Queue the prompt

### Character Swap (Wan 2.2 Animate)
1. Load `wan22_animate_faceswap.json`
2. Upload a **driving video** (performer whose motion you want to use)
3. Upload a **reference character image** (face to swap in)
4. Adjust settings:
   - Resolution: 832x480 recommended
   - Frame rate: 16 fps
5. Queue the prompt

**How it works:**
- Extracts pose/skeleton from driving video using YOLOv10 + ViTPose
- Segments the person using SAM2
- Generates new character with reference face matching the poses
- Optionally applies relight LoRA to match original lighting

**Required models:**
- `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors`
- `clip_vision_h.safetensors`
- `yolov10m.onnx` (in `models/detection/`)
- `vitpose-l-wholebody.onnx` (in `models/detection/`)
- `sam2.1_hiera_base_plus.safetensors`

See [VASTAI_SETUP.md](VASTAI_SETUP.md#wan-22-animate-vs-i2v) for more details.

### Talking Avatar (LivePortrait)
1. Load `sadtalker_lipsync.json`
2. Upload face image
3. Upload audio file (WAV/MP3)
4. Adjust expression scale
5. Queue the prompt

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Portrait
```bash
curl -X POST "http://localhost:8000/api/portrait" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "professional headshot of a young woman, blonde hair, blue eyes",
    "style": "photorealistic",
    "aspect_ratio": "9:16"
  }'
```

### Check Job Status
```bash
curl "http://localhost:8000/api/status/{job_id}"
```

### Download Result
```bash
curl "http://localhost:8000/api/result/{job_id}" --output result.png
```

---

## Installed Models

| Model | Location | Size |
|-------|----------|------|
| Flux Dev | `/workspace/ComfyUI/models/unet/flux1-dev.safetensors` | 23GB |
| Flux VAE | `/workspace/ComfyUI/models/vae/ae.safetensors` | 320MB |
| CLIP L | `/workspace/ComfyUI/models/clip/clip_l.safetensors` | 235MB |
| T5XXL | `/workspace/ComfyUI/models/clip/t5xxl_fp16.safetensors` | 9.4GB |
| PuLID-Flux | `/workspace/ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors` | 1.1GB |
| LivePortrait | `/workspace/ComfyUI/models/liveportrait/` | ~500MB |
| InsightFace | `/workspace/ComfyUI/models/insightface/inswapper_128.onnx` | 529MB |
| CodeFormer | `/workspace/ComfyUI/models/facerestore_models/codeformer-v0.1.0.pth` | 360MB |
| Wan 2.1 I2V | `/workspace/ComfyUI/models/diffusion_models/Wan2.1-I2V-14B-480P/` | 78GB |
| Wan 2.2 I2V | `/workspace/ComfyUI/models/diffusion_models/Wan2.2-I2V-A14B/` | 118GB |

---

## Checking Logs

```bash
# ComfyUI logs
ssh -p 55622 root@69.176.92.113 "tail -100 /workspace/comfyui.log"

# API logs
ssh -p 55622 root@69.176.92.113 "tail -100 /workspace/api.log"
```

---

## Troubleshooting

### ComfyUI Not Accessible
1. Ensure you're using SSH port forwarding (see above)
2. Check if ComfyUI is running: `pgrep -a python3`
3. Check logs: `tail /workspace/comfyui.log`

### Out of Memory
- The RTX 5090 has 32GB VRAM which should handle all models
- If issues occur, close other GPU processes first

### Model Not Found
- Verify model paths in workflow JSON match actual locations
- Check `/workspace/ComfyUI/models/` for installed models

---

## File Locations

- **Code**: `/workspace/myclone/ugc/`
- **Workflows**: `/workspace/ComfyUI/user/default/workflows/`
- **Models**: `/workspace/ComfyUI/models/`
- **Custom Nodes**: `/workspace/ComfyUI/custom_nodes/`
- **Outputs**: `/workspace/outputs/`
- **Logs**: `/workspace/comfyui.log`, `/workspace/api.log`

---

## Costs (Vast.ai)

- RTX 5090: ~$0.50-1.00/hour
- Storage: ~$0.05/GB/month
- Remember to stop your instance when not in use!

---

## Sources

- [Wan 2.2 I2V Model](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [PuLID-Flux Model](https://huggingface.co/guozinan/PuLID)
- [ComfyUI-PuLID-Flux Node](https://github.com/balazik/ComfyUI-PuLID-Flux)
- [Flux Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [LivePortrait](https://huggingface.co/Kijai/LivePortrait_safetensors)
- [ReActor Face Swap](https://github.com/Gourieff/comfyui-reactor-node)
