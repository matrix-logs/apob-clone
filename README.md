# APOB Clone - Open Source AI Portrait & Video Generation

An open-source alternative to [APOB.ai](https://apob.ai/) using state-of-the-art AI models. Generate photorealistic portraits, maintain consistent character identity, create videos from images, and produce talking avatars with lip sync.

## Features

| Feature | Model Used | Description |
|---------|------------|-------------|
| **AI Portrait Generation** | Flux Dev | Photorealistic human portraits from text |
| **Consistent Character** | PuLID-Flux | Same face across multiple images/poses |
| **Image to Video** | Wan 2.2 | Animate static images (same as APOB!) |
| **Talking Avatar** | LivePortrait/SadTalker | Lip-synced speaking videos |
| **Face Swap** | ReActor + CodeFormer | Swap faces with enhancement |

## Requirements

- **GPU**: NVIDIA RTX 5090 (32GB VRAM) recommended, RTX 4090 (24GB) minimum
- **CUDA**: 12.8+
- **Python**: 3.11+
- **Storage**: ~100GB for models

## Quick Start (Vast.ai)

### 1. Create Vast.ai Instance

1. Go to [Vast.ai](https://vast.ai/) and create an account
2. Select an RTX 5090 instance (or 4090 if 5090 unavailable)
3. Use the pre-built **ComfyUI** template
4. Ensure at least 100GB storage

### 2. Setup Instance

SSH into your instance and run:

```bash
# Clone this repository
git clone https://github.com/yourusername/apob-clone.git
cd apob-clone

# Run setup script
chmod +x scripts/setup_vastai.sh
bash scripts/setup_vastai.sh
```

### 3. Download Models

```bash
# Download all models (~80GB)
bash scripts/download_models.sh --all

# Or download selectively:
bash scripts/download_models.sh --flux      # Portrait only
bash scripts/download_models.sh --pulid     # + Character consistency
bash scripts/download_models.sh --wan       # + Video generation
bash scripts/download_models.sh --lipsync   # + Talking avatar
bash scripts/download_models.sh --faceswap  # + Face swap
```

### 4. Start Services

```bash
bash /workspace/start_all.sh
```

- **ComfyUI**: http://your-instance:8188
- **API**: http://your-instance:8000
- **API Docs**: http://your-instance:8000/docs

## API Usage

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

Response:
```json
{
  "job_id": "abc123...",
  "status": "queued",
  "message": "Portrait generation queued. Use /api/status/abc123... to check progress."
}
```

### Check Status

```bash
curl "http://localhost:8000/api/status/abc123..."
```

### Get Result

```bash
curl "http://localhost:8000/api/result/abc123..." --output portrait.png
```

### Consistent Character

```bash
curl -X POST "http://localhost:8000/api/character/upload" \
  -F "reference_image=@face.jpg" \
  -F "prompt=person standing on a beach at sunset" \
  -F "identity_strength=0.8"
```

### Image to Video

```bash
curl -X POST "http://localhost:8000/api/video/upload" \
  -F "image=@portrait.png" \
  -F "duration=4.0" \
  -F "motion_strength=0.5"
```

### Talking Avatar

```bash
curl -X POST "http://localhost:8000/api/talking/upload" \
  -F "image=@face.png" \
  -F "audio=@speech.wav" \
  -F "expression_scale=1.0"
```

### Face Swap

```bash
curl -X POST "http://localhost:8000/api/faceswap/upload" \
  -F "source_face=@my_face.jpg" \
  -F "target=@target_image.jpg" \
  -F "enhance_face=true"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/portrait` | POST | Generate AI portrait |
| `/api/character` | POST | Generate consistent character |
| `/api/video` | POST | Image to video |
| `/api/talking` | POST | Talking avatar |
| `/api/faceswap` | POST | Face swap |
| `/api/status/{job_id}` | GET | Check job status |
| `/api/result/{job_id}` | GET | Download result |
| `/api/results/{job_id}` | GET | List all result files |
| `/health` | GET | Health check |

## Project Structure

```
apob-clone/
├── api/
│   ├── main.py              # FastAPI entry point
│   ├── routers/
│   │   ├── portrait.py      # Portrait generation
│   │   ├── character.py     # Consistent character
│   │   ├── video.py         # Image to video
│   │   ├── talking.py       # Talking avatar
│   │   └── faceswap.py      # Face swap
│   ├── services/
│   │   ├── comfyui.py       # ComfyUI API client
│   │   └── workflow_loader.py
│   └── models/
│       └── schemas.py       # Pydantic models
├── workflows/
│   ├── flux_portrait.json
│   ├── pulid_character.json
│   ├── wan22_video.json
│   ├── sadtalker_lipsync.json
│   └── reactor_faceswap.json
├── scripts/
│   ├── setup_vastai.sh
│   └── download_models.sh
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

## Docker Deployment

```bash
# Build image
cd docker
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Models Used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| Flux Dev | Image generation | ~24GB | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| PuLID-Flux | Identity preservation | ~2GB | [HuggingFace](https://huggingface.co/guozinan/PuLID) |
| Wan 2.2 I2V | Video generation | ~28GB | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-480P) |
| LivePortrait | Lip sync | ~1.5GB | [HuggingFace](https://huggingface.co/Kijai/LivePortrait_safetensors) |
| InsightFace | Face analysis | ~500MB | [GitHub](https://github.com/deepinsight/insightface) |
| CodeFormer | Face enhancement | ~400MB | [GitHub](https://github.com/sczhou/CodeFormer) |

## Performance

Benchmarks on RTX 5090 (32GB):

| Task | Resolution | Time |
|------|------------|------|
| Portrait | 768x1024 | ~15-20s |
| Character | 768x1024 | ~20-25s |
| Video (4s) | 480p | ~2-3min |
| Talking (10s audio) | 512x512 | ~1-2min |
| Face Swap | Any | ~5-10s |

## Troubleshooting

### ComfyUI not starting
```bash
# Check if GPU is available
nvidia-smi

# Check ComfyUI logs
tail -f /workspace/ComfyUI/comfyui.log
```

### Model not found errors
```bash
# Verify models are downloaded
ls -la /workspace/ComfyUI/models/checkpoints/
ls -la /workspace/ComfyUI/models/unet/

# Re-run model download
bash scripts/download_models.sh --all
```

### Out of memory
- Reduce batch size in workflow
- Use smaller model variant (Wan 2.1 5B instead of 14B)
- Enable model offloading

## Contributing

Pull requests welcome! Areas for improvement:
- Additional workflow templates
- TTS integration for talking avatars
- Batch processing optimization
- Web frontend

## License

MIT License - See LICENSE file

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Flux](https://github.com/black-forest-labs/flux)
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2)
- [PuLID](https://github.com/ToTheBeginning/PuLID)
- [SadTalker](https://github.com/OpenTalker/SadTalker)
- [ReActor](https://github.com/Gourieff/comfyui-reactor-node)
