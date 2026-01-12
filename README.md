# APOB Clone - Open Source AI Portrait & Video Generation

An open-source alternative to [APOB.ai](https://apob.ai/) using state-of-the-art AI models. Generate photorealistic portraits, maintain consistent character identity, create videos from images, swap characters in videos, and produce talking avatars with lip sync.

## Features

| Feature | Model Used | Description |
|---------|------------|-------------|
| **AI Portrait Generation** | Flux Dev | Photorealistic human portraits from text |
| **Consistent Character** | PuLID-Flux | Same face across multiple images/poses |
| **Image to Video** | Wan 2.2 I2V | Animate static images with AI-generated motion |
| **Character Swap** | Wan 2.2 Animate | Replace character in video using driving video motion |
| **Talking Avatar** | LivePortrait | Lip-synced speaking videos |
| **Face Swap** | ReActor + CodeFormer | Swap faces with enhancement |

## What's New

### Wan 2.2 Animate - Character Swap Workflow

The latest addition enables **character replacement in videos** using motion from a driving video:

- Extract poses and faces from a performer video
- Apply motion to any reference character image
- Maintain character identity while matching performer movements
- Optional relighting to match original video's lighting

**Key difference from I2V:**
- I2V animates still images with AI-generated motion (text-prompted)
- Animate replaces characters using real motion from driving videos

See [Wan 2.2 Animate vs I2V](VASTAI_SETUP.md#wan-22-animate-vs-i2v) for detailed comparison.

---

## Requirements

- **GPU**: NVIDIA RTX 5090 (32GB VRAM) recommended, RTX 4090 (24GB) minimum
- **CUDA**: 12.1+ (12.8 for RTX 5090)
- **Python**: 3.10+
- **Storage**: ~100-300GB for models

---

## Quick Start

### Option 1: Vast.ai (Recommended)

```bash
# SSH into your Vast.ai instance
ssh -p <PORT> root@<IP_ADDRESS>

# Clone and setup
git clone https://github.com/matrix-logs/myclone.git
cd myclone/ugc
bash scripts/setup_vastai.sh

# Download models
bash scripts/setup_vastai.sh download_animate  # Character swap
bash scripts/setup_vastai.sh download_flux     # Portrait generation

# Start ComfyUI
bash /workspace/start_comfyui.sh
```

See [VASTAI_SETUP.md](VASTAI_SETUP.md) for detailed instructions.

### Option 2: Local Machine

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for Windows/Linux setup instructions.

---

## Available Workflows

| Workflow | File | Purpose |
|----------|------|---------|
| **Wan 2.2 Animate** | `wan22_animate_faceswap.json` | Swap characters in videos |
| **Wan 2.2 I2V** | `wan22_i2v_character.json` | Animate still images |
| **PuLID-Flux Character** | `pulid_flux_character.json` | Consistent face identity |
| **Flux Portrait** | `flux_portrait.json` | AI portrait generation |
| **ReActor Face Swap** | `reactor_faceswap.json` | Swap faces in images |
| **LivePortrait** | `liveportrait_animate.json` | Face animation/lip-sync |

---

## Documentation

| Document | Description |
|----------|-------------|
| [VASTAI_SETUP.md](VASTAI_SETUP.md) | Complete Vast.ai setup guide |
| [LOCAL_SETUP.md](LOCAL_SETUP.md) | Local Windows/Linux setup |
| [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) | Workflow usage guide |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues & fixes |

---

## Models Required

### For Wan 2.2 Animate (~35GB)

| Model | Size | Purpose |
|-------|------|---------|
| Wan2.2-Animate-14B (FP8) | 18GB | Character generation |
| UMT5-XXL | 9.4GB | Text encoding |
| CLIP Vision H | 2.4GB | Image encoding |
| SAM2 | 300MB | Segmentation |
| YOLOv10 | 59MB | Person detection |
| ViTPose-L | 1.2GB | Pose estimation |

### For Flux + PuLID (~35GB)

| Model | Size | Purpose |
|-------|------|---------|
| Flux Dev | 23GB | Image generation |
| T5XXL | 9.4GB | Text encoding |
| PuLID-Flux | 1.1GB | Face identity |
| InsightFace | 200MB | Face detection |

---

## Project Structure

```
ugc/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── routers/
│   └── services/
├── workflows/              # ComfyUI workflow JSONs
│   ├── wan22_animate_faceswap.json
│   ├── wan22_i2v_character.json
│   ├── pulid_flux_character.json
│   └── ...
├── scripts/
│   └── setup_vastai.sh     # Automated setup script
├── VASTAI_SETUP.md         # Vast.ai setup guide
├── LOCAL_SETUP.md          # Local setup guide
├── TROUBLESHOOTING.md      # Common issues & fixes
└── README.md
```

---

## Key Learnings

During development, we discovered several important details:

### Model Paths Matter
- Detection models (YOLOv10, ViTPose) must be in `models/detection/`, NOT `models/onnx/`
- CLIP Vision H is available from IP-Adapter repo, not Kijai's repo

### Workflow JSON Requirements
- ComfyUI requires valid UUID format for workflow `id` field
- Every node must have a `properties` field (even if empty: `{}`)

### Custom Nodes Required for Animate
- ComfyUI-WanVideoWrapper (core Wan support)
- ComfyUI-WanAnimatePreprocess (pose/face detection)
- ComfyUI-segment-anything-2 (SAM2 segmentation)
- ComfyUI-KJNodes (utility nodes)

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for all fixes.

---

## Performance

Benchmarks on RTX 5090 (32GB):

| Task | Resolution | Duration |
|------|------------|----------|
| Portrait (Flux) | 768x1024 | ~15-20s |
| PuLID Character | 768x1024 | ~20-25s |
| I2V (4s) | 480p | ~2-3min |
| Animate (4s) | 480p | ~3-5min |
| Talking (10s audio) | 512x512 | ~1-2min |
| Face Swap | Any | ~5-10s |

---

## Contributing

Pull requests welcome! Areas for improvement:
- Additional workflow templates
- TTS integration for talking avatars
- Web frontend
- Batch processing optimization

---

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based UI
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2) - Video generation models
- [Kijai](https://github.com/kijai) - ComfyUI node implementations
- [Flux](https://github.com/black-forest-labs/flux) - Image generation
- [PuLID](https://github.com/ToTheBeginning/PuLID) - Identity preservation
- [ReActor](https://github.com/Gourieff/comfyui-reactor-node) - Face swap

---

## License

MIT License - See LICENSE file
