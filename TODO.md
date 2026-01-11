# APOB.ai Clone - Implementation TODO

## Phase 1: Vast.ai Setup
- [ ] Create `scripts/setup_vastai.sh` - Instance configuration with CUDA 12.8, PyTorch 2.7.1+
- [ ] Create `scripts/download_models.sh` - Download all required models from HuggingFace

## Phase 2: ComfyUI Workflows
- [ ] `workflows/flux_portrait.json` - Flux Dev portrait generation with ControlNet
- [ ] `workflows/pulid_character.json` - PuLID-Flux consistent character identity
- [ ] `workflows/wan22_video.json` - Wan 2.2 image-to-video animation
- [ ] `workflows/sadtalker_lipsync.json` - SadTalker talking avatar + lip sync
- [ ] `workflows/reactor_faceswap.json` - ReActor/Roop face swap with CodeFormer

## Phase 3: FastAPI Backend
- [ ] `api/main.py` - FastAPI app entry point
- [ ] `api/services/comfyui.py` - ComfyUI WebSocket/REST API client
- [ ] `api/services/workflow_loader.py` - Load and parameterize workflow JSONs
- [ ] `api/services/queue_manager.py` - Async job queue management
- [ ] `api/models/schemas.py` - Pydantic request/response models
- [ ] `api/routers/portrait.py` - POST /api/portrait endpoint
- [ ] `api/routers/character.py` - POST /api/character endpoint
- [ ] `api/routers/video.py` - POST /api/video endpoint
- [ ] `api/routers/talking.py` - POST /api/talking endpoint
- [ ] `api/routers/faceswap.py` - POST /api/faceswap endpoint

## Phase 4: Deployment
- [ ] `docker/Dockerfile` - Container with ComfyUI + API
- [ ] `docker-compose.yml` - Multi-service orchestration
- [ ] `requirements.txt` - Python dependencies

## Phase 5: Documentation
- [ ] `README.md` - Setup and usage instructions
- [ ] API documentation with examples

---

## Models to Download

| Model | Source | Size |
|-------|--------|------|
| Flux Dev | `black-forest-labs/FLUX.1-dev` | ~24GB |
| Wan 2.2 T2V | `Wan-AI/Wan2.2-T2V-A14B` | ~28GB |
| PuLID-Flux | ComfyUI Manager | ~2GB |
| SadTalker | `OpenTalker/SadTalker` | ~1.5GB |
| InsightFace inswapper | InsightFace | ~500MB |
| CodeFormer | CodeFormer repo | ~400MB |

## ComfyUI Custom Nodes Required

- ComfyUI-Manager
- ComfyUI-Flux
- PuLID-ComfyUI
- ComfyUI-WAN
- ComfyUI-SadTalker / ComfyUI-MuseTalk
- ComfyUI-ReActor

---

## Progress Tracking

**Current Phase**: Implementation Complete
**Status**: All core features implemented

## Completed
- [x] Project structure created
- [x] Vast.ai setup script
- [x] Model download script
- [x] All ComfyUI workflow templates
- [x] FastAPI backend with all endpoints
- [x] ComfyUI API client service
- [x] Dockerfile and docker-compose
- [x] README documentation

## Next Steps (Manual)
1. Deploy to Vast.ai instance
2. Download models (~80GB)
3. Test each workflow in ComfyUI first
4. Fine-tune workflow parameters as needed
5. Test API endpoints
