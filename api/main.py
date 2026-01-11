"""
APOB Clone - FastAPI Backend

Main entry point for the API server that wraps ComfyUI workflows.
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from models import HealthResponse
from services import get_comfyui_client, init_comfyui_client

# Import routers
from routers import portrait, character, video, talking, faceswap


# Configuration
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs")
WORKFLOWS_DIR = os.getenv("WORKFLOWS_DIR", "workflows")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting APOB Clone API...")

    try:
        await init_comfyui_client(host=COMFYUI_HOST, port=COMFYUI_PORT)
        logger.info(f"Connected to ComfyUI at {COMFYUI_HOST}:{COMFYUI_PORT}")
    except Exception as e:
        logger.warning(f"Could not connect to ComfyUI: {e}")
        logger.warning("API will start but generation endpoints won't work until ComfyUI is available")

    yield

    # Shutdown
    logger.info("Shutting down APOB Clone API...")
    client = get_comfyui_client()
    await client.close()


# Create FastAPI app
app = FastAPI(
    title="APOB Clone API",
    description="Open source AI portrait, video, and avatar generation API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portrait.router, prefix="/api", tags=["Portrait Generation"])
app.include_router(character.router, prefix="/api", tags=["Consistent Character"])
app.include_router(video.router, prefix="/api", tags=["Image to Video"])
app.include_router(talking.router, prefix="/api", tags=["Talking Avatar"])
app.include_router(faceswap.router, prefix="/api", tags=["Face Swap"])


# ============================================
# Root & Health Endpoints
# ============================================
@app.get("/")
async def root():
    """API root - returns basic info"""
    return {
        "name": "APOB Clone API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    client = get_comfyui_client()
    comfyui_ok = await client.health_check()

    gpu_info = None
    gpu_available = False

    if comfyui_ok:
        try:
            stats = await client.get_system_stats()
            devices = stats.get("devices", [])
            if devices:
                gpu_info = devices[0].get("name", "Unknown GPU")
                gpu_available = True
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if comfyui_ok else "degraded",
        version="1.0.0",
        comfyui_connected=comfyui_ok,
        gpu_available=gpu_available,
        gpu_name=gpu_info
    )


# ============================================
# Job Status & Results
# ============================================
@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a generation job"""
    client = get_comfyui_client()
    job = client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "result_files": job.result_files,
        "error": job.error
    }


@app.get("/api/result/{job_id}")
async def get_job_result(job_id: str, file_index: int = 0):
    """
    Get the result file(s) from a completed job

    Args:
        job_id: The job ID
        file_index: Which file to return (default: first file)
    """
    client = get_comfyui_client()
    job = client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )

    if not job.result_files:
        raise HTTPException(status_code=404, detail="No result files found")

    if file_index >= len(job.result_files):
        raise HTTPException(
            status_code=400,
            detail=f"File index {file_index} out of range. Job has {len(job.result_files)} files."
        )

    file_info = job.result_files[file_index]

    try:
        file_data = await client.get_image(
            filename=file_info["filename"],
            subfolder=file_info.get("subfolder", ""),
            folder_type=file_info.get("folder_type", "output")
        )

        # Determine media type
        filename = file_info["filename"].lower()
        if filename.endswith(".mp4"):
            media_type = "video/mp4"
        elif filename.endswith(".webm"):
            media_type = "video/webm"
        elif filename.endswith(".gif"):
            media_type = "image/gif"
        elif filename.endswith(".png"):
            media_type = "image/png"
        else:
            media_type = "image/jpeg"

        from fastapi.responses import Response
        return Response(content=file_data, media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch result: {e}")


@app.get("/api/results/{job_id}")
async def list_job_results(job_id: str):
    """List all result files from a job"""
    client = get_comfyui_client()
    job = client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "files": [
            {
                "index": i,
                "filename": f["filename"],
                "type": f["type"],
                "url": f"/api/result/{job_id}?file_index={i}"
            }
            for i, f in enumerate(job.result_files)
        ]
    }


# ============================================
# Queue Management
# ============================================
@app.get("/api/queue")
async def get_queue():
    """Get current ComfyUI queue status"""
    client = get_comfyui_client()

    try:
        queue = await client.get_queue_status()
        return queue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue: {e}")


@app.delete("/api/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    client = get_comfyui_client()
    job = client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    if job.prompt_id:
        success = await client.cancel_prompt(job.prompt_id)
        if success:
            job.status = "cancelled"
            return {"message": f"Job {job_id} cancelled"}

    raise HTTPException(status_code=500, detail="Failed to cancel job")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
