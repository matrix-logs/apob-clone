"""
Portrait Generation Router

Handles AI portrait generation using Flux.
"""
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks

from models import PortraitRequest, PortraitResponse, AspectRatio
from services import get_comfyui_client, get_workflow_loader

router = APIRouter()


# Aspect ratio to dimensions mapping
ASPECT_RATIOS = {
    AspectRatio.SQUARE: (1024, 1024),
    AspectRatio.PORTRAIT: (768, 1024),
    AspectRatio.LANDSCAPE: (1024, 768),
    AspectRatio.PORTRAIT_4_3: (768, 1024),
    AspectRatio.LANDSCAPE_4_3: (1024, 768),
}


def build_portrait_prompt(request: PortraitRequest) -> str:
    """Build enhanced prompt from request parameters"""
    parts = []

    # Base style prefix
    if request.style == "photorealistic":
        parts.append("professional photograph, photorealistic, highly detailed skin texture")
    elif request.style == "artistic":
        parts.append("artistic portrait, painterly style")
    elif request.style == "anime":
        parts.append("anime style portrait, detailed illustration")
    elif request.style == "3d":
        parts.append("3D rendered portrait, octane render, unreal engine")

    # Add demographics if specified
    if request.age_range:
        parts.append(f"{request.age_range} person")
    if request.gender:
        parts.append(f"{request.gender}")
    if request.ethnicity:
        parts.append(f"{request.ethnicity}")

    # Add user prompt
    parts.append(request.prompt)

    # Quality boosters for photorealistic
    if request.style == "photorealistic":
        parts.append("studio lighting, 8k, ultra high definition, sharp focus")

    return ", ".join(parts)


@router.post("/portrait", response_model=PortraitResponse)
async def generate_portrait(
    request: PortraitRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate an AI portrait from a text description.

    Uses Flux Dev model for high-quality photorealistic portraits.

    Returns a job_id that can be used to check status and retrieve results.
    """
    job_id = str(uuid.uuid4())

    try:
        # Get dimensions from aspect ratio
        width, height = ASPECT_RATIOS.get(request.aspect_ratio, (768, 1024))

        # Build enhanced prompt
        full_prompt = build_portrait_prompt(request)

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_portrait_workflow(
            prompt=full_prompt,
            negative_prompt=request.negative_prompt or "",
            width=width,
            height=height,
            seed=request.seed or -1,
            num_images=request.num_images
        )

        # Queue the workflow
        client = get_comfyui_client()
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="flux_portrait"
        )

        return PortraitResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Portrait generation queued. Use /api/status/{job.job_id} to check progress."
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow not found. Please ensure flux_portrait.json exists. Error: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue portrait generation: {e}"
        )


@router.post("/portrait/sync")
async def generate_portrait_sync(request: PortraitRequest):
    """
    Generate a portrait and wait for completion.

    Warning: This endpoint blocks until generation is complete (can take 30-60s).
    For production use, prefer the async /portrait endpoint.
    """
    job_id = str(uuid.uuid4())

    try:
        width, height = ASPECT_RATIOS.get(request.aspect_ratio, (768, 1024))
        full_prompt = build_portrait_prompt(request)

        loader = get_workflow_loader()
        workflow = loader.get_portrait_workflow(
            prompt=full_prompt,
            negative_prompt=request.negative_prompt or "",
            width=width,
            height=height,
            seed=request.seed or -1,
            num_images=request.num_images
        )

        client = get_comfyui_client()
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="flux_portrait"
        )

        # Wait for completion
        completed_job = await client.wait_for_job(job_id, timeout=120.0)

        if completed_job.status == "failed":
            raise HTTPException(status_code=500, detail=completed_job.error)

        return {
            "job_id": completed_job.job_id,
            "status": completed_job.status,
            "files": [
                {
                    "index": i,
                    "url": f"/api/result/{completed_job.job_id}?file_index={i}"
                }
                for i in range(len(completed_job.result_files))
            ]
        }

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
