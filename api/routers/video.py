"""
Image to Video Router

Handles video generation using Wan 2.2.
"""
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from models import VideoRequest, VideoResponse
from services import get_comfyui_client, get_workflow_loader

router = APIRouter()


@router.post("/video", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """
    Generate a video from a static image using Wan 2.2.

    The image will be animated with natural motion based on the content
    and optional prompt describing desired motion.

    Args:
        request: VideoRequest with source image and video parameters.

    Returns:
        Job ID for tracking. Video generation can take 2-5 minutes.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload source image
        if request.image.startswith("data:") or len(request.image) > 500:
            upload_result = await client.upload_base64_image(
                request.image,
                filename=f"video_src_{job_id[:8]}.png"
            )
            image_name = upload_result["name"]
        else:
            image_name = request.image

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_video_workflow(
            image_name=image_name,
            prompt=request.prompt or "",
            duration=request.duration,
            fps=request.fps,
            motion_strength=request.motion_strength,
            seed=request.seed or -1
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="wan22_video"
        )

        # Estimate processing time based on duration
        estimated_time = int(request.duration * 30)  # ~30s per second of video

        return VideoResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Video generation queued. Duration: {request.duration}s at {request.fps}fps",
            estimated_time=estimated_time
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow not found: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue video generation: {e}"
        )


@router.post("/video/upload")
async def generate_video_upload(
    image: UploadFile = File(..., description="Source image to animate"),
    prompt: str = Form(default="", description="Motion description"),
    duration: float = Form(default=4.0, ge=1.0, le=10.0),
    fps: int = Form(default=24, ge=12, le=30),
    motion_strength: float = Form(default=0.5, ge=0.0, le=1.0),
    seed: Optional[int] = Form(default=None)
):
    """
    Generate video with file upload.

    Alternative endpoint for direct file uploads instead of base64.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload source image
        img_data = await image.read()
        img_filename = f"video_src_{job_id[:8]}.png"
        upload_result = await client.upload_image(img_data, img_filename)
        image_name = upload_result["name"]

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_video_workflow(
            image_name=image_name,
            prompt=prompt,
            duration=duration,
            fps=fps,
            motion_strength=motion_strength,
            seed=seed or -1
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="wan22_video"
        )

        estimated_time = int(duration * 30)

        return VideoResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Video generation queued from uploaded image.",
            estimated_time=estimated_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue video generation: {e}"
        )


@router.get("/video/presets")
async def get_video_presets():
    """Get recommended presets for different video types"""
    return {
        "presets": [
            {
                "name": "portrait_subtle",
                "description": "Subtle motion for portrait photos",
                "duration": 3.0,
                "fps": 24,
                "motion_strength": 0.3
            },
            {
                "name": "portrait_expressive",
                "description": "More animated portrait with expressions",
                "duration": 4.0,
                "fps": 24,
                "motion_strength": 0.6
            },
            {
                "name": "scene_cinematic",
                "description": "Cinematic scene animation",
                "duration": 5.0,
                "fps": 24,
                "motion_strength": 0.5
            },
            {
                "name": "action_dynamic",
                "description": "Dynamic action with strong motion",
                "duration": 4.0,
                "fps": 30,
                "motion_strength": 0.8
            }
        ]
    }
