"""
Consistent Character Router

Handles consistent character generation using PuLID-Flux.
"""
import uuid
import base64
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from models import CharacterRequest, CharacterResponse, AspectRatio
from services import get_comfyui_client, get_workflow_loader

router = APIRouter()


ASPECT_RATIOS = {
    AspectRatio.SQUARE: (1024, 1024),
    AspectRatio.PORTRAIT: (768, 1024),
    AspectRatio.LANDSCAPE: (1024, 768),
    AspectRatio.PORTRAIT_4_3: (768, 1024),
    AspectRatio.LANDSCAPE_4_3: (1024, 768),
}


@router.post("/character", response_model=CharacterResponse)
async def generate_character(request: CharacterRequest):
    """
    Generate consistent character images using PuLID-Flux.

    Requires a reference image to maintain character identity.
    The same face will be preserved across different poses and scenes.

    Args:
        request: CharacterRequest with reference_image (base64 or path),
                 prompt, and generation parameters.

    Returns:
        Job ID for tracking the generation.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload reference image
        if request.reference_image.startswith("data:") or len(request.reference_image) > 500:
            # Base64 image
            upload_result = await client.upload_base64_image(
                request.reference_image,
                filename=f"ref_{job_id[:8]}.png"
            )
            ref_image_name = upload_result["name"]
        else:
            # Assume it's already an uploaded filename
            ref_image_name = request.reference_image

        # Upload pose image if provided
        pose_image_name = None
        if request.pose_image:
            if request.pose_image.startswith("data:") or len(request.pose_image) > 500:
                pose_result = await client.upload_base64_image(
                    request.pose_image,
                    filename=f"pose_{job_id[:8]}.png"
                )
                pose_image_name = pose_result["name"]
            else:
                pose_image_name = request.pose_image

        # Get dimensions
        width, height = ASPECT_RATIOS.get(request.aspect_ratio, (768, 1024))

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_character_workflow(
            prompt=request.prompt,
            reference_image_name=ref_image_name,
            negative_prompt=request.negative_prompt or "",
            width=width,
            height=height,
            seed=request.seed or -1,
            identity_strength=request.identity_strength,
            pose_image_name=pose_image_name
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="pulid_character"
        )

        return CharacterResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Character generation queued. Identity strength: {request.identity_strength}"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow not found: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue character generation: {e}"
        )


@router.post("/character/upload")
async def generate_character_upload(
    reference_image: UploadFile = File(..., description="Reference face image"),
    prompt: str = Form(..., description="Scene/pose description"),
    negative_prompt: str = Form(default="", description="What to avoid"),
    identity_strength: float = Form(default=0.8, ge=0.0, le=1.0),
    aspect_ratio: str = Form(default="9:16"),
    seed: Optional[int] = Form(default=None),
    pose_image: Optional[UploadFile] = File(default=None, description="Optional pose reference")
):
    """
    Generate consistent character with file upload.

    Alternative endpoint that accepts file uploads directly instead of base64.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload reference image
        ref_data = await reference_image.read()
        ref_filename = f"ref_{job_id[:8]}.png"
        upload_result = await client.upload_image(ref_data, ref_filename)
        ref_image_name = upload_result["name"]

        # Upload pose image if provided
        pose_image_name = None
        if pose_image:
            pose_data = await pose_image.read()
            pose_filename = f"pose_{job_id[:8]}.png"
            pose_result = await client.upload_image(pose_data, pose_filename)
            pose_image_name = pose_result["name"]

        # Map aspect ratio string to enum
        ar_map = {
            "1:1": AspectRatio.SQUARE,
            "9:16": AspectRatio.PORTRAIT,
            "16:9": AspectRatio.LANDSCAPE,
            "3:4": AspectRatio.PORTRAIT_4_3,
            "4:3": AspectRatio.LANDSCAPE_4_3,
        }
        ar = ar_map.get(aspect_ratio, AspectRatio.PORTRAIT)
        width, height = ASPECT_RATIOS.get(ar, (768, 1024))

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_character_workflow(
            prompt=prompt,
            reference_image_name=ref_image_name,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed or -1,
            identity_strength=identity_strength,
            pose_image_name=pose_image_name
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="pulid_character"
        )

        return CharacterResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Character generation queued with uploaded reference."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue character generation: {e}"
        )
