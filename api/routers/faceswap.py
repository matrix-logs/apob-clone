"""
Face Swap Router

Handles face swapping using ReActor/Roop.
"""
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Literal

from models import FaceSwapRequest, FaceSwapResponse
from services import get_comfyui_client, get_workflow_loader

router = APIRouter()


@router.post("/faceswap", response_model=FaceSwapResponse)
async def swap_faces(request: FaceSwapRequest):
    """
    Swap faces between images or in video.

    Takes a source face and applies it to a target image/video.
    Optional face enhancement with CodeFormer or GFPGAN.

    Args:
        request: FaceSwapRequest with source face, target, and options.

    Returns:
        Job ID for tracking the face swap operation.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload source face image
        if request.source_image.startswith("data:") or len(request.source_image) > 500:
            src_result = await client.upload_base64_image(
                request.source_image,
                filename=f"src_face_{job_id[:8]}.png"
            )
            source_name = src_result["name"]
        else:
            source_name = request.source_image

        # Upload target image
        if request.target_image.startswith("data:") or len(request.target_image) > 500:
            tgt_result = await client.upload_base64_image(
                request.target_image,
                filename=f"target_{job_id[:8]}.png"
            )
            target_name = tgt_result["name"]
        else:
            target_name = request.target_image

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_faceswap_workflow(
            source_image_name=source_name,
            target_image_name=target_name,
            enhance_face=request.enhance_face,
            enhancer=request.enhancer,
            detection_threshold=request.detection_threshold
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="reactor_faceswap"
        )

        return FaceSwapResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Face swap queued. Enhancer: {request.enhancer if request.enhance_face else 'none'}"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow not found: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue face swap: {e}"
        )


@router.post("/faceswap/upload")
async def swap_faces_upload(
    source_face: UploadFile = File(..., description="Face to use"),
    target: UploadFile = File(..., description="Image to swap face into"),
    enhance_face: bool = Form(default=True),
    enhancer: Literal["codeformer", "gfpgan"] = Form(default="codeformer"),
    detection_threshold: float = Form(default=0.5, ge=0.0, le=1.0)
):
    """
    Face swap with file uploads.

    Upload source and target images directly.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload source face
        src_data = await source_face.read()
        src_filename = f"src_face_{job_id[:8]}.png"
        src_result = await client.upload_image(src_data, src_filename)
        source_name = src_result["name"]

        # Upload target
        tgt_data = await target.read()
        tgt_filename = f"target_{job_id[:8]}.png"
        tgt_result = await client.upload_image(tgt_data, tgt_filename)
        target_name = tgt_result["name"]

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_faceswap_workflow(
            source_image_name=source_name,
            target_image_name=target_name,
            enhance_face=enhance_face,
            enhancer=enhancer,
            detection_threshold=detection_threshold
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="reactor_faceswap"
        )

        return FaceSwapResponse(
            job_id=job.job_id,
            status=job.status,
            message="Face swap queued with uploaded images."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue face swap: {e}"
        )


@router.post("/faceswap/batch")
async def swap_faces_batch(
    source_face: UploadFile = File(..., description="Face to use"),
    targets: list[UploadFile] = File(..., description="Multiple target images"),
    enhance_face: bool = Form(default=True),
    enhancer: Literal["codeformer", "gfpgan"] = Form(default="codeformer")
):
    """
    Batch face swap - swap one face into multiple target images.

    Returns multiple job IDs, one per target image.
    """
    client = get_comfyui_client()

    # Upload source face once
    src_data = await source_face.read()
    src_id = str(uuid.uuid4())[:8]
    src_filename = f"src_face_{src_id}.png"
    src_result = await client.upload_image(src_data, src_filename)
    source_name = src_result["name"]

    jobs = []
    loader = get_workflow_loader()

    for target in targets:
        job_id = str(uuid.uuid4())

        try:
            # Upload target
            tgt_data = await target.read()
            tgt_filename = f"target_{job_id[:8]}.png"
            tgt_result = await client.upload_image(tgt_data, tgt_filename)
            target_name = tgt_result["name"]

            # Get workflow
            workflow = loader.get_faceswap_workflow(
                source_image_name=source_name,
                target_image_name=target_name,
                enhance_face=enhance_face,
                enhancer=enhancer
            )

            # Queue
            job = await client.queue_prompt(
                workflow=workflow,
                job_id=job_id,
                workflow_name="reactor_faceswap"
            )

            jobs.append({
                "job_id": job.job_id,
                "status": job.status,
                "target_filename": target.filename
            })

        except Exception as e:
            jobs.append({
                "job_id": job_id,
                "status": "failed",
                "target_filename": target.filename,
                "error": str(e)
            })

    return {
        "source_used": source_name,
        "jobs": jobs,
        "total": len(jobs),
        "queued": sum(1 for j in jobs if j["status"] != "failed")
    }


@router.get("/faceswap/enhancers")
async def get_enhancers():
    """Get available face enhancement options"""
    return {
        "enhancers": [
            {
                "id": "codeformer",
                "name": "CodeFormer",
                "description": "Best quality, preserves identity well. Recommended for most cases.",
                "speed": "medium"
            },
            {
                "id": "gfpgan",
                "name": "GFPGAN v1.4",
                "description": "Good quality, slightly faster than CodeFormer.",
                "speed": "fast"
            },
            {
                "id": "none",
                "name": "No Enhancement",
                "description": "Raw face swap without post-processing.",
                "speed": "fastest"
            }
        ],
        "default": "codeformer"
    }
