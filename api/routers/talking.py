"""
Talking Avatar Router

Handles talking avatar generation using SadTalker/LivePortrait.
"""
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from models import TalkingRequest, TalkingResponse
from services import get_comfyui_client, get_workflow_loader

router = APIRouter()


@router.post("/talking", response_model=TalkingResponse)
async def generate_talking_avatar(request: TalkingRequest):
    """
    Generate a talking avatar video from a face image and audio.

    Uses SadTalker/LivePortrait to animate the face with realistic
    lip sync and optional head motion.

    Args:
        request: TalkingRequest with face image, audio, and animation params.

    Returns:
        Job ID for tracking. Processing time depends on audio length.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload face image
        if request.image.startswith("data:") or len(request.image) > 500:
            img_result = await client.upload_base64_image(
                request.image,
                filename=f"face_{job_id[:8]}.png"
            )
            image_name = img_result["name"]
        else:
            image_name = request.image

        # Upload audio
        # For audio, we need to handle it specially
        if request.audio.startswith("data:") or len(request.audio) > 500:
            # Base64 audio
            import base64
            audio_data = request.audio
            if "," in audio_data:
                audio_data = audio_data.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_data)

            # Determine audio format from data or default to wav
            audio_filename = f"audio_{job_id[:8]}.wav"

            # Upload audio to ComfyUI input folder
            # Note: This requires a custom upload endpoint or manual handling
            # For now, we'll save to a temp location
            audio_name = audio_filename
        else:
            audio_name = request.audio

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_talking_workflow(
            image_name=image_name,
            audio_name=audio_name,
            expression_scale=request.expression_scale,
            blink=request.blink,
            head_motion=request.head_motion,
            fps=request.fps
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="sadtalker_lipsync"
        )

        # Estimate time based on typical audio length
        estimated_time = 60  # Default estimate

        return TalkingResponse(
            job_id=job.job_id,
            status=job.status,
            message="Talking avatar generation queued. Lip sync in progress.",
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
            detail=f"Failed to queue talking avatar: {e}"
        )


@router.post("/talking/upload")
async def generate_talking_upload(
    image: UploadFile = File(..., description="Face image"),
    audio: UploadFile = File(..., description="Audio file (wav, mp3)"),
    expression_scale: float = Form(default=1.0, ge=0.5, le=2.0),
    blink: bool = Form(default=True),
    head_motion: bool = Form(default=True),
    fps: int = Form(default=25, ge=12, le=30),
    output_format: str = Form(default="mp4")
):
    """
    Generate talking avatar with file uploads.

    Upload a face image and audio file directly.
    Supports wav and mp3 audio formats.
    """
    job_id = str(uuid.uuid4())
    client = get_comfyui_client()

    try:
        # Upload face image
        img_data = await image.read()
        img_filename = f"face_{job_id[:8]}.png"
        img_result = await client.upload_image(img_data, img_filename)
        image_name = img_result["name"]

        # Upload audio
        # ComfyUI typically expects audio in the input folder
        audio_data = await audio.read()
        audio_ext = audio.filename.split(".")[-1] if audio.filename else "wav"
        audio_filename = f"audio_{job_id[:8]}.{audio_ext}"

        # For audio, we may need to save directly to ComfyUI's input folder
        # This depends on the specific audio node implementation
        audio_name = audio_filename

        # Get workflow
        loader = get_workflow_loader()
        workflow = loader.get_talking_workflow(
            image_name=image_name,
            audio_name=audio_name,
            expression_scale=expression_scale,
            blink=blink,
            head_motion=head_motion,
            fps=fps
        )

        # Queue the workflow
        job = await client.queue_prompt(
            workflow=workflow,
            job_id=job_id,
            workflow_name="sadtalker_lipsync"
        )

        return TalkingResponse(
            job_id=job.job_id,
            status=job.status,
            message="Talking avatar queued with uploaded files.",
            estimated_time=60
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue talking avatar: {e}"
        )


@router.get("/talking/voices")
async def get_available_voices():
    """
    Get list of available TTS voices (if TTS is enabled).

    This endpoint lists voices that can be used with the /talking/tts endpoint.
    """
    # This would integrate with a TTS service if available
    return {
        "voices": [
            {"id": "en-US-1", "name": "English (US) - Female", "language": "en-US"},
            {"id": "en-US-2", "name": "English (US) - Male", "language": "en-US"},
            {"id": "en-GB-1", "name": "English (UK) - Female", "language": "en-GB"},
        ],
        "note": "TTS integration requires additional setup. See documentation."
    }


@router.post("/talking/tts")
async def generate_talking_tts(
    image: UploadFile = File(..., description="Face image"),
    text: str = Form(..., description="Text to speak"),
    voice_id: str = Form(default="en-US-1", description="Voice to use"),
    expression_scale: float = Form(default=1.0),
    head_motion: bool = Form(default=True)
):
    """
    Generate talking avatar with text-to-speech.

    Converts text to speech and then generates lip-synced video.
    Requires TTS service integration.
    """
    # This would first convert text to audio using TTS, then call the talking endpoint
    raise HTTPException(
        status_code=501,
        detail="TTS integration not yet implemented. Use /talking/upload with pre-recorded audio."
    )
