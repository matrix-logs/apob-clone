"""
Pydantic models for API request/response schemas
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AspectRatio(str, Enum):
    SQUARE = "1:1"
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"
    PORTRAIT_4_3 = "3:4"
    LANDSCAPE_4_3 = "4:3"


# ============================================
# Portrait Generation
# ============================================
class PortraitRequest(BaseModel):
    """Request for AI portrait generation"""
    prompt: str = Field(..., description="Text description of the portrait to generate")
    negative_prompt: Optional[str] = Field(
        default="ugly, blurry, low quality, distorted, deformed",
        description="What to avoid in the generation"
    )
    aspect_ratio: AspectRatio = Field(default=AspectRatio.PORTRAIT, description="Image aspect ratio")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    # Portrait-specific options
    gender: Optional[Literal["male", "female", "neutral"]] = None
    age_range: Optional[Literal["young", "adult", "middle-aged", "elderly"]] = None
    ethnicity: Optional[str] = None
    style: Optional[Literal["photorealistic", "artistic", "anime", "3d"]] = Field(default="photorealistic")


class PortraitResponse(BaseModel):
    """Response from portrait generation"""
    job_id: str
    status: JobStatus
    message: str


# ============================================
# Consistent Character
# ============================================
class CharacterRequest(BaseModel):
    """Request for consistent character generation"""
    reference_image: str = Field(..., description="Base64 encoded reference image or file path")
    prompt: str = Field(..., description="Scene/pose description for the character")
    negative_prompt: Optional[str] = Field(
        default="ugly, blurry, different person, wrong face",
        description="What to avoid"
    )
    aspect_ratio: AspectRatio = Field(default=AspectRatio.PORTRAIT)
    num_images: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = None

    # PuLID specific
    identity_strength: float = Field(default=0.8, ge=0.0, le=1.0, description="How strongly to preserve identity")
    pose_image: Optional[str] = Field(default=None, description="Optional pose reference image")


class CharacterResponse(BaseModel):
    """Response from character generation"""
    job_id: str
    status: JobStatus
    message: str


# ============================================
# Image to Video
# ============================================
class VideoRequest(BaseModel):
    """Request for image-to-video generation"""
    image: str = Field(..., description="Base64 encoded image or file path")
    prompt: Optional[str] = Field(default=None, description="Motion/action description")
    duration: float = Field(default=4.0, ge=1.0, le=10.0, description="Video duration in seconds")
    fps: int = Field(default=24, ge=12, le=30, description="Frames per second")
    motion_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of motion")
    seed: Optional[int] = None


class VideoResponse(BaseModel):
    """Response from video generation"""
    job_id: str
    status: JobStatus
    message: str
    estimated_time: Optional[int] = Field(default=None, description="Estimated processing time in seconds")


# ============================================
# Talking Avatar
# ============================================
class TalkingRequest(BaseModel):
    """Request for talking avatar generation"""
    image: str = Field(..., description="Base64 encoded face image or file path")
    audio: str = Field(..., description="Base64 encoded audio file or file path")

    # Lip sync options
    expression_scale: float = Field(default=1.0, ge=0.5, le=2.0, description="Expression intensity")
    blink: bool = Field(default=True, description="Enable natural blinking")
    head_motion: bool = Field(default=True, description="Enable subtle head motion")

    # Output options
    fps: int = Field(default=25, ge=12, le=30)
    output_format: Literal["mp4", "webm"] = Field(default="mp4")


class TalkingResponse(BaseModel):
    """Response from talking avatar generation"""
    job_id: str
    status: JobStatus
    message: str
    estimated_time: Optional[int] = None


# ============================================
# Face Swap
# ============================================
class FaceSwapRequest(BaseModel):
    """Request for face swap"""
    source_image: str = Field(..., description="Face to use (base64 or file path)")
    target_image: str = Field(..., description="Image/video to swap face into (base64 or file path)")

    # Options
    enhance_face: bool = Field(default=True, description="Apply face enhancement after swap")
    enhancer: Literal["codeformer", "gfpgan"] = Field(default="codeformer")
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # For video targets
    is_video: bool = Field(default=False)


class FaceSwapResponse(BaseModel):
    """Response from face swap"""
    job_id: str
    status: JobStatus
    message: str


# ============================================
# Job Status & Results
# ============================================
class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    created_at: str
    updated_at: str
    result_urls: Optional[List[str]] = None
    error: Optional[str] = None


class ResultResponse(BaseModel):
    """Response containing generated results"""
    job_id: str
    status: JobStatus
    files: List[str] = Field(default_factory=list, description="List of output file URLs")
    metadata: Optional[dict] = None


# ============================================
# Init file markers
# ============================================
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    comfyui_connected: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
