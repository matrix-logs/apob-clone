"""
Workflow Loader Service

Loads and parameterizes ComfyUI workflow JSON files.
"""
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class WorkflowLoader:
    """
    Loads ComfyUI workflow templates and injects parameters.

    Workflows are JSON files with placeholder values that get replaced
    with actual parameters at runtime.
    """

    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self._cache: Dict[str, Dict] = {}

    def _load_workflow(self, name: str) -> Dict[str, Any]:
        """Load a workflow from file, with caching"""
        if name in self._cache:
            return copy.deepcopy(self._cache[name])

        workflow_path = self.workflows_dir / f"{name}.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_path}")

        with open(workflow_path, "r") as f:
            workflow = json.load(f)

        self._cache[name] = workflow
        logger.info(f"Loaded workflow: {name}")

        return copy.deepcopy(workflow)

    def clear_cache(self):
        """Clear the workflow cache"""
        self._cache.clear()

    def _set_node_input(
        self,
        workflow: Dict,
        node_id: str,
        input_name: str,
        value: Any
    ):
        """Set an input value on a specific node"""
        if node_id in workflow:
            if "inputs" in workflow[node_id]:
                workflow[node_id]["inputs"][input_name] = value

    def _find_node_by_class(
        self,
        workflow: Dict,
        class_type: str
    ) -> Optional[str]:
        """Find the first node of a given class type"""
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                return node_id
        return None

    def _find_nodes_by_class(
        self,
        workflow: Dict,
        class_type: str
    ) -> list:
        """Find all nodes of a given class type"""
        nodes = []
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                nodes.append(node_id)
        return nodes

    # ============================================
    # Portrait Generation Workflow
    # ============================================
    def get_portrait_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 1024,
        seed: int = -1,
        steps: int = 20,
        cfg: float = 3.5,
        num_images: int = 1
    ) -> Dict[str, Any]:
        """
        Load and configure the Flux portrait generation workflow

        Args:
            prompt: Positive prompt describing the portrait
            negative_prompt: Things to avoid
            width: Output image width
            height: Output image height
            seed: Random seed (-1 for random)
            steps: Number of inference steps
            cfg: Classifier-free guidance scale
            num_images: Batch size
        """
        workflow = self._load_workflow("flux_portrait")

        # Find key nodes and set parameters
        # CLIP Text Encode (positive)
        clip_positive = self._find_node_by_class(workflow, "CLIPTextEncode")
        if clip_positive:
            self._set_node_input(workflow, clip_positive, "text", prompt)

        # KSampler
        sampler = self._find_node_by_class(workflow, "KSampler")
        if sampler:
            self._set_node_input(workflow, sampler, "seed", seed if seed >= 0 else -1)
            self._set_node_input(workflow, sampler, "steps", steps)
            self._set_node_input(workflow, sampler, "cfg", cfg)

        # Empty Latent Image (for dimensions)
        latent = self._find_node_by_class(workflow, "EmptyLatentImage")
        if latent:
            self._set_node_input(workflow, latent, "width", width)
            self._set_node_input(workflow, latent, "height", height)
            self._set_node_input(workflow, latent, "batch_size", num_images)

        return workflow

    # ============================================
    # Consistent Character Workflow (PuLID)
    # ============================================
    def get_character_workflow(
        self,
        prompt: str,
        reference_image_name: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 1024,
        seed: int = -1,
        identity_strength: float = 0.8,
        pose_image_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load and configure the PuLID consistent character workflow

        Args:
            prompt: Scene/pose description
            reference_image_name: Uploaded reference image filename
            negative_prompt: Things to avoid
            width: Output width
            height: Output height
            seed: Random seed
            identity_strength: How much to preserve identity (0-1)
            pose_image_name: Optional pose reference image
        """
        workflow = self._load_workflow("pulid_character")

        # Set prompt
        clip_positive = self._find_node_by_class(workflow, "CLIPTextEncode")
        if clip_positive:
            self._set_node_input(workflow, clip_positive, "text", prompt)

        # Set reference image
        load_image = self._find_node_by_class(workflow, "LoadImage")
        if load_image:
            self._set_node_input(workflow, load_image, "image", reference_image_name)

        # Set PuLID strength
        pulid_node = self._find_node_by_class(workflow, "ApplyPulidFlux")
        if pulid_node:
            self._set_node_input(workflow, pulid_node, "weight", identity_strength)

        # Set dimensions
        latent = self._find_node_by_class(workflow, "EmptyLatentImage")
        if latent:
            self._set_node_input(workflow, latent, "width", width)
            self._set_node_input(workflow, latent, "height", height)

        # Set seed
        sampler = self._find_node_by_class(workflow, "KSampler")
        if sampler:
            self._set_node_input(workflow, sampler, "seed", seed if seed >= 0 else -1)

        return workflow

    # ============================================
    # Image to Video Workflow (Wan 2.2)
    # ============================================
    def get_video_workflow(
        self,
        image_name: str,
        prompt: str = "",
        duration: float = 4.0,
        fps: int = 24,
        motion_strength: float = 0.5,
        seed: int = -1
    ) -> Dict[str, Any]:
        """
        Load and configure the Wan 2.2 image-to-video workflow

        Args:
            image_name: Uploaded source image filename
            prompt: Motion/action description
            duration: Video duration in seconds
            fps: Frames per second
            motion_strength: Amount of motion (0-1)
            seed: Random seed
        """
        workflow = self._load_workflow("wan22_video")

        # Calculate frame count
        num_frames = int(duration * fps)

        # Set source image
        load_image = self._find_node_by_class(workflow, "LoadImage")
        if load_image:
            self._set_node_input(workflow, load_image, "image", image_name)

        # Set prompt if Wan supports it
        text_node = self._find_node_by_class(workflow, "CLIPTextEncode")
        if text_node and prompt:
            self._set_node_input(workflow, text_node, "text", prompt)

        # Set frame count and other params
        wan_node = self._find_node_by_class(workflow, "WanI2VModel")
        if wan_node:
            self._set_node_input(workflow, wan_node, "num_frames", num_frames)
            self._set_node_input(workflow, wan_node, "fps", fps)

        # Set sampler seed
        sampler = self._find_node_by_class(workflow, "KSampler")
        if sampler:
            self._set_node_input(workflow, sampler, "seed", seed if seed >= 0 else -1)

        return workflow

    # ============================================
    # Talking Avatar Workflow (SadTalker/LivePortrait)
    # ============================================
    def get_talking_workflow(
        self,
        image_name: str,
        audio_name: str,
        expression_scale: float = 1.0,
        blink: bool = True,
        head_motion: bool = True,
        fps: int = 25
    ) -> Dict[str, Any]:
        """
        Load and configure the talking avatar workflow

        Args:
            image_name: Uploaded face image filename
            audio_name: Uploaded audio filename
            expression_scale: Expression intensity
            blink: Enable blinking
            head_motion: Enable head motion
            fps: Output video FPS
        """
        workflow = self._load_workflow("sadtalker_lipsync")

        # Set source image
        load_image = self._find_node_by_class(workflow, "LoadImage")
        if load_image:
            self._set_node_input(workflow, load_image, "image", image_name)

        # Set audio
        load_audio = self._find_node_by_class(workflow, "LoadAudio")
        if load_audio:
            self._set_node_input(workflow, load_audio, "audio", audio_name)

        # Set SadTalker/LivePortrait params
        sadtalker = self._find_node_by_class(workflow, "SadTalker")
        if sadtalker:
            self._set_node_input(workflow, sadtalker, "expression_scale", expression_scale)
            self._set_node_input(workflow, sadtalker, "still_mode", not head_motion)

        return workflow

    # ============================================
    # Face Swap Workflow (ReActor)
    # ============================================
    def get_faceswap_workflow(
        self,
        source_image_name: str,
        target_image_name: str,
        enhance_face: bool = True,
        enhancer: str = "codeformer",
        detection_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Load and configure the face swap workflow

        Args:
            source_image_name: Face to use
            target_image_name: Image to swap into
            enhance_face: Apply face enhancement
            enhancer: Which enhancer to use
            detection_threshold: Face detection confidence
        """
        workflow = self._load_workflow("reactor_faceswap")

        # Find all LoadImage nodes
        load_nodes = self._find_nodes_by_class(workflow, "LoadImage")

        # Set source and target images
        # Convention: first LoadImage is source, second is target
        if len(load_nodes) >= 2:
            self._set_node_input(workflow, load_nodes[0], "image", source_image_name)
            self._set_node_input(workflow, load_nodes[1], "image", target_image_name)

        # Set ReActor params
        reactor = self._find_node_by_class(workflow, "ReActorFaceSwap")
        if reactor:
            self._set_node_input(workflow, reactor, "detect_gender_source", "no")
            self._set_node_input(workflow, reactor, "detect_gender_input", "no")
            self._set_node_input(workflow, reactor, "face_restore_model", enhancer if enhance_face else "none")

        return workflow


# Global instance
_loader: Optional[WorkflowLoader] = None


def get_workflow_loader(workflows_dir: str = "workflows") -> WorkflowLoader:
    """Get or create the global workflow loader"""
    global _loader
    if _loader is None:
        _loader = WorkflowLoader(workflows_dir)
    return _loader
