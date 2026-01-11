"""
ComfyUI API Client Service

Handles all communication with ComfyUI's WebSocket and REST APIs.
"""
import json
import uuid
import asyncio
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import websockets
from loguru import logger


@dataclass
class ComfyJob:
    """Represents a job submitted to ComfyUI"""
    job_id: str
    prompt_id: Optional[str] = None
    status: str = "pending"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result_files: list = field(default_factory=list)
    error: Optional[str] = None
    workflow_name: str = ""


class ComfyUIClient:
    """
    Async client for ComfyUI API

    Handles:
    - Workflow submission via REST API
    - Real-time progress via WebSocket
    - File uploads/downloads
    - Queue management
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        client_id: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = client_id or str(uuid.uuid4())

        # Job tracking
        self.jobs: Dict[str, ComfyJob] = {}

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # WebSocket connection
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_callbacks: Dict[str, Callable] = {}

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0
            )
        return self._http_client

    async def close(self):
        """Close all connections"""
        if self._http_client:
            await self._http_client.aclose()
        if self._ws:
            await self._ws.close()
        if self._ws_task:
            self._ws_task.cancel()

    # ============================================
    # Health & Status
    # ============================================
    async def health_check(self) -> bool:
        """Check if ComfyUI is running and accessible"""
        try:
            client = await self.get_http_client()
            response = await client.get("/system_stats")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ComfyUI health check failed: {e}")
            return False

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get ComfyUI system statistics"""
        client = await self.get_http_client()
        response = await client.get("/system_stats")
        response.raise_for_status()
        return response.json()

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        client = await self.get_http_client()
        response = await client.get("/queue")
        response.raise_for_status()
        return response.json()

    # ============================================
    # Workflow Execution
    # ============================================
    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        job_id: Optional[str] = None,
        workflow_name: str = ""
    ) -> ComfyJob:
        """
        Queue a workflow for execution

        Args:
            workflow: The ComfyUI workflow JSON
            job_id: Optional custom job ID
            workflow_name: Name of the workflow for logging

        Returns:
            ComfyJob with tracking information
        """
        job_id = job_id or str(uuid.uuid4())

        # Create job tracking object
        job = ComfyJob(
            job_id=job_id,
            workflow_name=workflow_name
        )
        self.jobs[job_id] = job

        # Prepare request payload
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            client = await self.get_http_client()
            response = await client.post("/prompt", json=payload)
            response.raise_for_status()

            result = response.json()
            job.prompt_id = result.get("prompt_id")
            job.status = "queued"
            job.updated_at = datetime.utcnow()

            logger.info(f"Queued workflow '{workflow_name}' with prompt_id: {job.prompt_id}")
            return job

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Failed to queue workflow: {e}")
            raise

    async def get_prompt_status(self, prompt_id: str) -> Dict[str, Any]:
        """Get status of a specific prompt"""
        client = await self.get_http_client()
        response = await client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    async def cancel_prompt(self, prompt_id: str) -> bool:
        """Cancel a queued or running prompt"""
        try:
            client = await self.get_http_client()
            response = await client.post(
                "/queue",
                json={"delete": [prompt_id]}
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to cancel prompt: {e}")
            return False

    # ============================================
    # WebSocket Progress Tracking
    # ============================================
    async def connect_websocket(self):
        """Establish WebSocket connection for real-time updates"""
        if self._ws and self._ws.open:
            return

        url = f"{self.ws_url}?clientId={self.client_id}"
        self._ws = await websockets.connect(url)
        self._ws_task = asyncio.create_task(self._ws_listener())
        logger.info("WebSocket connected to ComfyUI")

    async def _ws_listener(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self._ws:
                await self._handle_ws_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _handle_ws_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            msg_data = data.get("data", {})

            if msg_type == "progress":
                # Update job progress
                prompt_id = msg_data.get("prompt_id")
                value = msg_data.get("value", 0)
                max_val = msg_data.get("max", 100)
                progress = (value / max_val) * 100 if max_val > 0 else 0

                for job in self.jobs.values():
                    if job.prompt_id == prompt_id:
                        job.progress = progress
                        job.status = "processing"
                        job.updated_at = datetime.utcnow()
                        break

            elif msg_type == "executing":
                prompt_id = msg_data.get("prompt_id")
                node = msg_data.get("node")

                if node is None:
                    # Execution completed
                    for job in self.jobs.values():
                        if job.prompt_id == prompt_id:
                            job.status = "completed"
                            job.progress = 100.0
                            job.updated_at = datetime.utcnow()
                            # Fetch results
                            await self._fetch_job_results(job)
                            break

            elif msg_type == "execution_error":
                prompt_id = msg_data.get("prompt_id")
                for job in self.jobs.values():
                    if job.prompt_id == prompt_id:
                        job.status = "failed"
                        job.error = msg_data.get("exception_message", "Unknown error")
                        job.updated_at = datetime.utcnow()
                        break

            # Call registered callbacks
            for callback in self._ws_callbacks.values():
                await callback(msg_type, msg_data)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from WebSocket: {message[:100]}")

    def register_callback(self, name: str, callback: Callable):
        """Register a callback for WebSocket messages"""
        self._ws_callbacks[name] = callback

    def unregister_callback(self, name: str):
        """Unregister a callback"""
        self._ws_callbacks.pop(name, None)

    # ============================================
    # File Operations
    # ============================================
    async def upload_image(
        self,
        image_data: bytes,
        filename: str,
        subfolder: str = "",
        overwrite: bool = True
    ) -> Dict[str, str]:
        """
        Upload an image to ComfyUI

        Returns:
            Dict with 'name', 'subfolder', 'type' keys
        """
        client = await self.get_http_client()

        files = {
            "image": (filename, image_data, "image/png")
        }
        data = {
            "overwrite": str(overwrite).lower()
        }
        if subfolder:
            data["subfolder"] = subfolder

        response = await client.post("/upload/image", files=files, data=data)
        response.raise_for_status()
        return response.json()

    async def upload_base64_image(
        self,
        base64_data: str,
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Upload a base64-encoded image"""
        # Remove data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_data)
        filename = filename or f"upload_{uuid.uuid4().hex[:8]}.png"

        return await self.upload_image(image_bytes, filename)

    async def get_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output"
    ) -> bytes:
        """Download an image from ComfyUI"""
        client = await self.get_http_client()

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }

        response = await client.get("/view", params=params)
        response.raise_for_status()
        return response.content

    async def _fetch_job_results(self, job: ComfyJob):
        """Fetch output files for a completed job"""
        if not job.prompt_id:
            return

        try:
            history = await self.get_prompt_status(job.prompt_id)

            if job.prompt_id in history:
                outputs = history[job.prompt_id].get("outputs", {})

                for node_id, node_output in outputs.items():
                    # Handle images
                    if "images" in node_output:
                        for img in node_output["images"]:
                            job.result_files.append({
                                "type": "image",
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder", ""),
                                "folder_type": img.get("type", "output")
                            })
                    # Handle videos/gifs
                    if "gifs" in node_output:
                        for vid in node_output["gifs"]:
                            job.result_files.append({
                                "type": "video",
                                "filename": vid["filename"],
                                "subfolder": vid.get("subfolder", ""),
                                "folder_type": vid.get("type", "output")
                            })

        except Exception as e:
            logger.error(f"Failed to fetch job results: {e}")

    # ============================================
    # Job Management
    # ============================================
    def get_job(self, job_id: str) -> Optional[ComfyJob]:
        """Get a job by ID"""
        return self.jobs.get(job_id)

    async def wait_for_job(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0
    ) -> ComfyJob:
        """
        Wait for a job to complete

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status

        Returns:
            Completed ComfyJob
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        start_time = asyncio.get_event_loop().time()

        while True:
            if job.status in ("completed", "failed"):
                return job

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                job.status = "failed"
                job.error = "Timeout waiting for completion"
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            await asyncio.sleep(poll_interval)

    def clear_completed_jobs(self, max_age_seconds: int = 3600):
        """Remove completed jobs older than max_age_seconds"""
        now = datetime.utcnow()
        to_remove = []

        for job_id, job in self.jobs.items():
            if job.status in ("completed", "failed"):
                age = (now - job.updated_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]

        return len(to_remove)


# Global client instance
_client: Optional[ComfyUIClient] = None


def get_comfyui_client() -> ComfyUIClient:
    """Get or create the global ComfyUI client"""
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client


async def init_comfyui_client(host: str = "127.0.0.1", port: int = 8188):
    """Initialize the global ComfyUI client with WebSocket connection"""
    global _client
    _client = ComfyUIClient(host=host, port=port)
    await _client.connect_websocket()
    return _client
