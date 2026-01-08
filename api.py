"""
LTX-2 FastAPI Service
RESTful API for video generation with preset support.

Features:
- Generate videos using presets or custom settings
- Thread-safe generation queue (one generation at a time)
- Preset management (CRUD operations)
- Health check and status endpoints

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    
    # Or with auto-reload for development
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import uuid
import base64
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

from presets import (
    get_preset_manager,
    PresetManager,
    GenerationPreset,
    DEFAULT_PRESET_NAME,
)

# Constants
MODELS_DIR = Path("./models")
OUTPUTS_DIR = Path("./outputs")
TEMPLATES_DIR = Path("./templates")
HF_REPO_ID = "Lightricks/LTX-2"

# Available checkpoints from HuggingFace
CHECKPOINTS = {
    "ltx-2-19b-dev": {
        "filename": "ltx-2-19b-dev.safetensors",
        "size": "43.3 GB",
        "description": "Full precision development model",
        "type": "checkpoint"
    },
    "ltx-2-19b-dev-fp8": {
        "filename": "ltx-2-19b-dev-fp8.safetensors",
        "size": "27.1 GB",
        "description": "FP8 quantized development model (recommended)",
        "type": "checkpoint"
    },
    "ltx-2-19b-dev-fp4": {
        "filename": "ltx-2-19b-dev-fp4.safetensors",
        "size": "20 GB",
        "description": "FP4 quantized development model (smallest)",
        "type": "checkpoint"
    },
    "ltx-2-19b-distilled": {
        "filename": "ltx-2-19b-distilled.safetensors",
        "size": "43.3 GB",
        "description": "Full precision distilled model",
        "type": "checkpoint"
    },
    "ltx-2-19b-distilled-fp8": {
        "filename": "ltx-2-19b-distilled-fp8.safetensors",
        "size": "27.1 GB",
        "description": "FP8 distilled model (fast inference)",
        "type": "checkpoint"
    },
    "ltx-2-19b-distilled-lora-384": {
        "filename": "ltx-2-19b-distilled-lora-384.safetensors",
        "size": "7.67 GB",
        "description": "Distilled LoRA adapter",
        "type": "lora"
    },
    "ltx-2-spatial-upscaler-x2": {
        "filename": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "size": "996 MB",
        "description": "2x spatial upscaler",
        "type": "upscaler"
    },
    "ltx-2-temporal-upscaler-x2": {
        "filename": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
        "size": "262 MB",
        "description": "2x temporal upscaler",
        "type": "upscaler"
    },
}


def ensure_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)
    (MODELS_DIR / "loras").mkdir(exist_ok=True)
    (MODELS_DIR / "upsamplers").mkdir(exist_ok=True)
    (MODELS_DIR / "gemma").mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def get_model_path(model_key: str) -> Optional[Path]:
    """Get local path for a model if it exists."""
    if model_key not in CHECKPOINTS:
        return None
    
    model_info = CHECKPOINTS[model_key]
    model_type = model_info["type"]
    
    if model_type == "checkpoint":
        path = MODELS_DIR / "checkpoints" / model_info["filename"]
    elif model_type == "lora":
        path = MODELS_DIR / "loras" / model_info["filename"]
    elif model_type == "upscaler":
        path = MODELS_DIR / "upsamplers" / model_info["filename"]
    else:
        path = MODELS_DIR / model_info["filename"]
    
    return path if path.exists() else None


def check_model_status(model_key: str) -> tuple:
    """Check if a model is downloaded. Returns (status, path)."""
    path = get_model_path(model_key)
    if path and path.exists():
        return "ready", str(path)
    return "missing", None


def download_model_file(model_key: str) -> str:
    """Download a model from HuggingFace. Returns the path."""
    if model_key not in CHECKPOINTS:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_info = CHECKPOINTS[model_key]
    model_type = model_info["type"]
    filename = model_info["filename"]
    
    # Determine target directory
    if model_type == "checkpoint":
        target_dir = MODELS_DIR / "checkpoints"
    elif model_type == "lora":
        target_dir = MODELS_DIR / "loras"
    elif model_type == "upscaler":
        target_dir = MODELS_DIR / "upsamplers"
    else:
        target_dir = MODELS_DIR
    
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    
    if target_path.exists():
        return str(target_path)
    
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=target_dir,
    )
    
    return downloaded_path


def get_default_checkpoint() -> Optional[str]:
    """Get the default checkpoint path (prefer distilled-fp8)."""
    checkpoint_dir = MODELS_DIR / "checkpoints"
    if checkpoint_dir.exists():
        # Prefer distilled-fp8
        for name in ["ltx-2-19b-distilled-fp8.safetensors", "ltx-2-19b-distilled.safetensors"]:
            path = checkpoint_dir / name
            if path.exists():
                return str(path)
        # Return any available checkpoint
        checkpoints = list(checkpoint_dir.glob("*.safetensors"))
        if checkpoints:
            return str(checkpoints[0])
    return None


def get_default_upsampler() -> Optional[str]:
    """Get the default spatial upsampler path."""
    upsampler_dir = MODELS_DIR / "upsamplers"
    if upsampler_dir.exists():
        upsamplers = list(upsampler_dir.glob("*.safetensors"))
        if upsamplers:
            return str(upsamplers[0])
    return None


def auto_download_required_models(preset) -> tuple:
    """
    Auto-download required models if not present.
    Returns (checkpoint_path, upsampler_path) or raises error.
    """
    checkpoint_path = preset.checkpoint_path
    upsampler_path = preset.spatial_upsampler_path
    
    # Check and download checkpoint
    if not checkpoint_path or not Path(checkpoint_path).exists():
        # Try to find existing checkpoint
        existing = get_default_checkpoint()
        if existing:
            checkpoint_path = existing
        else:
            # Download default checkpoint (distilled-fp8)
            print("Auto-downloading default checkpoint: ltx-2-19b-distilled-fp8")
            checkpoint_path = download_model_file("ltx-2-19b-distilled-fp8")
    
    # Check and download upsampler for pipelines that need it
    if preset.pipeline_type in ["distilled", "ic_lora", "ti2vid_two_stages", "keyframe_interpolation"]:
        if not upsampler_path or not Path(upsampler_path).exists():
            existing = get_default_upsampler()
            if existing:
                upsampler_path = existing
            else:
                # Download spatial upscaler
                print("Auto-downloading spatial upscaler")
                upsampler_path = download_model_file("ltx-2-spatial-upscaler-x2")
    
    return checkpoint_path, upsampler_path


# ============================================================================
# Pydantic Models for API
# ============================================================================

class GenerationStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationRequest(BaseModel):
    """Request model for video generation."""
    preset_name: Optional[str] = Field(
        default=None,
        description="Preset name to use. If not provided, uses default preset."
    )
    
    # Override settings (optional - override preset values)
    prompt: Optional[str] = Field(default=None, description="Override prompt from preset")
    negative_prompt: Optional[str] = Field(default=None, description="Override negative prompt")
    height: Optional[int] = Field(default=None, ge=256, le=2048, description="Override height")
    width: Optional[int] = Field(default=None, ge=256, le=2048, description="Override width")
    num_frames: Optional[int] = Field(default=None, ge=9, le=257, description="Override frame count")
    frame_rate: Optional[float] = Field(default=None, ge=8, le=60, description="Override FPS")
    num_inference_steps: Optional[int] = Field(default=None, ge=4, le=100, description="Override steps")
    cfg_guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=15.0, description="Override CFG")
    seed: Optional[int] = Field(default=None, description="Override seed (-1 for random)")
    
    # Image conditioning (base64 encoded)
    input_image_base64: Optional[str] = Field(default=None, description="Base64 encoded input image")
    image_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Image strength")


class GenerationJob(BaseModel):
    """A generation job with status."""
    job_id: str
    status: GenerationStatus
    preset_name: str
    prompt: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    position_in_queue: Optional[int] = None


class PresetCreate(BaseModel):
    """Model for creating/updating a preset."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    
    # Pipeline settings
    pipeline_type: str = Field(default="distilled")
    checkpoint_path: Optional[str] = None
    distilled_lora_path: str = Field(default="None")
    spatial_upsampler_path: Optional[str] = None
    gemma_path: str = Field(default="./models/gemma")
    
    # Generation parameters
    prompt: str = Field(default="")
    negative_prompt: str = Field(default="")
    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1536, ge=256, le=2048)
    num_frames: int = Field(default=121, ge=9, le=257)
    frame_rate: float = Field(default=24.0, ge=8, le=60)
    num_inference_steps: int = Field(default=40, ge=4, le=100)
    cfg_guidance_scale: float = Field(default=4.0, ge=1.0, le=15.0)
    seed: int = Field(default=-1)
    enable_fp8: bool = Field(default=True)
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0)


class PresetResponse(BaseModel):
    """Response model for preset data."""
    name: str
    description: str
    is_default: bool
    created_at: str
    updated_at: str
    pipeline_type: str
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    cfg_guidance_scale: float
    seed: int
    enable_fp8: bool
    image_strength: float
    # Model paths
    checkpoint_path: Optional[str] = None
    spatial_upsampler_path: Optional[str] = None
    distilled_lora_path: Optional[str] = None
    gemma_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    cuda_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    pipeline_loaded: bool
    queue_size: int
    current_job: Optional[str]


# ============================================================================
# Generation Queue and Worker
# ============================================================================

@dataclass
class QueuedJob:
    """Internal job representation."""
    job_id: str
    request: GenerationRequest
    preset: GenerationPreset
    status: GenerationStatus = GenerationStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    input_image_path: Optional[str] = None


class GenerationQueue:
    """
    Thread-safe generation queue that processes one job at a time.
    New requests wait in queue until the current generation completes.
    """
    
    def __init__(self):
        self._queue: Queue[QueuedJob] = Queue()
        self._jobs: Dict[str, QueuedJob] = {}
        self._current_job: Optional[str] = None
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._pipeline_cache = {
            "pipeline": None,
            "pipeline_type": None,
            "checkpoint_path": None,
            "spatial_upsampler_path": None,
            "gemma_path": None,
            "distilled_lora_path": None,
            "enable_fp8": None,
        }
    
    def start(self):
        """Start the worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def add_job(self, request: GenerationRequest, preset: GenerationPreset, input_image_path: Optional[str] = None) -> str:
        """Add a job to the queue. Returns job ID."""
        job_id = str(uuid.uuid4())
        job = QueuedJob(
            job_id=job_id,
            request=request,
            preset=preset,
            input_image_path=input_image_path,
        )
        
        with self._lock:
            self._jobs[job_id] = job
            self._queue.put(job)
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[QueuedJob]:
        """Get job status."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get position in queue (1-indexed, None if not queued)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != GenerationStatus.QUEUED:
                return None
            
            # Count jobs ahead in queue
            position = 1
            for jid, j in self._jobs.items():
                if j.status == GenerationStatus.QUEUED and j.created_at < job.created_at:
                    position += 1
            return position
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def get_current_job_id(self) -> Optional[str]:
        """Get the currently processing job ID."""
        return self._current_job
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Cannot cancel processing jobs."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == GenerationStatus.QUEUED:
                job.status = GenerationStatus.CANCELLED
                job.completed_at = datetime.now().isoformat()
                return True
        return False
    
    def _worker_loop(self):
        """Main worker loop - processes jobs one at a time."""
        while self._running:
            try:
                # Block waiting for a job (with timeout to allow checking _running)
                try:
                    job = self._queue.get(timeout=1.0)
                except:
                    continue
                
                # Skip cancelled jobs
                if job.status == GenerationStatus.CANCELLED:
                    self._queue.task_done()
                    continue
                
                # Process the job
                with self._lock:
                    self._current_job = job.job_id
                    job.status = GenerationStatus.PROCESSING
                    job.started_at = datetime.now().isoformat()
                
                try:
                    output_path = self._generate_video(job)
                    
                    with self._lock:
                        job.status = GenerationStatus.COMPLETED
                        job.output_path = output_path
                        job.completed_at = datetime.now().isoformat()
                        
                except Exception as e:
                    import traceback
                    with self._lock:
                        job.status = GenerationStatus.FAILED
                        job.error_message = f"{str(e)}\n{traceback.format_exc()}"
                        job.completed_at = datetime.now().isoformat()
                
                finally:
                    with self._lock:
                        self._current_job = None
                    self._queue.task_done()
                    
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _get_cached_pipeline(self, preset: GenerationPreset):
        """Get or create cached pipeline."""
        cache = self._pipeline_cache
        
        cache_valid = (
            cache["pipeline"] is not None
            and cache["pipeline_type"] == preset.pipeline_type
            and cache["checkpoint_path"] == preset.checkpoint_path
            and cache["spatial_upsampler_path"] == preset.spatial_upsampler_path
            and cache["gemma_path"] == preset.gemma_path
            and cache["distilled_lora_path"] == preset.distilled_lora_path
            and cache["enable_fp8"] == preset.enable_fp8
        )
        
        if cache_valid:
            return cache["pipeline"], None
        
        # Clear old pipeline
        if cache["pipeline"] is not None:
            del cache["pipeline"]
            cache["pipeline"] = None
            torch.cuda.empty_cache()
        
        try:
            if preset.pipeline_type == "distilled":
                from ltx_pipelines.distilled import DistilledPipeline
                
                if not preset.spatial_upsampler_path or not Path(preset.spatial_upsampler_path).exists():
                    return None, "Spatial upsampler is required for distilled pipeline."
                
                pipeline = DistilledPipeline(
                    checkpoint_path=preset.checkpoint_path,
                    spatial_upsampler_path=preset.spatial_upsampler_path,
                    gemma_root=preset.gemma_path,
                    loras=[],
                    fp8transformer=preset.enable_fp8,
                )
                
            elif preset.pipeline_type == "ti2vid_two_stages":
                from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
                from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
                
                if not preset.distilled_lora_path or preset.distilled_lora_path == "None":
                    return None, "Two-Stage Pipeline requires a distilled LoRA."
                
                distilled_lora_list = [
                    LoraPathStrengthAndSDOps(preset.distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
                ]
                
                pipeline = TI2VidTwoStagesPipeline(
                    checkpoint_path=preset.checkpoint_path,
                    distilled_lora=distilled_lora_list,
                    spatial_upsampler_path=preset.spatial_upsampler_path if preset.spatial_upsampler_path else None,
                    gemma_root=preset.gemma_path,
                    loras=[],
                    fp8transformer=preset.enable_fp8,
                )
                
            elif preset.pipeline_type == "ti2vid_one_stage":
                from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
                
                pipeline = TI2VidOneStagePipeline(
                    checkpoint_path=preset.checkpoint_path,
                    gemma_root=preset.gemma_path,
                    loras=[],
                    fp8transformer=preset.enable_fp8,
                )
                
            elif preset.pipeline_type == "ic_lora":
                from ltx_pipelines.ic_lora import ICLoraPipeline
                
                if not preset.spatial_upsampler_path or not Path(preset.spatial_upsampler_path).exists():
                    return None, "Spatial upsampler is required for IC-LoRA pipeline."
                
                loras = []
                if preset.distilled_lora_path and preset.distilled_lora_path != "None" and Path(preset.distilled_lora_path).exists():
                    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
                    loras = [LoraPathStrengthAndSDOps(preset.distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
                
                pipeline = ICLoraPipeline(
                    checkpoint_path=preset.checkpoint_path,
                    spatial_upsampler_path=preset.spatial_upsampler_path,
                    gemma_root=preset.gemma_path,
                    loras=loras,
                    fp8transformer=preset.enable_fp8,
                )
                
            elif preset.pipeline_type == "keyframe_interpolation":
                from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
                from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
                
                if not preset.distilled_lora_path or preset.distilled_lora_path == "None":
                    return None, "Keyframe Interpolation Pipeline requires a distilled LoRA."
                
                distilled_lora_list = [
                    LoraPathStrengthAndSDOps(preset.distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
                ]
                
                pipeline = KeyframeInterpolationPipeline(
                    checkpoint_path=preset.checkpoint_path,
                    distilled_lora=distilled_lora_list,
                    spatial_upsampler_path=preset.spatial_upsampler_path if preset.spatial_upsampler_path else None,
                    gemma_root=preset.gemma_path,
                    loras=[],
                    fp8transformer=preset.enable_fp8,
                )
                
            else:
                return None, f"Unknown pipeline type: {preset.pipeline_type}"
            
            # Cache the pipeline
            cache["pipeline"] = pipeline
            cache["pipeline_type"] = preset.pipeline_type
            cache["checkpoint_path"] = preset.checkpoint_path
            cache["spatial_upsampler_path"] = preset.spatial_upsampler_path
            cache["gemma_path"] = preset.gemma_path
            cache["distilled_lora_path"] = preset.distilled_lora_path
            cache["enable_fp8"] = preset.enable_fp8
            
            return pipeline, None
            
        except ImportError as e:
            return None, f"LTX Pipelines not installed: {e}"
        except Exception as e:
            import traceback
            return None, f"Failed to load pipeline: {e}\n{traceback.format_exc()}"
    
    def _generate_video(self, job: QueuedJob) -> str:
        """Generate video for a job. Returns output path."""
        import random
        from PIL import Image
        
        preset = job.preset
        request = job.request
        
        # Auto-download required models if not present
        try:
            checkpoint_path, upsampler_path = auto_download_required_models(preset)
            preset.checkpoint_path = checkpoint_path
            preset.spatial_upsampler_path = upsampler_path
        except Exception as e:
            raise RuntimeError(f"Failed to download required models: {e}")
        
        # Merge request overrides with preset
        prompt = request.prompt if request.prompt else preset.prompt
        negative_prompt = request.negative_prompt if request.negative_prompt is not None else preset.negative_prompt
        height = request.height if request.height else preset.height
        width = request.width if request.width else preset.width
        num_frames = request.num_frames if request.num_frames else preset.num_frames
        frame_rate = request.frame_rate if request.frame_rate else preset.frame_rate
        num_inference_steps = request.num_inference_steps if request.num_inference_steps else preset.num_inference_steps
        cfg_guidance_scale = request.cfg_guidance_scale if request.cfg_guidance_scale else preset.cfg_guidance_scale
        seed = request.seed if request.seed is not None else preset.seed
        image_strength = request.image_strength if request.image_strength is not None else preset.image_strength
        
        if not prompt:
            raise ValueError("Prompt is required")
        
        # Handle seed
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        
        # Set FP8 optimization
        if preset.enable_fp8:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Get pipeline
        pipeline, error = self._get_cached_pipeline(preset)
        if error:
            raise RuntimeError(error)
        
        # Prepare image conditioning
        images = []
        if job.input_image_path and Path(job.input_image_path).exists():
            images = [(job.input_image_path, 0, image_strength)]
        
        # Import utilities
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        
        tiling_config = TilingConfig.default()
        if preset.pipeline_type == "ti2vid_one_stage":
            video_chunks_number = 1
        else:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        # Generate
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUTS_DIR / f"ltx2_api_{timestamp}_{job.job_id[:8]}.mp4"
        
        with torch.no_grad():
            if preset.pipeline_type == "distilled":
                video, audio = pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(frame_rate),
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                )
            elif preset.pipeline_type == "ic_lora":
                video, audio = pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(frame_rate),
                    images=images,
                    video_conditioning=[],
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                )
            elif preset.pipeline_type == "ti2vid_one_stage":
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else "",
                    seed=seed,
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(frame_rate),
                    num_inference_steps=int(num_inference_steps),
                    cfg_guidance_scale=float(cfg_guidance_scale),
                    images=images,
                    enhance_prompt=False,
                )
            else:
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else "",
                    seed=seed,
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(frame_rate),
                    num_inference_steps=int(num_inference_steps),
                    cfg_guidance_scale=float(cfg_guidance_scale),
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                )
            
            encode_video(
                video=video,
                fps=float(frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )
        
        return str(output_path)


# ============================================================================
# FastAPI Application
# ============================================================================

# Global generation queue
generation_queue: Optional[GenerationQueue] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - start/stop worker."""
    global generation_queue
    
    # Startup
    ensure_directories()
    generation_queue = GenerationQueue()
    generation_queue.start()
    print("✅ LTX-2 API started. Generation worker running.")
    print(f"📂 Models directory: {MODELS_DIR.absolute()}")
    print(f"📂 Outputs directory: {OUTPUTS_DIR.absolute()}")
    print(f"🌐 Web UI: http://localhost:8000/")
    print(f"📚 API Docs: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    if generation_queue:
        generation_queue.stop()
    print("🛑 LTX-2 API stopped.")


app = FastAPI(
    title="LTX-2 Video Generation API",
    description="RESTful API for AI video generation using LTX-2 models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>LTX-2 API</title></head>
        <body style="background: #0a0a0f; color: #f0f0f5; font-family: sans-serif; padding: 2rem;">
            <h1>🎬 LTX-2 Video Generation API</h1>
            <p>Web UI template not found. Please ensure templates/index.html exists.</p>
            <p><a href="/docs" style="color: #a855f7;">📚 View API Documentation</a></p>
        </body>
        </html>
        """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    pipeline_loaded = (
        generation_queue is not None 
        and generation_queue._pipeline_cache.get("pipeline") is not None
    )
    
    return HealthResponse(
        status="healthy",
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
        pipeline_loaded=pipeline_loaded,
        queue_size=generation_queue.get_queue_size() if generation_queue else 0,
        current_job=generation_queue.get_current_job_id() if generation_queue else None,
    )


# ============================================================================
# Model Endpoints
# ============================================================================

@app.get("/models")
async def list_models():
    """List all available models and their download status."""
    result = {}
    for key, info in CHECKPOINTS.items():
        status, path = check_model_status(key)
        result[key] = {
            "filename": info["filename"],
            "size": info["size"],
            "description": info["description"],
            "type": info["type"],
            "status": status,
            "path": path,
        }
    return result


@app.get("/models/{model_key}")
async def get_model_info(model_key: str):
    """Get information about a specific model."""
    if model_key not in CHECKPOINTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    info = CHECKPOINTS[model_key]
    status, path = check_model_status(model_key)
    
    return {
        "key": model_key,
        "filename": info["filename"],
        "size": info["size"],
        "description": info["description"],
        "type": info["type"],
        "status": status,
        "path": path,
    }


@app.post("/models/{model_key}/download")
async def download_model(model_key: str, background_tasks: BackgroundTasks):
    """Download a model from HuggingFace."""
    if model_key not in CHECKPOINTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    # Check if already downloaded
    status, path = check_model_status(model_key)
    if status == "ready":
        return {"message": f"Model '{model_key}' is already downloaded", "path": path}
    
    try:
        # Download synchronously (could be made async with background task for large files)
        downloaded_path = download_model_file(model_key)
        return {"message": f"Model '{model_key}' downloaded successfully", "path": downloaded_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# ============================================================================
# Generation Endpoints
# ============================================================================

@app.post("/generate", response_model=GenerationJob)
async def generate_video(request: GenerationRequest):
    """
    Submit a video generation job.
    
    Uses the specified preset or default preset if not provided.
    The job is queued and processed sequentially (one at a time).
    Models are automatically downloaded on-demand if not present.
    """
    if not generation_queue:
        raise HTTPException(status_code=503, detail="Generation service not ready")
    
    # Get preset
    preset_manager = get_preset_manager()
    preset_name = request.preset_name or preset_manager.get_default_preset_name()
    preset = preset_manager.get_preset(preset_name)
    
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    # Note: Model validation is done at generation time with auto-download support
    # This allows jobs to be queued even if models aren't downloaded yet
    
    # Handle input image
    input_image_path = None
    if request.input_image_base64:
        try:
            import tempfile
            from PIL import Image
            import io
            
            image_data = base64.b64decode(request.input_image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Save to temp file
            temp_dir = OUTPUTS_DIR / "temp"
            temp_dir.mkdir(exist_ok=True)
            input_image_path = str(temp_dir / f"input_{uuid.uuid4().hex[:8]}.png")
            image.save(input_image_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid input image: {e}")
    
    # Add job to queue
    job_id = generation_queue.add_job(request, preset, input_image_path)
    job = generation_queue.get_job(job_id)
    
    prompt = request.prompt or preset.prompt
    
    return GenerationJob(
        job_id=job_id,
        status=job.status,
        preset_name=preset_name,
        prompt=prompt,
        created_at=job.created_at,
        position_in_queue=generation_queue.get_queue_position(job_id),
    )


@app.get("/generate/{job_id}", response_model=GenerationJob)
async def get_generation_status(job_id: str):
    """Get the status of a generation job."""
    if not generation_queue:
        raise HTTPException(status_code=503, detail="Generation service not ready")
    
    job = generation_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return GenerationJob(
        job_id=job.job_id,
        status=job.status,
        preset_name=job.preset.name,
        prompt=job.request.prompt or job.preset.prompt,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        output_path=job.output_path,
        error_message=job.error_message,
        position_in_queue=generation_queue.get_queue_position(job_id),
    )


@app.get("/generate/{job_id}/download")
async def download_video(job_id: str):
    """Download the generated video."""
    if not generation_queue:
        raise HTTPException(status_code=503, detail="Generation service not ready")
    
    job = generation_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.status != GenerationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")
    
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Output video not found")
    
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=Path(job.output_path).name,
    )


@app.delete("/generate/{job_id}")
async def cancel_generation(job_id: str):
    """Cancel a queued generation job."""
    if not generation_queue:
        raise HTTPException(status_code=503, detail="Generation service not ready")
    
    if generation_queue.cancel_job(job_id):
        return {"message": f"Job '{job_id}' cancelled"}
    else:
        raise HTTPException(status_code=400, detail="Cannot cancel job (not queued or already processing)")


# ============================================================================
# Preset Endpoints
# ============================================================================

@app.get("/presets", response_model=List[PresetResponse])
async def list_presets():
    """List all presets."""
    preset_manager = get_preset_manager()
    default_name = preset_manager.get_default_preset_name()
    
    presets = []
    for name in preset_manager.list_presets():
        preset = preset_manager.get_preset(name)
        if preset:
            presets.append(PresetResponse(
                name=preset.name,
                description=preset.description,
                is_default=(name == default_name),
                created_at=preset.created_at,
                updated_at=preset.updated_at,
                pipeline_type=preset.pipeline_type,
                prompt=preset.prompt,
                negative_prompt=preset.negative_prompt,
                height=preset.height,
                width=preset.width,
                num_frames=preset.num_frames,
                frame_rate=preset.frame_rate,
                num_inference_steps=preset.num_inference_steps,
                cfg_guidance_scale=preset.cfg_guidance_scale,
                seed=preset.seed,
                enable_fp8=preset.enable_fp8,
                image_strength=preset.image_strength,
                checkpoint_path=preset.checkpoint_path,
                spatial_upsampler_path=preset.spatial_upsampler_path,
                distilled_lora_path=preset.distilled_lora_path,
                gemma_path=preset.gemma_path,
            ))
    
    return presets


@app.get("/presets/{preset_name}", response_model=PresetResponse)
async def get_preset(preset_name: str):
    """Get a specific preset."""
    preset_manager = get_preset_manager()
    preset = preset_manager.get_preset(preset_name)
    
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        is_default=(preset_name == preset_manager.get_default_preset_name()),
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        pipeline_type=preset.pipeline_type,
        prompt=preset.prompt,
        negative_prompt=preset.negative_prompt,
        height=preset.height,
        width=preset.width,
        num_frames=preset.num_frames,
        frame_rate=preset.frame_rate,
        num_inference_steps=preset.num_inference_steps,
        cfg_guidance_scale=preset.cfg_guidance_scale,
        seed=preset.seed,
        enable_fp8=preset.enable_fp8,
        image_strength=preset.image_strength,
        checkpoint_path=preset.checkpoint_path,
        spatial_upsampler_path=preset.spatial_upsampler_path,
        distilled_lora_path=preset.distilled_lora_path,
        gemma_path=preset.gemma_path,
    )


@app.post("/presets", response_model=PresetResponse)
async def create_preset(preset_data: PresetCreate):
    """Create a new preset."""
    preset_manager = get_preset_manager()
    
    # Check if preset exists
    if preset_manager.get_preset(preset_data.name):
        raise HTTPException(status_code=409, detail=f"Preset '{preset_data.name}' already exists")
    
    preset = preset_manager.create_preset_from_settings(
        name=preset_data.name,
        description=preset_data.description,
        pipeline_type=preset_data.pipeline_type,
        checkpoint_path=preset_data.checkpoint_path or "",
        distilled_lora_path=preset_data.distilled_lora_path,
        spatial_upsampler_path=preset_data.spatial_upsampler_path or "",
        gemma_path=preset_data.gemma_path,
        prompt=preset_data.prompt,
        negative_prompt=preset_data.negative_prompt,
        height=preset_data.height,
        width=preset_data.width,
        num_frames=preset_data.num_frames,
        frame_rate=preset_data.frame_rate,
        num_inference_steps=preset_data.num_inference_steps,
        cfg_guidance_scale=preset_data.cfg_guidance_scale,
        seed=preset_data.seed,
        enable_fp8=preset_data.enable_fp8,
        image_strength=preset_data.image_strength,
    )
    
    preset_manager.save_preset(preset)
    
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        is_default=False,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        pipeline_type=preset.pipeline_type,
        prompt=preset.prompt,
        negative_prompt=preset.negative_prompt,
        height=preset.height,
        width=preset.width,
        num_frames=preset.num_frames,
        frame_rate=preset.frame_rate,
        num_inference_steps=preset.num_inference_steps,
        cfg_guidance_scale=preset.cfg_guidance_scale,
        seed=preset.seed,
        enable_fp8=preset.enable_fp8,
        image_strength=preset.image_strength,
        checkpoint_path=preset.checkpoint_path,
        spatial_upsampler_path=preset.spatial_upsampler_path,
        distilled_lora_path=preset.distilled_lora_path,
        gemma_path=preset.gemma_path,
    )


@app.put("/presets/{preset_name}", response_model=PresetResponse)
async def update_preset(preset_name: str, preset_data: PresetCreate):
    """Update an existing preset."""
    preset_manager = get_preset_manager()
    
    existing = preset_manager.get_preset(preset_name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    # Create updated preset (keep original name if data.name differs)
    preset = preset_manager.create_preset_from_settings(
        name=preset_name,  # Keep original name
        description=preset_data.description,
        pipeline_type=preset_data.pipeline_type,
        checkpoint_path=preset_data.checkpoint_path or existing.checkpoint_path,
        distilled_lora_path=preset_data.distilled_lora_path,
        spatial_upsampler_path=preset_data.spatial_upsampler_path or existing.spatial_upsampler_path,
        gemma_path=preset_data.gemma_path,
        prompt=preset_data.prompt,
        negative_prompt=preset_data.negative_prompt,
        height=preset_data.height,
        width=preset_data.width,
        num_frames=preset_data.num_frames,
        frame_rate=preset_data.frame_rate,
        num_inference_steps=preset_data.num_inference_steps,
        cfg_guidance_scale=preset_data.cfg_guidance_scale,
        seed=preset_data.seed,
        enable_fp8=preset_data.enable_fp8,
        image_strength=preset_data.image_strength,
    )
    preset.created_at = existing.created_at
    
    preset_manager.save_preset(preset, overwrite=True)
    
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        is_default=(preset_name == preset_manager.get_default_preset_name()),
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        pipeline_type=preset.pipeline_type,
        prompt=preset.prompt,
        negative_prompt=preset.negative_prompt,
        height=preset.height,
        width=preset.width,
        num_frames=preset.num_frames,
        frame_rate=preset.frame_rate,
        num_inference_steps=preset.num_inference_steps,
        cfg_guidance_scale=preset.cfg_guidance_scale,
        seed=preset.seed,
        enable_fp8=preset.enable_fp8,
        image_strength=preset.image_strength,
        checkpoint_path=preset.checkpoint_path,
        spatial_upsampler_path=preset.spatial_upsampler_path,
        distilled_lora_path=preset.distilled_lora_path,
        gemma_path=preset.gemma_path,
    )


@app.delete("/presets/{preset_name}")
async def delete_preset(preset_name: str):
    """Delete a preset. Cannot delete the 'default' preset."""
    preset_manager = get_preset_manager()
    
    if preset_name == DEFAULT_PRESET_NAME:
        raise HTTPException(status_code=400, detail="Cannot delete the default preset")
    
    if not preset_manager.get_preset(preset_name):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    preset_manager.delete_preset(preset_name)
    return {"message": f"Preset '{preset_name}' deleted"}


@app.post("/presets/{preset_name}/set-default")
async def set_default_preset(preset_name: str):
    """Set a preset as the default."""
    preset_manager = get_preset_manager()
    
    if not preset_manager.get_preset(preset_name):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    preset_manager.set_default(preset_name)
    return {"message": f"Preset '{preset_name}' is now the default"}


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
