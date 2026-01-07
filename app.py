"""
LTX-2 WebUI - Video Generation Interface
A beautiful web interface for Lightricks LTX-2 video generation models.

LTX-2 supports:
- Synchronized audio-video generation
- Native 4K resolution at up to 50 FPS
- Clips up to 20 seconds long
- Text-to-video, image-to-video, video-to-video, keyframe interpolation

Requires: Python >= 3.12, CUDA >= 12.7, PyTorch ~= 2.7
"""

import os
import sys
import json
import time
import torch
import gradio as gr
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
import tempfile
import shutil

# Constants
MODELS_DIR = Path("./models")
OUTPUTS_DIR = Path("./outputs")
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

# Pipeline types - based on README Pipeline Selection Guide
# Decision Tree:
# - Text-to-video only:
#   - Fastest inference → DistilledPipeline (8 sigmas, no CFG)
#   - Best quality → TI2VidTwoStagesPipeline (production recommended)
# - Image/Video conditioning:
#   - Reference videos → ICLoraPipeline
#   - Keyframe interpolation → KeyframeInterpolationPipeline
#   - Image-to-video → Any pipeline supports this
# Note: TI2VidOneStagePipeline is primarily for educational purposes
PIPELINE_TYPES = {
    "distilled": {
        "name": "⚡ Distilled Pipeline (Fastest)",
        "description": "🚀 Fastest inference with 8 predefined sigmas, no CFG needed. Best for: quick iterations, batch processing. Uses distilled checkpoint.",
        "recommended": True,
        "requires": ["distilled checkpoint", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": False, "upsampling": True, "conditioning": "Image"}
    },
    "ti2vid_two_stages": {
        "name": "🎬 Two-Stage Pipeline (Best Quality)",
        "description": "Production quality - Stage 1 with CFG guidance, Stage 2 upsamples 2x with distilled LoRA refinement. Best for: final renders, highest quality.",
        "recommended": False,
        "requires": ["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Image"}
    },
    "ti2vid_one_stage": {
        "name": "📚 One-Stage Pipeline (Educational)",
        "description": "⚠️ For learning/prototyping only. Single stage, no upsampling, lower resolution (512×768). NOT recommended for production.",
        "recommended": False,
        "requires": ["checkpoint", "gemma"],
        "features": {"stages": 1, "cfg": True, "upsampling": False, "conditioning": "Image"}
    },
    "ic_lora": {
        "name": "🎞️ IC-LoRA Pipeline (Video-to-Video)",
        "description": "Video-to-video transformations with reference video/image conditioning. Best for: style transfer, pose/depth control, video editing.",
        "recommended": False,
        "requires": ["checkpoint", "ic_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": False, "upsampling": True, "conditioning": "Image + Video"}
    },
    "keyframe_interpolation": {
        "name": "🎨 Keyframe Interpolation Pipeline",
        "description": "Interpolate between keyframe images for smooth animations. Uses guiding latents for smoother transitions. Best for: animation, motion graphics.",
        "recommended": False,
        "requires": ["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Keyframes"}
    },
}

# Custom CSS for a stunning dark theme
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary-hue: 265;
    --accent-hue: 340;
    --bg-dark: #0a0a0f;
    --bg-card: #12121a;
    --bg-hover: #1a1a25;
    --border-color: #2a2a3a;
    --text-primary: #f0f0f5;
    --text-secondary: #9090a5;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
    --accent-gradient: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f97316 100%);
    --glow-purple: 0 0 30px rgba(168, 85, 247, 0.3);
    --glow-pink: 0 0 30px rgba(236, 72, 153, 0.3);
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
}

/* Header styling */
.header-container {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 1rem;
    position: relative;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 100%;
    background: radial-gradient(ellipse at center, rgba(168, 85, 247, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.header-title {
    font-size: 3.5rem;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

/* Card styling */
.gr-panel, .gr-box, .gr-form {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
}

/* Tab styling */
.tab-nav {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    border: 1px solid var(--border-color) !important;
}

.tab-nav button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: var(--accent-gradient) !important;
    color: white !important;
}

/* Input styling */
input, textarea, select {
    background: var(--bg-dark) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: var(--glow-purple) !important;
}

/* Button styling */
.gr-button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.gr-button-primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: white !important;
}

.gr-button-primary:hover {
    box-shadow: var(--glow-purple), var(--glow-pink) !important;
    transform: translateY(-2px);
}

.gr-button-secondary {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* Slider styling */
.gr-slider input[type="range"] {
    accent-color: var(--accent-purple) !important;
}

/* Label styling */
label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* Accordion styling */
.gr-accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Progress bar */
.progress-bar {
    background: var(--accent-gradient) !important;
}

/* Model card styling */
.model-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.model-card:hover {
    border-color: var(--accent-purple);
    box-shadow: var(--glow-purple);
}

/* Status indicators */
.status-ready {
    color: #22c55e;
}

.status-downloading {
    color: #f59e0b;
}

.status-missing {
    color: #ef4444;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-dark) !important;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.9em;
}

/* Video output */
video {
    border-radius: 12px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
}

/* Gallery */
.gr-gallery {
    border-radius: 12px !important;
    overflow: hidden;
}

/* Markdown */
.gr-markdown {
    color: var(--text-secondary) !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-purple);
}

/* Animation */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(168, 85, 247, 0.3); }
    50% { box-shadow: 0 0 40px rgba(236, 72, 153, 0.5); }
}

.generating {
    animation: pulse-glow 2s ease-in-out infinite;
}
"""


def ensure_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)
    (MODELS_DIR / "loras").mkdir(exist_ok=True)
    (MODELS_DIR / "upsamplers").mkdir(exist_ok=True)
    (MODELS_DIR / "gemma").mkdir(exist_ok=True)


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


def check_model_status(model_key: str) -> Tuple[str, str]:
    """Check if a model is downloaded. Returns (status, status_text)."""
    path = get_model_path(model_key)
    if path and path.exists():
        size = path.stat().st_size / (1024 ** 3)  # GB
        return "ready", f"✅ Downloaded ({size:.2f} GB)"
    return "missing", "❌ Not downloaded"


def download_model(model_key: str, progress=gr.Progress()) -> str:
    """Download a model from HuggingFace."""
    if model_key not in CHECKPOINTS:
        return f"❌ Unknown model: {model_key}"
    
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
        return f"✅ Model already exists: {target_path}"
    
    try:
        progress(0, desc=f"Downloading {filename}...")
        
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=target_dir,
        )
        
        progress(1, desc="Download complete!")
        return f"✅ Successfully downloaded to: {downloaded_path}\n\n📋 Click 'Refresh Model Lists' in the Generate tab to see the new model."
    
    except Exception as e:
        return f"❌ Download failed: {str(e)}"


def get_available_models() -> dict:
    """Get status of all available models."""
    statuses = {}
    for key in CHECKPOINTS:
        status, text = check_model_status(key)
        statuses[key] = {
            "status": status,
            "text": text,
            "info": CHECKPOINTS[key]
        }
    return statuses


def refresh_model_status() -> str:
    """Generate HTML for model status display."""
    statuses = get_available_models()
    
    html = "<div style='display: grid; gap: 0.75rem;'>"
    
    for key, data in statuses.items():
        info = data["info"]
        status_class = "status-ready" if data["status"] == "ready" else "status-missing"
        
        html += f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: var(--text-primary);">{key}</strong>
                    <span style="color: var(--text-secondary); font-size: 0.9em;"> ({info['size']})</span>
                </div>
                <span class="{status_class}" style="font-size: 0.9em;">{data['text']}</span>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.85em; margin-top: 0.25rem;">
                {info['description']}
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def get_checkpoint_choices() -> List[str]:
    """Get list of available checkpoint files."""
    choices = []
    checkpoint_dir = MODELS_DIR / "checkpoints"
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.safetensors"):
            choices.append(str(f))
    # Sort with distilled-fp8 first (most memory efficient), then distilled, then others
    def sort_key(x):
        x_lower = x.lower()
        if "distilled-fp8" in x_lower:
            return (0, x)  # FP8 first (most memory efficient)
        elif "distilled" in x_lower and "lora" not in x_lower:
            return (1, x)  # Then full distilled
        elif "fp8" in x_lower:
            return (2, x)  # Then other FP8
        elif "fp4" in x_lower:
            return (3, x)  # Then FP4
        else:
            return (4, x)  # Then everything else
    choices.sort(key=sort_key)
    return choices if choices else ["No checkpoints found - download from Models tab"]


def get_default_checkpoint() -> str:
    """Get the default checkpoint path (prefer distilled-fp8 for memory efficiency)."""
    choices = get_checkpoint_choices()
    if choices and "No checkpoints" not in choices[0]:
        # Return first choice (distilled-fp8 is sorted first)
        return choices[0]
    # Fall back to expected default paths (prefer FP8)
    fp8_path = MODELS_DIR / "checkpoints" / "ltx-2-19b-distilled-fp8.safetensors"
    full_path = MODELS_DIR / "checkpoints" / "ltx-2-19b-distilled.safetensors"
    if fp8_path.exists():
        return str(fp8_path)
    return str(full_path)


def get_default_upsampler() -> str:
    """Get the default spatial upsampler path."""
    choices = get_upscaler_choices()
    if choices and "No upsamplers" not in choices[0]:
        return choices[0]
    # Fall back to expected default path
    default_path = str(MODELS_DIR / "upsamplers" / "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    return default_path


def get_lora_choices() -> List[str]:
    """Get list of available LoRA files."""
    choices = ["None"]
    lora_dir = MODELS_DIR / "loras"
    if lora_dir.exists():
        for f in lora_dir.glob("*.safetensors"):
            choices.append(str(f))
    return choices


def get_upscaler_choices() -> List[str]:
    """Get list of available upscaler files."""
    choices = []
    upscaler_dir = MODELS_DIR / "upsamplers"
    if upscaler_dir.exists():
        for f in upscaler_dir.glob("*.safetensors"):
            choices.append(str(f))
    return choices if choices else ["No upsamplers found - download from Models tab"]


# Global pipeline cache - keeps models in VRAM between generations
# The underlying ModelLedger also caches individual models (transformer, VAE, etc.)
# for even faster subsequent runs within the same pipeline configuration
_pipeline_cache = {
    "pipeline": None,
    "pipeline_type": None,
    "checkpoint_path": None,
    "spatial_upsampler_path": None,
    "gemma_path": None,
    "distilled_lora_path": None,
    "enable_fp8": None,
}


def clear_vram_cache() -> str:
    """Clear all cached models from VRAM."""
    global _pipeline_cache
    
    try:
        # Get VRAM usage before clearing
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / (1024 ** 3)
        else:
            vram_before = 0
        
        # Clear the pipeline cache
        if _pipeline_cache["pipeline"] is not None:
            # Clear ModelLedger cache(s) if the pipeline has them
            pipeline = _pipeline_cache["pipeline"]
            if hasattr(pipeline, 'model_ledger'):
                pipeline.model_ledger.clear_cache()
            # For two-stage pipelines with multiple ledgers
            if hasattr(pipeline, 'stage_1_model_ledger'):
                pipeline.stage_1_model_ledger.clear_cache()
            if hasattr(pipeline, 'stage_2_model_ledger'):
                pipeline.stage_2_model_ledger.clear_cache()
        
        # Clear the pipeline reference
        _pipeline_cache["pipeline"] = None
        _pipeline_cache["pipeline_type"] = None
        _pipeline_cache["checkpoint_path"] = None
        _pipeline_cache["spatial_upsampler_path"] = None
        _pipeline_cache["gemma_path"] = None
        _pipeline_cache["distilled_lora_path"] = None
        _pipeline_cache["enable_fp8"] = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram_after = torch.cuda.memory_allocated() / (1024 ** 3)
            freed = vram_before - vram_after
            return f"✅ VRAM cache cleared!\n\n📊 Freed: {freed:.2f} GB\n💾 Current usage: {vram_after:.2f} GB"
        else:
            return "✅ Cache cleared (no CUDA device detected)"
            
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"


def get_vram_status() -> str:
    """Get current VRAM usage status."""
    if not torch.cuda.is_available():
        return "No CUDA device detected"
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    # Check if pipeline is cached
    pipeline_status = "🟢 Pipeline cached" if _pipeline_cache["pipeline"] is not None else "⚪ No pipeline loaded"
    
    return f"""**VRAM Status:**
- Allocated: {allocated:.2f} GB
- Reserved: {reserved:.2f} GB  
- Total: {total:.1f} GB
- {pipeline_status}"""


def get_cached_pipeline(
    pipeline_type: str,
    checkpoint_path: str,
    spatial_upsampler_path: str,
    gemma_path: str,
    distilled_lora_path: str,
    enable_fp8: bool,
    progress=gr.Progress()
):
    """
    Get or create a cached pipeline. Keeps models in VRAM for faster subsequent generations.
    Only recreates pipeline if configuration changes.
    """
    global _pipeline_cache
    
    # Check if we can reuse the cached pipeline
    cache_valid = (
        _pipeline_cache["pipeline"] is not None
        and _pipeline_cache["pipeline_type"] == pipeline_type
        and _pipeline_cache["checkpoint_path"] == checkpoint_path
        and _pipeline_cache["spatial_upsampler_path"] == spatial_upsampler_path
        and _pipeline_cache["gemma_path"] == gemma_path
        and _pipeline_cache["distilled_lora_path"] == distilled_lora_path
        and _pipeline_cache["enable_fp8"] == enable_fp8
    )
    
    if cache_valid:
        progress(0.1, desc="Using cached pipeline (models already in VRAM)...")
        # Models are also cached in ModelLedger, making repeated generations instant
        return _pipeline_cache["pipeline"], None
    
    # Clear old pipeline to free VRAM before loading new one
    if _pipeline_cache["pipeline"] is not None:
        progress(0.1, desc="Clearing old pipeline from VRAM...")
        del _pipeline_cache["pipeline"]
        _pipeline_cache["pipeline"] = None
        torch.cuda.empty_cache()
    
    # Create new pipeline
    progress(0.15, desc=f"Loading {pipeline_type} pipeline (first run is slower)...")
    
    try:
        if pipeline_type == "distilled":
            from ltx_pipelines.distilled import DistilledPipeline
            
            if not spatial_upsampler_path or not Path(spatial_upsampler_path).exists():
                return None, "❌ Spatial upsampler is required for distilled pipeline.\n\nPlease download from the Models tab."
            
            pipeline = DistilledPipeline(
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root=gemma_path,
                loras=[],
                fp8transformer=enable_fp8,
            )
            
        elif pipeline_type == "ti2vid_two_stages":
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
            
            if not distilled_lora_path or distilled_lora_path == "None" or not Path(distilled_lora_path).exists():
                return None, "❌ Two-Stage Pipeline requires a distilled LoRA.\n\nPlease download 'ltx-2-19b-distilled-lora-384' from the Models tab."
            
            distilled_lora_list = [
                LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
            ]
            
            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=checkpoint_path,
                distilled_lora=distilled_lora_list,
                spatial_upsampler_path=spatial_upsampler_path if spatial_upsampler_path else None,
                gemma_root=gemma_path,
                loras=[],
                fp8transformer=enable_fp8,
            )
            
        elif pipeline_type == "ti2vid_one_stage":
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
            
            pipeline = TI2VidOneStagePipeline(
                checkpoint_path=checkpoint_path,
                gemma_root=gemma_path,
                loras=[],
                fp8transformer=enable_fp8,
            )
            
        elif pipeline_type == "ic_lora":
            from ltx_pipelines.ic_lora import ICLoraPipeline
            
            if not spatial_upsampler_path or not Path(spatial_upsampler_path).exists():
                return None, "❌ Spatial upsampler is required for IC-LoRA pipeline.\n\nPlease download from the Models tab."
            
            # IC-LoRA uses loras parameter, not distilled_lora
            # The user should provide the IC-LoRA model via the distilled_lora_path field
            loras = []
            if distilled_lora_path and distilled_lora_path != "None" and Path(distilled_lora_path).exists():
                from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
                loras = [LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
            
            pipeline = ICLoraPipeline(
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root=gemma_path,
                loras=loras,
                fp8transformer=enable_fp8,
            )
            
        elif pipeline_type == "keyframe_interpolation":
            from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
            from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
            
            if not distilled_lora_path or distilled_lora_path == "None" or not Path(distilled_lora_path).exists():
                return None, "❌ Keyframe Interpolation Pipeline requires a distilled LoRA.\n\nPlease download 'ltx-2-19b-distilled-lora-384' from the Models tab."
            
            distilled_lora_list = [
                LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
            ]
            
            pipeline = KeyframeInterpolationPipeline(
                checkpoint_path=checkpoint_path,
                distilled_lora=distilled_lora_list,
                spatial_upsampler_path=spatial_upsampler_path if spatial_upsampler_path else None,
                gemma_root=gemma_path,
                loras=[],
                fp8transformer=enable_fp8,
            )
            
        else:
            return None, f"❌ Unknown pipeline type: {pipeline_type}"
        
        # Cache the pipeline
        _pipeline_cache["pipeline"] = pipeline
        _pipeline_cache["pipeline_type"] = pipeline_type
        _pipeline_cache["checkpoint_path"] = checkpoint_path
        _pipeline_cache["spatial_upsampler_path"] = spatial_upsampler_path
        _pipeline_cache["gemma_path"] = gemma_path
        _pipeline_cache["distilled_lora_path"] = distilled_lora_path
        _pipeline_cache["enable_fp8"] = enable_fp8
        
        return pipeline, None
        
    except ImportError as e:
        return None, f"""❌ LTX Pipelines not installed.

Please install from the LTX-2 repository:
```
cd LTX-2
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
```

Error: {str(e)}"""
    except Exception as e:
        import traceback
        return None, f"❌ Failed to load pipeline:\n{str(e)}\n{traceback.format_exc()}"


def generate_video(
    pipeline_type: str,
    checkpoint_path: str,
    distilled_lora_path: str,
    spatial_upsampler_path: str,
    gemma_path: str,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    seed: int,
    enable_fp8: bool,
    input_image: Optional[Image.Image],
    image_strength: float,
    reference_video: Optional[str],
    keyframe_images: Optional[List[Image.Image]],
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Generate video using the selected pipeline. Keeps models in VRAM for faster subsequent runs."""
    
    # Validate inputs
    if not prompt:
        return None, "❌ Please enter a prompt"
    
    if not checkpoint_path or "No checkpoints" in checkpoint_path:
        return None, "❌ Please download and select a checkpoint from the Models tab"
    
    if not Path(checkpoint_path).exists():
        return None, f"❌ Checkpoint not found: {checkpoint_path}"
    
    # Check Gemma path
    if not gemma_path or not Path(gemma_path).exists():
        return None, """❌ Gemma text encoder not configured!

The Gemma 3 **12B** text encoder is **required** for all LTX-2 pipelines.

**To download Gemma 3 12B FP8 (no HF token required):**
```bash
huggingface-cli download pytorch/gemma-3-12b-it-FP8 --local-dir ./models/gemma
```

Then set the "Gemma Path" to `./models/gemma` in the Generate tab.

> ⚠️ **Important:** You must use Gemma 3 12B. Gemma 2 and Gemma 3 4B will NOT work!"""
    
    # Generate output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"ltx2_{timestamp}.mp4"
    
    # Handle seed - generate random if -1 or None
    import random
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)  # Ensure it's an integer
    
    # Set environment variable for FP8 optimization
    if enable_fp8:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        # Get or create cached pipeline (keeps models in VRAM)
        pipeline, error = get_cached_pipeline(
            pipeline_type=pipeline_type,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_path=gemma_path,
            distilled_lora_path=distilled_lora_path,
            enable_fp8=enable_fp8,
            progress=progress,
        )
        
        if error:
            return None, error
        
        progress(0.3, desc="Generating video...")
        
        # Prepare image conditioning
        images = []
        if input_image is not None:
            temp_img_path = OUTPUTS_DIR / f"temp_input_{timestamp}.png"
            input_image.save(temp_img_path)
            images = [(str(temp_img_path), 0, image_strength)]
        
        # Prepare keyframes for keyframe_interpolation pipeline
        if pipeline_type == "keyframe_interpolation" and keyframe_images:
            images = []  # Replace with keyframes
            for i, img in enumerate(keyframe_images):
                temp_path = OUTPUTS_DIR / f"temp_keyframe_{timestamp}_{i}.png"
                img.save(temp_path)
                frame_idx = int(i * (num_frames - 1) / max(1, len(keyframe_images) - 1))
                images.append((str(temp_path), frame_idx, 1.0))
        
        # Import utilities needed for video encoding
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        
        # TilingConfig and video_chunks_number for two-stage pipelines
        # One-stage uses video_chunks_number=1 (no chunking)
        tiling_config = TilingConfig.default()
        if pipeline_type == "ti2vid_one_stage":
            video_chunks_number = 1
        else:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        # Generate video using the cached pipeline
        # Each pipeline has a different API - match the source code exactly
        with torch.no_grad():
            if pipeline_type == "distilled":
                # DistilledPipeline: no CFG, no negative prompt, has tiling_config
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
            elif pipeline_type == "ic_lora":
                # ICLoraPipeline: video conditioning support, has tiling_config
                video_conditioning = []
                if reference_video and Path(reference_video).exists():
                    video_conditioning = [(reference_video, 1.0)]
                
                video, audio = pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(frame_rate),
                    images=images,
                    video_conditioning=video_conditioning,
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                )
            elif pipeline_type == "ti2vid_one_stage":
                # TI2VidOneStagePipeline: CFG + negative prompt, NO tiling_config
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
                # TI2VidTwoStagesPipeline, KeyframeInterpolationPipeline: CFG + negative prompt, has tiling_config
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
            
            # Encode and save video
            progress(0.9, desc="Encoding video...")
            encode_video(
                video=video,
                fps=float(frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )
        
        progress(1.0, desc="Complete!")
        
        if output_path.exists():
            return str(output_path), f"✅ Video generated successfully!\nSaved to: {output_path}\n\n💡 Tip: Models are cached in VRAM - subsequent generations are significantly faster!"
        else:
            return None, "❌ Video generation completed but output file not found"
            
    except Exception as e:
        import traceback
        return None, f"❌ Generation failed:\n{str(e)}\n\n{traceback.format_exc()}"


def create_ui():
    """Create the Gradio interface."""
    
    ensure_directories()
    
    with gr.Blocks(
        title="LTX-2 WebUI",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="purple",
            secondary_hue="pink",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Outfit"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ).set(
            body_background_fill="#0a0a0f",
            body_background_fill_dark="#0a0a0f",
            block_background_fill="#12121a",
            block_background_fill_dark="#12121a",
            border_color_primary="#2a2a3a",
            border_color_primary_dark="#2a2a3a",
        )
    ) as demo:
        
        # Header
        gr.HTML("""
            <div class="header-container">
                <h1 class="header-title">🎬 LTX-2 WebUI</h1>
                <p class="header-subtitle">
                    Generate stunning AI videos with Lightricks LTX-2 • 4K @ 50 FPS • Up to 20 seconds • Audio-Video Sync<br>
                    <a href="https://huggingface.co/Lightricks/LTX-2" target="_blank" style="color: #a855f7;">HuggingFace</a> • 
                    <a href="https://github.com/Lightricks/LTX-2" target="_blank" style="color: #ec4899;">GitHub</a> •
                    <a href="https://docs.ltx.video" target="_blank" style="color: #f97316;">Docs</a>
                </p>
            </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # ===== GENERATE TAB =====
            with gr.Tab("🎥 Generate", id="generate"):
                with gr.Row():
                    # Left Column - Settings
                    with gr.Column(scale=1):
                        
                        # Pipeline Selection
                        with gr.Group():
                            gr.Markdown("### 🔧 Pipeline")
                            pipeline_type = gr.Dropdown(
                                choices=list(PIPELINE_TYPES.keys()),
                                value="distilled",
                                label="Pipeline Type"
                            )
                            pipeline_info = gr.Markdown(
                                PIPELINE_TYPES["distilled"]["description"]
                            )
                        
                        # Model Selection
                        with gr.Group():
                            gr.Markdown("### 📦 Models")
                            gr.Markdown("*Models are auto-downloaded on first run*", elem_classes=["text-muted"])
                            
                            checkpoint_path = gr.Dropdown(
                                choices=get_checkpoint_choices(),
                                value=get_default_checkpoint(),
                                label="Checkpoint (Main model)",
                                allow_custom_value=True
                            )
                            
                            distilled_lora_path = gr.Dropdown(
                                choices=get_lora_choices(),
                                value="None",
                                label="Distilled LoRA (for two-stage pipeline only)",
                                allow_custom_value=True
                            )
                            
                            spatial_upsampler_path = gr.Dropdown(
                                choices=get_upscaler_choices(),
                                value=get_default_upsampler(),
                                label="Spatial Upsampler (2x resolution - required)",
                                allow_custom_value=True
                            )
                            
                            # Auto-detect Gemma path
                            default_gemma = "./models/gemma" if (MODELS_DIR / "gemma").exists() else "./models/gemma"
                            gemma_path = gr.Textbox(
                                label="Gemma Path (auto-downloaded)",
                                value=default_gemma,
                                placeholder="./models/gemma"
                            )
                            
                            refresh_btn = gr.Button("🔄 Refresh Model Lists", variant="secondary", size="sm")
                        
                        # Generation Settings (defaults match CLI: 1024x1536, 121 frames, 24fps)
                        with gr.Accordion("⚙️ Generation Settings", open=True):
                            with gr.Row():
                                height = gr.Slider(
                                    minimum=256, maximum=2048, value=1024, step=64,
                                    label="Height"
                                )
                                width = gr.Slider(
                                    minimum=256, maximum=2048, value=1536, step=64,
                                    label="Width"
                                )
                            
                            with gr.Row():
                                num_frames = gr.Slider(
                                    minimum=9, maximum=257, value=121, step=8,
                                    label="Frames (9, 17, 25, ... 257)"
                                )
                                frame_rate = gr.Slider(
                                    minimum=8, maximum=60, value=24, step=1,
                                    label="FPS"
                                )
                            
                            with gr.Row():
                                num_inference_steps = gr.Slider(
                                    minimum=4, maximum=100, value=40, step=1,
                                    label="Inference Steps (ignored by Distilled pipeline)"
                                )
                                cfg_guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=4.0, step=0.1,
                                    label="CFG Scale (ignored by Distilled pipeline)"
                                )
                            
                            seed = gr.Number(
                                value=-1, label="Seed (-1 for random)"
                            )
                            
                            enable_fp8 = gr.Checkbox(
                                value=True, label="Enable FP8 Optimization (reduces memory)"
                            )
                        
                        # Image Conditioning
                        with gr.Accordion("🖼️ Image Conditioning", open=False):
                            gr.Markdown("*Condition video on this starting image*")
                            input_image = gr.Image(
                                label="Input Image",
                                type="pil"
                            )
                            image_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=1.0, step=0.1,
                                label="Image Strength"
                            )
                        
                        # Keyframe Interpolation
                        with gr.Accordion("🎞️ Keyframe Images", open=False, visible=False) as keyframe_accordion:
                            gr.Markdown("*Upload images to interpolate between*")
                            keyframe_images = gr.Gallery(
                                label="Keyframe Images",
                                type="pil",
                                columns=4
                            )
                        
                        # Video Conditioning (for IC-LoRA)
                        with gr.Accordion("📹 Reference Video", open=False, visible=False) as video_accordion:
                            gr.Markdown("*Reference video for video-to-video*")
                            reference_video = gr.Video(
                                label="Reference Video"
                            )
                    
                    # Right Column - Prompt & Output
                    with gr.Column(scale=1):
                        
                        # Prompt
                        with gr.Group():
                            gr.Markdown("### ✨ Prompt")
                            prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="A beautiful sunset over the ocean with waves crashing on the shore...",
                                lines=4
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="blurry, low quality, distorted...",
                                lines=2
                            )
                        
                        # Generate Button
                        generate_btn = gr.Button(
                            "🚀 Generate Video",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Output
                        with gr.Group():
                            gr.Markdown("### 🎬 Output")
                            output_video = gr.Video(
                                label="Generated Video",
                                autoplay=True
                            )
                            output_status = gr.Markdown("")
                
                # Event handlers
                def update_pipeline_info(pipeline_type):
                    info = PIPELINE_TYPES.get(pipeline_type, {})
                    desc = info.get("description", "")
                    reqs = info.get("requires", [])
                    features = info.get("features", {})
                    
                    req_text = ", ".join(reqs)
                    feature_text = ""
                    if features:
                        stages = features.get("stages", "?")
                        cfg = "✅" if features.get("cfg") else "❌"
                        upsampling = "✅" if features.get("upsampling") else "❌"
                        conditioning = features.get("conditioning", "Image")
                        feature_text = f"\n\n**Features:** {stages} stages | CFG: {cfg} | Upsampling: {upsampling} | Conditioning: {conditioning}"
                    
                    return f"{desc}{feature_text}\n\n**Requires:** {req_text}"
                
                def update_visibility(pipeline_type):
                    show_keyframes = pipeline_type == "keyframe_interpolation"
                    show_video = pipeline_type == "ic_lora"
                    return (
                        gr.update(visible=show_keyframes),
                        gr.update(visible=show_video)
                    )
                
                pipeline_type.change(
                    update_pipeline_info,
                    inputs=[pipeline_type],
                    outputs=[pipeline_info]
                ).then(
                    update_visibility,
                    inputs=[pipeline_type],
                    outputs=[keyframe_accordion, video_accordion]
                )
                
                def refresh_models():
                    return (
                        gr.update(choices=get_checkpoint_choices()),
                        gr.update(choices=get_lora_choices()),
                        gr.update(choices=get_upscaler_choices())
                    )
                
                refresh_btn.click(
                    refresh_models,
                    outputs=[checkpoint_path, distilled_lora_path, spatial_upsampler_path]
                )
                
                generate_btn.click(
                    generate_video,
                    inputs=[
                        pipeline_type,
                        checkpoint_path,
                        distilled_lora_path,
                        spatial_upsampler_path,
                        gemma_path,
                        prompt,
                        negative_prompt,
                        height,
                        width,
                        num_frames,
                        frame_rate,
                        num_inference_steps,
                        cfg_guidance_scale,
                        seed,
                        enable_fp8,
                        input_image,
                        image_strength,
                        reference_video,
                        keyframe_images,
                    ],
                    outputs=[output_video, output_status]
                )
            
            # ===== MODELS TAB =====
            with gr.Tab("📦 Models", id="models"):
                gr.Markdown("""
                ### Download Models from HuggingFace
                
                Download the required model files from [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2).
                Models will be saved to the `./models` directory.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        model_status_html = gr.HTML(
                            value=refresh_model_status(),
                            label="Model Status"
                        )
                    
                    with gr.Column(scale=1):
                        model_to_download = gr.Dropdown(
                            choices=list(CHECKPOINTS.keys()),
                            label="Select Model to Download"
                        )
                        
                        download_btn = gr.Button(
                            "⬇️ Download Selected Model",
                            variant="primary"
                        )
                        
                        refresh_status_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary"
                        )
                        
                        download_status = gr.Markdown("")
                
                # Download handlers
                download_btn.click(
                    download_model,
                    inputs=[model_to_download],
                    outputs=[download_status]
                ).then(
                    refresh_model_status,
                    outputs=[model_status_html]
                ).then(
                    refresh_models,
                    outputs=[checkpoint_path, distilled_lora_path, spatial_upsampler_path]
                )
                
                refresh_status_btn.click(
                    refresh_model_status,
                    outputs=[model_status_html]
                ).then(
                    refresh_models,
                    outputs=[checkpoint_path, distilled_lora_path, spatial_upsampler_path]
                )
                
                # Gemma download section
                gr.Markdown("""
                ---
                ### Gemma 3 Text Encoder
                
                The Gemma 3 **12B** text encoder is required for all pipelines. We use the **PyTorch FP8** version which **does not require** HuggingFace authentication:
                
                ```bash
                # Download PyTorch FP8 version (no token required, ~13GB)
                huggingface-cli download pytorch/gemma-3-12b-it-FP8 --local-dir ./models/gemma
                
                # Or using Python
                from huggingface_hub import snapshot_download
                snapshot_download("pytorch/gemma-3-12b-it-FP8", local_dir="./models/gemma")
                ```
                
                After downloading, set the **Gemma Path** in the Generate tab to `./models/gemma`.
                
                > ⚠️ **Important:** LTX-2 requires the 12B model. Gemma 2 and Gemma 3 4B will cause dimension mismatch errors!
                """)
            
            # ===== GALLERY TAB =====
            with gr.Tab("🖼️ Gallery", id="gallery"):
                gr.Markdown("### Generated Videos")
                
                def get_gallery_videos():
                    videos = []
                    if OUTPUTS_DIR.exists():
                        for f in sorted(OUTPUTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)[:20]:
                            videos.append(str(f))
                    return videos
                
                gallery_videos = gr.Gallery(
                    value=get_gallery_videos,
                    label="Recent Generations",
                    columns=3,
                    object_fit="cover"
                )
                
                refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
                refresh_gallery_btn.click(
                    get_gallery_videos,
                    outputs=[gallery_videos]
                )
            
            # ===== SETTINGS TAB =====
            with gr.Tab("⚙️ Settings", id="settings"):
                gr.Markdown("""
                ### Configuration
                
                Configure default paths and settings for the LTX-2 WebUI.
                """)
                
                with gr.Group():
                    gr.Markdown("#### Directories")
                    models_dir_input = gr.Textbox(
                        value=str(MODELS_DIR),
                        label="Models Directory"
                    )
                    outputs_dir_input = gr.Textbox(
                        value=str(OUTPUTS_DIR),
                        label="Outputs Directory"
                    )
                
                with gr.Group():
                    gr.Markdown("#### 🧠 VRAM Management")
                    gr.Markdown("""
                    Models are cached in VRAM for faster subsequent generations.
                    Use the button below to clear the cache and free VRAM.
                    """)
                    
                    vram_status = gr.Markdown(value=get_vram_status())
                    
                    with gr.Row():
                        clear_vram_btn = gr.Button("🗑️ Clear VRAM Cache", variant="secondary")
                        refresh_vram_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    
                    clear_vram_result = gr.Markdown("")
                    
                    clear_vram_btn.click(
                        clear_vram_cache,
                        outputs=[clear_vram_result]
                    ).then(
                        get_vram_status,
                        outputs=[vram_status]
                    )
                    
                    refresh_vram_btn.click(
                        get_vram_status,
                        outputs=[vram_status]
                    )
                
                with gr.Group():
                    gr.Markdown("#### System Information")
                    
                    # GPU Info
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_info = f"✅ **{gpu_name}** ({gpu_memory:.1f} GB)"
                    else:
                        gpu_info = "❌ No CUDA GPU detected"
                    
                    gr.Markdown(f"**GPU:** {gpu_info}")
                    gr.Markdown(f"**PyTorch:** {torch.__version__}")
                    gr.Markdown(f"**CUDA Available:** {torch.cuda.is_available()}")
            
            # ===== HELP TAB =====
            with gr.Tab("❓ Help", id="help"):
                gr.Markdown("""
                ### LTX-2 WebUI Help
                
                #### 🚀 Quick Start
                
                Run `./run.sh` and all required models will be automatically downloaded:
                
                - ✅ LTX-2 19B Distilled FP8 Checkpoint (~27GB)
                - ✅ Gemma 3 12B FP8 Text Encoder (~13GB, no HF token required)
                - ✅ Spatial Upsampler (~1GB) - doubles resolution
                
                Then enter a prompt and click **Generate**!
                
                ---
                
                #### 🎯 Pipeline Selection Guide
                
                **Decision Tree:**
                ```
                Do you need to condition on existing images/videos?
                ├─ YES → Do you have reference videos for video-to-video?
                │  ├─ YES → Use IC-LoRA Pipeline
                │  └─ NO → Do you have keyframe images to interpolate?
                │     ├─ YES → Use Keyframe Interpolation Pipeline
                │     └─ NO → Use any pipeline (all support image conditioning)
                │
                └─ NO → Text-to-video only
                   ├─ Need best quality? → Use Two-Stage Pipeline (recommended for production)
                   └─ Need fastest inference? → Use Distilled Pipeline (8 sigmas, default)
                ```
                
                > **Note:** One-Stage Pipeline is for educational purposes only. Use two-stage pipelines for production.
                
                ---
                
                #### 📊 Features Comparison
                
                | Pipeline | Stages | CFG | Upsampling | Conditioning | Best For |
                |----------|--------|-----|------------|--------------|----------|
                | **Distilled** ⚡ | 2 | ❌ | ✅ | Image | Fastest inference (8 sigmas) |
                | **Two-Stage** 🎬 | 2 | ✅ | ✅ | Image | **Production quality** (recommended) |
                | **One-Stage** 📚 | 1 | ✅ | ❌ | Image | Educational, prototyping |
                | **IC-LoRA** 🎞️ | 2 | ❌ | ✅ | Image + Video | Video-to-video transformations |
                | **Keyframe** 🎨 | 2 | ✅ | ✅ | Keyframes | Animation, interpolation |
                
                ---
                
                #### System Requirements
                
                - **Python**: >= 3.12
                - **CUDA**: >= 12.7
                - **PyTorch**: ~= 2.7
                - **GPU Memory**: 24GB+ recommended (16GB with FP8/FP4)
                - **Disk Space**: ~55GB for default models (FP8)
                
                ---
                
                #### First-Time Setup (if not using run.sh)
                
                1. **Run the launcher** (downloads all models automatically):
                   ```bash
                   ./run.sh
                   ```
                   
                2. Or download models manually:
                   ```bash
                   # Gemma 3 12B FP8 (no token required)
                   huggingface-cli download pytorch/gemma-3-12b-it-FP8 --local-dir ./models/gemma
                   
                   # LTX-2 checkpoint
                   huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./models/checkpoints
                   
                   # Spatial upsampler
                   huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir ./models/upsamplers
                   ```
                
                ---
                
                #### Model Checkpoints
                
                | Model | Size | Description |
                |-------|------|-------------|
                | `ltx-2-19b-distilled-fp8` | 27.1 GB | **Recommended** - FP8 distilled (fastest + efficient) |
                | `ltx-2-19b-distilled` | 43.3 GB | Full precision distilled |
                | `ltx-2-19b-dev-fp8` | 27.1 GB | FP8 dev model (for Two-Stage) |
                | `ltx-2-19b-dev` | 43.3 GB | Full precision dev (flexible & trainable) |
                | `ltx-2-19b-dev-fp4` | 20 GB | FP4 quantized (smallest) |
                
                ---
                
                #### How Resolution Works (Two-Stage Pipelines)
                
                When you set a resolution like **1024×1536** in the UI:
                1. **Stage 1**: Generates at **half resolution** (512×768) - faster, uses less VRAM
                2. **Stage 2**: **Spatial upsampler** doubles resolution to 1024×1536 + refinement
                
                This is why spatial upsampler is **required** for most pipelines!
                
                ---
                
                #### Tips
                
                - **Memory Issues?** Enable FP8 optimization and use FP8/FP4 checkpoints
                - **Better Quality?** Use Two-Stage pipeline with 40+ inference steps and CFG scale 4.0
                - **Faster Generation?** Use Distilled pipeline (only 8 steps, no CFG needed!)
                - **Image Conditioning?** Upload a starting image - works with all pipelines
                - **Video-to-Video?** Use IC-LoRA pipeline with reference video
                
                ---
                
                #### Links
                
                - 📚 [LTX-2 Documentation](https://docs.ltx.video)
                - 🐙 [GitHub Repository](https://github.com/Lightricks/LTX-2)
                - 🤗 [HuggingFace Models](https://huggingface.co/Lightricks/LTX-2)
                - 💬 [Community](https://huggingface.co/Lightricks/LTX-2/discussions)
                """)
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary); font-size: 0.9rem;">
                <p>Built with ❤️ for the AI video generation community</p>
                <p style="margin-top: 0.5rem;">
                    <a href="https://github.com/Lightricks/LTX-2" target="_blank" style="color: var(--accent-purple);">GitHub</a> •
                    <a href="https://huggingface.co/Lightricks/LTX-2" target="_blank" style="color: var(--accent-pink);">HuggingFace</a>
                </p>
            </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

