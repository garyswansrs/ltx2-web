# 🎬 LTX-2 WebUI

A beautiful web interface and REST API for [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) video generation models.

[![LTX-2](https://img.shields.io/badge/Model-LTX--2-purple)](https://huggingface.co/Lightricks/LTX-2)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p align="center">
  <img src="https://img.shields.io/badge/4K-Resolution-orange" alt="4K">
  <img src="https://img.shields.io/badge/50-FPS-blue" alt="50 FPS">
  <img src="https://img.shields.io/badge/20s-Duration-green" alt="20 seconds">
  <img src="https://img.shields.io/badge/Audio-Sync-purple" alt="Audio Sync">
</p>

## ✨ Features

- **🎥 Multiple Generation Modes** - Text-to-video, image-to-video, video-to-video, keyframe interpolation
- **🔊 Audio-Video Sync** - Synchronized audio generation
- **📺 Native 4K** - Up to 4K resolution at 50 FPS
- **⚡ Fast Inference** - Distilled pipeline with only 8 steps
- **💾 Preset System** - Save and load generation settings
- **🔌 REST API** - Full-featured FastAPI service for integration
- **🧠 VRAM Caching** - Models stay in memory for faster subsequent generations
- **📦 Auto-Download** - All models downloaded automatically

## 🚀 Quick Start

**Just run one command** - everything is handled automatically:

```bash
./run.sh
```

That's it! The script will:

✅ Install all Python dependencies  
✅ Install LTX-2 core packages  
✅ Download Gemma 3 12B FP8 text encoder (~13GB)  
✅ Download LTX-2 19B Distilled FP8 checkpoint (~27GB)  
✅ Download Spatial upsampler (~1GB)  
✅ Start the server at **http://localhost:8000**

> **Note:** First run takes ~10-30 minutes depending on your internet speed (downloading ~40GB of models). Subsequent runs start instantly.

### Windows Users

```powershell
# If run.sh doesn't work, use these commands:
pip install -r requirements.txt
pip install -e LTX-2/packages/ltx-core
pip install -e LTX-2/packages/ltx-pipelines
python api.py  # Models auto-download on first generation
```

## 🌐 Interfaces

### Web UI

Access the web interface at **http://localhost:8000**

Features:
- Beautiful dark theme with gradient accents
- Generation mode selector (Fast, Quality, Image-to-Video, etc.)
- Real-time generation status with progress
- Preset management (save/load/delete)
- Model management with on-demand downloads
- Full API documentation built into the UI

### REST API

Full interactive API documentation at **http://localhost:8000/docs** (Swagger UI)

**Quick Examples:**

```bash
# Generate a video using default preset
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over the ocean"}'

# Generate with custom settings
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat walking through a garden",
    "height": 1024,
    "width": 1536,
    "num_frames": 121,
    "seed": 42
  }'

# Check job status
curl http://localhost:8000/generate/{job_id}

# Download completed video
curl http://localhost:8000/generate/{job_id}/download -o video.mp4

# List all presets
curl http://localhost:8000/presets

# Check system health
curl http://localhost:8000/health
```

### Gradio WebUI (Alternative)

```bash
python app.py
```

Access at **http://localhost:7860** with shareable link support.

## 📦 Models

### Required Models

| Model | Size | Description |
|-------|------|-------------|
| `ltx-2-19b-distilled-fp8` | 27 GB | **Recommended** - Fast FP8 distilled model |
| `Gemma 3 12B FP8` | 13 GB | Required text encoder |
| `ltx-2-spatial-upscaler-x2` | 1 GB | 2x resolution upscaler |

### Download Commands

```bash
# Gemma 3 12B FP8 (no HF token required)
huggingface-cli download MISHANM/google-gemma-3-12b-it-fp8 --local-dir ./models/gemma

# LTX-2 Checkpoint
huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./models/checkpoints

# Spatial Upsampler
huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir ./models/upsamplers
```

### Optional Models

| Model | Size | Use Case |
|-------|------|----------|
| `ltx-2-19b-dev-fp8` | 27 GB | Dev model for Two-Stage pipeline |
| `ltx-2-19b-distilled-lora-384` | 7.7 GB | Required for Two-Stage pipeline |
| `ltx-2-temporal-upscaler-x2` | 262 MB | Frame rate upscaling |

## 🔧 Pipelines

### Pipeline Selection Guide

```
Need video conditioning?
├─ YES → Reference videos for v2v? → Use IC-LoRA Pipeline
│        Keyframes to interpolate? → Use Keyframe Interpolation
│        Just image conditioning? → Any pipeline works
│
└─ NO → Text-to-video only
        ├─ Fastest inference? → Distilled Pipeline ⚡ (8 steps)
        └─ Best quality? → Two-Stage Pipeline 🎬 (40+ steps)
```

### Pipeline Comparison

| Pipeline | Steps | CFG | Upsampling | Best For |
|----------|-------|-----|------------|----------|
| **Distilled** ⚡ | 8 | ❌ | ✅ | Fast iterations |
| **Two-Stage** 🎬 | 40+ | ✅ | ✅ | Production quality |
| **IC-LoRA** 🎞️ | 40+ | ❌ | ✅ | Video-to-video |
| **Keyframe** 🎨 | 40+ | ✅ | ✅ | Animation |

## ⚙️ Configuration

### Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height` | 1024 | Output height (256-2048) |
| `width` | 1536 | Output width (256-2048) |
| `num_frames` | 121 | Frame count (9, 17, 25... 257) |
| `frame_rate` | 24 | FPS (8-60) |
| `num_inference_steps` | 40 | Denoising steps |
| `cfg_guidance_scale` | 4.0 | CFG scale (1.0-15.0) |
| `seed` | -1 | Random seed (-1 = random) |

### Environment Variables

```bash
# Enable FP8 memory optimization (recommended)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace token (for gated models)
export HF_TOKEN=your_token_here
```

## 💡 Tips

### Memory Optimization

- **Use FP8 checkpoints** - 30% smaller than full precision
- **Enable FP8 optimization** - Checkbox in UI or `enable_fp8=True`
- **Clear VRAM cache** - Settings tab when switching between tasks

### Best Quality

- Use **Two-Stage Pipeline** with 40+ inference steps
- CFG scale of 4.0-6.0 works well for most prompts
- Write detailed, cinematographic prompts

### Fastest Generation

- Use **Distilled Pipeline** - Only 8 steps required
- FP8 checkpoint reduces memory and improves speed
- Keep models cached between generations

## 📁 Project Structure

```
ltx2/
├── api.py              # FastAPI REST API server
├── app.py              # Gradio WebUI application
├── presets.py          # Preset management system
├── requirements.txt    # Python dependencies
├── run.sh              # One-command launcher
├── templates/          # HTML templates for API UI
├── models/             # Downloaded models (auto-created)
│   ├── checkpoints/    # LTX-2 model checkpoints
│   ├── gemma/          # Gemma text encoder
│   ├── loras/          # LoRA adapters
│   └── upsamplers/     # Spatial/temporal upsamplers
├── outputs/            # Generated videos
├── presets/            # Saved presets (JSON)
└── LTX-2/              # Official LTX-2 repository
    └── packages/
        ├── ltx-core/       # Core model implementation
        ├── ltx-pipelines/  # Pipeline implementations
        └── ltx-trainer/    # Training tools
```

## 📋 Requirements

- **Python**: >= 3.12
- **CUDA**: >= 12.7
- **PyTorch**: >= 2.7
- **GPU Memory**: 16GB+ (24GB+ recommended)
- **Disk Space**: ~55GB for models

## 🔗 Links

- [LTX-2 Repository](https://github.com/Lightricks/LTX-2)
- [HuggingFace Models](https://huggingface.co/Lightricks/LTX-2)
- [LTX Studio](https://ltx.studio)
- [API Documentation](http://localhost:8000/docs)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

LTX-2 models are subject to [Lightricks' licensing terms](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).

---

<p align="center">
  Built with ❤️ for the AI video generation community
</p>
