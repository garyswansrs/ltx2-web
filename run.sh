#!/bin/bash

#####################################################
#                CONFIGURATION                      #
#####################################################
# HuggingFace token (only needed for gated models like LTX-2 checkpoints)
# Get your token from: https://huggingface.co/settings/tokens
# NOTE: Gemma 3 12B FP8 from PyTorch does NOT require a token!

HF_TOKEN=""  # Optional - set if needed for downloading gated models

#####################################################

echo "========================================"
echo "       LTX-2 WebUI Launcher"
echo "========================================"
echo

# Set environment variable for FP8 optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install WebUI dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing WebUI dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Check if LTX-2 pipelines are installed
if ! python -c "import ltx_pipelines" 2>/dev/null; then
    echo
    echo "LTX-2 pipelines not found. Installing from local packages..."
    echo
    
    # Install ltx-core from local directory (already included)
    echo "Installing ltx-core..."
    pip install -e LTX-2/packages/ltx-core
    
    # Install ltx-pipelines from local directory (already included)
    echo "Installing ltx-pipelines..."
    pip install -e LTX-2/packages/ltx-pipelines
    
    echo
    echo "LTX-2 pipelines installed successfully!"
    echo
fi

# Check if Gemma 3 is downloaded (look for actual model files, not just README)
# LTX-2 requires Gemma 3, NOT Gemma 2!
GEMMA_OK=false
if [ -d "models/gemma" ]; then
    # Check if there are actual model files (*.safetensors or config.json)
    # Also verify it's Gemma 3 by checking config
    if [ -f "models/gemma/config.json" ]; then
        if grep -q "gemma3" "models/gemma/config.json" 2>/dev/null; then
            GEMMA_OK=true
        else
            echo "⚠️  Found Gemma 2 in models/gemma but LTX-2 requires Gemma 3!"
            echo "   Removing old model and downloading Gemma 3..."
            rm -rf models/gemma
        fi
    fi
fi

if [ "$GEMMA_OK" = false ]; then
    echo
    echo "================================================"
    echo "  Gemma 3 12B FP8 text encoder is REQUIRED for LTX-2"
    echo "================================================"
    echo
    echo "⚠️  LTX-2 requires Gemma 3 12B (MISHANM/google-gemma-3-12b-it-fp8)"
    echo "   Gemma 2 and Gemma 3 4B are NOT compatible!"
    echo "   Using PyTorch FP8 version - no HF token required!"
    echo
    
    mkdir -p models/gemma
    
    echo "Downloading MISHANM/google-gemma-3-12b-it-fp8 (this may take a while - ~13GB)..."
    huggingface-cli download MISHANM/google-gemma-3-12b-it-fp8 --local-dir ./models/gemma && {
        echo
        echo "✅ Gemma 3 12B FP8 downloaded successfully!"
    } || {
        echo
        echo "❌ Download failed. Please try manually:"
        echo "   huggingface-cli download MISHANM/google-gemma-3-12b-it-fp8 --local-dir ./models/gemma"
    }
fi

# Check if distilled checkpoint is downloaded (prefer FP8 for memory efficiency)
DISTILLED_FP8="models/checkpoints/ltx-2-19b-distilled-fp8.safetensors"
DISTILLED_FULL="models/checkpoints/ltx-2-19b-distilled.safetensors"

if [ ! -f "$DISTILLED_FP8" ] && [ ! -f "$DISTILLED_FULL" ]; then
    echo
    echo "================================================"
    echo "  Downloading LTX-2 19B Distilled FP8 Checkpoint"
    echo "================================================"
    echo
    echo "Downloading FP8 version (~27GB) - more memory efficient..."
    echo "(Use ltx-2-19b-distilled.safetensors for full precision if needed)"
    mkdir -p models/checkpoints
    
    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./models/checkpoints --token "$HF_TOKEN" && {
            echo "✅ Distilled FP8 checkpoint downloaded!"
        } || {
            huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./models/checkpoints && {
                echo "✅ Distilled FP8 checkpoint downloaded!"
            }
        }
    else
        huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./models/checkpoints && {
            echo "✅ Distilled FP8 checkpoint downloaded!"
        }
    fi
fi

# Check if spatial upsampler is downloaded (required for distilled pipeline)
if [ ! -f "models/upsamplers/ltx-2-spatial-upscaler-x2-1.0.safetensors" ]; then
    echo
    echo "Downloading spatial upsampler (required - increases resolution 2x)..."
    mkdir -p models/upsamplers
    
    huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir ./models/upsamplers && {
        echo "✅ Spatial upsampler downloaded!"
    } || {
        echo "❌ Failed to download spatial upsampler"
    }
fi


echo
echo "========================================"
echo "  All models ready! Starting WebUI..."
echo "========================================"
echo
echo "Open http://localhost:7860 in your browser"
echo
echo "Default setup: Distilled pipeline (fast 8-step inference)"
echo

python app.py
