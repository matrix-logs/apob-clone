# APOB Clone - Dockerfile
# Multi-stage build for ComfyUI + FastAPI backend

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create workspace
WORKDIR /workspace

# Install PyTorch with CUDA 12.8
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ============================================
# ComfyUI Installation
# ============================================
FROM base AS comfyui

WORKDIR /workspace

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ComfyUI requirements
WORKDIR /workspace/ComfyUI
RUN pip install -r requirements.txt

# Install ComfyUI Manager
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Install custom nodes
RUN git clone https://github.com/cubiq/PuLID_ComfyUI.git || true
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git || true
RUN git clone https://github.com/Gourieff/comfyui-reactor-node.git || true
RUN git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git || true
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git || true

# Install node dependencies
RUN for dir in */; do \
    if [ -f "${dir}requirements.txt" ]; then \
        pip install -r "${dir}requirements.txt" || true; \
    fi \
done

# ============================================
# API Installation
# ============================================
FROM comfyui AS api

WORKDIR /workspace

# Copy API code
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY workflows/ ./workflows/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p /workspace/outputs /workspace/inputs

# Create model directories
RUN mkdir -p /workspace/ComfyUI/models/checkpoints \
    /workspace/ComfyUI/models/clip \
    /workspace/ComfyUI/models/vae \
    /workspace/ComfyUI/models/loras \
    /workspace/ComfyUI/models/controlnet \
    /workspace/ComfyUI/models/pulid \
    /workspace/ComfyUI/models/insightface \
    /workspace/ComfyUI/models/wan

# ============================================
# Final Stage
# ============================================
FROM api AS final

WORKDIR /workspace

# Environment variables
ENV COMFYUI_HOST=127.0.0.1
ENV COMFYUI_PORT=8188
ENV OUTPUT_DIR=/workspace/outputs
ENV WORKFLOWS_DIR=/workspace/workflows

# Expose ports
EXPOSE 8188 8000

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting ComfyUI..."\n\
cd /workspace/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 &\n\
sleep 15\n\
echo "Starting API server..."\n\
cd /workspace/api && uvicorn main:app --host 0.0.0.0 --port 8000\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Default command
CMD ["/workspace/start.sh"]
