# ============================================================================
# Dockerfile for Gemini QA Pipeline on AWS GPU
# Base: NVIDIA CUDA + PyTorch
# Target: g4dn.xlarge (T4) or g5.xlarge (A10G)
# ============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="Ranjan-Aniket"
LABEL description="Video QA Generation Pipeline with GPU Support"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.10 \
    python3-pip \
    python3-dev \
    # Video/Audio processing
    ffmpeg \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Build tools
    git \
    wget \
    curl \
    build-essential \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements-aws-gpu.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements-aws-gpu.txt

# Download spacy model
RUN python3 -m spacy download en_core_web_sm

# Copy application code
COPY processing/ ./processing/
COPY templates/ ./templates/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env* ./

# Create output directories
RUN mkdir -p /app/outputs /app/uploads /app/logs /app/temp

# Expose ports (if running backend API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command (can be overridden)
CMD ["python3", "-c", "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"]

# ============================================================================
# BUILD & RUN INSTRUCTIONS
# ============================================================================
#
# Build:
#   docker build -t video-qa-pipeline .
#
# Run with GPU:
#   docker run --gpus all -it \
#     -e OPENAI_API_KEY="your-key" \
#     -e ANTHROPIC_API_KEY="your-key" \
#     -v $(pwd)/outputs:/app/outputs \
#     video-qa-pipeline \
#     python3 processing/smart_pipeline.py --video-url "https://youtube.com/..."
#
# Run interactively:
#   docker run --gpus all -it \
#     -e OPENAI_API_KEY="your-key" \
#     -e ANTHROPIC_API_KEY="your-key" \
#     video-qa-pipeline bash
#
# ============================================================================
