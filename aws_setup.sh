#!/bin/bash
# ============================================================================
# AWS GPU Instance Setup Script
# Run this after SSH'ing into your g4dn.xlarge or g5.xlarge instance
# ============================================================================

set -e  # Exit on error

echo "========================================="
echo "AWS GPU Setup for Video QA Pipeline"
echo "========================================="
echo ""

# Check if running on GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Are you on a GPU instance?"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y -qq \
    ffmpeg \
    tesseract-ocr \
    git \
    wget \
    curl \
    htop

# Clone repository
echo "📥 Cloning repository..."
if [ ! -d "Video_QA" ]; then
    git clone https://github.com/Ranjan-Aniket/Video_QA.git
fi
cd Video_QA

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -qq

# Install dependencies
echo "📦 Installing Python dependencies (this takes 5-10 minutes)..."
pip install -r requirements-aws-gpu.txt

# Download spacy model
echo "📦 Downloading spacy model..."
python -m spacy download en_core_web_sm

# Create .env file
echo "⚙️  Setting up environment variables..."
cat > .env << 'EOF'
# API Keys (UPDATE THESE!)
OPENAI_API_KEY=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
ANTHROPIC_API_KEY=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
HF_TOKEN=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

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp
GPT4_MODEL=gpt-4o
CLAUDE_MODEL=claude-sonnet-4-5-20250929

# Processing Configuration
GPU_ENABLED=true
MAX_PARALLEL_WORKERS=4

# Storage
OUTPUT_DIR=./outputs
UPLOAD_DIR=./uploads
LOG_DIR=./logs
EOF

# Create directories
echo "📁 Creating directories..."
mkdir -p outputs uploads logs temp

# Verify GPU
echo "🔍 Verifying GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
fi

# Test CUDA
echo "🔍 Testing CUDA with PyTorch..."
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: CUDA not available!")
PYEOF

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env if needed: nano .env"
echo "  2. Run pipeline:"
echo "     source venv/bin/activate"
echo '     python processing/smart_pipeline.py --video-url "https://youtube.com/watch?v=VIDEO_ID"'
echo ""
echo "  3. When done, stop instance to save money:"
echo "     # From AWS Console: Actions → Instance State → Stop"
echo ""
