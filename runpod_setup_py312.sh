#!/bin/bash
# RunPod Setup Script for Python 3.12
# Run this in RunPod terminal after cloning repository

set -e  # Exit on error

echo "========================================="
echo "RunPod Video QA Pipeline Setup (Py 3.12)"
echo "========================================="
echo ""

# Check Python version
python3 --version

# Check if in correct directory
if [ ! -f "requirements-runpod-py312.txt" ]; then
    echo "❌ Error: Not in Video_QA directory or requirements file missing"
    echo "Run: cd Video_QA"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies (this will take 5-10 minutes)..."
pip install -q -r requirements-runpod-py312.txt

# Download spacy model
echo "📦 Downloading spacy model..."
python -m spacy download en_core_web_sm

# Create .env file
echo "⚙️  Creating .env file..."
cat > .env << 'EOF'
# Database (SQLite - no PostgreSQL needed!)
DATABASE_URL=sqlite:///./video_qa.db

# API Keys
OPENAI_API_KEY=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
ANTHROPIC_API_KEY=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

# Configuration
GPU_ENABLED=true
MAX_PARALLEL_WORKERS=4
OUTPUT_DIR=./outputs
EOF

# Create directories
echo "📁 Creating directories..."
mkdir -p outputs uploads logs temp

# Verify GPU
echo ""
echo "🔍 Verifying GPU..."
nvidia-smi

echo ""
echo "🐍 Testing PyTorch CUDA..."
python3 << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "⚠️  NOTE: pyannote.audio was skipped (Python 3.12 incompatible)"
echo "   Audio diarization won't work, but Whisper transcription will"
echo ""
echo "Process a video:"
echo '  python processing/smart_pipeline.py --video-url "https://youtube.com/watch?v=VIDEO_ID"'
echo ""
echo "Example:"
echo '  python processing/smart_pipeline.py --video-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"'
echo ""
