# Video QA Generation Pipeline

Automated video question-answer generation system using GPT-4o, Claude Sonnet 4.5, and Gemini with GPU-accelerated processing.

## 🎯 Overview

This pipeline processes videos to generate high-quality question-answer pairs:
- **Input**: YouTube URL or video file
- **Output**: 30-35 context-aware Q&A pairs per video
- **Processing Time**: 15-25 minutes per video
- **Cost**: $2.25-4.50 per video

## 🚀 Quick Start

### Option 1: AWS GPU (Recommended)

```bash
# 1. Launch g4dn.xlarge instance on AWS
# 2. SSH into instance
# 3. Run automated setup:
curl -sSL https://raw.githubusercontent.com/Ranjan-Aniket/Video_QA/main/aws_setup.sh | bash

# 4. Process video:
source venv/bin/activate
python processing/smart_pipeline.py \
  --video-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

**See**: [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md) for detailed instructions.

### Option 2: Modal Serverless GPU

```bash
# 1. Clone repository
git clone https://github.com/Ranjan-Aniket/Video_QA.git
cd Video_QA

# 2. Install Modal
pip install modal

# 3. Authenticate
modal setup

# 4. Deploy
modal deploy modal_pipeline.py

# 5. Process video
modal run modal_pipeline.py --video-url "https://youtube.com/..."
```

## 📋 Features

### Multi-Pass Pipeline
- **Pass 1**: Smart frame filtering with CLIP and visual analysis
- **Pass 2A**: Moment selection with Claude Sonnet 4.5
- **Pass 2B**: Complex question type generation with Claude Opus 4
- **Pass 3**: Q&A generation with Claude Sonnet 4.5
- **Phase 8**: Vision-based questions with GPT-4o
- **Phase 9**: Validation with Gemini

### Question Types (20+ Categories)
- Temporal & Sequential reasoning
- Counting & Quantification
- Comparative inference
- Needle & Referential context
- Subscene holistic stitching
- Spurious correlations
- And more...

### GPU-Accelerated Processing
- CLIP (image-text alignment)
- YOLOv8 (object detection)
- Whisper (audio transcription)
- OCR (text extraction)
- Scene detection
- Pose estimation

## 🔧 Requirements

### GPU Requirements
- **Minimum**: 16GB VRAM (NVIDIA T4 or better)
- **Recommended**: 24GB VRAM (NVIDIA A10G)

### API Keys Required
- OpenAI API key (GPT-4o, Whisper)
- Anthropic API key (Claude Sonnet 4.5, Opus 4)
- Google API key (Gemini) - optional

### System Dependencies
- Python 3.9-3.11
- FFmpeg
- CUDA 12.1+ (for GPU)

## 📦 Installation

### AWS GPU Instance
```bash
./aws_setup.sh
```

### Docker
```bash
docker build -t video-qa-pipeline .
docker run --gpus all -it \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  -v $(pwd)/outputs:/app/outputs \
  video-qa-pipeline
```

### Manual Installation
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr git

# Install Python dependencies
pip install -r requirements-aws-gpu.txt

# Download models
python -m spacy download en_core_web_sm
```

## 💰 Cost Breakdown

### Per Video (20-25 minutes on g4dn.xlarge):
| Component | Cost |
|-----------|------|
| GPU (T4, 0.4hr) | $0.21 |
| LLM APIs | $2.20-3.90 |
| **Total** | **$2.41-4.11** |

### AWS Instance Options
| Instance | GPU | VRAM | Cost/hr | Best For |
|----------|-----|------|---------|----------|
| g4dn.xlarge | T4 | 16GB | $0.526 | Budget ✅ |
| g5.xlarge | A10G | 24GB | $1.006 | Speed |

## 📊 Output Format

Each video generates:

```
outputs/video_20251126_123456_VIDEO_ID/
├── frames/                          # Extracted frames (2fps)
├── phase1_scenes.json              # Scene boundaries
├── phase5_frame_selection.json     # Selected key frames
├── phase8_questions.json           # Generated questions
├── phase9_gemini_results.json      # Validation results
└── pipeline_results.json           # Complete output
```

### Sample Output
```json
{
  "video_id": "VIDEO_ID",
  "total_questions": 32,
  "questions": [
    {
      "id": "q_001",
      "question": "What action does the person perform after picking up the red ball?",
      "answer": "They throw it into the basket",
      "type": "temporal_sequential",
      "frames": [145, 156, 167],
      "confidence": 0.95
    }
  ]
}
```

## 🛠️ Configuration

Edit `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=...  # Optional

# Processing
GPU_ENABLED=true
MAX_PARALLEL_WORKERS=4

# Quality
MIN_CONFIDENCE_THRESHOLD=0.95
TARGET_QUESTIONS_PER_VIDEO=32
```

## 📚 Documentation

- [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md) - Complete AWS setup guide
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - Repository setup instructions
- [Dockerfile](Dockerfile) - Docker deployment
- [aws_setup.sh](aws_setup.sh) - Automated AWS setup script

## 🏗️ Architecture

```
Video Input
    ↓
Phase 1: Scene Detection (scenedetect)
    ↓
Phase 2: Visual Sampling (2fps, CLIP, YOLO)
    ↓
Pass 1: Smart Frame Filtering (free models)
    ↓
Pass 2A: Moment Selection (Claude Sonnet 4.5)
    ↓
Pass 2B: Complex Types (Claude Opus 4)
    ↓
Pass 3: Q&A Generation (Claude Sonnet 4.5)
    ↓
Phase 8: Vision Questions (GPT-4o)
    ↓
Phase 9: Validation (Gemini)
    ↓
Output: 32 Q&A pairs
```

## 🔍 Quality Assurance

- Template-based validation
- Hallucination filtering
- Answer grounding verification
- Hedging language detection
- Minimum confidence thresholds
- Multi-model consensus

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

## 📝 License

Private repository - All rights reserved.

## 👥 Team

- **Ranjan-Aniket** - Repository Owner
- **Jitendra-DataScientist** - Collaborator

## 🐛 Issues & Support

Report issues: https://github.com/Ranjan-Aniket/Video_QA/issues

## 🎯 Roadmap

- [ ] Add support for longer videos (>1 hour)
- [ ] Implement question difficulty scoring
- [ ] Add multi-language support
- [ ] Batch processing API
- [ ] Real-time processing dashboard

## 📊 Performance

| Metric | Value |
|--------|-------|
| Processing Speed | 15-25 min/video |
| Questions/Video | 30-35 |
| GPU Memory | ~12GB peak |
| Accuracy | >95% (validated) |
| Cost | $2.41-4.11/video |

---

**Ready to process videos?** Start with [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md)! 🚀
