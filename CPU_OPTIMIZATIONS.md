# CPU-Only Optimization Guide

## ⚠️ Reality Check

**CPU-only will NOT hit 20-25 minutes.** Even with optimizations, expect **60-120 minutes minimum** per video.

---

## 🔧 Optimizations to Reduce Time

### 1. Use Smaller Models

**Current (GPU)**:
```python
# Whisper
model = whisper.load_model("base")  # 74M params

# CLIP
model = "openai/clip-vit-base-patch32"  # 151M params
```

**CPU-Optimized**:
```python
# Whisper - Use tiny model
model = whisper.load_model("tiny")  # 39M params (2-3x faster)

# CLIP - Use smaller variant
model = "openai/clip-vit-base-patch16"  # Smaller, faster

# YOLO - Use nano
model = YOLO('yolov8n.pt')  # Already using smallest
```

### 2. Reduce Frame Sampling

**Current**: 2 fps (120 frames/minute)
**CPU-Optimized**: 0.5 fps (30 frames/minute) = 4x fewer frames

```python
# In .env or config
FRAME_EXTRACTION_FPS=0.5  # Instead of 2
```

### 3. Skip Heavy Models

Disable non-essential processing:

```python
# In .env
ENABLE_POSE_DETECTION=false  # Skip MediaPipe
ENABLE_EMOTION_DETECTION=false  # Skip FER
ENABLE_SCENE_DETECTION=false  # Use simple frame sampling
```

### 4. Parallel Processing

```python
# Use all CPU cores
MAX_PARALLEL_WORKERS=32  # For c6i.8xlarge (32 vCPUs)
```

### 5. Batch Processing

Process multiple videos sequentially to amortize startup costs.

---

## 🚀 Best CPU Instance for Optimized Pipeline

### AWS c6i.8xlarge (32 vCPUs)
- **Cost**: $1.36/hour
- **With optimizations**: ~60-90 minutes per video
- **Total cost**: $1.36-2.04 + $2.20-3.90 = **$3.56-5.94**
- **Still 3-4x slower than GPU but closer in cost**

---

## 📋 CPU Installation

```bash
# Use CPU requirements
pip install -r requirements.txt  # Not requirements-aws-gpu.txt

# Install CPU-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 💡 Hybrid Approach (Best of Both Worlds)

### Run Expensive Tasks on GPU, Others on CPU

1. **GPU Instance** (g4dn.xlarge): Run CLIP, Whisper, YOLO
   - Extract frames
   - Run all ML models
   - Save embeddings/detections
   - Time: ~10 minutes
   - Cost: ~$0.09

2. **CPU Instance** (t3.xlarge): Run LLM API calls
   - Process saved embeddings
   - Generate questions with Claude/GPT
   - Time: ~10-15 minutes
   - Cost: ~$0.03

**Total**: ~20-25 minutes, **$2.32-4.02** (similar to GPU-only but more complex)

---

## 🎯 Recommendation

### Don't Use CPU-Only

Here's why:

| Metric | GPU (g4dn.xlarge) | CPU (c6i.8xlarge optimized) |
|--------|-------------------|------------------------------|
| Time | 20-25 min ✅ | 60-90 min ❌ |
| Cost | $2.38-4.12 ✅ | $3.56-5.94 ❌ |
| Setup | Simple ✅ | Complex optimizations ❌ |
| Quality | Full models ✅ | Smaller models ❌ |

**CPU is slower AND more expensive!**

---

## 🔧 If You Must Use CPU

1. Use **c6i.8xlarge** ($1.36/hr, 32 vCPUs)
2. Apply all optimizations above
3. Expect **60-90 minutes** per video
4. Cost: **$3.56-5.94** per video
5. Lower quality (smaller models)

---

## ✅ Stick with GPU

**g4dn.xlarge is your best option:**
- ✅ Faster (20-25 min vs 60-90 min)
- ✅ Cheaper ($2.38-4.12 vs $3.56-5.94)
- ✅ Better quality (full models)
- ✅ Simpler setup

**There's no good reason to use CPU for this workload.**
