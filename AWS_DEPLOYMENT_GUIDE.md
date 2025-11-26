# AWS GPU Deployment Guide

## 🎯 Recommended Instance: g4dn.xlarge

**GPU**: NVIDIA T4 (16GB VRAM)
**Cost**: $0.526/hour (~$0.18-0.22 per video for 20-25 min)
**Total cost/video**: $2.38-4.12 (GPU + LLM APIs)

---

## 🚀 Quick Start (30 minutes)

### Step 1: Launch EC2 Instance

1. **Go to AWS Console**: https://console.aws.amazon.com/ec2/

2. **Click "Launch Instance"**

3. **Configure**:
   ```
   Name: video-qa-processor

   AMI: Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 22.04)
        (Search for: "Deep Learning AMI GPU PyTorch")

   Instance type: g4dn.xlarge

   Key pair: Create new or select existing
             (Download .pem file if creating new)

   Storage: 100 GB gp3

   Security group:
     - Allow SSH (port 22) from My IP
     - (Optional) Allow HTTP (port 8000) if running API
   ```

4. **Click "Launch Instance"**

5. **Wait 2-3 minutes** for instance to start

---

### Step 2: Connect to Instance

#### Option A: EC2 Instance Connect (Browser)
1. Select your instance
2. Click "Connect"
3. Click "EC2 Instance Connect" → "Connect"
4. Browser terminal opens ✅

#### Option B: SSH from Terminal
```bash
# Make key file secure
chmod 400 your-key.pem

# Connect
ssh -i your-key.pem ubuntu@YOUR_INSTANCE_PUBLIC_IP

# Find IP in AWS Console under "Public IPv4 address"
```

---

### Step 3: One-Command Setup

Once connected, run this **single command**:

```bash
curl -sSL https://raw.githubusercontent.com/Ranjan-Aniket/Video_QA/main/aws_setup.sh | bash
```

**Or manually**:
```bash
# Clone and run setup
git clone https://github.com/Ranjan-Aniket/Video_QA.git
cd Video_QA
chmod +x aws_setup.sh
./aws_setup.sh
```

This script will:
- ✅ Install all system dependencies (ffmpeg, tesseract, etc.)
- ✅ Install Python dependencies (~5-10 min)
- ✅ Download ML models (spacy)
- ✅ Create .env with your API keys
- ✅ Verify GPU is working
- ✅ Create output directories

---

### Step 4: Process Videos

```bash
# Activate virtual environment
source venv/bin/activate

# Process single video
python processing/smart_pipeline.py \
  --video-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Process from batch file
python processing/smart_pipeline.py \
  --batch-file videos.txt
```

**Output**: Results saved to `./outputs/video_TIMESTAMP_ID/`

---

### Step 5: Download Results

#### Option A: SCP (from your laptop)
```bash
# Download all outputs
scp -i your-key.pem -r \
  ubuntu@YOUR_INSTANCE_IP:/home/ubuntu/Video_QA/outputs \
  ./local-outputs/

# Download specific video
scp -i your-key.pem -r \
  ubuntu@YOUR_INSTANCE_IP:/home/ubuntu/Video_QA/outputs/video_20251126_123456 \
  ./
```

#### Option B: AWS S3
```bash
# On EC2 instance
pip install awscli
aws configure  # Enter your AWS credentials

# Upload to S3
aws s3 cp outputs/ s3://your-bucket/video-qa-outputs/ --recursive

# Download from your laptop
aws s3 cp s3://your-bucket/video-qa-outputs/ ./local-outputs/ --recursive
```

---

### Step 6: Stop Instance (Important! 💰)

**When done processing**, stop instance to avoid charges:

#### From AWS Console:
1. Select instance
2. **Actions** → **Instance State** → **Stop**

#### From Terminal:
```bash
# Get instance ID
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)

# Stop instance
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

**⚠️ IMPORTANT**:
- Stopped instances don't charge for compute ($0.526/hr saved)
- Storage still charges ($0.08/GB-month for 100GB = ~$8/month)
- Terminate instance if you don't need it anymore (deletes everything)

---

## 📊 Cost Breakdown

### Per Video (20-25 minutes on g4dn.xlarge):
- **GPU/Compute**: $0.18-0.22
- **LLM APIs** (GPT-4o, Claude, Gemini): $2.20-3.90
- **Total**: **$2.38-4.12**

### Daily Costs:
- **10 videos**: ~$24-41
- **20 videos**: ~$48-82
- **50 videos**: ~$119-206

### Instance Running Costs:
- **Per hour**: $0.526
- **Per day** (24h): $12.62
- **Per month** (stopped, just storage): ~$8

**💡 Tip**: Only run when processing, stop immediately after!

---

## 🐳 Docker Deployment (Optional)

If you prefer Docker:

```bash
# Build image
docker build -t video-qa-pipeline .

# Run with GPU
docker run --gpus all -it \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  -v $(pwd)/outputs:/app/outputs \
  video-qa-pipeline \
  python3 processing/smart_pipeline.py \
    --video-url "https://youtube.com/..."
```

---

## 🔍 Verify GPU Working

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA in PyTorch
python3 << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

**Expected output (g4dn.xlarge)**:
```
CUDA available: True
GPU: Tesla T4
VRAM: 15.7 GB
```

---

## 🛠️ Troubleshooting

### "CUDA not available"
```bash
# Check if on GPU instance
nvidia-smi

# If not found, you're on wrong instance type
# Need g4dn or g5 instance, not t2/t3/m5
```

### "Out of memory"
- g4dn.xlarge (16GB) should be enough
- If hitting OOM, reduce parallel workers in .env:
  ```
  MAX_PARALLEL_WORKERS=2
  ```

### "Module not found"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements-aws-gpu.txt
```

### "FFmpeg not found"
```bash
sudo apt-get install -y ffmpeg
```

### "Can't download YouTube video"
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

---

## 📁 File Structure on Instance

```
/home/ubuntu/Video_QA/
├── processing/           # Pipeline code
├── templates/           # Question templates
├── backend/             # API server
├── outputs/             # Generated results
│   └── video_20251126_123456/
│       ├── frames/
│       ├── phase8_questions.json
│       └── pipeline_results.json
├── venv/                # Python environment
├── .env                 # API keys
├── requirements-aws-gpu.txt
├── aws_setup.sh
└── Dockerfile
```

---

## 🔄 Update Code from GitHub

```bash
cd Video_QA
git pull origin main
source venv/bin/activate
pip install -r requirements-aws-gpu.txt  # If dependencies changed
```

---

## 📊 Monitor Processing

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch logs
tail -f logs/pipeline.log

# Monitor system
htop
```

---

## ⚡ Performance Tips

### 1. Use Instance Store (Faster I/O)
g4dn.xlarge has 125GB NVMe SSD:
```bash
# Mount instance store
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir /mnt/fast
sudo mount /dev/nvme1n1 /mnt/fast
sudo chown ubuntu:ubuntu /mnt/fast

# Use for temporary files
export TEMP_DIR=/mnt/fast/temp
```

### 2. Process Multiple Videos in Parallel
```bash
# Create videos.txt
cat > videos.txt << 'EOF'
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
https://youtube.com/watch?v=VIDEO3
EOF

# Process all (parallel)
python processing/smart_pipeline.py --batch-file videos.txt
```

### 3. Use Spot Instances (70% cheaper!)
- Same g4dn.xlarge for ~$0.16/hr (vs $0.526/hr)
- Can be interrupted (AWS takes it back)
- Good for non-urgent batch processing

---

## 🎯 Next Steps

1. ✅ Launch g4dn.xlarge instance
2. ✅ Run `./aws_setup.sh`
3. ✅ Process first video
4. ✅ Download results
5. ✅ **Stop instance**

**Total time**: ~30-40 minutes for first video (including setup)

---

## 📞 Need Help?

Check these files:
- `aws_setup.sh` - Automated setup script
- `requirements-aws-gpu.txt` - All dependencies
- `Dockerfile` - Docker deployment
- `.env` - Configuration (API keys)

Happy processing! 🚀
