# RunPod GPU Setup Guide

**Alternative to AWS** - No quota restrictions, instant access!

---

## 🚀 Quick Start (20 minutes)

### Step 1: Sign Up

1. Go to: https://runpod.io
2. Click "Sign Up"
3. Add payment method (credit card)
4. Add credits ($10 minimum, lasts 20-30 videos)

### Step 2: Deploy GPU Pod

1. **Click**: "Deploy" or "Rent GPU"

2. **Select GPU** (choose one):
   - **RTX A4000** (16GB) - $0.34/hr ✅ Best value
   - **RTX 4090** (24GB) - $0.44/hr (faster)
   - **A5000** (24GB) - $0.50/hr

3. **Select Template**:
   - Search: "PyTorch"
   - Choose: "PyTorch 2.1" or "RunPod PyTorch"

4. **Configure**:
   ```
   GPU Count: 1
   Container Disk: 50 GB
   Volume Disk: 50 GB (persistent storage)
   Expose HTTP Ports: 8000 (optional, for API)
   ```

5. **Click**: "Deploy On-Demand"

6. **Wait**: ~2-3 minutes for pod to start

### Step 3: Connect

1. **Click**: "Connect" on your pod

2. **Copy SSH command**:
   ```bash
   ssh root@X.X.X.X -p 12345 -i ~/.ssh/id_ed25519
   ```

3. **Or use Web Terminal**: Click "Start Web Terminal"

### Step 4: Setup Pipeline

Once connected:

```bash
# Clone repository
git clone https://github.com/Ranjan-Aniket/Video_QA.git
cd Video_QA

# Install dependencies
pip install -r requirements-aws-gpu.txt

# Download models
python -m spacy download en_core_web_sm

# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=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
ANTHROPIC_API_KEY=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
GPU_ENABLED=true
EOF

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 5: Process Video

```bash
python processing/smart_pipeline.py \
  --video-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Step 6: Download Results

**Option A: From Web UI**
1. Go to RunPod dashboard
2. Click your pod → "Files"
3. Navigate to `/Video_QA/outputs/`
4. Download results

**Option B: SCP**
```bash
# From your laptop
scp -P 12345 -r \
  root@X.X.X.X:/Video_QA/outputs/video_* \
  ./local-outputs/
```

### Step 7: Stop Pod (Save Money!)

**Important**: RunPod charges by the second, but stop when done:

1. Click "Stop" on your pod (not Terminate)
2. Or from terminal: `runpodctl stop pod POD_ID`

**Stopped pods**:
- ✅ Don't charge for compute
- ✅ Keep your data on volume disk
- ✅ Can restart anytime

---

## 💰 Cost Breakdown

### Per Video (RTX A4000, 20-25 min):
- **GPU**: $0.34/hr × 0.33-0.42hr = $0.11-0.14
- **LLM APIs**: $2.20-3.90
- **Total**: **$2.31-4.04**

### Per Video (RTX 4090, 15-20 min):
- **GPU**: $0.44/hr × 0.25-0.33hr = $0.11-0.15
- **LLM APIs**: $2.20-3.90
- **Total**: **$2.31-4.05**

**Cheaper than AWS!** ($2.38-4.12 on g4dn.xlarge)

---

## 🔧 Troubleshooting

### "No GPUs available"
- Try different region/datacenter
- Or wait a few minutes and retry
- Use "Spot" instances (70% cheaper, can be interrupted)

### "Out of credits"
- Add more credits in account settings
- Minimum: $10

### "Can't connect via SSH"
- Use Web Terminal instead (works always)
- Or check firewall settings

### "CUDA not found"
- Make sure you selected GPU template (not CPU)
- Verify: `nvidia-smi` should show GPU

---

## 📊 RunPod vs AWS

| Feature | RunPod | AWS g4dn.xlarge |
|---------|--------|-----------------|
| **GPU** | RTX A4000 (16GB) | T4 (16GB) |
| **Cost/hr** | $0.34 | $0.53 |
| **Setup time** | 5 min | 10 min |
| **Quota** | ✅ None | ⚠️ May need request |
| **Instant access** | ✅ Yes | ⚠️ Maybe |
| **Pay/second** | ✅ Yes | ❌ Pay/hour |
| **Cost/video** | $2.31-4.04 | $2.38-4.12 |

**Winner**: RunPod (cheaper, no quotas, instant access)

---

## 🎯 Recommendation

**Use RunPod if**:
- ✅ AWS/GCP quota is 0
- ✅ Want instant access
- ✅ Want cheaper option
- ✅ Don't want to wait for quota approval

**Use AWS if**:
- ✅ Already have AWS account with quota
- ✅ Want AWS ecosystem integration
- ✅ Need enterprise features

---

## 🔄 Persistent Workflow

### Save Money with Volumes

1. **Create volume** (one time): 50GB persistent storage
2. **Attach to pod**: Mounts at `/workspace`
3. **Store code & outputs** on volume
4. **Stop pod**: Volume persists
5. **Restart later**: Reattach same volume

**Benefit**: Pay $0.10/GB-month for storage (~$5/month) instead of keeping pod running

---

## 📚 Resources

- **RunPod Docs**: https://docs.runpod.io
- **Community**: https://discord.gg/runpod
- **Templates**: https://runpod.io/console/explore

---

**Ready to try?** Sign up at https://runpod.io and start in 10 minutes! 🚀
