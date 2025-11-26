# RunPod Full Stack Application Guide

**Backend (FastAPI) + Frontend (React/Vite) on RunPod**

---

## 🚀 Quick Start

### Step 1: Configure RunPod Ports

**Before starting, expose these ports in RunPod:**

1. Go to your RunPod pod settings
2. Under **"Expose HTTP Ports"**, add:
   - `8000` (Backend API)
   - `3000` (Frontend UI)
3. Click "Save"
4. Restart pod if needed

---

### Step 2: Install Frontend Dependencies

**In RunPod terminal:**

```bash
cd /workspace/Video_QA

# Make setup script executable
chmod +x runpod_fullstack_setup.sh

# Run setup (installs Node.js + npm dependencies)
./runpod_fullstack_setup.sh
```

**Time:** 3-5 minutes

---

### Step 3: Start Backend

**Terminal 1 (or use screen/tmux):**

```bash
cd /workspace/Video_QA

# Start FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**You should see:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 4: Start Frontend

**Terminal 2 (open new terminal or use screen/tmux):**

```bash
cd /workspace/Video_QA/frontend

# Start Vite dev server
npm run dev -- --host 0.0.0.0 --port 3000
```

**You should see:**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:3000/
➜  Network: http://0.0.0.0:3000/
```

---

### Step 5: Access the UI

**From your laptop browser:**

1. **Get your RunPod public URL**:
   - Go to RunPod dashboard
   - Click "Connect" on your pod
   - Find "HTTP Service" URLs for ports 3000 and 8000

2. **Open frontend**:
   - `https://XXXXXXXX-3000.proxy.runpod.net`

3. **The frontend should auto-connect to backend** at:
   - `https://XXXXXXXX-8000.proxy.runpod.net`

---

## 📤 Using the Application

1. **Upload Video**: Click "Upload Video" button
2. **Select Video**: Choose your `.mp4` file
3. **Start Processing**: Click "Process Video"
4. **Monitor Progress**: Watch real-time progress in UI
5. **Download Results**: Download JSON when complete

---

## 🔧 Using Screen for Background Processes

**To keep processes running when you close terminal:**

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start backend in screen
screen -S backend
cd /workspace/Video_QA
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Press Ctrl+A then D to detach

# Start frontend in screen
screen -S frontend
cd /workspace/Video_QA/frontend
npm run dev -- --host 0.0.0.0 --port 3000
# Press Ctrl+A then D to detach

# List screens
screen -ls

# Reattach to screen
screen -r backend  # or frontend
```

---

## 🐛 Troubleshooting

### "Connection refused" on frontend

**Check backend is running:**
```bash
curl http://localhost:8000/health
```

**Update frontend API URL if needed:**
```bash
# Check frontend/.env or vite.config.ts
# Make sure VITE_API_URL points to backend port 8000
```

### "Port already in use"

**Kill existing process:**
```bash
# Find process on port 8000
lsof -ti:8000 | xargs kill -9

# Find process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Can't access UI from browser

1. Verify ports 3000 and 8000 are exposed in RunPod settings
2. Use the proxy URLs from RunPod dashboard
3. Try accessing via direct IP: `http://RUNPOD_IP:3000`

---

## 💰 Cost Optimization

**Stop pod when not processing:**
1. Close browser
2. Stop backend: Ctrl+C in terminal 1
3. Stop frontend: Ctrl+C in terminal 2
4. RunPod dashboard → "Stop" pod

**Stopped pods:**
- ✅ Don't charge for GPU
- ✅ Keep your data
- ✅ Can restart anytime

---

## 📊 Expected Performance

**On RTX A4000:**
- **Processing**: 20-25 min per video
- **GPU Cost**: $0.11-0.14 per video
- **Total Cost**: $2.31-4.04 per video (GPU + LLM APIs)

---

## 🎯 Alternative: Local Laptop → RunPod Backend

**If you prefer running UI on your laptop:**

1. **On RunPod**: Only run backend
2. **On Laptop**: Run frontend locally
3. **Configure frontend** to connect to RunPod backend URL

```bash
# On laptop
cd frontend
echo "VITE_API_URL=https://XXXXXXXX-8000.proxy.runpod.net" > .env.local
npm run dev
```

Open `http://localhost:3000` on your laptop.

---

Ready to start? Run the setup script! 🚀
