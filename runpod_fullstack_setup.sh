#!/bin/bash
# RunPod Full Stack Application Setup
# Backend (FastAPI) + Frontend (React/Vite)

set -e

echo "========================================="
echo "Full Stack Video QA Application Setup"
echo "========================================="
echo ""

# Check if in correct directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: backend/main.py not found"
    echo "Run: cd /workspace/Video_QA"
    exit 1
fi

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "📦 Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies (2-3 minutes)..."
cd frontend
npm install
cd ..

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "To start the application:"
echo ""
echo "1️⃣  Start Backend (Terminal 1):"
echo "   cd /workspace/Video_QA"
echo "   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "2️⃣  Start Frontend (Terminal 2):"
echo "   cd /workspace/Video_QA/frontend"
echo "   npm run dev -- --host 0.0.0.0 --port 3000"
echo ""
echo "3️⃣  Access UI from your browser:"
echo "   http://RUNPOD_PUBLIC_IP:3000"
echo ""
echo "Make sure ports 8000 and 3000 are exposed in RunPod settings!"
echo ""
