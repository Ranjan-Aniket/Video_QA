# Push to GitHub - Quick Guide

## Step 1: Create GitHub Repository

1. **Go to**: https://github.com/new
2. **Repository name**: `Gemini_QA` (or any name you prefer)
3. **Visibility**:
   - ✅ **Private** (recommended - contains your pipeline)
   - ⚠️ Public (anyone can see, but .env is excluded)
4. **Don't initialize** with README, .gitignore, or license (we have them)
5. Click **"Create repository"**

## Step 2: Add Remote and Push

GitHub will show you commands. Copy the SSH or HTTPS URL, then run:

### Option A: HTTPS (Easier, works with Zscaler)
```bash
cd /Users/aranja14/Desktop/Gemini_QA

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Gemini_QA.git

# Push to GitHub
git push -u origin master
```

### Option B: SSH (Faster, but may need setup)
```bash
cd /Users/aranja14/Desktop/Gemini_QA

# Add remote (replace YOUR_USERNAME)
git remote add origin git@github.com:YOUR_USERNAME/Gemini_QA.git

# Push to GitHub
git push -u origin master
```

**If using SSH for first time**, you'll need to add SSH key:
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys
```

## Step 3: Verify

After pushing, go to:
```
https://github.com/YOUR_USERNAME/Gemini_QA
```

You should see:
- ✅ modal_pipeline.py
- ✅ processing/ folder
- ✅ templates/ folder
- ✅ requirements.txt
- ❌ .env (correctly excluded!)

## Step 4: Clone on Other Device

### On Personal Laptop/Computer:
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Gemini_QA.git
cd Gemini_QA

# Create .env file with your API keys
cat > .env << 'EOF'
OPENAI_API_KEY=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
ANTHROPIC_API_KEY=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
EOF

# Install Modal
pip install modal

# Authenticate with Modal
modal setup

# Deploy!
modal deploy modal_pipeline.py
```

## Troubleshooting

### "Authentication failed"
- HTTPS: GitHub may ask for Personal Access Token instead of password
- Go to: https://github.com/settings/tokens
- Generate new token with `repo` scope
- Use token as password when pushing

### "Permission denied (publickey)"
- You're using SSH but key not configured
- Either: Add SSH key to GitHub (see above)
- Or: Use HTTPS instead (easier)

### "Repository not found"
- Check spelling of repository name
- Make sure you created the repo on GitHub first
- Check if you're logged into correct GitHub account

---

## What's Committed (Safe)

✅ All Python code
✅ Modal deployment config
✅ Processing pipeline
✅ Templates
✅ Requirements.txt

## What's Excluded (Secure)

❌ .env (API keys - NOT in git!)
❌ .modal/ (tokens)
❌ outputs/ (generated files)
❌ .venv/ (Python environment)
❌ Test files
❌ Deployment scripts with tokens

**Your API keys are safe!** They're in .gitignore and won't be pushed.

---

Ready to push? Run the commands above!
