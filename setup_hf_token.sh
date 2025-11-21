#!/bin/bash
# Secure HuggingFace Token Setup Script

echo "=========================================="
echo "HuggingFace Token Setup"
echo "=========================================="
echo ""
echo "1. Visit: https://huggingface.co/settings/tokens"
echo "2. Create a new token with READ access"
echo "3. Copy the token (starts with hf_...)"
echo ""
echo -n "Paste your HuggingFace token here: "
read -s HF_TOKEN
echo ""

if [[ -z "$HF_TOKEN" ]]; then
    echo "❌ No token provided. Exiting."
    exit 1
fi

# Validate token format
if [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
    echo "⚠️  Warning: Token doesn't start with 'hf_'. Are you sure this is correct?"
    echo -n "Continue anyway? (y/N): "
    read CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

# Add to .env file
ENV_FILE="/Users/aranja14/Desktop/Gemini_QA/.env"

# Check if HF_TOKEN already exists in .env
if grep -q "^HF_TOKEN=" "$ENV_FILE" 2>/dev/null; then
    echo "Updating existing HF_TOKEN in .env..."
    # Use sed to replace the line (macOS compatible)
    sed -i '' "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" "$ENV_FILE"
else
    echo "Adding HF_TOKEN to .env..."
    echo "" >> "$ENV_FILE"
    echo "# HuggingFace Token for Speaker Diarization" >> "$ENV_FILE"
    echo "HF_TOKEN=$HF_TOKEN" >> "$ENV_FILE"
fi

echo ""
echo "✅ HuggingFace token saved to .env file!"
echo ""
echo "Testing authentication..."

# Test the token
cd /Users/aranja14/Desktop/Gemini_QA
source .venv/bin/activate
export HF_TOKEN="$HF_TOKEN"

python << EOF
from huggingface_hub import HfApi
import os

try:
    api = HfApi(token=os.getenv('HF_TOKEN'))
    user_info = api.whoami()
    print(f"✅ Authentication successful!")
    print(f"   Logged in as: {user_info['name']}")
    print(f"   Type: {user_info.get('type', 'user')}")
    print("")
    print("🎉 Speaker diarization is now enabled!")
    print("   You'll get labeled speakers: SPEAKER_01, SPEAKER_02, etc.")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    print("Please check your token and try again.")
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
