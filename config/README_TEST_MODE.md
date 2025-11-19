# Test Mode Configuration

## Overview

The HITL Evidence Review system supports both **Test Mode** (free) and **Production Mode** (paid AI APIs). This allows you to test the complete workflow without incurring costs.

---

## 🧪 Test Mode (Default) - **FREE**

**Current Status:** ✅ **Already Enabled**

By default, GPT-4 and Claude API calls are **disabled** and replaced with **mock predictions**. This allows you to:

- ✅ Test the complete HITL review workflow
- ✅ Train reviewers on the interface
- ✅ Validate consensus logic
- ✅ Debug without API costs

**Cost:** $0.00 (uses free local models only)

---

## 💰 Production Mode - **PAID APIs**

When you're ready for production, enable expensive AI APIs:

**Cost per video:**
- GPT-4 Vision: ~$0.01-0.05 per image
- Claude Sonnet 4.5 Vision: ~$0.01 per image
- GPT-4-mini (questions): ~$0.03 per question

---

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
# Test Mode (default - FREE)
ENABLE_GPT4_VISION=false
ENABLE_CLAUDE_VISION=false
ENABLE_GPT4_MINI=false
USE_MOCK_PREDICTIONS=true
MAX_API_COST=0.0

# Production Mode (paid)
# ENABLE_GPT4_VISION=true
# ENABLE_CLAUDE_VISION=true
# ENABLE_GPT4_MINI=true
# USE_MOCK_PREDICTIONS=false
# MAX_API_COST=5.0
```

### Method 2: Direct Code Configuration

Edit `/config/test_config.py`:

```python
TEST_CONFIG = TestConfig(
    # AI Model Flags
    enable_gpt4_vision=False,      # Set to True to enable GPT-4 Vision
    enable_claude_vision=False,    # Set to True to enable Claude Vision
    enable_gpt4_mini_generation=False,  # Set to True for GPT-4-mini questions

    # Mock Predictions
    use_mock_predictions=True,     # Keep True for testing
    mock_consensus="full_agreement",  # Options: "full_agreement", "majority", "disagreement"

    # Free Local Models (always enabled)
    enable_yolo=True,
    enable_ocr=True,
    enable_whisper=True,
)
```

---

## Models Used

### Free Local Models (Always Enabled)

These models run locally and are **100% free**:

| Model | Purpose | Cost |
|-------|---------|------|
| **YOLOv8n** | Object detection | FREE |
| **EasyOCR / PaddleOCR** | Text extraction | FREE |
| **Whisper** | Audio transcription | FREE |
| **OpenCV** | Frame extraction | FREE |
| **Scene Detector** | Scene classification | FREE |

### Paid Cloud APIs (Disabled by Default)

These require API keys and incur costs:

| Model | Purpose | Cost | Status |
|-------|---------|------|--------|
| **GPT-4 Vision** | Visual analysis | ~$0.03/image | ❌ Disabled (default) |
| **Claude Sonnet 4.5 Vision** | Visual analysis | ~$0.01/image | ❌ Disabled (default) |
| **GPT-4-mini** | Question generation | ~$0.03/question | ❌ Disabled (default) |

---

## Mock Prediction Modes

When `use_mock_predictions=True`, you can simulate different consensus scenarios:

### 1. Full Agreement (Default)
```python
mock_consensus="full_agreement"
```
- All 3 AI models agree
- High confidence (0.95)
- Low priority for human review

### 2. Majority Consensus
```python
mock_consensus="majority"
```
- 2 out of 3 models agree
- Medium confidence (0.85)
- Medium priority for human review

### 3. Disagreement
```python
mock_consensus="disagreement"
```
- All 3 models disagree
- Low confidence (0.60)
- High priority for human review

---

## Verifying Configuration

Run this command to see your current configuration:

```bash
cd /Users/aranja14/Desktop/Gemini_QA
python -c "from config.test_config import TEST_CONFIG; print(TEST_CONFIG.get_summary())"
```

**Expected output (Test Mode):**
```
🧪 TEST MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paid AI APIs:
  GPT-4 Vision:     ❌ DISABLED
  Claude Vision:    ❌ DISABLED
  GPT-4-mini Gen:   ❌ DISABLED

Free Local Models:
  YOLO Detection:   ✅ ENABLED
  OCR Extraction:   ✅ ENABLED
  Whisper Audio:    ✅ ENABLED
  Scene Detection:  ✅ ENABLED

Testing:
  Mock Predictions: ✅ ENABLED
  Max API Cost:     $0.00 per video

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Testing the HITL System

### 1. Create Test Evidence (with mocks)

```bash
# This will use mock predictions (no API costs)
python -c "
from processing.evidence_extractor import EvidenceExtractor
from pathlib import Path

extractor = EvidenceExtractor(
    video_path=Path('path/to/video.mp4'),
    video_id='test_video_001'
)

evidence_items = extractor.extract_evidence_for_hitl(
    video_path=Path('path/to/video.mp4'),
    interval_seconds=5.0,
    max_items=10
)

print(f'Created {len(evidence_items)} evidence items')
print('Cost: $0.00 (all mocked!)')
"
```

### 2. Test the Review UI

```bash
# Frontend should already be running on http://localhost:3000
# Navigate to: http://localhost:3000/evidence/review
```

### 3. Review Test Items

- Click "Evidence Review" in sidebar
- Review items using keyboard shortcuts (A/C/R/S)
- Check that consensus indicators work correctly
- Verify progress tracking updates

---

## Enabling Production APIs

⚠️ **Warning:** This will incur costs!

### Step 1: Add API Keys

Create `.env` file with your API keys:

```bash
# OpenAI API Key (for GPT-4 Vision)
OPENAI_API_KEY=sk-...

# Anthropic API Key (for Claude Sonnet)
ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2: Enable APIs

```bash
# In .env
ENABLE_GPT4_VISION=true
ENABLE_CLAUDE_VISION=true
USE_MOCK_PREDICTIONS=false
MAX_API_COST=5.0  # Maximum spend per video
```

### Step 3: Implement API Calls

Currently, the API call implementations are marked as TODO in:
- `/processing/evidence_extractor.py:686-690` (GPT-4 Vision)
- `/processing/evidence_extractor.py:719-723` (Claude Vision)

You'll need to implement these using the OpenAI and Anthropic SDKs.

---

## Cost Estimation

### Test Mode (Current)
- **Cost:** $0.00
- **Models:** YOLO, OCR, Whisper (local)
- **Predictions:** Mock data

### Production Mode (if enabled)
- **Per video (50 evidence items):**
  - GPT-4 Vision: 50 × $0.03 = $1.50
  - Claude Vision: 50 × $0.01 = $0.50
  - **Total:** ~$2.00 per video

- **Per 100 videos:** ~$200.00

---

## Troubleshooting

### "API key not found"
- Check that `.env` file exists in project root
- Verify API keys are valid
- Restart the backend server after adding keys

### "Still seeing mock data in production mode"
- Ensure `USE_MOCK_PREDICTIONS=false` in `.env`
- Verify API implementations are complete (not TODO)
- Check logs for API errors

### "Unexpected API costs"
- Set `MAX_API_COST` to limit spending
- Monitor usage in OpenAI/Anthropic dashboards
- Use test mode for development and QA

---

## Summary

✅ **For Testing/Development:** Use default configuration (Test Mode)
✅ **For Training:** Use Test Mode with different mock_consensus settings
✅ **For Production:** Enable paid APIs and implement API call logic

**Current Status:** 🧪 Test Mode (FREE) - No changes needed for testing!
