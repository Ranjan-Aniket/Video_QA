# Smart Pipeline Backend - File Map & Process Flow

**Date**: November 19, 2025
**Pipeline**: smart_pipeline.py (main orchestrator)

---

## 📋 **Complete File Map by Phase**

### **PHASE 1: Audio Analysis**

**Main File**: `processing/audio_analysis.py`

**Purpose**: Extract audio, transcribe with Whisper, detect speakers, find silence gaps

**Uses**:
- ✅ `whisper` (external) - Speech-to-text transcription
- ✅ `pyannote.audio` (external) - Speaker diarization
- ✅ `librosa` (external) - Audio processing, silence detection
- ✅ `pydub` (external) - Audio file handling

**Outputs**:
- `{video_id}_audio_analysis.json` - Contains:
  - Full transcript with word-level timestamps
  - Speaker segments (SPEAKER_00, SPEAKER_01, etc.)
  - Silence gaps (for scene change detection)
  - Audio events (music, sound effects)
  - Duration, language

**Process**:
```python
AudioAnalyzer(video_path)
  → extract_audio()           # Extract MP3 from video
  → transcribe_with_whisper() # Word-level timestamps
  → diarize_speakers()        # Identify who's speaking
  → detect_silence_gaps()     # Find scene changes (FIXED!)
  → detect_sound_effects()    # Music, crowd noise
```

---

### **PHASE 2: Smart Frame Extraction**

**Main File**: `processing/smart_frame_extractor.py`

**Purpose**: Extract frames strategically (template + premium + dense)

**Uses**:
- ✅ `cv2` (OpenCV) - Frame extraction from video
- ✅ `numpy` - Image processing
- ✅ `imagehash` - Deduplication

**Outputs**:
- `frames/{video_id}/frames_metadata.json` - Frame metadata
- `frames/{video_id}/*.jpg` - Actual frame images

**Frame Types**:
1. **Template frames** (40 frames) - Evenly distributed across video
2. **Premium frames** (54 frames) - Around opportunities (NOT USED YET - opportunities come later now)
3. **Dense frames** (varies) - Scene change clusters

**Process**:
```python
SmartFrameExtractor(video_path, opportunities, audio_analysis)
  → extract_template_frames()  # Evenly distributed
  → extract_premium_frames()   # Around opportunities (skipped if no opps)
  → extract_dense_frames()     # Scene change clusters
  → deduplicate_frames()       # Remove duplicates
```

---

### **PHASE 3: Hybrid Evidence Extraction**

**Main File**: `processing/bulk_frame_analyzer.py`

**Purpose**: Extract visual evidence from all frames using multiple AI models

**Uses 10+ Sub-Processors**:

#### **3.1 Object Detection**
- ✅ `processing/object_detector.py`
  - Uses: `ultralytics` (YOLOv8)
  - Detects: People, sports ball, objects
  - Output: Bounding boxes, confidence scores

#### **3.2 OCR (Text Recognition)**
- ✅ `processing/ocr_processor.py`
  - Uses: `paddleocr`
  - Extracts: On-screen text (scores, names, etc.)
  - Output: Text, bounding boxes, confidence

#### **3.3 Scene Classification**
- ✅ `processing/places365_processor.py`
  - Uses: ResNet50 (heuristic for now)
  - Detects: Basketball court, sports field, indoor/outdoor
  - Output: Scene type, confidence

#### **3.4 Image Captioning**
- ✅ `processing/blip2_processor.py`
  - Uses: `transformers` (BLIP-2 Flan-T5-XL, 15GB model)
  - Generates: Natural language captions
  - Output: "a basketball game with players on the court"

#### **3.5 Image Embeddings**
- ✅ `processing/clip_processor.py`
  - Uses: `transformers` (CLIP ViT-B/32)
  - Generates: 512-dim embeddings for similarity
  - Output: Embedding vectors

#### **3.6 Facial Expression Recognition**
- ✅ `processing/fer_processor.py`
  - Uses: `fer` library
  - Detects: Happy, sad, angry, etc.
  - Output: Emotion labels, confidence

#### **3.7 Jersey Number Detection**
- ✅ `processing/deepsport_processor.py`
  - Uses: DeepSport model
  - Detects: Player jersey numbers
  - Output: Number, bounding box

#### **3.8 Text Orientation**
- ✅ `processing/text_orientation_processor.py`
  - Detects: Upright, rotated, upside-down text
  - Output: Orientation angle

#### **3.9 Pose Detection**
- ✅ `processing/pose_detector.py`
  - Uses: `mediapipe` (Holistic)
  - Detects: Body pose, hand gestures, face landmarks
  - Output: Keypoints, skeleton

#### **3.10 Action Recognition** (optional)
- ⏸️ `processing/videomae_processor.py`
  - Uses: VideoMAE
  - Detects: Basketball actions (shoot, pass, dribble)
  - Status: Available but not enabled by default

**Sub-Phase 3A: Bulk Analysis** (All frames, local models)
```python
BulkFrameAnalyzer()
  → analyze_frames_batch()
    → YOLO (objects)
    → OCR (text)
    → Scene (classification)
    → BLIP-2 (captions)
    → CLIP (embeddings)
    → MediaPipe (poses)
    → FER (emotions)
    → DeepSport (jerseys)
```

**Sub-Phase 3B: AI Enhancement** (Template frames only, expensive)
```python
→ enhance_with_gpt4v()      # 40 template frames → GPT-4V
→ enhance_with_claude()     # 40 template frames → Claude Sonnet
```

**Outputs**:
- `{video_id}_evidence.json` - Contains:
  ```json
  {
    "frames": {
      "template_000": {
        "timestamp": 236.98,
        "ground_truth": {
          "yolo_objects": [...],
          "ocr_text": [...],
          "scene_type": "basketball_court_indoor",
          "blip2_caption": "...",
          "gpt4v_description": "```json {...}```",
          "claude_description": "```json {...}```",
          "clip_embeddings": [...],
          "facial_expressions": [...],
          "jersey_numbers": [...],
          "body_poses": [...]
        }
      }
    }
  }
  ```

---

### **PHASE 4: Evidence-Based Opportunity Detection** ⭐ **NEW!**

**Main File**: `processing/evidence_based_opportunity_detector.py`

**Purpose**: Find adversarial opportunities using ACTUAL visual evidence

**Uses**:
- ✅ Evidence from Phase 3 (reads `evidence.json`)
- ✅ Audio analysis from Phase 1 (optional context)

**Sub-Detectors**:

#### **4.1 Counting Detector**
```python
_detect_counting(evidence)
  → Parse GPT-4V/Claude JSON for players
  → Extract jersey numbers
  → Create: "Count players with jersey #23, #16"
```

#### **4.2 Comparative Detector**
```python
_detect_comparative(evidence)
  → Track score changes over time (from OCR/AI)
  → Create: "Compare score: WSH 52 → WSH 54"
```

#### **4.3 Object Interaction Detector**
```python
_detect_object_interaction(evidence)
  → Find player actions in AI descriptions
  → Create: "Player #13 dribbling basketball"
```

#### **4.4-4.9 More Detectors** (TODO)
- Inference
- Holistic Reasoning
- Needle in Haystack
- Context
- Subscene
- Spurious Correlations

**Outputs**:
- `{video_id}_opportunities.json` - Contains:
  ```json
  {
    "total_opportunities": 42,
    "validated_opportunities": 42,
    "opportunity_statistics": {
      "counting": 12,
      "comparative": 25,
      "object_interaction": 5
    },
    "opportunities": [
      {
        "opportunity_id": "count_0001",
        "opportunity_type": "counting",
        "description": "Count players with jersey #23, #16",
        "adversarial_score": 0.85,
        "evidence_type": "gpt4v_claude",
        "evidence_data": {
          "jersey_numbers": ["23", "16"]
        }
      }
    ]
  }
  ```

---

### **PHASE 5: Question Generation**

**Main File**: `processing/multimodal_question_generator_v2.py`

**Purpose**: Generate adversarial questions from opportunities

**Uses**:
- ✅ Opportunities from Phase 4
- ✅ Evidence from Phase 3
- ✅ Audio analysis from Phase 1
- ✅ Template registry (question templates)
- ✅ OpenAI GPT-4 (for question generation)

**Process**:
```python
MultimodalQuestionGeneratorV2()
  → For each opportunity:
    → Select template based on opportunity type
    → Extract visual cues from evidence
    → Extract audio cues from transcript
    → Generate question with GPT-4
    → Generate golden answer
    → Create adversarial distractors
```

**Outputs**:
- `{video_id}_questions.json` - Contains:
  ```json
  {
    "questions": [
      {
        "question_text": "How many players with visible jersey numbers are on court when score is WSH 52?",
        "golden_answer": "2 players (#23, #16)",
        "visual_cues": ["players #23, #16", "score WSH 52-TOR 57"],
        "audio_cues": ["And you see 11 first round picks..."],
        "task_type": "Counting",
        "complexity": "medium",
        "adversarial_score": 0.85
      }
    ]
  }
  ```

---

### **PHASE 6: Gemini Testing** (Optional)

**Main File**: `processing/gemini_tester.py` (or inline in smart_pipeline)

**Purpose**: Test Gemini 2.0 Flash with generated questions

**Uses**:
- ✅ Questions from Phase 5
- ✅ Video file
- ✅ Gemini API

**Outputs**:
- `{video_id}_gemini_results.json` - Contains:
  - Gemini's answers
  - Pass/fail for each question
  - Failure types
  - Overall accuracy

---

## 🗂️ **File Usage Summary**

### **Core Pipeline Files** (Always Used):
1. ✅ `processing/smart_pipeline.py` - Main orchestrator
2. ✅ `processing/audio_analysis.py` - Phase 1
3. ✅ `processing/smart_frame_extractor.py` - Phase 2
4. ✅ `processing/bulk_frame_analyzer.py` - Phase 3 (coordinator)
5. ✅ `processing/evidence_based_opportunity_detector.py` - Phase 4 ⭐
6. ✅ `processing/multimodal_question_generator_v2.py` - Phase 5

### **Evidence Extraction Files** (Used in Phase 3):
7. ✅ `processing/object_detector.py` - YOLO
8. ✅ `processing/ocr_processor.py` - PaddleOCR
9. ✅ `processing/places365_processor.py` - Scene classification
10. ✅ `processing/blip2_processor.py` - Image captioning
11. ✅ `processing/clip_processor.py` - Image embeddings
12. ✅ `processing/fer_processor.py` - Facial expressions
13. ✅ `processing/deepsport_processor.py` - Jersey numbers
14. ✅ `processing/text_orientation_processor.py` - Text rotation
15. ✅ `processing/pose_detector.py` - Body poses

### **Utility Files** (Used as needed):
16. ✅ `processing/opportunity_quality_filter.py` - Quality filtering (can be used post-Phase 4)
17. ⏸️ `processing/videomae_processor.py` - Action recognition (optional)

### **Deprecated Files** (No Longer Used):
18. ❌ `processing/opportunity_detector_v2.py` - OLD transcript-based (replaced)
19. ❌ `processing/adversarial_opportunity_detector.py` - OLD (replaced)

---

## 📊 **Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT: VIDEO FILE                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
   ┌─────────┐                    ┌─────────────┐
   │ PHASE 1 │                    │   PHASE 2   │
   │  Audio  │                    │   Frames    │
   └────┬────┘                    └──────┬──────┘
        │                                │
        │  audio_analysis.json           │  frames/*.jpg
        │  - transcript                  │  frames_metadata.json
        │  - speakers                    │
        │  - silence gaps                │
        │                                │
        └────────────┬───────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │   PHASE 3   │
              │  Evidence   │
              └──────┬──────┘
                     │
                     │  evidence.json
                     │  - YOLO objects
                     │  - OCR text
                     │  - GPT-4V/Claude descriptions
                     │  - Scene classification
                     │  - Poses, emotions, jerseys
                     │
                     ▼
              ┌─────────────┐
              │   PHASE 4   │ ⭐ NEW!
              │Opportunities│
              └──────┬──────┘
                     │
                     │  opportunities.json
                     │  - counting: 12
                     │  - comparative: 25
                     │  - object_interaction: 5
                     │
                     ▼
              ┌─────────────┐
              │   PHASE 5   │
              │  Questions  │
              └──────┬──────┘
                     │
                     │  questions.json
                     │  - Adversarial questions
                     │  - Golden answers
                     │  - Visual + audio cues
                     │
                     ▼
              ┌─────────────┐
              │   PHASE 6   │
              │Gemini Test  │
              └──────┬──────┘
                     │
                     ▼
              gemini_results.json
```

---

## 🔧 **External Dependencies**

### **AI/ML Models**:
- `whisper` - OpenAI Whisper (audio → text)
- `pyannote.audio` - Speaker diarization
- `ultralytics` - YOLOv8 (object detection)
- `paddleocr` - PaddleOCR (text extraction)
- `transformers` - BLIP-2, CLIP (Hugging Face)
- `mediapipe` - Pose detection
- `fer` - Facial expression recognition
- `librosa` - Audio analysis

### **APIs**:
- OpenAI GPT-4V (visual descriptions)
- Anthropic Claude Sonnet (visual descriptions)
- OpenAI GPT-4 (question generation)
- Google Gemini 2.0 Flash (testing)

---

## 📁 **Output File Structure**

```
outputs/
└── video_20251119_064141_Copy_of_WEnZKfBrKaE/
    ├── Copy_of_WEnZKfBrKaE_20251119_064141_audio_analysis.json   (Phase 1)
    ├── Copy_of_WEnZKfBrKaE_20251119_064141_evidence.json         (Phase 3)
    ├── Copy_of_WEnZKfBrKaE_20251119_064141_opportunities.json    (Phase 4) ⭐
    ├── Copy_of_WEnZKfBrKaE_20251119_064141_questions.json        (Phase 5)
    ├── Copy_of_WEnZKfBrKaE_20251119_064141_gemini_results.json   (Phase 6)
    ├── Copy_of_WEnZKfBrKaE_20251119_064141_pipeline_results.json (Summary)
    └── frames/
        └── Copy_of_WEnZKfBrKaE_20251119_064141/
            ├── frames_metadata.json                               (Phase 2)
            ├── template_000.jpg
            ├── template_001.jpg
            └── ...
```

---

## ⚡ **Performance Stats**

| Phase | Files Used | Time | Cost |
|-------|------------|------|------|
| 1. Audio | 1 file | ~30s | $0.006 (Whisper) |
| 2. Frames | 1 file | ~5s | $0 (OpenCV) |
| 3. Evidence | 15 files | ~120s | ~$2-3 (GPT-4V/Claude) |
| 4. Opportunities | 1 file | ~2s | $0 (reuses evidence) ⭐ |
| 5. Questions | 1 file | ~30s | ~$0.50 (GPT-4) |
| 6. Gemini | 1 file | ~60s | ~$0.10 (Gemini) |
| **TOTAL** | **19 files** | **~4min** | **~$2.60-3.60** |

---

**Generated**: November 19, 2025
**Status**: ✅ COMPLETE AND DOCUMENTED
