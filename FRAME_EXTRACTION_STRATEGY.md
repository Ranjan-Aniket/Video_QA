# Smart Frame Extraction Strategy

**File**: `processing/smart_frame_extractor.py`
**Current Status**: Uses OLD opportunities (needs update after Phase 4 reorder)

---

## 🎯 **Extraction Strategy Overview**

The smart frame extractor uses **3-tier approach** to extract frames strategically:

1. **Premium Frames** (Dense Clusters) - 10 frames per opportunity
2. **Template Frames** (Single Frame) - 1 frame per opportunity
3. **Scene Boundary Frames** (Not currently used)

---

## 📊 **Frame Types Explained**

### **1. Premium Frames (Dense Sampling)** 🔥

**What**: Extract 10 frames around high-value opportunities

**Strategy**:
- Select top 7 opportunities (by adversarial score)
- Extract 10 frames per opportunity
- Frames spaced 0.5 seconds apart
- Window: ±2.5 seconds around center timestamp

**Layout**:
```
Center timestamp: 100.0s

Frame positions:
-2.5s  -2.0s  -1.5s  -1.0s  -0.5s   0.0s  +0.5s  +1.0s  +1.5s  +2.0s
 97.5   98.0   98.5   99.0   99.5  100.0  100.5  101.0  101.5  102.0
  |      |      |      |      |      |      |      |      |      |
pos_00  pos_01 pos_02 pos_03 pos_04 pos_05 pos_06 pos_07 pos_08 pos_09
                                     ^^^^
                                   KEY FRAME
                              (gets GPT-4V + Claude)
```

**Key Frame**: Position 05 (center) marked as KEY
- Gets expensive AI analysis (GPT-4V + Claude)
- Other 9 frames get only cheap analysis (YOLO, OCR, BLIP-2)

**Code**:
```python
def _extract_dense_cluster(opp_id, center_timestamp):
    frames = []
    for i in range(-5, 5):  # -5 to +4 = 10 frames
        timestamp = center_timestamp + (i * 0.5)  # 0.5s spacing
        position = i + 5  # Convert to 0-9
        is_key = (i == 0)  # Center is key

        frame = extract_frame_at_timestamp(
            timestamp=timestamp,
            frame_id=f"{opp_id}_dense_{position:02d}",
            is_key_frame=is_key,
            cluster_id=opp_id
        )
        frames.append(frame)

    return frames  # Returns 10 frames
```

**Example Output**:
```
premium_001_dense_00.jpg  (97.5s)
premium_001_dense_01.jpg  (98.0s)
premium_001_dense_02.jpg  (98.5s)
premium_001_dense_03.jpg  (99.0s)
premium_001_dense_04.jpg  (99.5s)
premium_001_dense_05.jpg  (100.0s) ⭐ KEY FRAME
premium_001_dense_06.jpg  (100.5s)
premium_001_dense_07.jpg  (101.0s)
premium_001_dense_08.jpg  (101.5s)
premium_001_dense_09.jpg  (102.0s)
```

**Total**: 7 opportunities × 10 frames = **70 premium frames**
- 7 KEY frames (get expensive AI)
- 63 context frames (get cheap analysis only)

---

### **2. Template Frames (Single Frame)** 📋

**What**: Extract 1 frame per template opportunity

**Strategy**:
- Take first 40 non-premium opportunities
- Extract exactly 1 frame per opportunity
- Use precise timestamp (key_word_timestamp > visual_timestamp > audio_start)
- **ALL marked as KEY frames** (get expensive AI analysis)

**Code**:
```python
template_count = 0
for opp in opportunities:
    if opp.get("requires_premium_frame", False):
        continue  # Skip premium opportunities

    if template_count >= 40:
        break  # Stop at 40

    timestamp = (
        opp.get("key_word_timestamp") or
        opp.get("visual_timestamp") or
        opp.get("audio_start", 0)
    )

    frame = extract_frame_at_timestamp(
        timestamp=timestamp,
        frame_id=f"template_{template_count:03d}",
        frame_type="template",
        is_key_frame=True  # All templates are KEY
    )

    template_count += 1
```

**Example Output**:
```
template_000.jpg  (10.5s) ⭐ KEY FRAME
template_001.jpg  (25.3s) ⭐ KEY FRAME
template_002.jpg  (42.7s) ⭐ KEY FRAME
...
template_039.jpg  (545.2s) ⭐ KEY FRAME
```

**Total**: **40 template frames**
- ALL 40 are KEY frames (get expensive AI)

---

### **3. Scene Boundary Frames** (Not Currently Used)

**What**: Extract frames at detected scene changes

**Status**: Code exists but not used in current pipeline

**How it would work**:
- Use audio silence gaps with type="scene_change"
- Extract frame at gap boundary
- Useful for detecting editing/transitions

---

## 📈 **Frame Extraction Statistics**

### **Total Frames Extracted**:

| Type | Count | KEY Frames | Cheap Analysis | Expensive AI |
|------|-------|------------|----------------|--------------|
| Premium Dense | 70 | 7 | 63 | 7 |
| Template | 40 | 40 | 0 | 40 |
| **TOTAL** | **110** | **47** | **63** | **47** |

### **What Each Frame Gets**:

**All 110 Frames** (Cheap - Phase 3 Bulk Analysis):
- ✅ YOLO object detection
- ✅ OCR text extraction
- ✅ Scene classification
- ✅ BLIP-2 captioning
- ✅ CLIP embeddings
- ✅ MediaPipe pose detection
- ✅ FER emotions
- ✅ DeepSport jersey numbers
- ✅ Text orientation

**47 KEY Frames Only** (Expensive - Phase 3 AI Enhancement):
- ✅ GPT-4V detailed description ($0.01 per frame)
- ✅ Claude Sonnet detailed description ($0.01 per frame)
- Total cost: 47 × $0.02 = **$0.94**

---

## 🔧 **Frame Extraction Process**

### **Step 1: Video Analysis**
```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```

### **Step 2: Extract Frame at Timestamp**
```python
def _extract_frame_at_timestamp(timestamp):
    cap = cv2.VideoCapture(video_path)

    # Seek to timestamp
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read frame
    ret, frame_img = cap.read()

    if ret:
        # Save as JPEG
        cv2.imwrite(f"frame_{timestamp:.2f}s.jpg", frame_img)

        # Create metadata
        frame = ExtractedFrame(
            frame_id="frame_001",
            timestamp=timestamp,
            frame_type="premium",
            image_path="frame_001.jpg",
            is_key_frame=True,
            width=1920,
            height=1080
        )

        return frame

    return None
```

### **Step 3: Save Metadata**
```python
{
  "video_id": "video_001",
  "extraction_strategy": "dense_sampling",
  "total_frames": 110,
  "key_frames": 47,
  "frames": [
    {
      "frame_id": "premium_001_dense_05",
      "timestamp": 100.0,
      "frame_type": "premium",
      "image_path": "frames/premium_001_dense_05_100.00s.jpg",
      "is_key_frame": true,
      "cluster_id": "premium_001",
      "cluster_position": 5,
      "width": 1920,
      "height": 1080
    }
  ]
}
```

---

## ⚠️ **Current Issue: Phase Order Problem**

### **The Problem**:

Currently, frame extraction happens in **Phase 2** using opportunities from **OLD Phase 2** (transcript-based):

```
OLD Order:
Phase 1: Audio Analysis ✅
Phase 2: Opportunities (transcript-based, 86% garbage) ❌
Phase 3: Frame Extraction (uses Phase 2 opportunities) ⚠️
Phase 4: Evidence Extraction
```

**Result**: Premium frames selected based on garbage opportunities!

### **The Fix** (After Phase Reordering):

New order moves opportunities AFTER evidence:

```
NEW Order:
Phase 1: Audio Analysis ✅
Phase 2: Frame Extraction (uses what opportunities?) ⚠️
Phase 3: Evidence Extraction ✅
Phase 4: Opportunities (evidence-based, 0% garbage) ✅
```

**Problem**: Phase 2 needs opportunities but they don't exist until Phase 4!

### **Solutions**:

#### **Option A**: Extract all frames evenly (no opportunities)
```python
# In Phase 2 (before opportunities exist):
def extract_template_frames_evenly():
    # Extract 40 frames evenly distributed
    interval = video_duration / 40
    for i in range(40):
        timestamp = i * interval
        extract_frame(timestamp)
```

#### **Option B**: Two-pass extraction
```python
# Phase 2: Extract template frames only (40 frames)
extract_template_frames_evenly()

# Phase 4.5: After opportunities, extract premium dense clusters
for premium_opp in top_7_opportunities:
    extract_dense_cluster(premium_opp.timestamp)
```

#### **Option C**: Skip premium frames for now
```python
# Phase 2: Just extract template frames
# Phase 4: Opportunities generated
# Phase 5: Questions use template frames only (no premium)
```

---

## 📊 **Frame Quality Metrics**

The extractor can optionally check frame quality:

```python
@dataclass
class FrameQualityMetrics:
    blur_score: float        # Laplacian variance (higher = sharper)
    brightness: float        # Mean pixel intensity (0-1)
    motion_level: float      # Difference from previous frame
    is_shot_boundary: bool   # Detected cut/transition
    is_black_frame: bool     # Mostly black
    is_transition: bool      # Fade/wipe/dissolve
    overall_quality: float   # Combined score

def is_good_quality(min_quality=0.5):
    return (
        overall_quality >= min_quality and
        not is_black_frame and
        not is_transition
    )
```

**Thresholds**:
- `MIN_BLUR_SCORE = 0.15` - Reject very blurry frames
- `MIN_BRIGHTNESS = 0.05` - Reject too dark
- `MAX_BRIGHTNESS = 0.95` - Reject too bright (overexposed)
- `SCENE_CHANGE_THRESHOLD = 0.3` - Detect cuts

---

## 🔄 **Data Flow**

```
INPUT: opportunities.json (from Phase 4 in new order)
  ↓
READ:
  - premium_frames (top 7 opportunities)
  - opportunities (all opportunities)
  ↓
EXTRACT:
  ├─ Premium Dense Clusters (7 × 10 = 70 frames)
  │  └─ Mark center frame as KEY
  │
  └─ Template Frames (40 × 1 = 40 frames)
     └─ Mark ALL as KEY
  ↓
SAVE:
  - frames/*.jpg (110 image files)
  - frames_metadata.json
  ↓
OUTPUT: List[ExtractedFrame] (110 frames, 47 KEY)
```

---

## 💡 **Key Insights**

1. **Dense Sampling = Context**
   - 10 frames give temporal context around opportunity
   - Center frame is main analysis point
   - Surrounding frames show before/after

2. **KEY Frame Optimization**
   - Only 47/110 frames (43%) get expensive AI
   - Saves ~$1.26 per video (63 frames × $0.02)
   - Still maintains quality (all key moments analyzed)

3. **Template vs Premium**
   - Template: Broad coverage (40 moments)
   - Premium: Deep dive (7 moments × 10 frames)
   - Balance between breadth and depth

4. **Timestamp Precision**
   - Uses word-level timestamps when available
   - Falls back to visual timestamp or audio start
   - OpenCV seeks to nearest frame (1/fps precision)

---

## 🚧 **TODO: Fix Phase Order Integration**

After reordering phases, frame extraction needs adjustment:

1. **Phase 2 can't use Phase 4 opportunities** (they don't exist yet)
2. **Solution**: Extract template frames evenly in Phase 2
3. **Premium frames**: Either skip or add Phase 4.5 for dense sampling

**Recommended**: Use Option A (extract evenly) for now, add premium sampling later.

---

**Generated**: November 19, 2025
**Status**: ⚠️ NEEDS UPDATE FOR NEW PHASE ORDER
