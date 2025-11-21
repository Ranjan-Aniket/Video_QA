# Phase 5 Frame Selection Prompt Improvements

**Date**: 2025-11-20
**File**: `/Users/aranja14/Desktop/Gemini_QA/processing/llm_frame_selector.py`
**Lines Modified**: 203-342, 368-386, 429-473

---

## Summary

Rewrote Phase 5 frame selection prompt to eliminate generic filler frames and improve question type coverage using a **Two-Pass Selection Strategy** with **Objective Criteria** for all 13 question types.

---

## Key Changes

### 1. Objective Criteria for All 13 Question Types (Lines 219-273)

Each question type now has:
- **Checkable evidence requirements** (e.g., "Needle = readable text in `text_detected` field")
- **Concrete examples** showing good evidence-based reasoning

Examples:
- **Needle**: `✓ Evidence: Readable text in text_detected field`
- **Counting**: `✓ Evidence: Multiple instances of same object type (people, chairs, screens)`
- **Audio-Visual Stitching**: `✓ Evidence: High audio_score + rich visual context at same timestamp`
- **Tackling Spurious Correlations**: `✓ Evidence: Unusual/unexpected context (surprising scene_type, atypical objects)`

### 2. Two-Pass Selection Strategy (Lines 276-294)

Replaced hard "Select exactly 150 frames" with adaptive approach:

**Pass 1: Must-Haves (Tier 1)**
- Priority ≥0.85 (auto-include)
- Readable text detected (Needle questions)
- Dense clusters: 4+ highlights within 8 seconds

**Pass 2: Smart Gap-Filling (Tier 2) - ONLY IF NEEDED**
- Fill gaps >40 seconds
- Only if gap contains frames with priority ≥0.75
- Select 1-2 frames from gap

### 3. Flexible Frame Range (Lines 203-205, 291-294)

Changed from hard constraint to quality-based range:
- **Old**: "Select exactly 150 frames"
- **New**: "Select 80-120 frames based on quality"
- Explicitly allows stopping at 90 frames if that's all the quality content available

### 4. Positive Framing with Examples (Lines 297-308)

Replaced "BANNED" language with:
- ✓ **Good examples**: "Audio peak (0.92) + speaker pointing at slide with 'Budget' text → Audio-Visual Stitching + Needle"
- ✗ **Avoid examples**: "Mid-section coverage (vague, not evidence-based)"

### 5. Updated Validation Logic (Lines 368-386)

**Old behavior**:
- Required 80-100% of budget (120-150 frames)
- Would force padding below 120 frames

**New behavior**:
- Target range: 80-120 frames (53-80% of 150 budget)
- Below 80: Accept as-is (trusts Claude's judgment on low-quality videos)
- Above 120: Trim lowest priority frames
- No forced padding

### 6. Improved Fallback Selection (Lines 429-473)

**Old behavior**:
- Always filled to exactly `frame_budget` (150 frames)
- Used evenly spaced frames as filler

**New behavior**:
- Uses top highlights up to 120 frames (80% of budget)
- Only adds evenly spaced frames if below 80 minimum
- Prefers quality over quota

---

## Expected Outcomes

### Problem 1: Generic Filler Reasoning ✅ FIXED
- **Before**: 56.7% of frames had "temporal distribution", "mid-section coverage", "between highlights"
- **After**: Two-pass strategy eliminates gap-filling unless gaps >40s exist with priority ≥0.75 frames

### Problem 2: Missing Question Type Coverage ✅ IMPROVED
- **Before**: Only 76.9% coverage (10/13 types) despite relevant frames existing
- **After**: Objective criteria ensure frames are tagged correctly (e.g., text frames → Needle)

### Problem 3: Budget Underutilization ✅ FIXED
- **Before**: Selected 54 frames + 96-frame dense cluster (gaming the system)
- **After**: 80-120 single frames + max 2 dense clusters (6-8 frames each)

### Problem 4: Conflicting Directives ✅ FIXED
- **Before**: "Select exactly 150" vs "Prefer 120 quality frames"
- **After**: Clear range target (80-120) with explicit permission to stop early

---

## Testing Plan

1. **Re-run pipeline** on current video (Copy of w-A-4ckmFJo)
2. **Analyze Phase 5 output** for:
   - Frame count (should be 80-120)
   - Generic reasoning percentage (should be <10%)
   - Question type coverage (should be >85% with correct tagging)
   - Dense cluster usage (should be 0-2 clusters, <20 total frames)
3. **Test on 3-5 additional videos** to validate consistency
4. **Add validation** ONLY if persistent problems emerge

---

## Files Modified

- `processing/llm_frame_selector.py` (203-342, 368-386, 429-473)

---

## Rollback Plan

If the new prompt performs worse:
```bash
git diff processing/llm_frame_selector.py  # Review changes
git checkout HEAD -- processing/llm_frame_selector.py  # Revert
```

Checkpoint file available: `PHASE5_PROMPT_IMPROVEMENTS.md`
