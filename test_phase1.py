#!/Users/aranja14/Desktop/Gemini_QA/.venv/bin/python3
"""Test Phase 1 only"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, '/Users/aranja14/Desktop/Gemini_QA')

# Setup logging to show output immediately
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("="*80, flush=True)
print("PHASE 1 TEST: Enhanced Scene Detector", flush=True)
print("="*80, flush=True)

video_path = "/Users/aranja14/Desktop/Gemini_QA/uploads/Copy of w1cBUA1N2ds_20251120_053447.mp4"

try:
    # Test 1: Audio Analysis
    print("\n🎵 Step 1: Audio Analysis...", flush=True)
    from processing.audio_analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(video_path)
    audio_result = analyzer.analyze(save_json=False)

    print(f"✓ Audio analysis complete:", flush=True)
    print(f"  Duration: {audio_result['duration']:.1f}s", flush=True)
    print(f"  Segments: {len(audio_result['segments'])}", flush=True)
    print(f"  Speakers: {audio_result['speaker_count']}", flush=True)

    # Test 2: Scene Detection
    print("\n🎬 Step 2: Enhanced Scene Detection...", flush=True)
    from processing.scene_detector_enhanced import SceneDetectorEnhanced

    detector = SceneDetectorEnhanced(
        base_threshold=0.3,
        min_scene_duration=1.0,
        enable_adaptive=True,
        enable_motion=True
    )

    scenes_result = detector.detect_scenes(video_path)

    print(f"\n✓ Scene detection complete:", flush=True)
    print(f"  Total scenes: {scenes_result['total_scenes']}", flush=True)
    print(f"  Calibrated threshold: {scenes_result['calibrated_threshold']:.3f}", flush=True)
    print(f"  Avg scene duration: {scenes_result['avg_scene_duration']:.1f}s", flush=True)

    # Show first 5 scenes
    print(f"\n  First 5 scenes:", flush=True)
    for scene in scenes_result['scenes'][:5]:
        print(f"    Scene {scene['scene_id']}: {scene['start']:.1f}s-{scene['end']:.1f}s "
              f"[{scene['transition_type']}, qual: {scene['avg_quality']:.2f}]", flush=True)

    # Test 3: Quality Mapping
    print("\n📊 Step 3: Quality Mapping...", flush=True)
    from processing.quality_mapper import QualityMapper

    quality_mapper = QualityMapper()
    quality_result = quality_mapper.map_quality(video_path)

    print(f"✓ Quality mapping complete:", flush=True)
    print(f"  Samples: {len(quality_result['quality_scores'])}", flush=True)
    print(f"  Average quality: {quality_result['average_quality']:.2f}", flush=True)

    print("\n" + "="*80, flush=True)
    print("✅ PHASE 1 COMPLETE!", flush=True)
    print("="*80, flush=True)

except Exception as e:
    print(f"\n❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
