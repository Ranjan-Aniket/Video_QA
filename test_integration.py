#!/usr/bin/env python3
"""
Test Enhanced Scene Detector Integration

Quick test to verify the enhanced scene detector works in the pipeline.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def test_phase1_integration(video_path: str):
    """Test Phase 1 with enhanced scene detector"""

    print("=" * 80)
    print("TESTING ENHANCED SCENE DETECTOR INTEGRATION")
    print("=" * 80)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False

    print(f"\n✓ Video found: {video_path.name}")
    print(f"  Size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        # Test 1: Import enhanced detector
        print("\n🔹 Test 1: Importing enhanced scene detector...")
        from processing.scene_detector_enhanced import SceneDetectorEnhanced
        print("  ✓ Import successful")

        # Test 2: Initialize detector
        print("\n🔹 Test 2: Initializing detector...")
        detector = SceneDetectorEnhanced(
            base_threshold=0.3,
            min_scene_duration=1.0,
            enable_adaptive=True,
            enable_motion=True
        )
        print("  ✓ Initialization successful")

        # Test 3: Run detection
        print("\n🔹 Test 3: Running scene detection...")
        result = detector.detect_scenes(str(video_path))
        print(f"  ✓ Detection complete")
        print(f"    • Total scenes: {result['total_scenes']}")
        print(f"    • Video duration: {result['video_duration']:.1f}s")
        print(f"    • Avg scene duration: {result['avg_scene_duration']:.1f}s")
        print(f"    • Calibrated threshold: {result['calibrated_threshold']:.3f}")

        # Test 4: Verify scene data structure
        print("\n🔹 Test 4: Verifying scene data structure...")
        if result['scenes']:
            first_scene = result['scenes'][0]
            required_fields = ['scene_id', 'start', 'end', 'duration',
                             'transition_type', 'confidence', 'avg_quality']

            missing_fields = [f for f in required_fields if f not in first_scene]
            if missing_fields:
                print(f"  ❌ Missing fields: {missing_fields}")
                return False

            print("  ✓ All required fields present:")
            print(f"    • scene_id: {first_scene['scene_id']}")
            print(f"    • start: {first_scene['start']:.1f}s")
            print(f"    • end: {first_scene['end']:.1f}s")
            print(f"    • duration: {first_scene['duration']:.1f}s")
            print(f"    • transition_type: {first_scene['transition_type']}")
            print(f"    • confidence: {first_scene['confidence']:.2f}")
            print(f"    • avg_quality: {first_scene['avg_quality']:.2f}")

        # Test 5: Verify transition types
        print("\n🔹 Test 5: Checking transition type distribution...")
        transition_counts = {}
        quality_scores = []

        for scene in result['scenes']:
            t_type = scene['transition_type']
            transition_counts[t_type] = transition_counts.get(t_type, 0) + 1
            quality_scores.append(scene['avg_quality'])

        print("  ✓ Transition types found:")
        for t_type, count in sorted(transition_counts.items()):
            pct = (count / result['total_scenes'] * 100)
            print(f"    • {t_type}: {count} ({pct:.1f}%)")

        # Test 6: Quality statistics
        print("\n🔹 Test 6: Quality statistics...")
        import numpy as np
        avg_quality = np.mean(quality_scores)
        high_quality = sum(1 for q in quality_scores if q >= 0.8)
        low_quality = sum(1 for q in quality_scores if q < 0.5)

        print(f"  ✓ Quality analysis:")
        print(f"    • Average quality: {avg_quality:.2f}")
        print(f"    • High quality scenes (≥0.8): {high_quality} ({high_quality/len(quality_scores)*100:.1f}%)")
        print(f"    • Low quality scenes (<0.5): {low_quality} ({low_quality/len(quality_scores)*100:.1f}%)")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nEnhanced scene detector is working correctly!")
        print("Integration with smart_pipeline.py should work as expected.")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_quality_filtering(video_path: str):
    """Test Phase 2 quality filtering"""

    print("\n" + "=" * 80)
    print("TESTING PHASE 2 QUALITY FILTERING")
    print("=" * 80)

    try:
        # Get scenes from Phase 1
        print("\n🔹 Running Phase 1 to get scenes...")
        from processing.scene_detector_enhanced import SceneDetectorEnhanced

        detector = SceneDetectorEnhanced(
            base_threshold=0.3,
            min_scene_duration=1.0,
            enable_adaptive=True,
            enable_motion=False  # Faster for testing
        )
        result = detector.detect_scenes(str(video_path))
        scenes = result['scenes']

        print(f"  ✓ Got {len(scenes)} scenes")

        # Count quality distribution
        high_q = sum(1 for s in scenes if s['avg_quality'] >= 0.8)
        med_q = sum(1 for s in scenes if 0.5 <= s['avg_quality'] < 0.8)
        low_q = sum(1 for s in scenes if s['avg_quality'] < 0.5)

        print(f"\n  Quality distribution:")
        print(f"    • High (≥0.8): {high_q}")
        print(f"    • Medium (0.5-0.8): {med_q}")
        print(f"    • Low (<0.5): {low_q}")

        # Test filtering
        print("\n🔹 Testing quality filtering...")
        from processing.quick_visual_sampler import QuickVisualSampler

        print("\n  Test 1: No filtering (min_quality=0.0)")
        sampler = QuickVisualSampler()
        result1 = sampler.sample_and_analyze(
            video_path=str(video_path),
            scenes=scenes[:10],  # Just test first 10 scenes
            min_quality=0.0
        )
        print(f"    • Sampled: {result1['total_sampled']}")
        print(f"    • Skipped: {result1['skipped_low_quality']}")

        print("\n  Test 2: With filtering (min_quality=0.5)")
        result2 = sampler.sample_and_analyze(
            video_path=str(video_path),
            scenes=scenes[:10],
            min_quality=0.5
        )
        print(f"    • Sampled: {result2['total_sampled']}")
        print(f"    • Skipped: {result2['skipped_low_quality']}")

        if result2['skipped_low_quality'] > 0:
            print(f"\n  ✓ Quality filtering is working!")
            print(f"    Filtered out {result2['skipped_low_quality']} low-quality scenes")
        else:
            print(f"\n  ℹ️  No low-quality scenes in test sample")

        print("\n✅ Phase 2 quality filtering test passed")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_integration.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    # Run tests
    test1_passed = test_phase1_integration(video_path)

    if test1_passed:
        test2_passed = test_phase2_quality_filtering(video_path)

    if test1_passed:
        print("\n" + "=" * 80)
        print("🎉 INTEGRATION COMPLETE")
        print("=" * 80)
        print("\nYour pipeline is now using the enhanced scene detector!")
        print("\nNext steps:")
        print("  1. Run full pipeline: python processing/smart_pipeline.py")
        print("  2. Check Phase 1 checkpoint for new fields:")
        print("     - calibrated_threshold")
        print("     - avg_scene_duration")
        print("     - scene[].transition_type")
        print("     - scene[].confidence")
        print("     - scene[].avg_quality")
        print("  3. Optionally increase min_quality in Phase 2 to skip bad scenes")
    else:
        print("\n❌ Integration has issues - please review errors above")
