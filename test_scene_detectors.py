#!/usr/bin/env python3
"""
Test and compare Basic vs Enhanced Scene Detectors

Usage:
    python test_scene_detectors.py path/to/video.mp4
"""

import sys
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def compare_detectors(video_path: str):
    """Compare basic vs enhanced scene detectors"""

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    print("=" * 80)
    print(f"SCENE DETECTOR COMPARISON")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)

    # Test 1: Basic Detector
    print("\n🔹 TEST 1: BASIC DETECTOR")
    print("-" * 80)

    try:
        from processing.scene_detector import SceneDetector

        basic = SceneDetector(threshold=0.3)

        start = time.time()
        basic_result = basic.detect_scenes(str(video_path))
        basic_time = time.time() - start

        print(f"\n✓ Basic detector completed in {basic_time:.1f}s")
        print(f"  Scenes detected: {basic_result['total_scenes']}")
        print(f"  Video duration: {basic_result['video_duration']:.1f}s")
        print(f"  Avg scene duration: {basic_result['video_duration'] / basic_result['total_scenes']:.1f}s")

        # Show first 5 scenes
        print(f"\n  First 5 scenes:")
        for scene in basic_result['scenes'][:5]:
            print(f"    Scene {scene['scene_id']}: "
                  f"{scene['start']:.1f}s → {scene['end']:.1f}s "
                  f"({scene['duration']:.1f}s)")

    except Exception as e:
        print(f"❌ Basic detector failed: {e}")
        basic_result = None
        basic_time = 0

    # Test 2: Enhanced Detector
    print("\n🔹 TEST 2: ENHANCED DETECTOR")
    print("-" * 80)

    try:
        from processing.scene_detector_enhanced import SceneDetectorEnhanced

        enhanced = SceneDetectorEnhanced(
            base_threshold=0.3,
            min_scene_duration=1.0,
            enable_adaptive=True,
            enable_motion=True
        )

        start = time.time()
        enhanced_result = enhanced.detect_scenes(str(video_path))
        enhanced_time = time.time() - start

        print(f"\n✓ Enhanced detector completed in {enhanced_time:.1f}s")
        print(f"  Scenes detected: {enhanced_result['total_scenes']}")
        print(f"  Video duration: {enhanced_result['video_duration']:.1f}s")
        print(f"  Avg scene duration: {enhanced_result['avg_scene_duration']:.1f}s")
        print(f"  Calibrated threshold: {enhanced_result['calibrated_threshold']:.3f}")

        # Show first 5 scenes with extra info
        print(f"\n  First 5 scenes:")
        for scene in enhanced_result['scenes'][:5]:
            print(f"    Scene {scene['scene_id']}: "
                  f"{scene['start']:.1f}s → {scene['end']:.1f}s "
                  f"({scene['duration']:.1f}s) "
                  f"[{scene['transition_type']}, "
                  f"conf: {scene['confidence']:.2f}, "
                  f"qual: {scene['avg_quality']:.2f}]")

    except Exception as e:
        print(f"❌ Enhanced detector failed: {e}")
        enhanced_result = None
        enhanced_time = 0

    # Comparison
    if basic_result and enhanced_result:
        print("\n" + "=" * 80)
        print("📊 COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\n{'Metric':<30} {'Basic':<15} {'Enhanced':<15} {'Difference':<15}")
        print("-" * 80)

        # Scene count
        basic_count = basic_result['total_scenes']
        enhanced_count = enhanced_result['total_scenes']
        diff_count = enhanced_count - basic_count
        diff_pct = (diff_count / basic_count * 100) if basic_count > 0 else 0
        print(f"{'Total scenes':<30} {basic_count:<15} {enhanced_count:<15} "
              f"{diff_count:+d} ({diff_pct:+.0f}%)")

        # Avg scene duration
        basic_avg = basic_result['video_duration'] / basic_count if basic_count > 0 else 0
        enhanced_avg = enhanced_result['avg_scene_duration']
        diff_avg = enhanced_avg - basic_avg
        diff_avg_pct = (diff_avg / basic_avg * 100) if basic_avg > 0 else 0
        print(f"{'Avg scene duration (s)':<30} {basic_avg:<15.1f} {enhanced_avg:<15.1f} "
              f"{diff_avg:+.1f}s ({diff_avg_pct:+.0f}%)")

        # Processing time
        diff_time = enhanced_time - basic_time
        diff_time_pct = (diff_time / basic_time * 100) if basic_time > 0 else 0
        print(f"{'Processing time (s)':<30} {basic_time:<15.1f} {enhanced_time:<15.1f} "
              f"{diff_time:+.1f}s ({diff_time_pct:+.0f}%)")

        # Speed
        basic_fps = basic_result['video_duration'] / basic_time if basic_time > 0 else 0
        enhanced_fps = enhanced_result['video_duration'] / enhanced_time if enhanced_time > 0 else 0
        print(f"{'Processing speed (x real-time)':<30} {basic_fps:<15.1f} {enhanced_fps:<15.1f}")

        # Additional enhanced features
        print("\n" + "-" * 80)
        print("Enhanced-only features:")

        # Transition types
        transition_counts = {}
        for scene in enhanced_result['scenes']:
            t_type = scene['transition_type']
            transition_counts[t_type] = transition_counts.get(t_type, 0) + 1

        print(f"  Transition types detected:")
        for t_type, count in sorted(transition_counts.items()):
            pct = (count / enhanced_count * 100) if enhanced_count > 0 else 0
            print(f"    - {t_type}: {count} ({pct:.1f}%)")

        # Quality distribution
        quality_scores = [s['avg_quality'] for s in enhanced_result['scenes']]
        high_quality = sum(1 for q in quality_scores if q >= 0.8)
        medium_quality = sum(1 for q in quality_scores if 0.5 <= q < 0.8)
        low_quality = sum(1 for q in quality_scores if q < 0.5)

        print(f"\n  Scene quality distribution:")
        print(f"    - High (≥0.8): {high_quality} ({high_quality/enhanced_count*100:.1f}%)")
        print(f"    - Medium (0.5-0.8): {medium_quality} ({medium_quality/enhanced_count*100:.1f}%)")
        print(f"    - Low (<0.5): {low_quality} ({low_quality/enhanced_count*100:.1f}%)")

        # Recommendations
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATION")
        print("=" * 80)

        if diff_count < -20:  # Enhanced found 20+ fewer scenes
            print("✅ Enhanced detector significantly reduced false positives")
            print("   → Use Enhanced for better accuracy")
        elif diff_count > 20:  # Enhanced found 20+ more scenes
            print("⚠️  Enhanced detector found many more scenes")
            print("   → May be detecting gradual transitions")
            print("   → Review results to verify")
        else:
            print("✅ Both detectors found similar scene counts")
            print(f"   → Enhanced is {diff_time_pct:.0f}% slower but provides:")
            print("     • Transition type classification")
            print("     • Per-scene quality scores")
            print("     • Adaptive thresholding")
            print("   → Recommended for production use")

        if enhanced_time > basic_time * 3:
            print(f"\n⚠️  Enhanced is {diff_time_pct:.0f}% slower")
            print("   → Consider disabling motion detection if speed critical:")
            print("     SceneDetectorEnhanced(enable_motion=False)")

    print("\n" + "=" * 80)
    print("✓ Comparison complete")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_scene_detectors.py path/to/video.mp4")
        sys.exit(1)

    compare_detectors(sys.argv[1])
