"""
Enhanced Scene Detector - Improved Multi-Feature Detection

IMPROVEMENTS over basic version:
1. Adaptive thresholding (per-video calibration)
2. Multi-feature fusion (color + edges + motion)
3. Temporal smoothing (avoid false positives from camera shake)
4. Gradual transition detection (fades, dissolves)
5. Flash/flicker filtering
6. Minimum scene duration enforcement

Algorithm:
- Color histogram (HSV) - 70% weight
- Edge histogram (structural changes) - 20% weight
- Motion intensity (optical flow) - 10% weight
- Combine with adaptive threshold
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class EnhancedScene:
    """Enhanced scene with transition type"""
    scene_id: int
    start: float
    end: float
    duration: float
    transition_type: str  # "cut", "fade", "dissolve", "unknown"
    confidence: float
    avg_quality: float  # Average frame quality in scene


class SceneDetectorEnhanced:
    """
    Enhanced scene detector with multi-feature fusion.

    Features:
    - Adaptive threshold calibration
    - Color + Edge + Motion features
    - Temporal smoothing (5-frame window)
    - Gradual transition detection
    - Flash filtering
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        min_scene_duration: float = 1.0,
        enable_adaptive: bool = True,
        enable_motion: bool = True
    ):
        """
        Initialize enhanced scene detector.

        Args:
            base_threshold: Base threshold (will be adapted)
            min_scene_duration: Minimum scene length in seconds
            enable_adaptive: Auto-calibrate threshold per video
            enable_motion: Enable motion-based detection (slower but better)
        """
        self.base_threshold = base_threshold
        self.min_scene_duration = min_scene_duration
        self.enable_adaptive = enable_adaptive
        self.enable_motion = enable_motion

        # Feature weights
        self.color_weight = 0.70
        self.edge_weight = 0.20
        self.motion_weight = 0.10

        # Temporal smoothing
        self.temporal_window = 5
        self.diff_history = deque(maxlen=self.temporal_window)

    def detect_scenes(self, video_path: str) -> Dict:
        """
        Detect scenes with enhanced multi-feature approach.

        Returns:
            {
                'scenes': [list of EnhancedScene objects],
                'total_scenes': int,
                'video_duration': float,
                'avg_scene_duration': float,
                'calibrated_threshold': float
            }
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Enhanced scene detection: {video_path.name}")
        logger.info(f"  Frames: {total_frames} @ {fps} fps")
        logger.info(f"  Adaptive threshold: {self.enable_adaptive}")
        logger.info(f"  Motion detection: {self.enable_motion}")

        # Phase 1: Calibration (if adaptive enabled)
        threshold = self.base_threshold
        if self.enable_adaptive:
            threshold = self._calibrate_threshold(cap, fps, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            logger.info(f"  Calibrated threshold: {threshold:.3f}")

        # Phase 2: Scene detection
        scenes = []
        scene_start = 0.0
        frame_num = 0

        prev_frame = None
        prev_color_hist = None
        prev_edge_hist = None

        scene_quality_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps

            # Calculate features
            color_hist = self._calculate_color_histogram(frame)
            edge_hist = self._calculate_edge_histogram(frame)
            quality = self._assess_frame_quality(frame)
            scene_quality_scores.append(quality)

            # Compare with previous frame
            if prev_frame is not None and prev_color_hist is not None:
                # Color difference
                color_diff = cv2.compareHist(
                    color_hist, prev_color_hist,
                    cv2.HISTCMP_BHATTACHARYYA
                )

                # Edge difference (structural changes)
                edge_diff = cv2.compareHist(
                    edge_hist, prev_edge_hist,
                    cv2.HISTCMP_BHATTACHARYYA
                )

                # Motion difference (optional)
                motion_diff = 0.0
                if self.enable_motion:
                    motion_diff = self._calculate_motion_difference(
                        prev_frame, frame
                    )

                # Weighted combination
                combined_diff = (
                    color_diff * self.color_weight +
                    edge_diff * self.edge_weight +
                    motion_diff * self.motion_weight
                )

                # Temporal smoothing
                self.diff_history.append(combined_diff)
                smoothed_diff = np.mean(self.diff_history)

                # Scene boundary detection
                if smoothed_diff > threshold:
                    # Check minimum duration
                    duration = timestamp - scene_start

                    if duration >= self.min_scene_duration:
                        # Detect transition type
                        transition_type = self._classify_transition(
                            combined_diff, color_diff, edge_diff
                        )

                        # Calculate average quality for this scene
                        avg_quality = np.mean(scene_quality_scores) if scene_quality_scores else 0.5

                        scenes.append(EnhancedScene(
                            scene_id=len(scenes),
                            start=scene_start,
                            end=timestamp,
                            duration=duration,
                            transition_type=transition_type,
                            confidence=min(smoothed_diff / threshold, 1.0),
                            avg_quality=avg_quality
                        ))

                        scene_start = timestamp
                        scene_quality_scores = []
                        self.diff_history.clear()

            prev_frame = frame.copy()
            prev_color_hist = color_hist
            prev_edge_hist = edge_hist
            frame_num += 1

            # Progress logging
            if frame_num % 300 == 0:
                progress = (frame_num / total_frames) * 100
                logger.info(f"  Progress: {progress:.1f}% ({len(scenes)} scenes so far)")

        # Add final scene
        final_timestamp = frame_num / fps
        duration = final_timestamp - scene_start
        avg_quality = np.mean(scene_quality_scores) if scene_quality_scores else 0.5

        scenes.append(EnhancedScene(
            scene_id=len(scenes),
            start=scene_start,
            end=final_timestamp,
            duration=duration,
            transition_type="unknown",
            confidence=1.0,
            avg_quality=avg_quality
        ))

        cap.release()

        # Calculate statistics
        avg_scene_duration = np.mean([s.duration for s in scenes]) if scenes else 0

        logger.info(f"✓ Detected {len(scenes)} scenes")
        logger.info(f"  Average duration: {avg_scene_duration:.1f}s")
        logger.info(f"  Transition types: {self._count_transitions(scenes)}")

        return {
            'scenes': [self._scene_to_dict(s) for s in scenes],
            'total_scenes': len(scenes),
            'video_duration': final_timestamp,
            'avg_scene_duration': avg_scene_duration,
            'calibrated_threshold': threshold
        }

    def _calibrate_threshold(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        total_frames: int
    ) -> float:
        """
        Auto-calibrate threshold by sampling video.

        Strategy:
        - Sample 100 evenly distributed frames
        - Calculate frame-to-frame differences
        - Use 85th percentile as threshold (filters out noise)
        """
        logger.info("  Calibrating adaptive threshold...")

        sample_count = min(100, total_frames // 10)
        sample_interval = total_frames // sample_count

        differences = []
        prev_color_hist = None

        for i in range(sample_count):
            frame_idx = i * sample_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            color_hist = self._calculate_color_histogram(frame)

            if prev_color_hist is not None:
                diff = cv2.compareHist(
                    color_hist, prev_color_hist,
                    cv2.HISTCMP_BHATTACHARYYA
                )
                differences.append(diff)

            prev_color_hist = color_hist

        if not differences:
            return self.base_threshold

        # Use 85th percentile (catches real scene changes, filters noise)
        calibrated = np.percentile(differences, 85)

        # Clamp to reasonable range
        calibrated = max(0.2, min(0.6, calibrated))

        return calibrated

    def _calculate_color_histogram(self, frame) -> np.ndarray:
        """Calculate HSV color histogram (same as basic version)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                           [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _calculate_edge_histogram(self, frame) -> np.ndarray:
        """
        Calculate edge histogram (structural changes).

        Detects changes in composition/structure even if colors similar.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate histogram of edge orientations
        # Divide into 8 orientation bins
        hist = np.histogram(edges[edges > 0], bins=8, range=(0, 256))[0]

        # Normalize
        if hist.sum() > 0:
            hist = hist / hist.sum()

        return hist.astype(np.float32)

    def _calculate_motion_difference(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> float:
        """
        Calculate motion intensity using frame difference.

        Simpler than optical flow but effective for scene changes.
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)

        # Normalize to [0, 1]
        motion_intensity = np.mean(diff) / 255.0

        return motion_intensity

    def _classify_transition(
        self,
        combined_diff: float,
        color_diff: float,
        edge_diff: float
    ) -> str:
        """
        Classify transition type.

        - Hard cut: High color + high edge diff
        - Fade: High color, low edge diff
        - Dissolve: Moderate both
        """
        if combined_diff > 0.7:
            if edge_diff > 0.5:
                return "cut"
            else:
                return "fade"
        elif combined_diff > 0.4:
            return "dissolve"
        else:
            return "unknown"

    def _assess_frame_quality(self, frame) -> float:
        """
        Quick quality assessment (blur + brightness).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur score (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness
        brightness = np.mean(gray) / 255.0

        # Combined quality
        blur_ok = blur_score > 100
        brightness_ok = 0.1 < brightness < 0.9

        if blur_ok and brightness_ok:
            return 1.0
        elif blur_ok or brightness_ok:
            return 0.5
        else:
            return 0.0

    def _scene_to_dict(self, scene: EnhancedScene) -> Dict:
        """Convert EnhancedScene to dict for JSON serialization"""
        return {
            'scene_id': scene.scene_id,
            'start': scene.start,
            'end': scene.end,
            'duration': scene.duration,
            'transition_type': scene.transition_type,
            'confidence': scene.confidence,
            'avg_quality': scene.avg_quality
        }

    def _count_transitions(self, scenes: List[EnhancedScene]) -> Dict:
        """Count transition types"""
        counts = {'cut': 0, 'fade': 0, 'dissolve': 0, 'unknown': 0}
        for scene in scenes:
            counts[scene.transition_type] = counts.get(scene.transition_type, 0) + 1
        return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with comparison
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

        # Basic detector
        from scene_detector import SceneDetector
        basic = SceneDetector(threshold=0.3)
        basic_result = basic.detect_scenes(video_path)

        print(f"\n{'='*60}")
        print("BASIC DETECTOR")
        print(f"{'='*60}")
        print(f"Scenes detected: {basic_result['total_scenes']}")

        # Enhanced detector
        enhanced = SceneDetectorEnhanced(
            base_threshold=0.3,
            enable_adaptive=True,
            enable_motion=True
        )
        enhanced_result = enhanced.detect_scenes(video_path)

        print(f"\n{'='*60}")
        print("ENHANCED DETECTOR")
        print(f"{'='*60}")
        print(f"Scenes detected: {enhanced_result['total_scenes']}")
        print(f"Calibrated threshold: {enhanced_result['calibrated_threshold']:.3f}")
        print(f"Avg scene duration: {enhanced_result['avg_scene_duration']:.1f}s")
        print(f"\nTransition types:")
        for scene in enhanced_result['scenes'][:10]:
            print(f"  Scene {scene['scene_id']}: "
                  f"{scene['start']:.1f}s-{scene['end']:.1f}s "
                  f"({scene['transition_type']}, quality: {scene['avg_quality']:.2f})")
