"""
Quick Visual Sampler - Phase 2: Extract frames ONLY

Fast frame extraction at 2fps with basic quality filtering.
All vision models (CLIP, YOLO, MediaPipe, Places365) run in Phase 3.

Cost: $0
Time: ~30 seconds for typical video
Output: Frame metadata (timestamp, frame_id, frame_path, quality)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
import json

logger = logging.getLogger(__name__)


class QuickVisualSampler:
    """Sample frames and run FREE models for visual context"""

    def __init__(self, enable_blip2: bool = False):
        """
        Initialize sampler (NO MODELS - just frame extraction)

        Args:
            enable_blip2: DEPRECATED - no longer used
        """
        self.enable_blip2 = enable_blip2
        self.models_loaded = True  # No models needed anymore
        logger.info("QuickVisualSampler initialized (frame extraction only)")
    
    def sample_and_analyze(
        self,
        video_path: str,
        scenes: List[Dict],
        min_quality: float = 0.0,
        mode: str = "scene",  # "scene" or "fps"
        fps_rate: float = 2.0,  # For fps mode: frames per second to extract
        frames_output_dir: str = None  # Where to save extracted frames
    ) -> Dict:
        """
        Sample frames and run all FREE models.

        Args:
            video_path: Path to video
            scenes: List of scene dicts from scene_detector
            min_quality: Minimum scene quality (0.0-1.0). Skip scenes below this (default: 0.0 = all scenes)
            mode: "scene" for 1 frame per scene, "fps" for uniform FPS sampling
            fps_rate: For fps mode, how many frames per second to extract (default: 2.0)
            frames_output_dir: Directory to save extracted frame images (required for Pass 2A/2B)

        Returns:
            {
                'samples': [list of analyzed frames with frame_id and frame_path],
                'total_sampled': int,
                'skipped_low_quality': int
            }
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Check initialization.")

        # Create frames output directory if specified
        if frames_output_dir:
            frames_dir = Path(frames_output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving frames to: {frames_dir}")
        else:
            frames_dir = None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        samples = []
        skipped_count = 0

        if mode == "fps":
            # FPS Mode: Extract frames ONLY (no vision models)
            logger.info(f"Extracting frames at {fps_rate} FPS (vision models will run in Phase 3)...")

            # Calculate timestamps
            interval = 1.0 / fps_rate
            timestamps = np.arange(0, video_duration, interval)

            logger.info(f"Will extract ~{len(timestamps)} frames")

            for i, timestamp in enumerate(timestamps):
                # Extract frame
                frame_num = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Quick quality check
                quality = self._assess_quality(frame)
                if quality < min_quality:
                    skipped_count += 1
                    continue

                # Save frame to disk (REQUIRED for Phase 3)
                frame_path = None
                if frames_dir:
                    frame_filename = f"frame_{frame_num:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                else:
                    logger.warning("No frames_output_dir specified - frames won't be saved!")
                    continue

                # Store minimal metadata (Phase 3 will run all vision models)
                samples.append({
                    'frame_id': frame_num,
                    'frame_path': str(frame_path),
                    'timestamp': timestamp,
                    'quality': quality
                })

                if len(samples) % 100 == 0:
                    logger.info(f"Extracted {len(samples)}/{len(timestamps)} frames...")

        else:
            # Scene Mode: 1 frame per scene (frame extraction only)
            logger.info(f"Extracting {len(scenes)} scene frames...")
            if min_quality > 0:
                logger.info(f"  Filtering scenes with quality < {min_quality:.2f}")

            for scene in scenes:
                # Skip low-quality scenes if threshold set
                scene_quality = scene.get('avg_quality', 1.0)
                if scene_quality < min_quality:
                    skipped_count += 1
                    continue

                # Pick middle of scene
                mid_timestamp = (scene['start'] + scene['end']) / 2

                # Extract frame
                frame_num = int(mid_timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Quick quality check
                quality = self._assess_quality(frame)

                # Save frame to disk (REQUIRED for Phase 3)
                frame_path = None
                if frames_dir:
                    frame_filename = f"frame_{frame_num:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                else:
                    logger.warning("No frames_output_dir specified - frames won't be saved!")
                    continue

                # Store minimal metadata
                samples.append({
                    'frame_id': frame_num,
                    'frame_path': str(frame_path),
                    'timestamp': mid_timestamp,
                    'quality': quality
                })

                if len(samples) % 10 == 0:
                    logger.info(f"Extracted {len(samples)}/{len(scenes)} frames...")

        cap.release()

        logger.info(f"✅ Extracted {len(samples)} frames (vision models will run in Phase 3)")
        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} low-quality frames (< {min_quality:.2f})")

        return {
            'samples': samples,
            'total_sampled': len(samples),
            'skipped_low_quality': skipped_count
        }
    
    def _assess_quality(self, frame) -> float:
        """Quick quality assessment"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur score
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray) / 255.0
        
        # Quality score (0-1)
        blur_ok = blur_score > 100
        brightness_ok = 0.1 < brightness < 0.9
        
        if blur_ok and brightness_ok:
            return 1.0
        elif blur_ok or brightness_ok:
            return 0.5
        else:
            return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    import sys
    if len(sys.argv) > 1:
        sampler = QuickVisualSampler()
        
        # Mock scenes
        scenes = [{'start': 0, 'end': 10}, {'start': 10, 'end': 20}]
        
        result = sampler.sample_and_analyze(sys.argv[1], scenes)
        print(f"Sampled {result['total_sampled']} frames")
