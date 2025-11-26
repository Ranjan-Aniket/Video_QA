"""
Quick Visual Sampler - Phase 2: Sample frames + Run FREE models

Samples 1 frame per scene (~50-100 frames) and runs ALL FREE models:
- BLIP-2 (captions)
- CLIP (embeddings)
- Places365 (scene classification)
- YOLO (object detection)
- OCR (text extraction)
- Pose (human poses)
- FER (emotions)

Cost: $0 (all local models)
Output: visual_context.json with rich visual information
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
        Initialize sampler and load models

        Args:
            enable_blip2: Enable BLIP-2 captioning (SLOW - 90s/frame, adds 2+ hours)
        """
        self.enable_blip2 = enable_blip2
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load all FREE models"""
        try:
            # Import all processors
            from .clip_processor import CLIPProcessor
            from .places365_processor import Places365Processor
            from .object_detector import ObjectDetector
            from .ocr_processor import OCRProcessor
            from .pose_detector import PoseDetector

            # Auto-detect GPU availability
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("ℹ️  No GPU detected - using CPU mode")
            except ImportError:
                logger.info("ℹ️  PyTorch not available - using CPU mode")

            device = "cuda" if gpu_available else "cpu"

            # BLIP-2 is OPTIONAL (very slow - 90s per frame!)
            if self.enable_blip2:
                from .blip2_processor import BLIP2Processor
                self.blip2 = BLIP2Processor()
                logger.info("⚠️  BLIP-2 enabled - expect 2+ hours for Phase 2")
            else:
                self.blip2 = None
                logger.info("✓ BLIP-2 disabled - Phase 2 will be ~15 minutes instead of 2+ hours")

            # Load models with auto-detected device
            self.clip = CLIPProcessor(device=device)
            self.places365 = Places365Processor()  # CPU only (no GPU support)
            self.yolo = ObjectDetector(device=device)
            self.ocr = OCRProcessor()  # GPU enabled in ocr_processor.py if available
            self.pose = PoseDetector()  # MediaPipe auto-detects GPU
            
            # Optional: FER (if available)
            try:
                from .fer_processor import FERProcessor
                self.fer = FERProcessor()
            except:
                self.fer = None
                logger.warning("FER processor not available")
            
            self.models_loaded = True
            logger.info("All FREE models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
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
            # FPS Mode: Extract frames at uniform FPS rate
            logger.info(f"Sampling at {fps_rate} FPS with FREE models...")

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

                # Quality check
                quality = self._assess_quality(frame)
                if quality < min_quality:
                    skipped_count += 1
                    continue

                # Save frame to disk if output directory specified
                frame_path = None
                if frames_dir:
                    frame_filename = f"frame_{frame_num:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)

                # Run all FREE models on this frame
                analysis = self._analyze_frame(frame, timestamp, frame_num, frame_path)
                samples.append(analysis)

                if len(samples) % 100 == 0:
                    logger.info(f"Processed {len(samples)}/{len(timestamps)} frames...")

        else:
            # Scene Mode: 1 frame per scene (original behavior)
            logger.info(f"Sampling {len(scenes)} scenes with FREE models...")
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

                # Save frame to disk if output directory specified
                frame_path = None
                if frames_dir:
                    frame_filename = f"frame_{frame_num:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)

                # Run all FREE models on this frame
                analysis = self._analyze_frame(frame, mid_timestamp, frame_num, frame_path)
                samples.append(analysis)

                if len(samples) % 10 == 0:
                    logger.info(f"Processed {len(samples)}/{len(scenes)} samples...")

        cap.release()

        logger.info(f"Completed: {len(samples)} frames analyzed with FREE models")
        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} low-quality scenes (< {min_quality:.2f})")

        return {
            'samples': samples,
            'total_sampled': len(samples),
            'skipped_low_quality': skipped_count
        }
    
    def _analyze_frame(self, frame, timestamp: float, frame_num: int, frame_path: Path = None) -> Dict:
        """Run all FREE models on single frame"""

        # 1. BLIP-2: Generate caption (OPTIONAL - very slow!)
        blip2_caption = None
        if self.blip2:
            blip2_caption = self.blip2.generate_caption(frame)

        # 2. CLIP: Extract embeddings + attributes
        clip_embedding = self.clip.encode_image(frame)

        # 3. Places365: Classify scene
        scene_result = self.places365.classify_scene(frame)

        # 4. YOLO: Detect objects (JIT)
        object_result = self.yolo.detect_objects_jit(frame, timestamp)

        # 5. OCR: DISABLED - Vision models (Claude/GPT-4o) will detect text in Phase 8
        # No need for redundant OCR processing - saves 20+ minutes per video
        ocr_result = {'text_blocks': [], 'block_count': 0}  # Skip OCR entirely

        # 6. Pose: Detect human poses
        pose_result = self.pose.detect_pose(frame)

        # 7. FER: Detect emotions (optional)
        emotions = []
        if self.fer:
            emotions = self.fer.detect_emotions(frame)

        # 8. Quality assessment
        quality = self._assess_quality(frame)

        return {
            'frame_id': frame_num,  # Critical for Pass 2A to find the frame file
            'frame_path': str(frame_path) if frame_path else None,
            'timestamp': timestamp,
            'blip2_caption': blip2_caption,
            'clip_embedding': clip_embedding.tolist() if clip_embedding is not None else [],
            'scene_type': scene_result.scene_category,
            'objects': [obj.to_dict() for obj in object_result.detections],
            'text_detected': ocr_result.get('all_text', '') if isinstance(ocr_result, dict) else ocr_result.all_text,
            'poses': pose_result.to_dict() if pose_result else {},
            'emotions': emotions,
            'quality': quality
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
