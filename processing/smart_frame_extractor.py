"""
Smart Frame Extractor - ENHANCED VERSION with Dense Sampling Support

DENSE SAMPLING ARCHITECTURE:
- Premium opportunities: Extract 10 frames each (0.5s intervals, ±2.5s window)
  * Total: 7 opportunities × 10 frames = 70 frames
  * Center frame marked as KEY FRAME for AI analysis
- Template opportunities: Extract 1 frame each
  * Total: 40 opportunities × 1 frame = 40 frames  
  * All marked as KEY FRAMES for AI analysis

KEY FRAMES (47 total):
- 7 premium center frames (GPT-4o + Claude)
- 40 template frames (GPT-4o + Claude)

Integration with smart_pipeline.py Phase 4:
- ALL frames get BLIP-2 + YOLO + OCR + Pose (FREE)
- Only KEY frames get GPT-4o + Claude ($0.02 each)
- Cost: 47 × $0.02 = $0.94
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FrameQualityMetrics:
    """Quality metrics for a frame"""
    blur_score: float
    brightness: float
    motion_level: float
    is_shot_boundary: bool
    is_black_frame: bool
    is_transition: bool
    overall_quality: float

    def is_good_quality(self, min_quality: float = 0.5) -> bool:
        return (
            self.overall_quality >= min_quality and
            not self.is_black_frame and
            not self.is_transition
        )


@dataclass
class ExtractedFrame:
    """Represents a single extracted frame with dense sampling support"""
    frame_id: str
    timestamp: float
    frame_type: str  # "premium", "template", "scene_boundary", "bulk"
    priority: str
    image_path: str
    
    opportunity_type: Optional[str] = None
    audio_cue: Optional[str] = None
    reason: Optional[str] = None
    analysis_needed: Optional[List[str]] = None
    recommended_models: Optional[List[str]] = None
    quality_metrics: Optional[FrameQualityMetrics] = None
    adjacent_frames: Optional[Dict[str, str]] = None
    
    # NEW: Dense sampling support
    is_key_frame: bool = False  # Mark for AI analysis
    cluster_id: Optional[str] = None  # Dense cluster ID
    cluster_position: Optional[int] = None  # Position 0-9
    
    width: Optional[int] = None
    height: Optional[int] = None
    frame_number: Optional[int] = None


class SmartFrameExtractorEnhanced:
    """Extract frames with dense sampling for premium opportunities"""
    
    MIN_BLUR_SCORE = 0.15
    MIN_BRIGHTNESS = 0.05
    MAX_BRIGHTNESS = 0.95
    SCENE_CHANGE_THRESHOLD = 0.3
    MIN_SCENE_DURATION = 1.0
    
    # NEW: Dense sampling configuration
    DENSE_SAMPLING_ENABLED = True  # Always enabled
    DENSE_WINDOW_SIZE = 2.5  # ±2.5s window
    DENSE_INTERVAL = 0.5  # 0.5s spacing
    DENSE_FRAMES_PER_CLUSTER = 10  # 10 frames per premium
    
    def __init__(self, video_path: str, output_dir: Optional[str] = None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        if output_dir is None:
            self.output_dir = self.video_path.parent / "frames" / self.video_path.stem
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps: Optional[float] = None
        self.total_frames: Optional[int] = None
        self.duration: Optional[float] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        
        self._prev_frame = None
        self._prev_hist = None
        self._frame_quality_cache = {}
        
        self._load_video_properties()
        
        logger.info(f"SmartFrameExtractorEnhanced initialized (DENSE SAMPLING ENABLED)")
        logger.info(f"  Video: {self.video_path.name}")
        logger.info(f"  FPS: {self.fps:.2f}")
        logger.info(f"  Duration: {self.duration:.1f}s")
        logger.info(f"  Dense config: {self.DENSE_FRAMES_PER_CLUSTER} frames/cluster @ {self.DENSE_INTERVAL}s intervals")
    
    def _load_video_properties(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    
    def _extract_dense_cluster(
        self,
        opp_id: str,
        center_timestamp: float,
        opportunity_type: str,
        audio_cue: str
    ) -> List[ExtractedFrame]:
        """
        Extract dense cluster of 10 frames around timestamp
        
        Frame spacing: 0.5s intervals
        Window: ±2.5s (5 frames before, 5 frames after... but center is 0, so 0-9)
        Center frame (position 5) marked as KEY FRAME
        """
        frames = []
        half_frames = self.DENSE_FRAMES_PER_CLUSTER // 2
        
        for i in range(-half_frames, half_frames):
            timestamp = center_timestamp + (i * self.DENSE_INTERVAL)
            
            # Skip if out of bounds
            if timestamp < 0 or timestamp > self.duration:
                continue
            
            # Position in cluster (0-9)
            position = i + half_frames
            
            # Center frame is key frame
            is_key = (i == 0)
            
            # Frame ID includes cluster and position
            frame_id = f"{opp_id}_dense_{position:02d}"
            
            frame = self._extract_frame_at_timestamp(
                timestamp=timestamp,
                frame_id=frame_id,
                frame_type="premium",
                priority="critical" if is_key else "high",
                opportunity_type=opportunity_type,
                audio_cue=audio_cue,
                reason=f"Dense frame {position}/10 for {opp_id}" + (" [KEY]" if is_key else ""),
                is_key_frame=is_key,
                cluster_id=opp_id,
                cluster_position=position
            )
            
            if frame:
                frames.append(frame)
        
        return frames
    
    def extract_from_opportunities(self, opportunities_path: str) -> List[ExtractedFrame]:
        """
        Extract frames with DENSE SAMPLING for premium opportunities
        
        NEW BEHAVIOR:
        - Premium opportunities: Extract 10 dense frames each
        - Template opportunities: Extract 1 frame each (marked as key)
        - Total: ~110 frames (70 dense + 40 template)
        - Key frames: 47 (7 premium centers + 40 template)
        """
        logger.info("=" * 80)
        logger.info("EXTRACTING FRAMES WITH DENSE SAMPLING")
        logger.info("=" * 80)
        
        with open(opportunities_path, 'r') as f:
            opps_data = json.load(f)
        
        extracted_frames = []
        frame_counter = {"premium_clusters": 0, "template": 0, "total_dense": 0}
        
        # Get premium frames list
        premium_frames_list = opps_data.get("premium_frames", [])
        logger.info(f"\n📍 Step 1: Extracting DENSE clusters for {len(premium_frames_list)} premium opportunities...")
        logger.info(f"   Expected: {len(premium_frames_list)} × 10 = {len(premium_frames_list) * 10} frames")
        
        for pf in premium_frames_list:
            timestamp = pf.get("timestamp", pf.get("visual_timestamp", 0))
            opp_id = pf.get("opportunity_id", f"premium_{frame_counter['premium_clusters']:03d}")
            opp_type = pf.get("opportunity_type", "premium_keyframe")
            audio_quote = pf.get("audio_quote", "")

            # Validate timestamp (skip if invalid - common with Whisper bugs)
            if timestamp < 0 or timestamp > self.duration:
                logger.warning(f"   ✗ Skipping {opp_id}: invalid timestamp {timestamp:.1f}s (video duration: {self.duration:.1f}s)")
                continue

            # Extract dense cluster
            cluster_frames = self._extract_dense_cluster(
                opp_id=opp_id,
                center_timestamp=timestamp,
                opportunity_type=opp_type,
                audio_cue=audio_quote
            )
            
            extracted_frames.extend(cluster_frames)
            frame_counter["premium_clusters"] += 1
            frame_counter["total_dense"] += len(cluster_frames)
            
            # Find key frame
            key_frames = [f for f in cluster_frames if f.is_key_frame]
            key_info = f" [KEY at {key_frames[0].timestamp:.1f}s]" if key_frames else ""
            
            logger.info(f"   [{frame_counter['premium_clusters']}] {opp_id}: {len(cluster_frames)} frames @ {timestamp:.1f}s{key_info}")
        
        logger.info(f"   ✓ Total dense frames: {frame_counter['total_dense']}")
        
        # Extract template opportunities (single frame each, all key frames)
        opportunities = opps_data.get("opportunities", [])
        logger.info(f"\n🎯 Step 2: Extracting template frames (1 per opportunity, all KEY)...")
        logger.info(f"   Target: 40 template frames")
        
        template_count = 0
        for opp in opportunities:
            # Skip if this is a premium opportunity
            if opp.get("requires_premium_frame", False):
                continue
            
            # Stop at 40 template frames
            if template_count >= 40:
                break
            
            timestamp = opp.get("key_word_timestamp") or opp.get("visual_timestamp") or opp.get("audio_start", 0)
            opp_id = opp.get("opportunity_id", f"template_{template_count:03d}")
            opp_type = opp.get("opportunity_type", "template")
            audio_quote = opp.get("audio_quote", "")
            
            frame = self._extract_frame_at_timestamp(
                timestamp=timestamp,
                frame_id=f"template_{template_count:03d}",
                frame_type="template",
                priority="high",
                opportunity_type=opp_type,
                audio_cue=audio_quote,
                reason=f"Template opportunity {opp_id} [KEY]",
                is_key_frame=True  # All template frames are KEY
            )
            
            if frame:
                extracted_frames.append(frame)
                template_count += 1
                
                if template_count % 10 == 0:
                    logger.info(f"   Progress: {template_count}/40 template frames")
        
        logger.info(f"   ✓ Template frames: {template_count}")
        
        # Summary
        total = len(extracted_frames)
        premium_frames = [f for f in extracted_frames if f.frame_type == "premium"]
        template_frames = [f for f in extracted_frames if f.frame_type == "template"]
        key_frames = [f for f in extracted_frames if f.is_key_frame]
        
        logger.info("=" * 80)
        logger.info(f"✅ EXTRACTED {total} FRAMES (DENSE SAMPLING)")
        logger.info(f"   Premium (dense): {len(premium_frames)} frames in {frame_counter['premium_clusters']} clusters")
        logger.info(f"   Template: {len(template_frames)} frames")
        logger.info(f"   KEY frames (for AI): {len(key_frames)}")
        logger.info(f"     - Premium center: {len([f for f in key_frames if f.frame_type == 'premium'])}")
        logger.info(f"     - Template: {len([f for f in key_frames if f.frame_type == 'template'])}")
        logger.info("=" * 80)
        
        return extracted_frames
    
    def _extract_frame_at_timestamp(
        self,
        timestamp: float,
        frame_id: str,
        frame_type: str,
        priority: str,
        opportunity_type: Optional[str] = None,
        audio_cue: Optional[str] = None,
        reason: Optional[str] = None,
        analysis_needed: Optional[List[str]] = None,
        recommended_models: Optional[List[str]] = None,
        quality_metrics: Optional[FrameQualityMetrics] = None,
        is_key_frame: bool = False,
        cluster_id: Optional[str] = None,
        cluster_position: Optional[int] = None
    ) -> Optional[ExtractedFrame]:
        """Extract single frame with all metadata"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video")
            return None
        
        frame_number = int(timestamp * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_img = cap.read()
        cap.release()
        
        if not ret:
            logger.warning(f"Failed to extract frame at {timestamp:.1f}s")
            return None
        
        # Save frame
        frame_filename = f"{frame_id}_{timestamp:.2f}s.jpg"
        frame_path = self.output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame_img)
        
        return ExtractedFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_type=frame_type,
            priority=priority,
            image_path=str(frame_path),
            opportunity_type=opportunity_type,
            audio_cue=audio_cue,
            reason=reason,
            analysis_needed=analysis_needed,
            recommended_models=recommended_models,
            quality_metrics=quality_metrics,
            is_key_frame=is_key_frame,
            cluster_id=cluster_id,
            cluster_position=cluster_position,
            width=frame_img.shape[1],
            height=frame_img.shape[0],
            frame_number=frame_number
        )
    
    def save_frame_metadata(self, frames: List[ExtractedFrame], output_path: Optional[Path] = None):
        """Save frame metadata to JSON"""
        if output_path is None:
            output_path = self.output_dir / "frames_metadata.json"
        
        frames_data = []
        for frame in frames:
            frame_dict = asdict(frame)
            if frame.quality_metrics:
                frame_dict["quality_metrics"] = asdict(frame.quality_metrics)
            frames_data.append(frame_dict)
        
        # Count clusters
        clusters = {}
        for f in frames:
            if f.cluster_id:
                if f.cluster_id not in clusters:
                    clusters[f.cluster_id] = []
                clusters[f.cluster_id].append(f)
        
        metadata = {
            "video_path": str(self.video_path),
            "extraction_time": datetime.now().isoformat(),
            "dense_sampling_enabled": self.DENSE_SAMPLING_ENABLED,
            "dense_config": {
                "frames_per_cluster": self.DENSE_FRAMES_PER_CLUSTER,
                "interval": self.DENSE_INTERVAL,
                "window_size": self.DENSE_WINDOW_SIZE
            },
            "summary": {
                "total_frames": len(frames),
                "premium_frames": len([f for f in frames if f.frame_type == "premium"]),
                "template_frames": len([f for f in frames if f.frame_type == "template"]),
                "key_frames": len([f for f in frames if f.is_key_frame]),
                "dense_clusters": len(clusters)
            },
            "clusters": {
                cluster_id: {
                    "frame_count": len(cluster_frames),
                    "center_timestamp": [f.timestamp for f in cluster_frames if f.is_key_frame][0] if any(f.is_key_frame for f in cluster_frames) else None,
                    "time_range": [min(f.timestamp for f in cluster_frames), max(f.timestamp for f in cluster_frames)]
                }
                for cluster_id, cluster_frames in clusters.items()
            },
            "video_properties": {
                "fps": self.fps,
                "duration": self.duration,
                "width": self.width,
                "height": self.height
            },
            "frames": frames_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Saved frame metadata to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        video_path = sys.argv[1]
        opportunities_path = sys.argv[2]
        
        extractor = SmartFrameExtractorEnhanced(video_path)
        frames = extractor.extract_from_opportunities(opportunities_path)
        extractor.save_frame_metadata(frames)
        
        # Show summary
        key_frames = [f for f in frames if f.is_key_frame]
        clusters = {}
        for f in frames:
            if f.cluster_id:
                clusters[f.cluster_id] = clusters.get(f.cluster_id, 0) + 1
        
        print(f"\n✅ Extracted {len(frames)} total frames")
        print(f"   - Dense clusters: {len(clusters)}")
        print(f"   - Frames per cluster: {list(clusters.values())}")
        print(f"   - Key frames: {len(key_frames)}")
    else:
        print("Usage: python smart_frame_extractor.py <video_path> <opportunities_json>")