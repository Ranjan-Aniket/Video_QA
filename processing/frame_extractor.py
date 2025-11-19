"""
Frame Extractor - Smart Frame Sampling

Purpose: Extract frames from video with intelligent sampling strategies
Compliance: JIT extraction per question, minimize costs
Architecture: Evidence-first, configurable sampling rates

Sampling Strategies:
1. Uniform: Extract frames at fixed intervals (e.g., 1 frame/second)
2. Scene-based: Extract key frames from each scene
3. On-demand: Extract frames only for specific timestamps (JIT)
4. Adaptive: Higher sampling for complex scenes, lower for simple ones
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Frame sampling strategies"""
    UNIFORM = "uniform"  # Fixed interval
    SCENE_BASED = "scene_based"  # Key frames per scene
    ON_DEMAND = "on_demand"  # Specific timestamps only (JIT)
    ADAPTIVE = "adaptive"  # Variable rate based on content


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction"""
    strategy: SamplingStrategy = SamplingStrategy.ON_DEMAND
    
    # Uniform strategy settings
    uniform_fps: float = 1.0  # Extract 1 frame per second
    
    # Scene-based strategy settings
    frames_per_scene: int = 3  # Extract 3 key frames per scene
    
    # Adaptive strategy settings
    min_fps: float = 0.5  # Minimum sampling rate
    max_fps: float = 2.0  # Maximum sampling rate
    
    # Quality settings
    resize_width: Optional[int] = None  # Resize frames (None = original)
    resize_height: Optional[int] = None
    jpeg_quality: int = 85  # JPEG compression quality (0-100)
    
    # Cost optimization
    max_frames_per_video: int = 300  # Hard limit to control costs


@dataclass
class ExtractedFrame:
    """Single extracted frame with metadata"""
    frame_index: int  # Frame number in video
    timestamp: float  # Time in seconds
    frame_data: np.ndarray  # Actual frame image (HxWxC)
    scene_id: Optional[int] = None  # Which scene this frame belongs to
    is_key_frame: bool = False  # True if this is a scene key frame
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get frame dimensions (height, width, channels)"""
        return self.frame_data.shape
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without frame data)"""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "scene_id": self.scene_id,
            "is_key_frame": self.is_key_frame,
            "shape": self.shape
        }


@dataclass
class FrameExtractionResult:
    """Result of frame extraction operation"""
    video_id: str
    frames: List[ExtractedFrame]
    total_frames_in_video: int
    extraction_strategy: SamplingStrategy
    cost_estimate: float
    
    @property
    def frame_count(self) -> int:
        """Number of frames extracted"""
        return len(self.frames)
    
    @property
    def timestamps(self) -> List[float]:
        """List of all frame timestamps"""
        return [f.timestamp for f in self.frames]


class FrameExtractor:
    """
    Extract frames from video using various sampling strategies.
    
    Optimized for JIT evidence extraction to minimize costs.
    """
    
    def __init__(self, config: Optional[FrameExtractionConfig] = None):
        """
        Initialize frame extractor.
        
        Args:
            config: Frame extraction configuration
        """
        self.config = config or FrameExtractionConfig()
        
        logger.info(
            f"FrameExtractor initialized with strategy: {self.config.strategy.value}"
        )
    
    def extract_frames(
        self,
        video_path: Path,
        video_id: str,
        scene_boundaries: Optional[List[Tuple[float, float]]] = None,
        specific_timestamps: Optional[List[float]] = None
    ) -> FrameExtractionResult:
        """
        Extract frames from video based on configured strategy.
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            scene_boundaries: Optional list of (start, end) timestamps per scene
            specific_timestamps: Optional specific timestamps to extract (for JIT)
        
        Returns:
            FrameExtractionResult with extracted frames
        """
        logger.info(
            f"Extracting frames from {video_id} using {self.config.strategy.value}"
        )
        
        # Get video metadata
        fps, total_frames, duration = self._get_video_info(video_path)
        
        # Select extraction method based on strategy
        if self.config.strategy == SamplingStrategy.ON_DEMAND:
            if specific_timestamps is None:
                raise ValueError(
                    "ON_DEMAND strategy requires specific_timestamps parameter"
                )
            frames = self._extract_at_timestamps(
                video_path, specific_timestamps, fps
            )
        
        elif self.config.strategy == SamplingStrategy.UNIFORM:
            frames = self._extract_uniform(video_path, fps, duration)
        
        elif self.config.strategy == SamplingStrategy.SCENE_BASED:
            if scene_boundaries is None:
                raise ValueError(
                    "SCENE_BASED strategy requires scene_boundaries parameter"
                )
            frames = self._extract_scene_based(
                video_path, scene_boundaries, fps
            )
        
        elif self.config.strategy == SamplingStrategy.ADAPTIVE:
            frames = self._extract_adaptive(
                video_path, fps, duration, scene_boundaries
            )
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.strategy}")
        
        # Enforce frame limit
        if len(frames) > self.config.max_frames_per_video:
            logger.warning(
                f"Extracted {len(frames)} frames exceeds limit "
                f"{self.config.max_frames_per_video}, truncating"
            )
            frames = frames[:self.config.max_frames_per_video]
        
        # Calculate cost estimate
        cost = self._calculate_extraction_cost(len(frames))
        
        result = FrameExtractionResult(
            video_id=video_id,
            frames=frames,
            total_frames_in_video=total_frames,
            extraction_strategy=self.config.strategy,
            cost_estimate=cost
        )
        
        logger.info(
            f"Extracted {len(frames)}/{total_frames} frames "
            f"(cost: ${cost:.4f})"
        )
        
        return result
    
    def extract_single_frame(
        self, video_path: Path, timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at specific timestamp (JIT).
        
        This is the most cost-effective method for on-demand extraction.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds to extract frame
        
        Returns:
            Frame as numpy array or None if extraction failed
        """
        # TODO: Implement with cv2.VideoCapture
        # import cv2
        # cap = cv2.VideoCapture(str(video_path))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # frame_number = int(timestamp * fps)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # ret, frame = cap.read()
        # cap.release()
        # 
        # if ret:
        #     return self._preprocess_frame(frame)
        # return None
        
        logger.warning("extract_single_frame not implemented - placeholder")
        return None
    
    def _get_video_info(
        self, video_path: Path
    ) -> Tuple[float, int, float]:
        """
        Get video metadata (FPS, total frames, duration).
        
        Returns:
            (fps, total_frames, duration_seconds)
        """
        # TODO: Implement with cv2.VideoCapture
        # import cv2
        # cap = cv2.VideoCapture(str(video_path))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration = total_frames / fps
        # cap.release()
        # return fps, total_frames, duration
        
        # Placeholder
        return 30.0, 1800, 60.0
    
    def _extract_at_timestamps(
        self,
        video_path: Path,
        timestamps: List[float],
        fps: float
    ) -> List[ExtractedFrame]:
        """
        Extract frames at specific timestamps (JIT extraction).
        
        This is the recommended approach for cost optimization.
        """
        frames = []
        
        # TODO: Implement with cv2.VideoCapture
        # import cv2
        # cap = cv2.VideoCapture(str(video_path))
        # 
        # for timestamp in sorted(timestamps):
        #     frame_number = int(timestamp * fps)
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        #     ret, frame_data = cap.read()
        #     
        #     if ret:
        #         frame_data = self._preprocess_frame(frame_data)
        #         frames.append(ExtractedFrame(
        #             frame_index=frame_number,
        #             timestamp=timestamp,
        #             frame_data=frame_data
        #         ))
        # 
        # cap.release()
        
        logger.warning("_extract_at_timestamps not implemented - placeholder")
        return frames
    
    def _extract_uniform(
        self, video_path: Path, fps: float, duration: float
    ) -> List[ExtractedFrame]:
        """Extract frames at uniform intervals"""
        frames = []
        interval = 1.0 / self.config.uniform_fps  # Time between frames
        
        # TODO: Implement with cv2.VideoCapture
        # timestamps = np.arange(0, duration, interval)
        # return self._extract_at_timestamps(video_path, timestamps.tolist(), fps)
        
        logger.warning("_extract_uniform not implemented - placeholder")
        return frames
    
    def _extract_scene_based(
        self,
        video_path: Path,
        scene_boundaries: List[Tuple[float, float]],
        fps: float
    ) -> List[ExtractedFrame]:
        """Extract key frames from each scene"""
        frames = []
        
        for scene_id, (start, end) in enumerate(scene_boundaries):
            scene_duration = end - start
            
            # Extract evenly spaced frames within scene
            timestamps = np.linspace(
                start, end, self.config.frames_per_scene
            )
            
            # TODO: Extract frames at these timestamps
            # scene_frames = self._extract_at_timestamps(
            #     video_path, timestamps.tolist(), fps
            # )
            # for frame in scene_frames:
            #     frame.scene_id = scene_id
            #     frame.is_key_frame = True
            # frames.extend(scene_frames)
        
        logger.warning("_extract_scene_based not implemented - placeholder")
        return frames
    
    def _extract_adaptive(
        self,
        video_path: Path,
        fps: float,
        duration: float,
        scene_boundaries: Optional[List[Tuple[float, float]]]
    ) -> List[ExtractedFrame]:
        """
        Extract frames with adaptive sampling rate.
        
        Higher sampling for complex scenes, lower for simple ones.
        """
        frames = []
        
        # TODO: Implement adaptive sampling based on:
        # 1. Scene complexity (motion, objects, changes)
        # 2. Scene duration
        # 3. Overall video characteristics
        
        # For now, fall back to uniform sampling
        logger.warning(
            "Adaptive sampling not implemented, falling back to uniform"
        )
        return self._extract_uniform(video_path, fps, duration)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame (resize, compress, normalize).
        
        Args:
            frame: Raw frame from video (HxWxC, BGR format)
        
        Returns:
            Preprocessed frame
        """
        # TODO: Implement preprocessing
        # import cv2
        # 
        # # Resize if configured
        # if self.config.resize_width and self.config.resize_height:
        #     frame = cv2.resize(
        #         frame,
        #         (self.config.resize_width, self.config.resize_height),
        #         interpolation=cv2.INTER_LANCZOS4
        #     )
        # 
        # # Convert BGR to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 
        # return frame
        
        return frame
    
    def _calculate_extraction_cost(self, num_frames: int) -> float:
        """
        Calculate cost of frame extraction.
        
        Cost model:
        - $0.001 per frame (storage + processing)
        - Scales with number of frames
        
        Args:
            num_frames: Number of frames extracted
        
        Returns:
            Estimated cost in dollars
        """
        cost_per_frame = 0.001
        return num_frames * cost_per_frame


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: JIT extraction (recommended for cost optimization)
    config_jit = FrameExtractionConfig(
        strategy=SamplingStrategy.ON_DEMAND
    )
    extractor_jit = FrameExtractor(config_jit)
    
    # Extract frames only for specific question timestamps
    question_timestamps = [5.2, 12.8, 23.1, 45.7]  # From Q&A generation
    
    result = extractor_jit.extract_frames(
        video_path=Path("sample_video.mp4"),
        video_id="vid_abc123",
        specific_timestamps=question_timestamps
    )
    
    print(f"✓ Extracted {result.frame_count} frames")
    print(f"  Cost: ${result.cost_estimate:.4f}")
    print(f"  Timestamps: {result.timestamps}")
    
    # Example 2: Scene-based extraction
    config_scene = FrameExtractionConfig(
        strategy=SamplingStrategy.SCENE_BASED,
        frames_per_scene=3
    )
    extractor_scene = FrameExtractor(config_scene)
    
    scenes = [(0.0, 15.2), (15.2, 32.5), (32.5, 48.1)]  # From scene detection
    
    # result = extractor_scene.extract_frames(
    #     video_path=Path("sample_video.mp4"),
    #     video_id="vid_abc123",
    #     scene_boundaries=scenes
    # )
