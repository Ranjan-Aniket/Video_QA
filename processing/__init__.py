"""
Processing Package - Block 4: Evidence & Video Processing

This package provides JIT (Just-In-Time) evidence extraction for cost-optimized
video processing.
"""

from .video_processor import (
    VideoProcessor,
    VideoMetadata,
    ProcessingCosts,
    ProcessingResult
)

from .frame_extractor import (
    FrameExtractor,
    FrameExtractionConfig,
    ExtractedFrame,
    FrameExtractionResult,
    SamplingStrategy
)

from .audio_processor import (
    AudioProcessor,
    AudioSegment,
    AudioExtractionResult
)

from .ocr_processor import (
    OCRProcessor,
    TextBlock,
    FrameOCRResult,
    OCRExtractionResult
)

from .object_detector import (
    ObjectDetector,
    DetectedObject,
    ObjectCategory,
    BoundingBox,
    FrameDetectionResult,
    ObjectDetectionResult
)

from .scene_detector import (
    SceneDetector,
    Scene,
    SceneType,
    SceneDetectionResult
)

from .evidence_extractor import (
    EvidenceExtractor,
    EvidenceRequest,
    ExtractedEvidence,
    EvidenceType
)

from .cost_optimizer import (
    CostOptimizer,
    CostLimits,
    CostTracking,
    CostCategory,
    CostOptimizationResult
)

__all__ = [
    # Video Processor
    'VideoProcessor',
    'VideoMetadata',
    'ProcessingCosts',
    'ProcessingResult',
    
    # Frame Extractor
    'FrameExtractor',
    'FrameExtractionConfig',
    'ExtractedFrame',
    'FrameExtractionResult',
    'SamplingStrategy',
    
    # Audio Processor
    'AudioProcessor',
    'AudioSegment',
    'AudioExtractionResult',
    
    # OCR Processor
    'OCRProcessor',
    'TextBlock',
    'FrameOCRResult',
    'OCRExtractionResult',
    
    # Object Detector
    'ObjectDetector',
    'DetectedObject',
    'ObjectCategory',
    'BoundingBox',
    'FrameDetectionResult',
    'ObjectDetectionResult',
    
    # Scene Detector
    'SceneDetector',
    'Scene',
    'SceneType',
    'SceneDetectionResult',
    
    # Evidence Extractor
    'EvidenceExtractor',
    'EvidenceRequest',
    'ExtractedEvidence',
    'EvidenceType',
    
    # Cost Optimizer
    'CostOptimizer',
    'CostLimits',
    'CostTracking',
    'CostCategory',
    'CostOptimizationResult',
]

__version__ = '1.0.0'
