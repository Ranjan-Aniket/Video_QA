"""
Object Detector - Object & Action Detection in Frames

Purpose: Detect objects, people, and actions in video frames
Compliance: JIT detection per question, minimize compute costs
Architecture: Evidence-first, supports batch processing

Detection Capabilities:
- Object detection (YOLO v8 or similar)
- Person detection and attributes (clothing, pose)
- Action recognition
- Scene understanding

Cost Model:
- Local YOLO inference: ~$0.002/frame (GPU compute)
- Target: ~$1.00 per video for object detection
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ObjectCategory(Enum):
    """Object category types"""
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    OBJECT = "object"
    FURNITURE = "furniture"
    FOOD = "food"
    ELECTRONICS = "electronics"
    SPORTS = "sports"
    OTHER = "other"


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class DetectedObject:
    """Single detected object with attributes"""
    class_name: str  # Object class (e.g., "person", "car", "dog")
    confidence: float  # Detection confidence (0.0-1.0)
    bbox: BoundingBox  # Bounding box
    category: ObjectCategory  # Object category

    # Optional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    # For persons: clothing color, pose, action
    # For vehicles: type, color
    # For objects: material, state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2
            },
            "category": self.category.value,
            "attributes": self.attributes
        }


@dataclass
class FrameDetectionResult:
    """Detection results for a single frame"""
    frame_index: int
    timestamp: float
    detections: List[DetectedObject]

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def unique_classes(self) -> Set[str]:
        return {obj.class_name for obj in self.detections}

    @property
    def has_person(self) -> bool:
        return any(obj.category == ObjectCategory.PERSON for obj in self.detections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "num_detections": self.num_detections,
            "detections": [obj.to_dict() for obj in self.detections]
        }


@dataclass
class ObjectDetectionResult:
    """Complete object detection results for a video"""
    video_id: str
    frame_results: List[FrameDetectionResult]
    total_detections: int
    unique_classes: Set[str]
    detection_cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_id": self.video_id,
            "total_frames": len(self.frame_results),
            "total_detections": self.total_detections,
            "unique_classes": list(self.unique_classes),
            "detection_cost": self.detection_cost,
            "frames": [frame.to_dict() for frame in self.frame_results]
        }


class ObjectDetector:
    """
    Object detector using YOLO for video frame analysis.

    Features:
    - Lazy loading (model loaded on first use)
    - GPU/CPU support
    - Batch processing for efficiency
    - Person attribute extraction
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.25,
        device: str = None
    ):
        """
        Initialize object detector.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, etc.)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            device: Device to run on ('cpu', 'cuda', 'mps', auto-detected if None)
        """
        import torch

        # Auto-detect best device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Apple Silicon MPS")
        else:
            self.device = "cpu"
            logger.warning("No GPU available, using CPU (will be slow)")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None  # Lazy loading

        logger.info(
            f"ObjectDetector initialized "
            f"(model={model_name}, conf={confidence_threshold}, device={self.device})"
        )

    def _init_model(self):
        """Initialize YOLO model (lazy loading)"""
        if self.model is not None:
            return  # Already initialized

        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model {self.model_name}...")
            self.model = YOLO(f"{self.model_name}.pt")

            # Move to specified device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model.to(self.device)
                    logger.info(f"YOLO model loaded on GPU")
                else:
                    logger.warning("CUDA requested but not available, using CPU")
                    self.device = "cpu"
            else:
                logger.info(f"YOLO model loaded on CPU")

            logger.info(f"YOLO model {self.model_name} initialized successfully")

        except ImportError:
            logger.error("Ultralytics YOLO not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            self.model = None

    def detect_objects_in_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        video_id: str
    ) -> ObjectDetectionResult:
        """
        Detect objects in multiple frames.

        Args:
            frames: List of frame images (HxWxC numpy arrays)
            timestamps: Timestamp for each frame
            video_id: Unique video identifier

        Returns:
            ObjectDetectionResult with all detections
        """
        logger.info(f"Detecting objects in {len(frames)} frames")

        frame_results = []
        all_unique_classes = set()
        total_detections = 0
        detection_cost = 0.0

        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Detect objects in single frame
            detections = self._detect_in_frame(frame)

            result = FrameDetectionResult(
                frame_index=i,
                timestamp=timestamp,
                detections=detections
            )

            frame_results.append(result)
            all_unique_classes.update(result.unique_classes)
            total_detections += len(detections)

            # Calculate cost (GPU compute)
            detection_cost += 0.002  # ~$0.002 per frame

        detection_result = ObjectDetectionResult(
            video_id=video_id,
            frame_results=frame_results,
            total_detections=total_detections,
            unique_classes=all_unique_classes,
            detection_cost=detection_cost
        )

        logger.info(
            f"Detected {total_detections} objects, "
            f"{len(all_unique_classes)} unique classes "
            f"(cost: ${detection_cost:.4f})"
        )

        return detection_result

    def detect_objects_jit(
        self, frame: np.ndarray, timestamp: float
    ) -> FrameDetectionResult:
        """
        Detect objects in single frame on-demand (JIT).

        This is the most cost-effective method for question-specific detection.

        Args:
            frame: Frame image (HxWxC numpy array)
            timestamp: Frame timestamp in video

        Returns:
            FrameDetectionResult with detections
        """
        detections = self._detect_in_frame(frame)

        return FrameDetectionResult(
            frame_index=-1,  # Unknown frame index for JIT
            timestamp=timestamp,
            detections=detections
        )

    def _detect_in_frame(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in single frame using YOLO.

        Args:
            frame: Frame image (HxWxC numpy array, RGB format)

        Returns:
            List of detected objects
        """
        # Initialize model if needed
        if self.model is None:
            self._init_model()

        if self.model is None:
            logger.warning("YOLO model not available, returning empty detections")
            return []

        try:
            # Run YOLO inference
            # Retry once if AttributeError occurs (batchnorm fusion issue)
            try:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    verbose=False  # Suppress YOLO console output
                )
            except AttributeError as attr_err:
                if "bn" in str(attr_err):
                    logger.warning("YOLO fusion error, reinitializing model...")
                    self.model = None
                    self._init_model()
                    if self.model is None:
                        return []
                    # Retry inference
                    results = self.model(
                        frame,
                        conf=self.confidence_threshold,
                        verbose=False
                    )
                else:
                    raise

            # Parse detections
            detections = []

            if len(results) > 0:
                result = results[0]  # First result (single frame)

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for box in boxes:
                        # Get box coordinates (xyxy format)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy

                        # Get class and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = result.names[cls]

                        # Categorize object
                        category = self._categorize_object(class_name)

                        # Extract attributes (if person)
                        attributes = {}
                        if category == ObjectCategory.PERSON:
                            attributes = self._extract_person_attributes(
                                frame, int(x1), int(y1), int(x2), int(y2)
                            )

                        # Create DetectedObject
                        detection = DetectedObject(
                            class_name=class_name,
                            confidence=conf,
                            bbox=BoundingBox(
                                x1=int(x1),
                                y1=int(y1),
                                x2=int(x2),
                                y2=int(y2)
                            ),
                            category=category,
                            attributes=attributes
                        )

                        detections.append(detection)

            logger.debug(f"Detected {len(detections)} objects in frame")
            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _categorize_object(self, class_name: str) -> ObjectCategory:
        """Categorize detected object into high-level category"""
        class_name_lower = class_name.lower()

        if class_name_lower == "person":
            return ObjectCategory.PERSON
        elif class_name_lower in ["car", "truck", "bus", "motorcycle", "bicycle"]:
            return ObjectCategory.VEHICLE
        elif class_name_lower in ["dog", "cat", "bird", "horse"]:
            return ObjectCategory.ANIMAL
        elif class_name_lower in ["chair", "couch", "bed", "table"]:
            return ObjectCategory.FURNITURE
        elif class_name_lower in ["pizza", "sandwich", "apple", "banana"]:
            return ObjectCategory.FOOD
        elif class_name_lower in ["laptop", "phone", "tv", "keyboard"]:
            return ObjectCategory.ELECTRONICS
        elif class_name_lower in ["ball", "racket", "skateboard", "sports ball"]:
            return ObjectCategory.SPORTS
        else:
            return ObjectCategory.OTHER

    def _extract_person_attributes(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Dict[str, Any]:
        """
        Extract attributes for detected person.

        Attributes include:
        - Clothing color (top, bottom)
        - Pose (standing, sitting, etc.)
        - Action (walking, running, etc.)

        Args:
            frame: Full frame image
            x1, y1, x2, y2: Person bounding box

        Returns:
            Dictionary of extracted attributes
        """
        # TODO: Implement advanced person attribute extraction
        # For now, return basic attributes

        attributes = {
            "bbox_area": (x2 - x1) * (y2 - y1),
            "position": {
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2)
            }
        }

        # Could add:
        # - Dominant clothing colors (using color quantization)
        # - Pose estimation (using MediaPipe or similar)
        # - Action recognition (using temporal analysis)

        return attributes

    def _detect_in_frames_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[DetectedObject]]:
        """
        Detect objects in multiple frames with batching for better performance.

        Args:
            frames: List of frame images
            batch_size: Number of frames to process in parallel

        Returns:
            List of detection lists (one per frame)
        """
        if self.model is None:
            self._init_model()

        if self.model is None:
            return [[] for _ in frames]

        all_detections = []

        # Process in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            try:
                # YOLO can process batches efficiently
                results = self.model(
                    batch,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                # Parse each frame's results
                for j, result in enumerate(results):
                    frame_idx = i + j
                    frame = batch[j]
                    frame_detections = []

                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes

                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy

                            cls = int(box.cls[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            class_name = result.names[cls]

                            category = self._categorize_object(class_name)

                            attributes = {}
                            if category == ObjectCategory.PERSON:
                                attributes = self._extract_person_attributes(
                                    frame, int(x1), int(y1), int(x2), int(y2)
                                )

                            detection = DetectedObject(
                                class_name=class_name,
                                confidence=conf,
                                bbox=BoundingBox(
                                    x1=int(x1), y1=int(y1),
                                    x2=int(x2), y2=int(y2)
                                ),
                                category=category,
                                attributes=attributes
                            )

                            frame_detections.append(detection)

                    all_detections.append(frame_detections)

            except Exception as e:
                logger.error(f"Error in batch detection: {e}")
                # Add empty results for this batch
                all_detections.extend([[] for _ in batch])

        return all_detections
