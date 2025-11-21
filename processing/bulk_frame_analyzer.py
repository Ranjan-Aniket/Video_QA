"""
Bulk Frame Analyzer - COMPLETE Multi-Model Analysis for Non-Premium Frames

Analyzes template, bulk, and scene boundary frames using ALL available open-source models:

CORE MODELS (Always enabled):
- YOLOv8n: Fast object detection
- PaddleOCR: Text extraction
- Places365: Scene classification

ADVANCED MODELS (Optional):
- BLIP-2: Image captioning and visual question answering
- CLIP: Image-text similarity matching
- FER: Facial expression recognition
- DeepSport: Sports jersey number detection
- VideoMAE: Action recognition
- Text Orientation: Auto-detect text orientation
- Pose Detection: MediaPipe pose, hand gestures, gaze direction

Cost: FREE (all models run locally)
Speed: ~0.5-2s per frame depending on enabled models

Used in Phase 4 for comprehensive evidence extraction without API costs.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BulkAnalysisResult:
    """Result from bulk frame analysis with all models"""
    frame_id: str
    timestamp: float
    frame_type: str
    opportunity_type: Optional[str]
    audio_cue: Optional[str]

    # Core analysis results
    yolo_objects: List[Dict[str, Any]] = field(default_factory=list)
    ocr_text: List[Dict[str, Any]] = field(default_factory=list)
    scene_type: str = "unknown"
    scene_confidence: float = 0.0

    # Advanced analysis results
    blip2_caption: Optional[str] = None
    blip2_confidence: float = 0.0
    clip_embeddings: Optional[List[float]] = None
    facial_expressions: List[Dict[str, Any]] = field(default_factory=list)
    jersey_numbers: List[Dict[str, Any]] = field(default_factory=list)
    text_orientation: Optional[str] = None
    detected_actions: List[Dict[str, Any]] = field(default_factory=list)
    body_poses: List[Dict[str, Any]] = field(default_factory=list)
    hand_gestures: List[Dict[str, Any]] = field(default_factory=list)
    face_landmarks: Optional[Dict[str, Any]] = None

    # Metadata
    analysis_method: str = "bulk_analyzer_complete"
    models_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to evidence format expected by pipeline"""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "frame_type": self.frame_type,
            "opportunity_type": self.opportunity_type,
            "audio_cue": self.audio_cue,
            "ground_truth": {
                # Core models
                "yolo_objects": self.yolo_objects,
                "ocr_text": self.ocr_text,
                "scene_type": self.scene_type,
                "scene_confidence": self.scene_confidence,
                # Advanced models
                "blip2_caption": self.blip2_caption,
                "blip2_confidence": self.blip2_confidence,
                "clip_embeddings": self.clip_embeddings,
                "facial_expressions": self.facial_expressions,
                "jersey_numbers": self.jersey_numbers,
                "text_orientation": self.text_orientation,
                "detected_actions": self.detected_actions,
                "body_poses": self.body_poses,
                "hand_gestures": self.hand_gestures,
                "face_landmarks": self.face_landmarks,
                # Metadata
                "analysis_method": self.analysis_method,
                "models_used": self.models_used
            }
        }


class BulkFrameAnalyzer:
    """
    Comprehensive frame analyzer using ALL available open-source models

    CORE MODELS (enabled by default):
    - YOLOv8n (lightweight, fast) for object detection
    - PaddleOCR for text extraction
    - Places365 for scene classification

    ADVANCED MODELS (optional):
    - BLIP-2 for image captioning
    - CLIP for image-text embeddings
    - FER for facial expression recognition
    - DeepSport for sports jersey number detection
    - VideoMAE for action recognition
    - Text Orientation for auto-rotation
    - Pose Detection for body pose, hand gestures, gaze direction

    All models run locally (no API costs)
    """

    def __init__(
        self,
        # Core models
        enable_yolo: bool = True,
        enable_ocr: bool = True,
        enable_scene: bool = True,
        # Advanced models
        enable_blip2: bool = False,
        enable_clip: bool = False,
        enable_fer: bool = False,
        enable_deepsport: bool = False,
        enable_videomae: bool = False,
        enable_text_orientation: bool = False,
        enable_pose: bool = False,
        # Model variants
        yolo_model: str = "yolov8n",
        ocr_languages: List[str] = None
    ):
        """
        Initialize comprehensive bulk frame analyzer

        Args:
            enable_yolo: Enable YOLO object detection (recommended)
            enable_ocr: Enable OCR text extraction (recommended)
            enable_scene: Enable scene classification (recommended)
            enable_blip2: Enable BLIP-2 image captioning (slow, ~2GB)
            enable_clip: Enable CLIP embeddings (useful for retrieval)
            enable_fer: Enable facial expression recognition
            enable_deepsport: Enable sports jersey detection (domain-specific)
            enable_videomae: Enable action recognition (requires video context)
            enable_text_orientation: Enable auto text orientation
            enable_pose: Enable pose detection (body pose, hand gestures, gaze)
            yolo_model: YOLO variant (yolov8n=fast, yolov8s=balanced, yolov8m=accurate)
            ocr_languages: OCR languages (default: ["en"])
        """
        # Core models
        self.enable_yolo = enable_yolo
        self.enable_ocr = enable_ocr
        self.enable_scene = enable_scene

        # Advanced models
        self.enable_blip2 = enable_blip2
        self.enable_clip = enable_clip
        self.enable_fer = enable_fer
        self.enable_deepsport = enable_deepsport
        self.enable_videomae = enable_videomae
        self.enable_text_orientation = enable_text_orientation
        self.enable_pose = enable_pose

        # Configuration
        self.yolo_model_name = yolo_model
        self.ocr_languages = ocr_languages or ["en"]

        # Lazy load processors (only when needed)
        self.object_detector = None
        self.ocr_processor = None
        self.scene_detector = None
        self.blip2_processor = None
        self.clip_processor = None
        self.fer_processor = None
        self.deepsport_processor = None
        self.videomae_processor = None
        self.text_orientation_processor = None
        self.pose_detector = None

        # Statistics
        self.frames_processed = 0
        self.total_objects_detected = 0
        self.total_text_extracted = 0
        self.total_faces_detected = 0
        self.total_jerseys_detected = 0
        self.total_poses_detected = 0
        self.total_gestures_detected = 0

        logger.info(f"BulkFrameAnalyzer initialized (COMPLETE - All Models)")
        logger.info(f"  Core Models:")
        logger.info(f"    YOLO: {'enabled' if enable_yolo else 'disabled'} ({yolo_model})")
        logger.info(f"    OCR: {'enabled' if enable_ocr else 'disabled'}")
        logger.info(f"    Scene: {'enabled' if enable_scene else 'disabled'}")
        logger.info(f"  Advanced Models:")
        logger.info(f"    BLIP-2: {'enabled' if enable_blip2 else 'disabled'}")
        logger.info(f"    CLIP: {'enabled' if enable_clip else 'disabled'}")
        logger.info(f"    FER: {'enabled' if enable_fer else 'disabled'}")
        logger.info(f"    DeepSport: {'enabled' if enable_deepsport else 'disabled'}")
        logger.info(f"    VideoMAE: {'enabled' if enable_videomae else 'disabled'}")
        logger.info(f"    Text Orientation: {'enabled' if enable_text_orientation else 'disabled'}")
        logger.info(f"    Pose Detection: {'enabled' if enable_pose else 'disabled'}")

    # ==================== LAZY LOADERS ====================

    def _init_object_detector(self):
        """Initialize YOLO object detector"""
        if self.object_detector is None and self.enable_yolo:
            try:
                from .object_detector import ObjectDetector
                self.object_detector = ObjectDetector(model_name=self.yolo_model_name)
                logger.info(f"✓ YOLOv8 loaded ({self.yolo_model_name})")
            except Exception as e:
                logger.warning(f"⚠ YOLOv8 unavailable: {e}")
                self.enable_yolo = False

    def _init_ocr_processor(self):
        """Initialize OCR processor"""
        if self.ocr_processor is None and self.enable_ocr:
            try:
                from .ocr_processor import OCRProcessor
                self.ocr_processor = OCRProcessor(use_local_ocr=True)
                logger.info(f"✓ PaddleOCR loaded")
            except Exception as e:
                logger.warning(f"⚠ PaddleOCR unavailable: {e}")
                self.enable_ocr = False

    def _init_scene_detector(self):
        """Initialize scene classifier"""
        if self.scene_detector is None and self.enable_scene:
            try:
                from .places365_processor import Places365Processor
                self.scene_detector = Places365Processor()
                logger.info(f"✓ Places365 scene classifier loaded")
            except Exception as e:
                logger.warning(f"⚠ Places365 unavailable: {e}")
                self.enable_scene = False

    def _init_blip2_processor(self):
        """Initialize BLIP-2 processor"""
        if self.blip2_processor is None and self.enable_blip2:
            try:
                from .blip2_processor import BLIP2Processor
                self.blip2_processor = BLIP2Processor()
                logger.info(f"✓ BLIP-2 loaded (~2GB)")
            except Exception as e:
                logger.warning(f"⚠ BLIP-2 unavailable: {e}")
                self.enable_blip2 = False

    def _init_clip_processor(self):
        """Initialize CLIP processor"""
        if self.clip_processor is None and self.enable_clip:
            try:
                from .clip_processor import CLIPProcessor
                self.clip_processor = CLIPProcessor()
                logger.info(f"✓ CLIP loaded")
            except Exception as e:
                logger.warning(f"⚠ CLIP unavailable: {e}")
                self.enable_clip = False

    def _init_fer_processor(self):
        """Initialize FER processor"""
        if self.fer_processor is None and self.enable_fer:
            try:
                from .fer_processor import FERProcessor
                self.fer_processor = FERProcessor()
                logger.info(f"✓ FER loaded")
            except Exception as e:
                logger.warning(f"⚠ FER unavailable: {e}")
                self.enable_fer = False

    def _init_deepsport_processor(self):
        """Initialize DeepSport processor"""
        if self.deepsport_processor is None and self.enable_deepsport:
            try:
                from .deepsport_processor import DeepSportProcessor
                self.deepsport_processor = DeepSportProcessor()
                logger.info(f"✓ DeepSport loaded")
            except Exception as e:
                logger.warning(f"⚠ DeepSport unavailable: {e}")
                self.enable_deepsport = False

    def _init_videomae_processor(self):
        """Initialize VideoMAE processor"""
        if self.videomae_processor is None and self.enable_videomae:
            try:
                from .videomae_processor import VideoMAEProcessor
                self.videomae_processor = VideoMAEProcessor()
                logger.info(f"✓ VideoMAE loaded")
            except Exception as e:
                logger.warning(f"⚠ VideoMAE unavailable: {e}")
                self.enable_videomae = False

    def _init_text_orientation_processor(self):
        """Initialize text orientation processor"""
        if self.text_orientation_processor is None and self.enable_text_orientation:
            try:
                from .text_orientation_processor import TextOrientationProcessor
                self.text_orientation_processor = TextOrientationProcessor()
                logger.info(f"✓ Text Orientation loaded")
            except Exception as e:
                logger.warning(f"⚠ Text Orientation unavailable: {e}")
                self.enable_text_orientation = False

    def _init_pose_detector(self):
        """Initialize pose detector"""
        if self.pose_detector is None and self.enable_pose:
            try:
                from .pose_detector import PoseDetector
                self.pose_detector = PoseDetector()
                logger.info(f"✓ MediaPipe Pose loaded (~30MB)")
            except Exception as e:
                logger.warning(f"⚠ MediaPipe Pose unavailable: {e}")
                self.enable_pose = False

    # ==================== MAIN ANALYSIS ====================

    def analyze_frame(self, frame, prev_frame=None, next_frame=None) -> BulkAnalysisResult:
        """
        Analyze a single frame with ALL enabled models

        Args:
            frame: ExtractedFrame object from SmartFrameExtractor
            prev_frame: Previous frame (for VideoMAE temporal context)
            next_frame: Next frame (for VideoMAE temporal context)

        Returns:
            BulkAnalysisResult with comprehensive evidence
        """
        frame_path = Path(frame.image_path)

        if not frame_path.exists():
            logger.warning(f"Frame not found: {frame_path}")
            return self._create_empty_result(frame)

        # Load image
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.error(f"Failed to load: {frame_path}")
            return self._create_empty_result(frame)

        # Track which models are used
        models_used = []

        # ==================== CORE MODELS ====================

        # 1. YOLO Object Detection
        yolo_objects = []
        if self.enable_yolo:
            self._init_object_detector()
            if self.object_detector:
                try:
                    yolo_objects = self._detect_objects(img, frame.frame_id, frame.timestamp)
                    self.total_objects_detected += len(yolo_objects)
                    models_used.append("yolov8n")
                except Exception as e:
                    logger.error(f"YOLO failed for {frame.frame_id}: {e}")

        # 2. OCR Text Extraction
        ocr_text = []
        if self.enable_ocr:
            self._init_ocr_processor()
            if self.ocr_processor:
                try:
                    ocr_text = self._extract_text(img, frame.frame_id, frame.timestamp)
                    self.total_text_extracted += len(ocr_text)
                    models_used.append("paddleocr")
                except Exception as e:
                    logger.error(f"OCR failed for {frame.frame_id}: {e}")

        # 3. Scene Classification
        scene_type = "unknown"
        scene_confidence = 0.0
        if self.enable_scene:
            self._init_scene_detector()
            if self.scene_detector:
                try:
                    scene_type, scene_confidence = self._classify_scene(img)
                    models_used.append("places365")
                except Exception as e:
                    logger.error(f"Scene classification failed for {frame.frame_id}: {e}")

        # ==================== ADVANCED MODELS ====================

        # 4. BLIP-2 Image Captioning
        blip2_caption = None
        blip2_confidence = 0.0
        if self.enable_blip2:
            self._init_blip2_processor()
            if self.blip2_processor:
                try:
                    blip2_caption, blip2_confidence = self._generate_caption(img)
                    models_used.append("blip2")
                except Exception as e:
                    logger.error(f"BLIP-2 failed for {frame.frame_id}: {e}")

        # 5. CLIP Embeddings
        clip_embeddings = None
        if self.enable_clip:
            self._init_clip_processor()
            if self.clip_processor:
                try:
                    clip_embeddings = self._compute_embeddings(img)
                    models_used.append("clip")
                except Exception as e:
                    logger.error(f"CLIP failed for {frame.frame_id}: {e}")

        # 6. Facial Expression Recognition
        facial_expressions = []
        if self.enable_fer:
            self._init_fer_processor()
            if self.fer_processor:
                try:
                    facial_expressions = self._detect_expressions(img)
                    self.total_faces_detected += len(facial_expressions)
                    models_used.append("fer")
                except Exception as e:
                    logger.error(f"FER failed for {frame.frame_id}: {e}")

        # 7. Sports Jersey Number Detection
        jersey_numbers = []
        if self.enable_deepsport:
            self._init_deepsport_processor()
            if self.deepsport_processor:
                try:
                    jersey_numbers = self._detect_jerseys(img)
                    self.total_jerseys_detected += len(jersey_numbers)
                    models_used.append("deepsport")
                except Exception as e:
                    logger.error(f"DeepSport failed for {frame.frame_id}: {e}")

        # 8. Text Orientation Detection
        text_orientation = None
        if self.enable_text_orientation:
            self._init_text_orientation_processor()
            if self.text_orientation_processor:
                try:
                    text_orientation = self._detect_orientation(img)
                    models_used.append("text_orientation")
                except Exception as e:
                    logger.error(f"Text orientation failed for {frame.frame_id}: {e}")

        # 9. Action Recognition (requires temporal context)
        detected_actions = []
        if self.enable_videomae:
            self._init_videomae_processor()
            if self.videomae_processor and prev_frame and next_frame:
                try:
                    detected_actions = self._recognize_actions(prev_frame, img, next_frame)
                    models_used.append("videomae")
                except Exception as e:
                    logger.error(f"VideoMAE failed for {frame.frame_id}: {e}")

        # 10. Pose Detection (body pose, hand gestures, gaze)
        body_poses = []
        hand_gestures = []
        face_landmarks = None
        if self.enable_pose:
            self._init_pose_detector()
            if self.pose_detector:
                try:
                    pose_result = self._detect_pose(img)
                    body_poses = pose_result.get("body_poses", [])
                    hand_gestures = pose_result.get("hand_gestures", [])
                    face_landmarks = pose_result.get("face_landmarks")
                    self.total_poses_detected += len(body_poses)
                    self.total_gestures_detected += len(hand_gestures)
                    models_used.append("mediapipe_pose")
                except Exception as e:
                    logger.error(f"Pose detection failed for {frame.frame_id}: {e}")

        self.frames_processed += 1

        return BulkAnalysisResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            frame_type=frame.frame_type,
            opportunity_type=frame.opportunity_type,
            audio_cue=frame.audio_cue,
            yolo_objects=yolo_objects,
            ocr_text=ocr_text,
            scene_type=scene_type,
            scene_confidence=scene_confidence,
            blip2_caption=blip2_caption,
            blip2_confidence=blip2_confidence,
            clip_embeddings=clip_embeddings,
            facial_expressions=facial_expressions,
            jersey_numbers=jersey_numbers,
            text_orientation=text_orientation,
            detected_actions=detected_actions,
            body_poses=body_poses,
            hand_gestures=hand_gestures,
            face_landmarks=face_landmarks,
            models_used=models_used
        )

    # ==================== MODEL-SPECIFIC METHODS ====================

    def _detect_objects(self, img: np.ndarray, frame_id: str = "unknown", timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """YOLO object detection"""
        if not self.object_detector:
            return []

        # detect_objects_in_frames expects (frames, timestamps, video_id)
        results = self.object_detector.detect_objects_in_frames(
            frames=[img],
            timestamps=[timestamp],
            video_id=frame_id
        )

        # Extract detections from result
        if not results:
            return []

        objects = []

        # Handle ObjectDetectionResult object
        # ObjectDetectionResult has 'frame_results' not 'detections'
        if hasattr(results, 'frame_results') and results.frame_results:
            # Get first frame result (we only sent 1 frame)
            frame_result = results.frame_results[0]
            detections = frame_result.detections if hasattr(frame_result, 'detections') else []
        else:
            detections = []

        for det in detections:
            # Handle both DetectedObject objects and dicts
            if hasattr(det, 'to_dict'):
                det_dict = det.to_dict()
            else:
                det_dict = det

            objects.append({
                "class": det_dict.get("class_name", "unknown"),
                "confidence": float(det_dict.get("confidence", 0.0)),
                "bbox": det_dict.get("bbox", [0, 0, 0, 0]),
                "area": det_dict.get("area", 0)
            })
        return objects

    def _extract_text(self, img: np.ndarray, frame_id: str = "unknown", timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """OCR text extraction"""
        if not self.ocr_processor:
            return []

        # extract_text_from_frames expects (frames, timestamps, video_id)
        results = self.ocr_processor.extract_text_from_frames(
            frames=[img],
            timestamps=[timestamp],
            video_id=frame_id
        )

        # Extract text regions from result
        if not results:
            return []

        text_items = []

        # Handle OCRExtractionResult object
        # OCRExtractionResult has 'frame_results' not 'text_regions'
        if hasattr(results, 'frame_results') and results.frame_results:
            # Get first frame result (we only sent 1 frame)
            frame_result = results.frame_results[0]
            text_blocks = frame_result.text_blocks if hasattr(frame_result, 'text_blocks') else []
        else:
            text_blocks = []

        for ocr_item in text_blocks:
            # Handle both TextRegion objects and dicts
            if hasattr(ocr_item, 'to_dict'):
                ocr_dict = ocr_item.to_dict()
            else:
                ocr_dict = ocr_item

            text_items.append({
                "text": ocr_dict.get("text", ""),
                "confidence": float(ocr_dict.get("confidence", 0.0)),
                "bbox": ocr_dict.get("bounding_box", [[0, 0], [0, 0], [0, 0], [0, 0]]),  # TextBlock uses 'bounding_box'
                "orientation": ocr_dict.get("orientation", 0)
            })
        return text_items

    def _classify_scene(self, img: np.ndarray) -> tuple:
        """Scene classification using Places365"""
        if not self.scene_detector:
            return "unknown", 0.0

        try:
            # Places365Processor.classify_scene() returns SceneClassification object
            result = self.scene_detector.classify_scene(img)
            return result.scene_category, result.confidence
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return "unknown", 0.0

    def _generate_caption(self, img: np.ndarray) -> tuple:
        """BLIP-2 image captioning"""
        if not self.blip2_processor:
            return None, 0.0

        # generate_caption returns a string directly
        caption = self.blip2_processor.generate_caption(img)
        if caption:
            # BLIP-2 doesn't provide confidence scores, default to 0.9
            return caption, 0.9
        return None, 0.0

    def _compute_embeddings(self, img: np.ndarray) -> Optional[List[float]]:
        """CLIP embeddings"""
        if not self.clip_processor:
            return None

        embeddings = self.clip_processor.encode_image(img)
        if embeddings is not None:
            return embeddings.tolist()
        return None

    def _detect_expressions(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Facial expression recognition"""
        if not self.fer_processor:
            return []

        expressions = self.fer_processor.detect_emotions(img)
        results = []
        for expr in expressions:
            results.append({
                "emotion": expr.get("dominant_emotion", "neutral"),
                "confidence": float(expr.get("confidence", 0.0)),
                "bbox": expr.get("bbox", [0, 0, 0, 0]),
                "all_emotions": expr.get("emotions", {})
            })
        return results

    def _detect_jerseys(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Sports jersey number detection"""
        if not self.deepsport_processor:
            return []

        jerseys = self.deepsport_processor.detect_jersey_numbers(img)
        results = []
        for jersey in jerseys:
            results.append({
                "number": jersey.get("number", ""),
                "confidence": float(jersey.get("confidence", 0.0)),
                "bbox": jersey.get("bbox", [0, 0, 0, 0]),
                "team_color": jersey.get("team_color", "unknown")
            })
        return results

    def _detect_orientation(self, img: np.ndarray) -> Optional[str]:
        """Text orientation detection"""
        if not self.text_orientation_processor:
            return None

        orientation = self.text_orientation_processor.detect_orientation(img)
        return orientation.get("orientation", "0") if orientation else None

    def _recognize_actions(self, prev_img: np.ndarray, curr_img: np.ndarray, next_img: np.ndarray) -> List[Dict[str, Any]]:
        """Action recognition (requires temporal context)"""
        if not self.videomae_processor:
            return []

        # VideoMAE needs sequence of frames
        frame_sequence = [prev_img, curr_img, next_img]
        actions = self.videomae_processor.recognize_actions(frame_sequence)
        results = []
        for action in actions:
            results.append({
                "action": action.get("action_name", "unknown"),
                "confidence": float(action.get("confidence", 0.0))
            })
        return results

    def _detect_pose(self, img: np.ndarray) -> Dict[str, Any]:
        """Pose detection (body pose, hand gestures, gaze)"""
        if not self.pose_detector:
            return {"body_poses": [], "hand_gestures": [], "face_landmarks": None}

        pose_result = self.pose_detector.detect_pose(img)

        # Convert to dict format with bbox calculation
        body_poses = []
        for pose in pose_result.body_poses:
            # Calculate bbox from landmarks
            if pose.landmarks:
                x_coords = [lm["x"] * img.shape[1] for lm in pose.landmarks]
                y_coords = [lm["y"] * img.shape[0] for lm in pose.landmarks]
                bbox = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords) - min(x_coords),
                    max(y_coords) - min(y_coords)
                ]
            else:
                bbox = [0, 0, 0, 0]
            
            body_poses.append({
                "pose_type": pose.pose_type,
                "confidence": float(pose.confidence),
                "landmarks": pose.landmarks,
                "keypoints": {k: {"x": v[0], "y": v[1]} for k, v in pose.keypoints.items()},
                "bbox": bbox
            })

        hand_gestures = []
        for gesture in pose_result.hand_gestures:
            # Calculate bbox from landmarks
            if gesture.landmarks:
                x_coords = [lm["x"] * img.shape[1] for lm in gesture.landmarks]
                y_coords = [lm["y"] * img.shape[0] for lm in gesture.landmarks]
                bbox = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords) - min(x_coords),
                    max(y_coords) - min(y_coords)
                ]
            else:
                bbox = [0, 0, 0, 0]
            
            hand_gestures.append({
                "hand": gesture.hand,
                "gesture": gesture.gesture,
                "confidence": float(gesture.confidence),
                "landmarks": gesture.landmarks,
                "fingertips": {k: {"x": v[0], "y": v[1]} for k, v in gesture.fingertips.items()},
                "bbox": bbox
            })

        face_landmarks = None
        if pose_result.face_landmarks:
            fl = pose_result.face_landmarks
            # Calculate bbox from landmarks
            if fl.landmarks:
                x_coords = [lm["x"] * img.shape[1] for lm in fl.landmarks]
                y_coords = [lm["y"] * img.shape[0] for lm in fl.landmarks]
                bbox = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords) - min(x_coords),
                    max(y_coords) - min(y_coords)
                ]
            else:
                bbox = [0, 0, 0, 0]
            
            face_landmarks = {
                "gaze_direction": fl.gaze_direction,
                "gaze_confidence": float(fl.gaze_confidence),
                "eye_contact": fl.eye_contact,
                "head_pose": fl.head_pose,
                "num_landmarks": fl.num_landmarks,
                "bbox": bbox
            }

        return {
            "body_poses": body_poses,
            "hand_gestures": hand_gestures,
            "face_landmarks": face_landmarks
        }

    def _create_empty_result(self, frame) -> BulkAnalysisResult:
        """Create empty result when analysis fails"""
        return BulkAnalysisResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            frame_type=frame.frame_type,
            opportunity_type=frame.opportunity_type,
            audio_cue=frame.audio_cue,
            analysis_method="bulk_analyzer_failed"
        )

    def analyze_frames(
        self,
        frames: List,
        audio_analysis: Dict,
        video_path: str
    ) -> Dict:
        """
        Analyze multiple frames (batch processing for Phase 7)

        Args:
            frames: List of extracted frames with frame_id, timestamp, etc.
            audio_analysis: Audio analysis dict (from Phase 1)
            video_path: Path to video file

        Returns:
            Dict with evidence for all frames:
            {
                "frames": {
                    "frame_001": {...},
                    "frame_002": {...},
                    ...
                },
                "evidence_count": int
            }
        """
        logger.info(f"\n📊 Analyzing {len(frames)} frames with bulk analyzer...")

        evidence_frames = {}

        for i, frame in enumerate(frames):
            frame_id = frame.frame_id

            try:
                # Analyze this frame
                result = self.analyze_frame(frame)

                # Convert to evidence format
                evidence_frames[frame_id] = result.to_dict()

                if (i + 1) % 10 == 0:
                    logger.info(f"   Progress: {i+1}/{len(frames)} frames")

            except Exception as e:
                logger.error(f"   Error analyzing {frame_id}: {e}")
                # Add empty result
                evidence_frames[frame_id] = self._create_empty_result(frame).to_dict()

        logger.info(f"✅ Completed analysis of {len(frames)} frames")

        return {
            "frames": evidence_frames,
            "evidence_count": len(evidence_frames)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        return {
            "frames_processed": self.frames_processed,
            "total_objects_detected": self.total_objects_detected,
            "total_text_extracted": self.total_text_extracted,
            "total_faces_detected": self.total_faces_detected,
            "total_jerseys_detected": self.total_jerseys_detected,
            "total_poses_detected": self.total_poses_detected,
            "total_gestures_detected": self.total_gestures_detected,
            "avg_objects_per_frame": (
                self.total_objects_detected / self.frames_processed
                if self.frames_processed > 0 else 0
            ),
            "avg_text_per_frame": (
                self.total_text_extracted / self.frames_processed
                if self.frames_processed > 0 else 0
            ),
            "avg_faces_per_frame": (
                self.total_faces_detected / self.frames_processed
                if self.frames_processed > 0 else 0
            ),
            "avg_poses_per_frame": (
                self.total_poses_detected / self.frames_processed
                if self.frames_processed > 0 else 0
            ),
            "avg_gestures_per_frame": (
                self.total_gestures_detected / self.frames_processed
                if self.frames_processed > 0 else 0
            )
        }


if __name__ == "__main__":
    # Test the complete analyzer
    import sys

    if len(sys.argv) > 1:
        from dataclasses import dataclass

        @dataclass
        class DummyFrame:
            frame_id: str = "test_001"
            timestamp: float = 10.0
            frame_type: str = "bulk"
            image_path: str = sys.argv[1]
            opportunity_type: str = "test"
            audio_cue: str = ""

        # Test with all models enabled
        analyzer = BulkFrameAnalyzer(
            enable_yolo=True,
            enable_ocr=True,
            enable_scene=True,
            enable_blip2=True,
            enable_clip=True,
            enable_fer=True,
            enable_deepsport=False,  # Only for sports videos
            enable_videomae=False,   # Needs temporal context
            enable_text_orientation=True,
            enable_pose=True  # MediaPipe pose detection
        )

        frame = DummyFrame(image_path=sys.argv[1])

        print("\nAnalyzing frame with ALL models...")
        result = analyzer.analyze_frame(frame)

        print("\n" + "=" * 80)
        print("COMPLETE BULK FRAME ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Frame: {result.frame_id}")
        print(f"Models Used: {', '.join(result.models_used)}")

        print(f"\n[YOLO] Objects: {len(result.yolo_objects)}")
        for obj in result.yolo_objects[:3]:
            print(f"  - {obj['class']}: {obj['confidence']:.2f}")

        print(f"\n[OCR] Text: {len(result.ocr_text)}")
        for text in result.ocr_text[:3]:
            print(f"  - \"{text['text']}\": {text['confidence']:.2f}")

        print(f"\n[Scene] {result.scene_type} ({result.scene_confidence:.2f})")

        if result.blip2_caption:
            print(f"\n[BLIP-2] Caption: {result.blip2_caption} ({result.blip2_confidence:.2f})")

        if result.clip_embeddings:
            print(f"\n[CLIP] Embedding dimension: {len(result.clip_embeddings)}")

        if result.facial_expressions:
            print(f"\n[FER] Faces: {len(result.facial_expressions)}")
            for face in result.facial_expressions[:3]:
                print(f"  - {face['emotion']}: {face['confidence']:.2f}")

        if result.body_poses:
            print(f"\n[Pose] Body poses: {len(result.body_poses)}")
            for pose in result.body_poses[:3]:
                print(f"  - {pose['pose_type']}: {pose['confidence']:.2f}")

        if result.hand_gestures:
            print(f"\n[Pose] Hand gestures: {len(result.hand_gestures)}")
            for gesture in result.hand_gestures[:3]:
                print(f"  - {gesture['hand']} hand: {gesture['gesture']} ({gesture['confidence']:.2f})")

        if result.face_landmarks:
            print(f"\n[Pose] Gaze: {result.face_landmarks['gaze_direction']} ({result.face_landmarks['gaze_confidence']:.2f})")

        print("=" * 80)
    else:
        print("Usage: python bulk_frame_analyzer.py <image_path>")