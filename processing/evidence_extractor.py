"""
Evidence Extractor - JIT Evidence Extraction Orchestrator

Purpose: Orchestrate on-demand evidence extraction for questions
Compliance: Minimize costs through JIT extraction strategy
Architecture: Evidence-first, extracts only what's needed when needed
"""

# Standard library imports
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from datetime import datetime

# Block 4 internal imports (relative imports)
from .frame_extractor import FrameExtractor, SamplingStrategy, FrameExtractionConfig
from .audio_processor import AudioProcessor
from .ocr_processor import OCRProcessor
from .object_detector import ObjectDetector
from .scene_detector import SceneDetector

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Types of evidence that can be extracted"""
    VISUAL_CUE = "visual_cue"
    AUDIO_CUE = "audio_cue"
    OBJECT = "object"
    OCR_TEXT = "ocr_text"
    SCENE_CONTEXT = "scene_context"
    ACTION = "action"


@dataclass
class EvidenceRequest:
    """Request for specific evidence extraction"""
    question_id: str
    question_type: str
    timestamp: float
    evidence_types: List[EvidenceType]
    time_window: Tuple[float, float] = None
    required_objects: List[str] = field(default_factory=list)
    required_text: List[str] = field(default_factory=list)


@dataclass
class ExtractedEvidence:
    """Extracted evidence for a question"""
    question_id: str
    timestamp: float
    evidence: Dict[EvidenceType, Any]
    extraction_cost: float
    cached: bool = False
    
    def has_evidence_type(self, evidence_type: EvidenceType) -> bool:
        """Check if specific evidence type was extracted"""
        return evidence_type in self.evidence and self.evidence[evidence_type] is not None
    
    def get_visual_cue(self) -> Optional[Any]:
        """Get visual cue frame"""
        return self.evidence.get(EvidenceType.VISUAL_CUE)
    
    def get_audio_cue(self) -> Optional[str]:
        """Get audio transcription"""
        return self.evidence.get(EvidenceType.AUDIO_CUE)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "question_id": self.question_id,
            "timestamp": self.timestamp,
            "evidence_types": [e.value for e in self.evidence.keys()],
            "extraction_cost": self.extraction_cost,
            "cached": self.cached
        }


class EvidenceExtractor:
    """
    JIT evidence extraction orchestrator.
    
    Extracts evidence on-demand per question to minimize costs.
    Coordinates frame extraction, audio processing, OCR, and object detection.
    """
    
    def __init__(
        self,
        video_path: Path,
        video_id: str,
        cache_dir: Optional[Path] = None,
        enable_caching: bool = True
    ):
        """
        Initialize evidence extractor for a video.
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            cache_dir: Directory for caching evidence
            enable_caching: Whether to cache extracted evidence
        """
        self.video_path = video_path
        self.video_id = video_id
        self.cache_dir = cache_dir or Path("./cache/evidence")
        self.enable_caching = enable_caching
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-processors (lazy loading)
        self.frame_extractor: Optional[FrameExtractor] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.ocr_processor: Optional[OCRProcessor] = None
        self.object_detector: Optional[ObjectDetector] = None
        
        # Track total costs
        self.total_extraction_cost = 0.0
        self.extraction_count = 0
        
        logger.info(f"EvidenceExtractor initialized for {video_id}")
    
    def extract_evidence_jit(
        self, request: EvidenceRequest
    ) -> ExtractedEvidence:
        """
        Extract evidence on-demand for a specific question.
        
        This is the core JIT extraction method - extracts only what's needed!
        
        Args:
            request: Evidence extraction request
        
        Returns:
            ExtractedEvidence with requested evidence
        """
        logger.info(
            f"JIT extracting evidence for question {request.question_id} "
            f"at {request.timestamp:.1f}s"
        )
        
        # Check cache first
        if self.enable_caching:
            cached = self._load_from_cache(request.question_id)
            if cached:
                logger.info(f"Loaded cached evidence for {request.question_id}")
                return cached
        
        evidence = {}
        cost = 0.0
        
        # Extract each requested evidence type
        for evidence_type in request.evidence_types:
            if evidence_type == EvidenceType.VISUAL_CUE:
                frame, frame_cost = self._extract_visual_cue(request)
                evidence[EvidenceType.VISUAL_CUE] = frame
                cost += frame_cost
            
            elif evidence_type == EvidenceType.AUDIO_CUE:
                audio, audio_cost = self._extract_audio_cue(request)
                evidence[EvidenceType.AUDIO_CUE] = audio
                cost += audio_cost
            
            elif evidence_type == EvidenceType.OBJECT:
                objects, obj_cost = self._extract_objects(request)
                evidence[EvidenceType.OBJECT] = objects
                cost += obj_cost
            
            elif evidence_type == EvidenceType.OCR_TEXT:
                text, ocr_cost = self._extract_ocr_text(request)
                evidence[EvidenceType.OCR_TEXT] = text
                cost += ocr_cost
            
            elif evidence_type == EvidenceType.SCENE_CONTEXT:
                scene, scene_cost = self._extract_scene_context(request)
                evidence[EvidenceType.SCENE_CONTEXT] = scene
                cost += scene_cost
            
            elif evidence_type == EvidenceType.ACTION:
                action, action_cost = self._extract_action(request)
                evidence[EvidenceType.ACTION] = action
                cost += action_cost
        
        # Create result
        result = ExtractedEvidence(
            question_id=request.question_id,
            timestamp=request.timestamp,
            evidence=evidence,
            extraction_cost=cost,
            cached=False
        )
        
        # Update totals
        self.total_extraction_cost += cost
        self.extraction_count += 1
        
        # Cache result
        if self.enable_caching:
            self._save_to_cache(result)
        
        logger.info(
            f"Extracted evidence for {request.question_id} "
            f"(cost: ${cost:.4f}, total: ${self.total_extraction_cost:.2f})"
        )
        
        return result
    
    def _init_frame_extractor(self):
        """Initialize frame extractor (lazy)"""
        if self.frame_extractor is None:
            config = FrameExtractionConfig(strategy=SamplingStrategy.ON_DEMAND)
            self.frame_extractor = FrameExtractor(config)
            logger.debug("FrameExtractor initialized")
    
    def _init_audio_processor(self):
        """Initialize audio processor (lazy)"""
        if self.audio_processor is None:
            self.audio_processor = AudioProcessor(enable_caching=True)
            logger.debug("AudioProcessor initialized")
    
    def _init_ocr_processor(self):
        """Initialize OCR processor (lazy)"""
        if self.ocr_processor is None:
            self.ocr_processor = OCRProcessor(use_local_ocr=True)
            logger.debug("OCRProcessor initialized")
    
    def _init_object_detector(self):
        """Initialize object detector (lazy)"""
        if self.object_detector is None:
            self.object_detector = ObjectDetector(
                model_name="yolov8n",
                confidence_threshold=0.5
            )
            logger.debug("ObjectDetector initialized")
    
    def _extract_visual_cue(
        self, request: EvidenceRequest
    ) -> Tuple[Any, float]:
        """Extract visual cue frame"""
        self._init_frame_extractor()
        
        frame = self.frame_extractor.extract_single_frame(
            self.video_path, request.timestamp
        )
        cost = 0.001  # ~$0.001 per frame
        
        return frame, cost
    
    def _extract_audio_cue(
        self, request: EvidenceRequest
    ) -> Tuple[Optional[str], float]:
        """Extract audio transcription"""
        self._init_audio_processor()
        
        # Determine time window
        if request.time_window:
            start, end = request.time_window
        else:
            # Default: ±2 seconds around timestamp
            start = max(0, request.timestamp - 2.0)
            end = request.timestamp + 2.0
        
        segment = self.audio_processor.transcribe_segment_jit(
            self.video_path, start, end
        )
        transcription = segment.transcription if segment else None
        duration_minutes = (end - start) / 60.0
        cost = duration_minutes * 0.006  # $0.006/minute
        
        return transcription, cost
    
    def _extract_objects(
        self, request: EvidenceRequest
    ) -> Tuple[List[Any], float]:
        """Extract detected objects"""
        self._init_object_detector()
        self._init_frame_extractor()
        
        frame = self.frame_extractor.extract_single_frame(
            self.video_path, request.timestamp
        )
        
        if frame is not None:
            result = self.object_detector.detect_objects_jit(frame, request.timestamp)
            objects = result.detections
        else:
            objects = []
        
        cost = 0.003  # Frame + detection
        
        return objects, cost
    
    def _extract_ocr_text(
        self, request: EvidenceRequest
    ) -> Tuple[List[str], float]:
        """Extract OCR text from frame"""
        self._init_ocr_processor()
        self._init_frame_extractor()
        
        frame = self.frame_extractor.extract_single_frame(
            self.video_path, request.timestamp
        )
        
        if frame is not None:
            result = self.ocr_processor.extract_text_jit(frame, request.timestamp)
            texts = [block.text for block in result.text_blocks]
        else:
            texts = []
        
        cost = 0.001  # Frame + OCR (free with EasyOCR)
        
        return texts, cost
    
    def _extract_scene_context(
        self, request: EvidenceRequest
    ) -> Tuple[Optional[Any], float]:
        """Extract scene context information"""
        # Scene detection done upfront, no additional cost
        scene = None
        cost = 0.0
        
        return scene, cost
    
    def _extract_action(
        self, request: EvidenceRequest
    ) -> Tuple[Optional[str], float]:
        """Extract detected action"""
        # Similar to object detection
        action = None
        cost = 0.003
        
        return action, cost
    
    def _load_from_cache(self, question_id: str) -> Optional[ExtractedEvidence]:
        """Load cached evidence"""
        cache_file = self.cache_dir / f"{self.video_id}_{question_id}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            return ExtractedEvidence(
                question_id=data["question_id"],
                timestamp=data["timestamp"],
                evidence={},  # Simplified
                extraction_cost=data["extraction_cost"],
                cached=True
            )
        except Exception as e:
            logger.warning(f"Failed to load cache for {question_id}: {e}")
            return None
    
    def _save_to_cache(self, result: ExtractedEvidence) -> None:
        """Save extracted evidence to cache"""
        cache_file = self.cache_dir / f"{self.video_id}_{result.question_id}.json"
        
        try:
            cache_data = result.to_dict()
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Cached evidence for {result.question_id}")
        except Exception as e:
            logger.warning(f"Failed to cache evidence: {e}")
    
    def extract(self, video_path: str) -> Dict[str, Any]:
        """
        Bulk extraction method for initial evidence gathering.
        Returns data compatible with EvidenceDatabase format.

        This performs basic extraction of transcript and video metadata
        needed for question generation. Detailed evidence is extracted JIT.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with evidence data for EvidenceDatabase
        """
        logger.info(f"Performing bulk extraction for: {video_path}")

        # Initialize audio processor if needed
        if not self.audio_processor:
            self.audio_processor = AudioProcessor(
                cache_dir=self.cache_dir / "audio",
                enable_caching=True
            )

        # Extract transcript (critical for most questions)
        try:
            # Use extract_audio_segments with full_video=True to get full transcript
            extraction_result = self.audio_processor.extract_audio_segments(
                video_path=Path(video_path),
                video_id=self.video_id,
                full_video=True
            )

            # Convert AudioSegment objects to dictionaries
            transcript_segments = [seg.to_dict() for seg in extraction_result.segments]
            logger.info(f"Extracted {len(transcript_segments)} transcript segments")
        except Exception as e:
            logger.warning(f"Transcript extraction failed: {e}")
            transcript_segments = []

        # Get video duration
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        except Exception as e:
            logger.warning(f"Duration extraction failed: {e}")
            duration = 0.0

        # Return evidence in EvidenceDatabase-compatible format
        evidence_data = {
            "video_id": self.video_id,
            "duration": duration,

            # Audio evidence
            "transcript_segments": transcript_segments,
            "music_segments": [],  # JIT extraction
            "sound_effects": [],  # JIT extraction
            "ambient_sounds": [],  # JIT extraction
            "tone_changes": [],  # JIT extraction

            # Visual evidence (JIT extraction)
            "person_detections": [],
            "object_detections": [],
            "scene_detections": [],
            "ocr_detections": [],
            "action_detections": [],

            # Temporal evidence
            "scene_changes": [],  # JIT extraction
            "event_timeline": [],

            # Detected names (for blocking)
            "character_names": [],
            "team_names": [],
            "media_names": [],
            "brand_names": [],

            # Video segments
            "intro_end": None,
            "outro_start": None
        }

        logger.info(f"Bulk extraction complete for {self.video_id}")
        return evidence_data

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            "total_cost": self.total_extraction_cost,
            "extraction_count": self.extraction_count,
            "avg_cost_per_extraction": (
                self.total_extraction_cost / self.extraction_count
                if self.extraction_count > 0 else 0.0
            )
        }

    def extract_evidence_for_hitl(
        self,
        video_path: Path,
        interval_seconds: float = 5.0,
        max_items: int = 50
    ) -> List['EvidenceItem']:
        """
        Extract evidence items for Human-in-the-Loop review

        Samples video at regular intervals, runs deterministic models (YOLO, OCR, Whisper),
        gets AI predictions from GPT-4/Claude, analyzes consensus, and creates
        structured EvidenceItem objects.

        Args:
            video_path: Path to video file
            interval_seconds: Sampling interval (default: 5s)
            max_items: Maximum evidence items to create (default: 50)

        Returns:
            List of EvidenceItem objects ready for review
        """
        from processing.evidence_item import EvidenceItem, AIPrediction, GroundTruth
        from processing.ai_consensus import AIConsensusEngine
        from processing.evidence_verification import EvidenceVerifier
        from database.evidence_operations import EvidenceOperations
        from config.test_config import TEST_CONFIG, MockPredictions

        logger.info("=" * 80)
        logger.info("EXTRACT_EVIDENCE_FOR_HITL - START")
        logger.info("=" * 80)
        logger.info(f"Video ID: {self.video_id}")
        logger.info(f"Video Path: {video_path}")
        logger.info(f"Video Path Exists: {Path(video_path).exists()}")
        logger.info(f"Sampling Interval: {interval_seconds}s")
        logger.info(f"Max Items: {max_items}")
        logger.info(TEST_CONFIG.get_summary())
        logger.info("=" * 80)

        evidence_items: List[EvidenceItem] = []
        logger.info("Initializing consensus engine and verifier...")
        consensus_engine = AIConsensusEngine()
        verifier = EvidenceVerifier()
        logger.info("✓ Consensus engine and verifier initialized")

        # Get video duration
        logger.info("Getting video duration...")
        try:
            import cv2
            logger.info(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Failed to open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            logger.info(f"✓ Video opened successfully")
            logger.info(f"  - FPS: {fps}")
            logger.info(f"  - Frame Count: {frame_count}")
            logger.info(f"  - Duration: {duration:.1f}s")
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}", exc_info=True)
            duration = 60.0  # Default fallback
            logger.info(f"Using default duration: {duration}s")

        # Sample timestamps
        logger.info("Calculating sample timestamps...")
        timestamps = []
        current = interval_seconds
        while current < duration and len(timestamps) < max_items:
            timestamps.append(current)
            current += interval_seconds

        logger.info(f"✓ Will sample {len(timestamps)} frames from {duration:.1f}s video")
        if len(timestamps) > 5:
            logger.info(f"  - First 5 timestamps: {[f'{t:.1f}s' for t in timestamps[:5]]}")
        else:
            logger.info(f"  - All timestamps: {[f'{t:.1f}s' for t in timestamps]}")
        logger.info("=" * 80)

        # Extract evidence at each timestamp
        logger.info("Starting evidence extraction loop...")
        for i, timestamp in enumerate(timestamps):
            try:
                logger.info("=" * 80)
                logger.info(f"PROCESSING TIMESTAMP {timestamp:.1f}s ({i+1}/{len(timestamps)})")
                logger.info("=" * 80)

                # Extract frame
                logger.info(f"Step 1: Extracting frame at {timestamp:.1f}s...")
                frame = self._extract_single_frame(video_path, timestamp)
                if frame is None:
                    logger.warning(f"✗ Failed to extract frame at {timestamp}s")
                    continue
                logger.info(f"✓ Frame extracted: shape={frame.shape}")

                # Run deterministic models (ground truth)
                logger.info(f"Step 2: Running 10 AI models for ground truth extraction...")
                ground_truth = self._extract_ground_truth(frame, timestamp, video_path)
                logger.info(f"✓ Ground truth extraction complete")

                # Get AI predictions (respects test configuration)
                logger.info(f"Step 3: Getting AI predictions...")
                gpt4_pred = self._get_gpt4_prediction(timestamp, ground_truth)
                logger.info(f"  - GPT-4: {'Got prediction' if gpt4_pred else 'None (disabled/mock)'}")
                claude_pred = self._get_claude_prediction(timestamp, ground_truth)
                logger.info(f"  - Claude: {'Got prediction' if claude_pred else 'None (disabled/mock)'}")
                open_model_pred = self._get_open_model_prediction(timestamp, ground_truth)
                logger.info(f"  - Open Model: {'Got prediction' if open_model_pred else 'None (disabled/mock)'}")

                # Create evidence item
                logger.info(f"Step 4: Creating evidence item...")
                evidence_item = EvidenceItem(
                    video_id=self.video_id,
                    evidence_type='scene',  # Could be 'ocr', 'object_detection', etc.
                    timestamp_start=timestamp,
                    timestamp_end=timestamp + 1.0,
                    gpt4_prediction=gpt4_pred,
                    claude_prediction=claude_pred,
                    open_model_prediction=open_model_pred,
                    ground_truth=ground_truth
                )
                logger.info(f"✓ Evidence item created")

                # Run consensus analysis
                logger.info(f"Step 5: Running consensus analysis...")
                if gpt4_pred or claude_pred or open_model_pred:
                    consensus = consensus_engine.analyze_consensus(
                        gpt4_pred, claude_pred, open_model_pred, ground_truth
                    )
                    evidence_item.consensus = consensus  # Already a dataclass object
                    logger.info(f"✓ Consensus analysis complete")

                    # Verify against ground truth
                    logger.info(f"Step 6: Verifying evidence against ground truth...")
                    verified, flag_reason, priority = verifier.verify_evidence(
                        evidence_item, evidence_item.consensus
                    )

                    if not verified:
                        evidence_item.consensus.flag_reason = flag_reason
                        evidence_item.consensus.priority_level = priority
                        logger.info(f"⚠ Evidence flagged: {flag_reason} (priority: {priority})")
                    else:
                        logger.info(f"✓ Evidence verified")
                else:
                    logger.info(f"⊘ Skipping consensus (no AI predictions)")

                # Store in database
                logger.info(f"Step 7: Storing evidence item in database...")
                evidence_id = self._store_evidence_item(evidence_item)
                evidence_item.evidence_id = evidence_id
                logger.info(f"✓ Evidence item stored with ID: {evidence_id}")

                evidence_items.append(evidence_item)
                logger.info(f"✓ Evidence item added to list (total: {len(evidence_items)})")
                logger.info("=" * 80)

            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"✗ FAILED to process timestamp {timestamp}s")
                logger.error(f"Error: {e}")
                logger.error("=" * 80, exc_info=True)
                continue

        logger.info("=" * 80)
        logger.info("EXTRACT_EVIDENCE_FOR_HITL - COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Total evidence items created: {len(evidence_items)}")
        logger.info(f"✓ Processed {len(timestamps)} timestamps")
        logger.info(f"✓ Success rate: {len(evidence_items)}/{len(timestamps)} ({100*len(evidence_items)/len(timestamps):.1f}%)")
        logger.info("=" * 80)
        return evidence_items

    def _extract_single_frame(self, video_path: Path, timestamp: float):
        """Extract a single frame at timestamp"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None

    def _extract_ground_truth(
        self,
        frame,
        timestamp: float,
        video_path: Path
    ) -> 'GroundTruth':
        """
        Extract ground truth from ALL deterministic models

        Models used:
        1. YOLOv8x (131MB) - 90% object detection
        2. CLIP ViT-L/14 (1.7GB) - 90% clothing/attributes
        3. Places365-R152 (800MB) - 92% scene classification
        4. PaddleOCR (300MB) - 92% text extraction
        5. VideoMAE (1.2GB) - 88% action recognition
        6. BLIP-2 Flan-T5-XL (4GB) - 85% contextual understanding
        7. Whisper base (150MB) - 95% audio transcription
        8. DeepSport (300MB) - 85% jersey number OCR
        9. FER+ (100MB) - 80% emotion detection
        10. Auto-Orient + Tesseract - text orientation
        """
        from processing.evidence_item import GroundTruth
        from processing.clip_processor import CLIPProcessor
        from processing.places365_processor import Places365Processor
        from processing.videomae_processor import VideoMAEProcessor
        from processing.blip2_processor import BLIP2Processor
        from processing.deepsport_processor import DeepSportProcessor
        from processing.fer_processor import FERProcessor
        from processing.text_orientation_processor import TextOrientationProcessor

        ground_truth_data = {}
        logger.info("  Starting 10-model ground truth extraction...")
        logger.info("  " + "-" * 76)

        # 1. YOLOv8x - Object Detection (131MB, 90% accuracy)
        logger.info("  [1/10] YOLOv8x - Object Detection (131MB, 90% accuracy)...")
        try:
            if self.object_detector is None:
                from processing.object_detector import ObjectDetector
                logger.info("    ⋯ Initializing YOLOv8x detector...")
                self.object_detector = ObjectDetector(model_name="yolov8x")
                logger.info("    ✓ YOLOv8x initialized")

            detection_result = self.object_detector.detect_objects_jit(frame, timestamp)
            ground_truth_data['yolov8x_objects'] = [d.to_dict() for d in detection_result.detections]
            ground_truth_data['object_count'] = detection_result.detection_count
            ground_truth_data['person_count'] = detection_result.person_count

            logger.info(f"    ✓ Detected {detection_result.detection_count} objects, {detection_result.person_count} persons")

            # Extract person bboxes for other processors
            person_bboxes = [
                (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
                for d in detection_result.detections
                if d.class_name == "person"
            ]
        except Exception as e:
            logger.warning(f"    ✗ YOLOv8x detection failed: {e}")
            person_bboxes = []

        # 2. CLIP ViT-L/14 - Clothing/Attributes (1.7GB, 90% accuracy)
        logger.info("  [2/10] CLIP ViT-L/14 - Clothing/Attributes (1.7GB, 90% accuracy)...")
        try:
            clip_processor = CLIPProcessor(model_name="ViT-L/14")
            if person_bboxes:
                clothing_attrs = clip_processor.detect_clothing_attributes(frame, person_bboxes)
                ground_truth_data['clip_clothing'] = [attr.to_dict() for attr in clothing_attrs]
                logger.info(f"    ✓ Detected {len(clothing_attrs)} clothing attributes")
            else:
                logger.info(f"    ⊘ No person bboxes, skipping CLIP")
        except Exception as e:
            logger.warning(f"    ✗ CLIP attribute detection failed: {e}")

        # 3. Places365-R152 - Scene Classification (800MB, 92% accuracy)
        logger.info("  [3/10] Places365-R152 - Scene Classification (800MB, 92% accuracy)...")
        try:
            places_processor = Places365Processor(model_name="resnet152")
            scene_result = places_processor.classify_scene(frame)
            ground_truth_data['places365_scene'] = scene_result.to_dict()
            ground_truth_data['is_indoor'] = scene_result.attributes.get('indoor', False)
            ground_truth_data['is_sports_venue'] = places_processor.is_sports_venue(frame)
            logger.info(f"    ✓ Scene: {scene_result.scene_type}, Indoor: {ground_truth_data['is_indoor']}, Sports Venue: {ground_truth_data['is_sports_venue']}")
        except Exception as e:
            logger.warning(f"    ✗ Places365 scene classification failed: {e}")

        # 4. PaddleOCR - Text Extraction (300MB, 92% accuracy)
        logger.info("  [4/10] PaddleOCR + Auto-Orient - Text Extraction (300MB, 92% accuracy)...")
        try:
            # First, auto-orient text for better OCR
            text_orient_processor = TextOrientationProcessor()
            oriented_frame, orient_metadata = text_orient_processor.auto_orient_for_ocr(frame)
            ground_truth_data['text_orientation'] = orient_metadata

            # Then extract text
            if self.ocr_processor is None:
                from processing.ocr_processor import OCRProcessor
                logger.info("    ⋯ Initializing PaddleOCR...")
                self.ocr_processor = OCRProcessor(use_local_ocr=True)  # Uses PaddleOCR
                logger.info("    ✓ PaddleOCR initialized")

            ocr_results = self.ocr_processor.extract_text_from_frame(oriented_frame)
            if ocr_results:
                ground_truth_data['paddleocr_text'] = [r.get('text', '') for r in ocr_results if r.get('text')]
                ground_truth_data['ocr_blocks'] = ocr_results
                logger.info(f"    ✓ Extracted {len(ground_truth_data['paddleocr_text'])} text blocks")
            else:
                logger.info(f"    ⊘ No text detected")
        except Exception as e:
            logger.warning(f"    ✗ PaddleOCR extraction failed: {e}")

        # 5. VideoMAE - Action Recognition (1.2GB, 88% accuracy)
        # Note: Requires temporal context (multiple frames)
        logger.info("  [5/10] VideoMAE - Action Recognition (1.2GB, 88% accuracy)...")
        try:
            videomae_processor = VideoMAEProcessor(model_name="videomae-base")
            # For single frame, we can't do full action recognition
            # This would be better used in a temporal window
            ground_truth_data['action_recognition_note'] = "Requires temporal window of frames"
            logger.info(f"    ⊘ Skipped (requires temporal context)")
        except Exception as e:
            logger.warning(f"    ✗ VideoMAE action recognition failed: {e}")

        # 6. BLIP-2 Flan-T5-XL - Contextual Understanding (4GB, 85% accuracy)
        logger.info("  [6/10] BLIP-2 Flan-T5-XL - Contextual Understanding (4GB, 85% accuracy)...")
        try:
            blip2_processor = BLIP2Processor(model_name="Salesforce/blip2-flan-t5-xl")
            context_result = blip2_processor.generate_description(frame)
            ground_truth_data['blip2_description'] = context_result.to_dict()
            ground_truth_data['image_caption'] = context_result.description
            logger.info(f"    ✓ Generated caption: '{context_result.description[:60]}...'")
        except Exception as e:
            logger.warning(f"    ✗ BLIP-2 contextual understanding failed: {e}")

        # 7. Whisper base - Audio Transcription (150MB, 95% accuracy)
        logger.info("  [7/10] Whisper base - Audio Transcription (150MB, 95% accuracy)...")
        try:
            if self.audio_processor is None:
                from processing.audio_processor import AudioProcessor
                logger.info("    ⋯ Initializing Whisper...")
                self.audio_processor = AudioProcessor()
                logger.info("    ✓ Whisper initialized")

            # Extract audio segment around timestamp
            segment = self.audio_processor.extract_segment(
                str(video_path),
                timestamp - 2.0,
                timestamp + 2.0
            )
            if segment and hasattr(segment, 'text'):
                ground_truth_data['whisper_transcript'] = segment.text
                logger.info(f"    ✓ Transcribed: '{segment.text[:60]}...'")
            else:
                logger.info(f"    ⊘ No speech detected")
        except Exception as e:
            logger.warning(f"    ✗ Whisper transcription failed: {e}")

        # 8. DeepSport - Jersey Number OCR (300MB, 85% accuracy)
        logger.info("  [8/10] DeepSport - Jersey Number OCR (300MB, 85% accuracy)...")
        try:
            deepsport_processor = DeepSportProcessor(model_name="deepsport-jersey-ocr")
            if person_bboxes:
                jersey_numbers = deepsport_processor.detect_jersey_numbers(frame, person_bboxes)
                ground_truth_data['deepsport_jerseys'] = [num.to_dict() for num in jersey_numbers]
                ground_truth_data['player_numbers'] = [num.number for num in jersey_numbers]
                logger.info(f"    ✓ Detected {len(jersey_numbers)} jersey numbers")
            else:
                logger.info(f"    ⊘ No person bboxes, skipping DeepSport")
        except Exception as e:
            logger.warning(f"    ✗ DeepSport jersey detection failed: {e}")

        # 9. FER+ - Facial Expression/Emotion (100MB, 80% accuracy)
        logger.info("  [9/10] FER+ - Facial Expression/Emotion (100MB, 80% accuracy)...")
        try:
            fer_processor = FERProcessor(model_name="fer+")
            facial_expressions = fer_processor.detect_emotions(frame)
            ground_truth_data['fer_emotions'] = [expr.to_dict() for expr in facial_expressions]
            if facial_expressions:
                ground_truth_data['dominant_emotion'] = fer_processor.get_dominant_emotion(frame)
                logger.info(f"    ✓ Detected {len(facial_expressions)} faces, dominant emotion: {ground_truth_data.get('dominant_emotion', 'N/A')}")
            else:
                logger.info(f"    ⊘ No faces detected")
        except Exception as e:
            logger.warning(f"    ✗ FER+ emotion detection failed: {e}")

        # 10. Scene Detector (legacy compatibility)
        logger.info("  [10/10] Scene Detector - Legacy compatibility...")
        try:
            if hasattr(self, 'scene_detector') and self.scene_detector:
                scene = self.scene_detector.classify_scene(frame)
                if scene:
                    ground_truth_data['legacy_scene'] = scene
                    logger.info(f"    ✓ Legacy scene: {scene}")
            else:
                logger.info(f"    ⊘ No legacy scene detector")
        except Exception as e:
            logger.warning(f"    ✗ Legacy scene classification failed: {e}")

        logger.info("  " + "-" * 76)
        logger.info(f"  ✓ Ground truth extraction complete using 10 models at {timestamp:.1f}s")
        logger.info("  " + "-" * 76)
        return GroundTruth(**ground_truth_data)

    def _get_gpt4_prediction(
        self,
        timestamp: float,
        ground_truth: 'GroundTruth'
    ) -> Optional[Dict[str, Any]]:
        """
        Get GPT-4 Vision prediction (or mock if testing)

        Args:
            timestamp: Video timestamp
            ground_truth: Ground truth data from deterministic models

        Returns:
            GPT-4 prediction dict or None
        """
        from config.test_config import TEST_CONFIG, MockPredictions

        if not TEST_CONFIG.enable_gpt4_vision:
            # Use mock prediction
            if TEST_CONFIG.use_mock_predictions:
                return MockPredictions.get_mock_gpt4_prediction(
                    evidence_type='visual_text',
                    timestamp=timestamp,
                    consensus_type=TEST_CONFIG.mock_consensus
                )
            return None

        # TODO: Call actual GPT-4 Vision API
        # import openai
        # response = openai.ChatCompletion.create(...)
        logger.warning("GPT-4 Vision API not implemented yet")
        return None

    def _get_claude_prediction(
        self,
        timestamp: float,
        ground_truth: 'GroundTruth'
    ) -> Optional[Dict[str, Any]]:
        """
        Get Claude Sonnet 4.5 prediction (or mock if testing)

        Args:
            timestamp: Video timestamp
            ground_truth: Ground truth data from deterministic models

        Returns:
            Claude prediction dict or None
        """
        from config.test_config import TEST_CONFIG, MockPredictions

        if not TEST_CONFIG.enable_claude_vision:
            # Use mock prediction
            if TEST_CONFIG.use_mock_predictions:
                return MockPredictions.get_mock_claude_prediction(
                    evidence_type='visual_text',
                    timestamp=timestamp,
                    consensus_type=TEST_CONFIG.mock_consensus
                )
            return None

        # TODO: Call actual Claude Vision API
        # import anthropic
        # response = anthropic.messages.create(...)
        logger.warning("Claude Vision API not implemented yet")
        return None

    def _get_open_model_prediction(
        self,
        timestamp: float,
        ground_truth: 'GroundTruth'
    ) -> Optional[Dict[str, Any]]:
        """
        Get open model prediction (YOLO, OCR, Whisper combined)

        This uses the actual ground truth from deterministic models

        Args:
            timestamp: Video timestamp
            ground_truth: Ground truth data from deterministic models

        Returns:
            Open model prediction dict
        """
        from config.test_config import TEST_CONFIG, MockPredictions

        # In test mode, return mock data
        if TEST_CONFIG.use_mock_predictions:
            return MockPredictions.get_mock_open_model_prediction(
                evidence_type='combined',
                timestamp=timestamp,
                consensus_type=TEST_CONFIG.mock_consensus
            )

        # In production, combine actual ground truth data
        if ground_truth:
            return ground_truth.to_dict() if hasattr(ground_truth, 'to_dict') else {}

        return None

    def _store_evidence_item(self, evidence_item: 'EvidenceItem') -> int:
        """Store evidence item in database"""
        from database.evidence_operations import EvidenceOperations

        try:
            # Convert predictions to dicts (handle both dict and object types)
            def to_dict_safe(pred):
                if pred is None:
                    return None
                if isinstance(pred, dict):
                    return pred
                if hasattr(pred, 'to_dict'):
                    return pred.to_dict()
                return pred

            gpt4_dict = to_dict_safe(evidence_item.gpt4_prediction)
            claude_dict = to_dict_safe(evidence_item.claude_prediction)
            open_dict = to_dict_safe(evidence_item.open_model_prediction)

            # Get consensus info
            confidence = 0.0
            needs_review = False
            priority = 'low'
            if evidence_item.consensus:
                confidence = evidence_item.consensus.confidence_score
                needs_review = evidence_item.consensus.needs_human_review
                priority = evidence_item.consensus.priority_level

            # Create evidence item in database
            evidence_id = EvidenceOperations.create_evidence_item(
                video_id=evidence_item.video_id,
                evidence_type=evidence_item.evidence_type,
                timestamp_start=evidence_item.timestamp_start,
                timestamp_end=evidence_item.timestamp_end,
                gpt4_prediction=gpt4_dict,
                claude_prediction=claude_dict,
                open_model_prediction=open_dict,
                ground_truth=evidence_item.ground_truth.to_dict() if evidence_item.ground_truth else None,
                ai_consensus_reached=(evidence_item.consensus.consensus_level.value != 'none') if evidence_item.consensus else False,
                consensus_answer=evidence_item.consensus.consensus_answer if evidence_item.consensus else None,
                confidence_score=confidence,
                needs_review=needs_review,
                priority=priority
            )

            return evidence_id

        except Exception as e:
            logger.error(f"Failed to store evidence item: {e}", exc_info=True)
            return 0


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = EvidenceExtractor(
        video_path=Path("sample_video.mp4"),
        video_id="vid_abc123",
        enable_caching=True
    )
    
    request = EvidenceRequest(
        question_id="q1",
        question_type="temporal",
        timestamp=10.5,
        evidence_types=[
            EvidenceType.VISUAL_CUE,
            EvidenceType.AUDIO_CUE
        ],
        time_window=(8.5, 12.5)
    )
    
    evidence = extractor.extract_evidence_jit(request)
    
    print(f"✓ Extracted evidence for {evidence.question_id}")
    print(f"  Cost: ${evidence.extraction_cost:.4f}")
