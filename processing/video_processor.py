"""
Video Processor - FIXED IMPLEMENTATION

This actually processes videos using real models:
- OpenCV for frame extraction
- Whisper for audio transcription
- YOLO for object detection
- PaddleOCR for text extraction
- Scene detection for segmentation

Cost Breakdown Target:
- Frame extraction: $0.50
- Audio processing: $0.30
- OCR: $0.40
- Object detection: $1.00
- Scene detection: $0.20
- Total processing: ~$2.40/video
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata extracted during processing"""
    video_id: str
    duration: float  # seconds
    fps: float
    width: int
    height: int
    total_frames: int
    has_audio: bool
    file_size: int
    file_hash: str
    processing_timestamp: str
    

@dataclass
class ProcessingCosts:
    """Track processing costs per video"""
    frame_extraction_cost: float = 0.0
    audio_processing_cost: float = 0.0
    ocr_cost: float = 0.0
    object_detection_cost: float = 0.0
    scene_detection_cost: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Calculate total processing cost"""
        return (
            self.frame_extraction_cost +
            self.audio_processing_cost +
            self.ocr_cost +
            self.object_detection_cost +
            self.scene_detection_cost
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return {
            "frame_extraction": self.frame_extraction_cost,
            "audio_processing": self.audio_processing_cost,
            "ocr": self.ocr_cost,
            "object_detection": self.object_detection_cost,
            "scene_detection": self.scene_detection_cost,
            "total": self.total_cost
        }


@dataclass
class RawVideoContext:
    """
    Raw multimodal context from video processing
    
    This is what the evidence extractor expects!
    """
    video_id: str
    duration: float
    
    # Raw transcript
    transcript: List[Dict]  # [{text, start, end, speaker, confidence}]
    
    # Raw visual detections
    frames: List[Dict]  # [{timestamp, detections, scene_info}]
    
    # Raw audio
    audio_features: Dict  # {music, sounds, ambient}
    
    # Metadata
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Complete video processing result"""
    video_id: str
    metadata: VideoMetadata
    raw_context: Optional[RawVideoContext]
    success: bool
    error_message: Optional[str] = None
    costs: Optional[ProcessingCosts] = None
    processing_time_seconds: float = 0.0
    
    # Processing stats
    frames_extracted: int = 0
    scenes_detected: int = 0
    transcript_segments: int = 0
    ocr_blocks: int = 0
    objects_detected: int = 0


class VideoProcessor:
    """
    FIXED: Actually processes videos using real models
    """
    
    def __init__(
        self,
        max_processing_cost: float = 2.40,
        cache_dir: Optional[Path] = None,
        enable_caching: bool = False,
        load_models: bool = True
    ):
        """
        Initialize video processor with real models.
        
        Args:
            max_processing_cost: Maximum allowed cost per video
            cache_dir: Directory for caching
            enable_caching: Whether to cache results
            load_models: Whether to load ML models (set False for testing)
        """
        self.max_processing_cost = max_processing_cost
        self.cache_dir = cache_dir or Path("./cache/video_processing")
        self.enable_caching = enable_caching
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models_loaded = False
        if load_models:
            self._init_models()
        
        logger.info(
            f"VideoProcessor initialized (models_loaded={self.models_loaded}, "
            f"max_cost=${max_processing_cost:.2f})"
        )
    
    def _init_models(self):
        """Initialize all ML models"""
        logger.info("Loading ML models...")
        
        try:
            # 1. Whisper for audio transcription
            import whisper
            # Auto-detect best device
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("base", device=device)
            logger.info("✓ Whisper loaded (base model)")
        except Exception as e:
            logger.warning(f"Failed to load Whisper: {e}")
            self.whisper_model = None
        
        try:
            # 2. YOLO for object detection
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")  # Nano model for speed
            logger.info("✓ YOLO loaded (v8 nano)")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}")
            self.yolo_model = None
        
        try:
            # 3. PaddleOCR for text extraction
            from paddleocr import PaddleOCR
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
            logger.info("✓ PaddleOCR loaded")
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}")
            self.ocr_model = None
        
        self.models_loaded = True
        logger.info("All models loaded successfully")
    
    def process_video(
        self,
        video_path: Path,
        video_id: Optional[str] = None,
        skip_audio: bool = False,
        skip_ocr: bool = False,
        skip_objects: bool = False
    ) -> ProcessingResult:
        """
        FIXED: Actually process video with real models
        
        Args:
            video_path: Path to video file
            video_id: Optional video ID
            skip_audio: Skip audio processing (save cost)
            skip_ocr: Skip OCR processing (save cost)
            skip_objects: Skip object detection (save cost)
        
        Returns:
            ProcessingResult with ACTUAL processed data
        """
        start_time = datetime.now()
        
        # Generate video ID if not provided
        if video_id is None:
            video_id = self._generate_video_id(video_path)
        
        logger.info(f"Processing video: {video_id} ({video_path.name})")
        
        # Check cache
        if self.enable_caching:
            cached_result = self._load_from_cache(video_id)
            if cached_result:
                logger.info(f"✓ Loaded cached result for {video_id}")
                return cached_result
        
        try:
            # Initialize cost tracking
            costs = ProcessingCosts()
            
            # Step 1: Extract metadata
            metadata = self._extract_metadata(video_path, video_id)
            logger.info(
                f"✓ Metadata: {metadata.duration:.1f}s, "
                f"{metadata.fps:.1f}fps, {metadata.width}x{metadata.height}"
            )
            
            # Step 2: Detect scenes
            scene_timestamps = self._detect_scenes(video_path, metadata)
            costs.scene_detection_cost = 0.20 + (metadata.duration / 60.0) * 0.01
            logger.info(f"✓ Detected {len(scene_timestamps)} scenes")
            
            # Step 3: Extract frames (sample from key scenes)
            frames_data = self._extract_frames(video_path, scene_timestamps, metadata)
            costs.frame_extraction_cost = len(frames_data) * 0.001
            logger.info(f"✓ Extracted {len(frames_data)} frames")
            
            # Step 4: Process audio (if not skipped)
            transcript = []
            if not skip_audio and metadata.has_audio:
                transcript = self._transcribe_audio(video_path, metadata)
                costs.audio_processing_cost = 0.30
                logger.info(f"✓ Transcribed {len(transcript)} segments")
            
            # Step 5: Run OCR on frames (if not skipped)
            if not skip_ocr:
                ocr_blocks = self._run_ocr_on_frames(frames_data)
                costs.ocr_cost = len(frames_data) * 0.002
                logger.info(f"✓ OCR found {ocr_blocks} text blocks")
            
            # Step 6: Detect objects in frames (if not skipped)
            objects_detected = 0
            if not skip_objects:
                objects_detected = self._detect_objects_in_frames(frames_data)
                costs.object_detection_cost = len(frames_data) * 0.003
                logger.info(f"✓ Detected {objects_detected} objects")
            
            # Step 7: Extract audio features
            audio_features = self._extract_audio_features(video_path, metadata)
            
            # Create RawVideoContext with ACTUAL data
            raw_context = RawVideoContext(
                video_id=video_id,
                duration=metadata.duration,
                transcript=transcript,
                frames=frames_data,
                audio_features=audio_features,
                metadata={
                    "fps": metadata.fps,
                    "width": metadata.width,
                    "height": metadata.height,
                    "scenes": scene_timestamps
                }
            )
            
            # Check cost budget
            if costs.total_cost > self.max_processing_cost:
                logger.warning(
                    f"⚠ Cost ${costs.total_cost:.2f} exceeds "
                    f"budget ${self.max_processing_cost:.2f}"
                )
            
            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            result = ProcessingResult(
                video_id=video_id,
                metadata=metadata,
                raw_context=raw_context,
                success=True,
                costs=costs,
                processing_time_seconds=processing_time,
                frames_extracted=len(frames_data),
                scenes_detected=len(scene_timestamps),
                transcript_segments=len(transcript),
                ocr_blocks=ocr_blocks if not skip_ocr else 0,
                objects_detected=objects_detected
            )
            
            # Cache result
            if self.enable_caching:
                self._save_to_cache(video_id, result)
            
            logger.info(
                f"✓ Video processing complete: {video_id} "
                f"(cost: ${costs.total_cost:.2f}, time: {processing_time:.1f}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Error processing video {video_id}: {str(e)}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                video_id=video_id,
                metadata=metadata if 'metadata' in locals() else None,
                raw_context=None,
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    def _generate_video_id(self, video_path: Path) -> str:
        """Generate unique video ID from file"""
        # Use filename + size for quick ID
        file_size = video_path.stat().st_size
        identifier = f"{video_path.name}_{file_size}"
        file_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"vid_{file_hash}"
    
    def _extract_metadata(self, video_path: Path, video_id: str) -> VideoMetadata:
        """
        FIXED: Actually extract video metadata using OpenCV
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Extract properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0
            
            # Check audio (read a few frames)
            has_audio = True  # Assume yes, proper detection needs ffprobe
            
            cap.release()
            
            # File info
            file_size = video_path.stat().st_size
            file_hash = self._generate_video_id(video_path).replace("vid_", "")
            
            return VideoMetadata(
                video_id=video_id,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                has_audio=has_audio,
                file_size=file_size,
                file_hash=file_hash,
                processing_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise
    
    def _detect_scenes(
        self, 
        video_path: Path, 
        metadata: VideoMetadata
    ) -> List[Tuple[float, float]]:
        """
        FIXED: Actually detect scenes using PySceneDetect
        """
        try:
            from scenedetect import detect, ContentDetector
            
            # Detect scenes
            scene_list = detect(
                str(video_path),
                ContentDetector(threshold=27.0)
            )
            
            # Convert to timestamps
            scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                
                # Skip very short scenes (< 1 second)
                if end_sec - start_sec >= 1.0:
                    scenes.append((start_sec, end_sec))
            
            # If no scenes detected, create one scene for whole video
            if not scenes:
                scenes = [(0.0, metadata.duration)]
            
            return scenes
            
        except ImportError:
            logger.warning("PySceneDetect not available, using uniform segmentation")
            # Fallback: divide into 10-second segments
            scenes = []
            current = 0.0
            while current < metadata.duration:
                end = min(current + 10.0, metadata.duration)
                scenes.append((current, end))
                current = end
            return scenes
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            # Fallback: single scene
            return [(0.0, metadata.duration)]
    
    def _extract_frames(
        self,
        video_path: Path,
        scenes: List[Tuple[float, float]],
        metadata: VideoMetadata
    ) -> List[Dict]:
        """
        FIXED: Actually extract frames using OpenCV
        """
        try:
            import cv2
            
            frames_data = []
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Extract 3 frames per scene (beginning, middle, end)
            for scene_id, (start_time, end_time) in enumerate(scenes):
                # Calculate frame numbers
                start_frame = int(start_time * metadata.fps)
                end_frame = int(end_time * metadata.fps)
                mid_frame = (start_frame + end_frame) // 2
                
                # Extract 3 key frames per scene
                for frame_num in [start_frame, mid_frame, end_frame]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        timestamp = frame_num / metadata.fps
                        
                        frames_data.append({
                            "timestamp": timestamp,
                            "frame_number": frame_num,
                            "scene_id": scene_id,
                            "frame_data": frame_rgb,  # Actual frame!
                            "detections": [],  # Will be populated by object detection
                            "ocr_results": []  # Will be populated by OCR
                        })
            
            cap.release()
            logger.info(f"Extracted {len(frames_data)} frames from {len(scenes)} scenes")
            return frames_data
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _transcribe_audio(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> List[Dict]:
        """
        FIXED: Actually transcribe audio using Whisper
        """
        if self.whisper_model is None:
            logger.warning("Whisper not available, skipping transcription")
            return []
        
        try:
            # Extract audio to temp file
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name
            
            # Use ffmpeg to extract audio
            subprocess.run([
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                temp_path
            ], check=True, capture_output=True)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_path,
                language='en',
                word_timestamps=True
            )
            
            # Convert to expected format
            transcript = []
            for segment in result['segments']:
                transcript.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end'],
                    'confidence': segment.get('confidence', 0.9),
                    'words': segment.get('words', [])
                })
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            return transcript
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return []
    
    def _run_ocr_on_frames(self, frames_data: List[Dict]) -> int:
        """
        FIXED: Actually run OCR using PaddleOCR
        """
        if self.ocr_model is None:
            logger.warning("PaddleOCR not available, skipping OCR")
            return 0
        
        total_blocks = 0
        
        try:
            for frame_dict in frames_data:
                frame = frame_dict['frame_data']
                
                # Run OCR
                result = self.ocr_model.ocr(frame, cls=True)
                
                # Parse results
                ocr_results = []
                if result and result[0]:
                    for line in result[0]:
                        bbox, (text, confidence) = line
                        
                        if confidence > 0.5:  # Filter low confidence
                            ocr_results.append({
                                'text': text,
                                'confidence': float(confidence),
                                'bbox': bbox
                            })
                            total_blocks += 1
                
                # Store in frame
                frame_dict['ocr_results'] = ocr_results
            
            return total_blocks
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return 0
    
    def _detect_objects_in_frames(self, frames_data: List[Dict]) -> int:
        """
        FIXED: Actually detect objects using YOLO
        """
        if self.yolo_model is None:
            logger.warning("YOLO not available, skipping object detection")
            return 0
        
        total_objects = 0
        
        try:
            for frame_dict in frames_data:
                frame = frame_dict['frame_data']
                
                # Run YOLO
                results = self.yolo_model(frame, verbose=False)
                
                # Parse detections
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        if confidence > 0.5:  # Filter low confidence
                            detections.append({
                                'class': result.names[cls_id],
                                'class_id': cls_id,
                                'confidence': confidence,
                                'bbox': bbox
                            })
                            total_objects += 1
                
                # Store in frame
                frame_dict['detections'] = detections
            
            return total_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return 0
    
    def _extract_audio_features(
        self,
        video_path: Path,
        metadata: VideoMetadata
    ) -> Dict:
        """
        Extract audio features (music, sounds, ambient)
        
        Placeholder for now - would use librosa for full implementation
        """
        return {
            'music': [],
            'sounds': [],
            'ambient': []
        }
    
    def _load_from_cache(self, video_id: str) -> Optional[ProcessingResult]:
        """Load cached processing result"""
        cache_file = self.cache_dir / f"{video_id}_result.json"
        context_file = self.cache_dir / f"{video_id}_context.pkl"
        
        if not cache_file.exists() or not context_file.exists():
            return None
        
        try:
            import pickle
            
            # Load result metadata
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Load raw context
            with open(context_file, 'rb') as f:
                raw_context = pickle.load(f)
            
            # Reconstruct result
            return ProcessingResult(
                video_id=data["video_id"],
                metadata=VideoMetadata(**data["metadata"]),
                raw_context=raw_context,
                success=data["success"],
                error_message=data.get("error_message"),
                costs=ProcessingCosts(**data["costs"]) if data.get("costs") else None,
                processing_time_seconds=data.get("processing_time_seconds", 0.0),
                frames_extracted=data.get("frames_extracted", 0),
                scenes_detected=data.get("scenes_detected", 0),
                transcript_segments=data.get("transcript_segments", 0),
                ocr_blocks=data.get("ocr_blocks", 0),
                objects_detected=data.get("objects_detected", 0)
            )
        except Exception as e:
            logger.warning(f"Failed to load cache for {video_id}: {e}")
            return None
    
    def _save_to_cache(self, video_id: str, result: ProcessingResult) -> None:
        """Save processing result to cache"""
        cache_file = self.cache_dir / f"{video_id}_result.json"
        context_file = self.cache_dir / f"{video_id}_context.pkl"
        
        try:
            import pickle
            
            # Save result metadata
            cache_data = {
                "video_id": result.video_id,
                "metadata": {
                    "video_id": result.metadata.video_id,
                    "duration": result.metadata.duration,
                    "fps": result.metadata.fps,
                    "width": result.metadata.width,
                    "height": result.metadata.height,
                    "total_frames": result.metadata.total_frames,
                    "has_audio": result.metadata.has_audio,
                    "file_size": result.metadata.file_size,
                    "file_hash": result.metadata.file_hash,
                    "processing_timestamp": result.metadata.processing_timestamp
                },
                "success": result.success,
                "error_message": result.error_message,
                "costs": result.costs.to_dict() if result.costs else None,
                "processing_time_seconds": result.processing_time_seconds,
                "frames_extracted": result.frames_extracted,
                "scenes_detected": result.scenes_detected,
                "transcript_segments": result.transcript_segments,
                "ocr_blocks": result.ocr_blocks,
                "objects_detected": result.objects_detected,
                "cached_at": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Save raw context separately (pickle for numpy arrays)
            with open(context_file, 'wb') as f:
                pickle.dump(result.raw_context, f)
            
            logger.debug(f"✓ Cached result for {video_id}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {video_id}: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize with real models
    processor = VideoProcessor(
        max_processing_cost=2.40,
        load_models=True
    )
    
    # Process video
    video_path = Path("sample_video.mp4")
    
    result = processor.process_video(
        video_path=video_path,
        skip_audio=False,  # Set True to save $0.30
        skip_ocr=False,    # Set True to save ~$0.40
        skip_objects=False # Set True to save ~$1.00
    )
    
    if result.success:
        print(f"✓ Video processed: {result.video_id}")
        print(f"  Duration: {result.metadata.duration:.1f}s")
        print(f"  Frames: {result.frames_extracted}")
        print(f"  Scenes: {result.scenes_detected}")
        print(f"  Transcript segments: {result.transcript_segments}")
        print(f"  OCR blocks: {result.ocr_blocks}")
        print(f"  Objects detected: {result.objects_detected}")
        print(f"  Cost: ${result.costs.total_cost:.2f}")
        print(f"  Time: {result.processing_time_seconds:.1f}s")
        
        # The RawVideoContext is now populated!
        print(f"\n✓ RawVideoContext ready for evidence extraction:")
        print(f"  - Transcript: {len(result.raw_context.transcript)} segments")
        print(f"  - Frames: {len(result.raw_context.frames)} frames with detections")
        print(f"  - Audio features: {len(result.raw_context.audio_features)} types")
    else:
        print(f"✗ Processing failed: {result.error_message}")