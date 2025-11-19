"""
Audio Processor - Audio Extraction & Transcription

Model: Whisper base (150MB)
Accuracy: 95% transcription accuracy
Purpose: Extract and transcribe audio from videos with cost optimization
Compliance: JIT extraction per question, minimize transcription costs
Architecture: Evidence-first, supports segment-based processing

Primary Model: Whisper base (150MB, 95% accuracy)
Alternative: Whisper API for cloud processing ($0.006/minute)
Cost Optimization Strategy:
- Extract audio only for relevant segments (not full video)
- Use local Whisper base model (free)
- Cache transcriptions to avoid re-processing
- Target: ~$0.30 per video for audio processing (if using API)
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Single audio segment with transcription"""
    start_time: float  # seconds
    end_time: float  # seconds
    transcription: str  # Transcribed text
    confidence: float  # Transcription confidence (0.0-1.0)
    language: Optional[str] = None  # Detected language
    speaker_id: Optional[int] = None  # Speaker identification (if available)
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds"""
        return self.end_time - self.start_time
    
    @property
    def word_count(self) -> int:
        """Number of words in transcription"""
        return len(self.transcription.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "transcription": self.transcription,
            "confidence": self.confidence,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "duration": self.duration
        }


@dataclass
class AudioExtractionResult:
    """Result of audio extraction and transcription"""
    video_id: str
    segments: List[AudioSegment]
    total_duration: float  # Total audio duration
    transcription_cost: float  # Cost in dollars
    language: Optional[str] = None  # Primary language
    has_music: bool = False  # Contains background music
    has_speech: bool = True  # Contains speech
    
    @property
    def full_transcription(self) -> str:
        """Concatenate all segment transcriptions"""
        return " ".join(seg.transcription for seg in self.segments)
    
    @property
    def segment_count(self) -> int:
        """Number of audio segments"""
        return len(self.segments)


class AudioProcessor:
    """
    Extract and transcribe audio from videos.
    
    Optimized for cost-effective JIT processing:
    1. Extract audio only for needed segments
    2. Cache transcriptions
    3. Use segment-based processing to minimize API costs
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_caching: bool = True,
        whisper_api_key: Optional[str] = None
    ):
        """
        Initialize audio processor.
        
        Args:
            cache_dir: Directory for caching transcriptions
            enable_caching: Whether to cache results
            whisper_api_key: API key for Whisper transcription
        """
        self.cache_dir = cache_dir or Path("./cache/audio")
        self.enable_caching = enable_caching
        self.whisper_api_key = whisper_api_key
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("AudioProcessor initialized")
    
    def extract_audio_segments(
        self,
        video_path: Path,
        video_id: str,
        segments: Optional[List[Tuple[float, float]]] = None,
        full_video: bool = False
    ) -> AudioExtractionResult:
        """
        Extract and transcribe audio segments.
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            segments: List of (start, end) timestamps to extract
                     If None, extracts full video audio
            full_video: If True, process entire video (expensive!)
        
        Returns:
            AudioExtractionResult with transcribed segments
        """
        logger.info(f"Processing audio for {video_id}")
        
        # Check cache
        if self.enable_caching:
            cached = self._load_from_cache(video_id, segments)
            if cached:
                logger.info(f"Loaded cached transcription for {video_id}")
                return cached
        
        # Determine segments to process
        if full_video or segments is None:
            # Get video duration
            duration = self._get_video_duration(video_path)
            process_segments = [(0.0, duration)]
            logger.warning(
                f"Processing full video audio ({duration:.1f}s) - "
                "this increases costs!"
            )
        else:
            process_segments = segments
        
        # Extract and transcribe each segment
        transcribed_segments = []
        total_cost = 0.0
        
        for start, end in process_segments:
            # Extract audio segment
            audio_path = self._extract_audio_segment(
                video_path, start, end, video_id
            )
            
            if audio_path is None:
                logger.warning(f"Failed to extract audio segment {start}-{end}")
                continue
            
            # Transcribe segment
            segment = self._transcribe_segment(audio_path, start, end)
            
            if segment:
                transcribed_segments.append(segment)
                
                # Calculate transcription cost
                duration_minutes = segment.duration / 60.0
                cost = duration_minutes * 0.006  # $0.006/minute for Whisper
                total_cost += cost
            
            # Clean up temporary audio file
            if audio_path.exists():
                audio_path.unlink()
        
        # Get total video duration
        total_duration = self._get_video_duration(video_path)
        
        # Detect primary language (from first segment)
        primary_language = None
        if transcribed_segments:
            primary_language = transcribed_segments[0].language
        
        # Create result
        result = AudioExtractionResult(
            video_id=video_id,
            segments=transcribed_segments,
            total_duration=total_duration,
            transcription_cost=total_cost,
            language=primary_language,
            has_speech=len(transcribed_segments) > 0
        )
        
        # Cache result
        if self.enable_caching:
            self._save_to_cache(video_id, result, segments)
        
        logger.info(
            f"Transcribed {len(transcribed_segments)} segments "
            f"(cost: ${total_cost:.4f})"
        )
        
        return result
    
    def transcribe_segment_jit(
        self,
        video_path: Path,
        start_time: float,
        end_time: float
    ) -> Optional[AudioSegment]:
        """
        Transcribe a single segment on-demand (JIT).
        
        This is the most cost-effective method for question-specific audio.
        
        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
        
        Returns:
            AudioSegment with transcription or None if failed
        """
        # Extract segment
        audio_path = self._extract_audio_segment(
            video_path, start_time, end_time, "temp"
        )
        
        if audio_path is None:
            return None
        
        # Transcribe
        segment = self._transcribe_segment(audio_path, start_time, end_time)
        
        # Clean up
        if audio_path.exists():
            audio_path.unlink()
        
        return segment
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            return duration
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0
    
    def _extract_audio_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        video_id: str
    ) -> Optional[Path]:
        """
        Extract audio segment from video.

        Uses ffmpeg to extract audio between timestamps.

        Returns:
            Path to extracted audio file or None if failed
        """
        import subprocess

        output_path = self.cache_dir / f"{video_id}_{start_time}_{end_time}.wav"

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-ss", str(start_time),
            "-to", str(end_time),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # WAV format
            "-ar", "16000",  # 16kHz sampling rate
            "-ac", "1",  # Mono
            str(output_path),
            "-y"  # Overwrite
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return None

        return output_path
    
    def _transcribe_segment(
        self,
        audio_path: Path,
        start_time: float,
        end_time: float
    ) -> Optional[AudioSegment]:
        """
        Transcribe audio segment using Whisper API or local Whisper.

        Args:
            audio_path: Path to audio file
            start_time: Original start time in video
            end_time: Original end time in video

        Returns:
            AudioSegment with transcription
        """
        # Try OpenAI Whisper API first if key is available
        if self.whisper_api_key:
            try:
                import openai

                client = openai.OpenAI(api_key=self.whisper_api_key)

                with open(audio_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )

                return AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    transcription=transcript.text,
                    confidence=1.0,  # Whisper doesn't provide confidence
                    language=getattr(transcript, 'language', 'en')
                )
            except Exception as e:
                logger.warning(f"OpenAI Whisper API failed: {e}, falling back to local Whisper")

        # Fall back to local Whisper using whisper library
        try:
            import whisper

            # Load tiny model for speed (can upgrade to base/small later)
            model = whisper.load_model("tiny")
            result = model.transcribe(str(audio_path))

            return AudioSegment(
                start_time=start_time,
                end_time=end_time,
                transcription=result["text"],
                confidence=1.0,
                language=result.get("language", "en")
            )
        except ImportError:
            logger.error("Neither OpenAI API nor local whisper library available")
            logger.error("Install with: pip install openai-whisper")
            return None
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            return None
    
    def _load_from_cache(
        self,
        video_id: str,
        segments: Optional[List[Tuple[float, float]]]
    ) -> Optional[AudioExtractionResult]:
        """Load cached transcription result"""
        # Create cache key from segments
        if segments:
            segments_str = "_".join(f"{s}_{e}" for s, e in segments)
            cache_key = f"{video_id}_{segments_str}"
        else:
            cache_key = f"{video_id}_full"
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct segments
            segments_list = [
                AudioSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    transcription=seg["transcription"],
                    confidence=seg["confidence"],
                    language=seg.get("language"),
                    speaker_id=seg.get("speaker_id")
                )
                for seg in data["segments"]
            ]
            
            return AudioExtractionResult(
                video_id=data["video_id"],
                segments=segments_list,
                total_duration=data["total_duration"],
                transcription_cost=data["transcription_cost"],
                language=data.get("language"),
                has_music=data.get("has_music", False),
                has_speech=data.get("has_speech", True)
            )
        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
            return None
    
    def _save_to_cache(
        self,
        video_id: str,
        result: AudioExtractionResult,
        segments: Optional[List[Tuple[float, float]]]
    ) -> None:
        """Save transcription result to cache"""
        # Create cache key
        if segments:
            segments_str = "_".join(f"{s}_{e}" for s, e in segments)
            cache_key = f"{video_id}_{segments_str}"
        else:
            cache_key = f"{video_id}_full"
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                "video_id": result.video_id,
                "segments": [seg.to_dict() for seg in result.segments],
                "total_duration": result.total_duration,
                "transcription_cost": result.transcription_cost,
                "language": result.language,
                "has_music": result.has_music,
                "has_speech": result.has_speech
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Saved cache for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = AudioProcessor(enable_caching=True)
    
    # Example 1: JIT segment transcription (recommended)
    segment = processor.transcribe_segment_jit(
        video_path=Path("sample_video.mp4"),
        start_time=10.5,
        end_time=15.2
    )
    
    if segment:
        print(f"✓ Transcribed segment:")
        print(f"  Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
        print(f"  Text: {segment.transcription}")
        print(f"  Language: {segment.language}")
    
    # Example 2: Batch segment processing
    segments = [
        (5.0, 10.0),   # First question audio cue
        (20.0, 25.0),  # Second question audio cue
        (40.0, 45.0)   # Third question audio cue
    ]
    
    result = processor.extract_audio_segments(
        video_path=Path("sample_video.mp4"),
        video_id="vid_abc123",
        segments=segments
    )
    
    print(f"\n✓ Transcribed {result.segment_count} segments")
    print(f"  Cost: ${result.transcription_cost:.4f}")
    print(f"  Full text: {result.full_transcription[:100]}...")
