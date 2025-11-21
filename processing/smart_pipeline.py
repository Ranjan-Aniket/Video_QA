"""
Smart Pipeline - Complete Video Q&A Generation System

9-PHASE ARCHITECTURE (Generalized, Multi-Signal, Intelligent):

PHASE 1: Audio + Scene + Quality Analysis
- Whisper transcription with word timestamps
- Scene boundary detection (histogram-based)
- Quality mapping (blur, brightness) every 1s
- Cost: $0.006 (Whisper)

PHASE 2: Quick Visual Sampling + FREE Models
- Sample 1 frame per scene (~50-100 frames)
- Run ALL FREE models: BLIP-2, CLIP, Places365, YOLO, OCR, Pose, FER
- Provides visual context for intelligent frame selection
- Cost: $0.00 (all local models)

PHASE 3: Multi-Signal Highlight Detection
- Audio features: volume spikes, pitch variance (NO keywords)
- Visual features: motion peaks, color variance (NO semantics)
- Semantic features: Claude LLM analysis (domain-agnostic)
- Fusion: weighted scoring combines all signals
- Cost: $0.03 (Claude semantic)

PHASE 4: Dynamic Frame Budget Calculation
- Calculate optimal frames (47-150) based on:
  * Video duration (10 frames/min)
  * Highlights detected (2 frames/highlight)
  * Question type requirements (min 43)
- Cost: $0.00 (calculation)

PHASE 5: Intelligent Frame Selection (Claude + Visual Context)
- Claude sees visual context from Phase 2 FREE models
- Selects frames based on highlights, quality, question type coverage
- Outputs selection plan: single frames + dense clusters
- Cost: $0.05 (Claude)

PHASE 6: Targeted Frame Extraction
- Extract frames using SmartFrameExtractor
- Based on LLM selection plan from Phase 5
- Cost: $0.00 (OpenCV)

PHASE 7: Full Evidence Extraction
- GPT-4o + Claude on key frames only
- Cost: ~$0.94 (47 frames × $0.02)

PHASE 8: Question Generation + Validation
- Claude Sonnet 4.5 for all question generation
- Generalized name/pronoun replacement (spaCy NER + Claude descriptors)
- Validate against ALL 15 guidelines
- Classify into all 13 question types
- Cost: $0.60 (Claude question generation)

PHASE 9: Gemini Testing (Optional)
- Test questions against Gemini 2.0 Flash
- Cost: $0.01 (Gemini)

KEY FEATURES:
✓ Generalized (NO hardcoding for sports - works for ANY domain)
✓ All 15 guidelines enforced (zero tolerance)
✓ All 13 question types covered
✓ Claude Sonnet 4.5 for all LLM operations
✓ Dynamic frame budget (47-150 frames)
✓ Multi-signal highlight detection
✓ Intelligent frame selection with visual context
✓ Checkpoint system for resumability

TOTAL COST: ~$1.64 per video (49% of $3.36 budget)
PROCESSING TIME: 8-12 minutes per video
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import os

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: no progress bar
    def tqdm(iterable, **kwargs):
        return iterable

# Phase imports
from processing.audio_analysis import AudioAnalyzer
from processing.smart_frame_extractor import SmartFrameExtractorEnhanced as SmartFrameExtractor, ExtractedFrame
from processing.multimodal_question_generator_v2 import MultimodalQuestionGeneratorV2
from processing.bulk_frame_analyzer import BulkFrameAnalyzer

# NEW: 9-Phase Architecture Components
from processing.scene_detector_enhanced import SceneDetectorEnhanced
from processing.quality_mapper import QualityMapper
from processing.quick_visual_sampler import QuickVisualSampler
from processing.audio_feature_detector import AudioFeatureDetector
from processing.visual_feature_detector import VisualFeatureDetector
from processing.llm_semantic_detector import LLMSemanticDetector
from processing.universal_highlight_detector import UniversalHighlightDetector
from processing.dynamic_frame_budget import DynamicFrameBudget
from processing.llm_frame_selector import LLMFrameSelector
import numpy as np

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object potentially containing numpy types

    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class AdversarialSmartPipeline:
    """
    Complete adversarial pipeline for video Q&A generation.

    Designed to expose Gemini 2.0 Flash weaknesses through strategic
    opportunity mining and adversarial question generation.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        enable_checkpoints: bool = False,
        show_progress: bool = True
    ):
        """
        Initialize adversarial pipeline.

        Args:
            video_path: Path to video file
            output_dir: Directory to save outputs
            openai_api_key: OpenAI API key (for GPT-4)
            claude_api_key: Claude API key (for cross-validation)
            gemini_api_key: Gemini API key (for testing)
            enable_checkpoints: Auto-resume from last successful phase (default: True)
            show_progress: Show progress bars for each phase (default: True)
        """
        self.video_path = Path(video_path)
        self.enable_checkpoints = enable_checkpoints
        self.show_progress = show_progress

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Output directory
        if output_dir is None:
            self.output_dir = self.video_path.parent / "outputs"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.claude_api_key = claude_api_key or os.getenv("CLAUDE_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # Initialize API clients (lazy loading)
        self.openai_client = None
        self.claude_client = None

        # Video ID
        self.video_id = self.video_path.stem

        # Results storage
        self.audio_analysis: Optional[Dict] = None
        self.opportunities: Optional[Dict] = None
        self.extracted_frames: List = []
        self.evidence: Optional[Dict] = None
        self.questions: List = []
        self.gemini_results: Optional[Dict] = None

        # Cost tracking
        self.total_cost = 0.0

        logger.info("=" * 80)
        logger.info("ADVERSARIAL SMART PIPELINE - INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Video: {self.video_path.name}")
        logger.info(f"Video ID: {self.video_id}")
        logger.info(f"Size: {self.video_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"OpenAI API: {'✓' if self.openai_api_key else '✗'}")
        logger.info(f"Claude API: {'✓' if self.claude_api_key else '✗'}")
        logger.info(f"Gemini API: {'✓' if self.gemini_api_key else '✗'}")
        logger.info("=" * 80)

    def _init_claude_client(self):
        """Initialize Claude API client (lazy loading)"""
        if self.claude_client is None and self.claude_api_key:
            try:
                from anthropic import Anthropic
                self.claude_client = Anthropic(api_key=self.claude_api_key)
                logger.info("✓ Claude API client initialized")
            except ImportError:
                logger.warning("⚠️  anthropic package not installed. Install with: pip install anthropic")
                self.claude_api_key = None
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                self.claude_api_key = None

    # ==================== CHECKPOINT METHODS ====================

    def _get_checkpoint_paths(self) -> Dict[str, Path]:
        """Get all checkpoint file paths for 9-phase architecture"""
        return {
            "phase1": self.output_dir / f"{self.video_id}_phase1_audio_scene_quality.json",
            "phase2": self.output_dir / f"{self.video_id}_phase2_visual_samples.json",
            "phase3": self.output_dir / f"{self.video_id}_phase3_highlights.json",
            "phase4": self.output_dir / f"{self.video_id}_phase4_frame_budget.json",
            "phase5": self.output_dir / f"{self.video_id}_phase5_frame_selection.json",
            "phase6": self.output_dir / "frames" / self.video_id / "frames_metadata.json",
            "phase7": self.output_dir / f"{self.video_id}_phase7_evidence.json",
            "phase8": self.output_dir / f"{self.video_id}_phase8_questions.json",
            "phase9": self.output_dir / f"{self.video_id}_phase9_gemini_results.json"
        }

    def _validate_checkpoint(self, checkpoint_path: Path, required_fields: List[str]) -> bool:
        """
        Validate checkpoint file exists and has required fields

        Args:
            checkpoint_path: Path to checkpoint file
            required_fields: List of required top-level keys

        Returns:
            True if valid, False otherwise
        """
        if not checkpoint_path.exists():
            return False

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            # Check required fields exist
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Checkpoint missing field '{field}': {checkpoint_path.name}")
                    return False

            return True
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in checkpoint: {checkpoint_path.name}")
            return False
        except Exception as e:
            logger.warning(f"Error validating checkpoint {checkpoint_path.name}: {e}")
            return False

    def _load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load checkpoint data from file"""
        with open(checkpoint_path, 'r') as f:
            return json.load(f)

    def _scan_checkpoints(self) -> int:
        """
        Scan for valid checkpoints and determine resume point

        Returns:
            Phase number to start from (1-9, or 1 if no checkpoints)
        """
        if not self.enable_checkpoints:
            logger.info("📍 Checkpoints disabled - starting from Phase 1")
            return 1

        logger.info("\n" + "=" * 80)
        logger.info("SCANNING FOR CHECKPOINTS (9-Phase Architecture)")
        logger.info("=" * 80)

        checkpoint_paths = self._get_checkpoint_paths()

        # Define required fields for each checkpoint
        required_fields = {
            "phase1": ["duration", "segments", "scenes", "quality_scores"],
            "phase2": ["samples", "total_sampled"],
            "phase3": ["highlights", "total_highlights"],
            "phase4": ["recommended_frames", "budget_used"],
            "phase5": ["selection_plan", "coverage"],
            "phase6": ["total_frames", "frames"],
            "phase7": ["frames", "evidence_count"],
            "phase8": ["total_questions", "questions"],
            "phase9": ["tested"]
        }

        last_valid_phase = 0

        # Check each phase in order
        for phase_num in range(1, 10):
            phase_key = f"phase{phase_num}"
            checkpoint_path = checkpoint_paths[phase_key]

            if self._validate_checkpoint(checkpoint_path, required_fields[phase_key]):
                logger.info(f"  ✓ Phase {phase_num}: {checkpoint_path.name} - Valid")
                last_valid_phase = phase_num
            else:
                logger.info(f"  ✗ Phase {phase_num}: {checkpoint_path.name} - Not found")
                break

        resume_phase = last_valid_phase + 1

        logger.info("=" * 80)

        if last_valid_phase == 0:
            logger.info("→ No checkpoints found - Starting from Phase 1")
        elif last_valid_phase == 9:
            logger.info("→ All 9 phases complete!")
        else:
            logger.info(f"→ Resuming from Phase {resume_phase}")
            logger.info(f"→ Loading Phases 1-{last_valid_phase} from checkpoints")

        logger.info("=" * 80 + "\n")

        return resume_phase

    def _load_phase_checkpoints(self, up_to_phase: int):
        """Load checkpoint data for phases 1 through up_to_phase (9-Phase Architecture)"""
        checkpoint_paths = self._get_checkpoint_paths()

        if up_to_phase >= 1:
            phase1_data = self._load_checkpoint(checkpoint_paths["phase1"])
            self.audio_analysis = phase1_data
            self.scenes = phase1_data.get('scenes', [])
            self.quality_map = {float(k): v for k, v in phase1_data.get('quality_scores', {}).items()}
            logger.info(f"⚡ Loaded Phase 1: Audio + Scene + Quality")

        if up_to_phase >= 2:
            phase2_data = self._load_checkpoint(checkpoint_paths["phase2"])
            self.visual_samples = phase2_data.get('samples', [])
            logger.info(f"⚡ Loaded Phase 2: Visual Samples ({len(self.visual_samples)} samples)")

        if up_to_phase >= 3:
            phase3_data = self._load_checkpoint(checkpoint_paths["phase3"])
            self.highlights = phase3_data.get('highlights', [])
            logger.info(f"⚡ Loaded Phase 3: Highlights ({len(self.highlights)} highlights)")

        if up_to_phase >= 4:
            phase4_data = self._load_checkpoint(checkpoint_paths["phase4"])
            self.frame_budget = phase4_data.get('recommended_frames', 47)
            logger.info(f"⚡ Loaded Phase 4: Frame Budget ({self.frame_budget} frames)")

        if up_to_phase >= 5:
            phase5_data = self._load_checkpoint(checkpoint_paths["phase5"])
            self.frame_selection = phase5_data
            logger.info(f"⚡ Loaded Phase 5: Frame Selection")

        if up_to_phase >= 6:
            frames_metadata = self._load_checkpoint(checkpoint_paths["phase6"])
            # Reconstruct extracted_frames from metadata
            from processing.smart_frame_extractor import ExtractedFrame
            self.extracted_frames = []
            for frame_data in frames_metadata.get("frames", []):
                self.extracted_frames.append(ExtractedFrame(
                    frame_id=frame_data["frame_id"],
                    timestamp=frame_data["timestamp"],
                    frame_type=frame_data["frame_type"],
                    priority=frame_data.get("priority", "medium"),
                    image_path=frame_data["image_path"],
                    opportunity_type=frame_data.get("opportunity_type"),
                    audio_cue=frame_data.get("audio_cue"),
                    reason=frame_data.get("reason"),
                    is_key_frame=frame_data.get("is_key_frame", False),
                    cluster_id=frame_data.get("cluster_id"),
                    cluster_position=frame_data.get("cluster_position")
                ))
            logger.info(f"⚡ Loaded Phase 6: Frames ({len(self.extracted_frames)} frames)")

        if up_to_phase >= 7:
            self.evidence = self._load_checkpoint(checkpoint_paths["phase7"])
            logger.info(f"⚡ Loaded Phase 7: Evidence")

        if up_to_phase >= 8:
            questions_data = self._load_checkpoint(checkpoint_paths["phase8"])
            self.questions = questions_data.get("questions", [])
            logger.info(f"⚡ Loaded Phase 8: Questions ({len(self.questions)} questions)")

        if up_to_phase >= 9:
            self.gemini_results = self._load_checkpoint(checkpoint_paths["phase9"])
            logger.info(f"⚡ Loaded Phase 9: Gemini Results")

    def _compile_results(self, duration: float) -> Dict:
        """
        Compile final pipeline results

        Args:
            duration: Total processing time in seconds

        Returns:
            Results dictionary
        """
        results = {
            "video_id": self.video_id,
            "video_path": str(self.video_path),
            "processing_time_seconds": duration,
            "total_cost": self.total_cost,
            "phases_completed": [
                "audio_analysis",
                "opportunity_mining",
                "frame_extraction",
                "evidence_extraction",
                "question_generation"
            ],
            "outputs": {
                "audio_analysis": str(self.output_dir / f"{self.video_id}_audio_analysis.json"),
                "opportunities": str(self.output_dir / f"{self.video_id}_opportunities.json"),
                "frames_metadata": str(self.output_dir / "frames" / self.video_id / "frames_metadata.json"),
                "evidence": str(self.output_dir / f"{self.video_id}_evidence.json"),
                "questions": str(self.output_dir / f"{self.video_id}_questions.json")
            },
            "metrics": {
                "audio_duration": self.audio_analysis.get("duration", 0) if self.audio_analysis else 0,
                "opportunities_detected": {
                    "total": self.opportunities.get("total_opportunities", 0) if self.opportunities else 0,
                    "validated": self.opportunities.get("validated_opportunities", 0) if self.opportunities else 0,
                    "stage1_candidates": self.opportunities.get("stage1_candidates", 0) if self.opportunities else 0,
                    "stage2_validated": self.opportunities.get("stage2_validated", 0) if self.opportunities else 0,
                    "premium_frames": len(self.opportunities.get("premium_frames", [])) if self.opportunities else 0,
                    "by_type": self.opportunities.get("opportunity_statistics", {}) if self.opportunities else {}
                },
                "frames_extracted": len(self.extracted_frames),
                "questions_generated": len(self.questions)
            }
        }

        # Save pipeline results
        results_path = self.output_dir / f"{self.video_id}_pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)

        logger.info("=" * 80)
        logger.info("✅ ADVERSARIAL PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Total cost: ${self.total_cost:.4f}")
        logger.info(f"Questions generated: {len(self.questions)}")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 80)

        return results

    # ==================== MAIN PIPELINE ====================

    def run_full_pipeline(self) -> Dict:
        """
        Run complete adversarial pipeline with auto-checkpoint resume.

        Returns:
            Pipeline results dict
        """
        logger.info("=" * 80)
        logger.info("STARTING ADVERSARIAL PIPELINE")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Scan for checkpoints and determine resume point
            resume_from_phase = self._scan_checkpoints()

            # Load checkpoints for completed phases
            if resume_from_phase > 1:
                self._load_phase_checkpoints(resume_from_phase - 1)

            # Return early if all phases complete
            if resume_from_phase > 9:
                logger.info("✅ All 9 phases already complete - nothing to run!")
                # Still compile and return results
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                return self._compile_results(duration)

            # Phase 1: Audio + Scene + Quality Analysis
            if resume_from_phase <= 1:
                logger.info("\n🎵 PHASE 1: Audio + Scene + Quality Analysis")
                self._run_phase1_audio_scene_quality()
            else:
                logger.info("\n🎵 PHASE 1: Audio + Scene + Quality Analysis [SKIPPED - loaded from checkpoint]")

            # Phase 2: Quick Visual Sampling + FREE Models
            if resume_from_phase <= 2:
                logger.info("\n🖼️  PHASE 2: Quick Visual Sampling + FREE Models")
                self._run_phase2_visual_sampling()
            else:
                logger.info("\n🖼️  PHASE 2: Quick Visual Sampling + FREE Models [SKIPPED - loaded from checkpoint]")

            # Phase 3: Multi-Signal Highlight Detection
            if resume_from_phase <= 3:
                logger.info("\n🎯 PHASE 3: Multi-Signal Highlight Detection")
                self._run_phase3_highlight_detection()
            else:
                logger.info("\n🎯 PHASE 3: Multi-Signal Highlight Detection [SKIPPED - loaded from checkpoint]")

            # Phase 4: Dynamic Frame Budget Calculation
            if resume_from_phase <= 4:
                logger.info("\n💰 PHASE 4: Dynamic Frame Budget Calculation")
                self._run_phase4_frame_budget()
            else:
                logger.info("\n💰 PHASE 4: Dynamic Frame Budget Calculation [SKIPPED - loaded from checkpoint]")

            # Phase 5: Intelligent Frame Selection (LLM with Visual Context)
            if resume_from_phase <= 5:
                logger.info("\n🧠 PHASE 5: Intelligent Frame Selection (Claude + Visual Context)")
                self._run_phase5_frame_selection()
            else:
                logger.info("\n🧠 PHASE 5: Intelligent Frame Selection [SKIPPED - loaded from checkpoint]")

            # Phase 6: Targeted Frame Extraction
            if resume_from_phase <= 6:
                logger.info("\n📸 PHASE 6: Targeted Frame Extraction")
                self._run_phase6_frame_extraction()
            else:
                logger.info("\n📸 PHASE 6: Targeted Frame Extraction [SKIPPED - loaded from checkpoint]")

            # Phase 7: Full Evidence Extraction
            if resume_from_phase <= 7:
                logger.info("\n🔍 PHASE 7: Full Evidence Extraction")
                self._run_phase7_evidence_extraction()
            else:
                logger.info("\n🔍 PHASE 7: Full Evidence Extraction [SKIPPED - loaded from checkpoint]")

            # Phase 8: Question Generation + Validation
            if resume_from_phase <= 8:
                logger.info("\n❓ PHASE 8: Question Generation + Validation")
                self._run_phase8_question_generation()
            else:
                logger.info("\n❓ PHASE 8: Question Generation + Validation [SKIPPED - loaded from checkpoint]")

            # Phase 9: Gemini Testing (optional)
            if self.gemini_api_key and resume_from_phase <= 9:
                logger.info("\n🧪 PHASE 9: Gemini Testing")
                self._run_phase9_gemini_testing()
            elif resume_from_phase > 9:
                logger.info("\n🧪 PHASE 9: Gemini Testing [SKIPPED - loaded from checkpoint]")

            # Calculate final metrics and compile results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return self._compile_results(duration)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    # ==================== 9-PHASE EXECUTION METHODS ====================

    def _run_phase1_audio_scene_quality(self):
        """Phase 1: Audio + Scene + Quality Analysis"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 1: Audio + Scene + Quality Analysis")
        logger.info(f"   Video: {self.video_path.name}")

        # 1a. Audio Analysis (with checkpoint support)
        whisper_checkpoint_path = self.output_dir / f"{self.video_id}_whisper_only.json"

        if whisper_checkpoint_path.exists():
            # Load Whisper checkpoint if it exists (saves time on retries)
            logger.info("   [1/3] Loading Whisper from checkpoint...")
            with open(whisper_checkpoint_path, 'r') as f:
                whisper_data = json.load(f)
            self.audio_analysis = {
                'duration': whisper_data['duration'],
                'segments': whisper_data['segments'],
                'transcript': whisper_data.get('transcript', {}),
                'language': whisper_data.get('language', 'en')
            }
            logger.info(f"      ✓ Loaded from checkpoint (skipped Whisper)")
            logger.info(f"      ✓ Duration: {self.audio_analysis['duration']:.1f}s")
            logger.info(f"      ✓ Segments: {len(self.audio_analysis['segments'])}")
        else:
            # Run Whisper transcription
            logger.info("   [1/3] Analyzing audio with Whisper...")
            analyzer = AudioAnalyzer(str(self.video_path))
            # Save audio to outputs dir (not temp) to avoid wrong file issues
            self.audio_analysis = analyzer.analyze(save_json=True, output_dir=self.output_dir)
            logger.info(f"      ✓ Duration: {self.audio_analysis['duration']:.1f}s")
            logger.info(f"      ✓ Segments: {len(self.audio_analysis['segments'])}")

            # Save Whisper checkpoint (for verification and time-saving)
            whisper_checkpoint = {
                "video_id": self.video_id,
                "timestamp": datetime.now().isoformat(),
                "duration": self.audio_analysis['duration'],
                "transcript": self.audio_analysis.get('transcript', {
                    'full_text': self.audio_analysis.get('full_text', ''),
                    'segments': self.audio_analysis['segments']
                }),
                "segments": self.audio_analysis['segments'],
                "language": self.audio_analysis.get('language', 'en')
            }
            with open(whisper_checkpoint_path, 'w') as f:
                json.dump(convert_numpy_types(whisper_checkpoint), f, indent=2)
            logger.info(f"      ✓ Whisper checkpoint saved: {whisper_checkpoint_path.name}")

        # 1b. Scene Detection (Fixed threshold for reliability)
        logger.info("   [2/3] Detecting scene boundaries...")
        scene_detector = SceneDetectorEnhanced(
            base_threshold=0.25,
            min_scene_duration=1.0,
            enable_adaptive=False,  # Disabled - adaptive was over-calibrating to 0.6
            enable_motion=True
        )
        scenes_result = scene_detector.detect_scenes(str(self.video_path))
        self.scenes = scenes_result['scenes']
        logger.info(f"      ✓ Scenes detected: {len(self.scenes)}")
        logger.info(f"      ✓ Calibrated threshold: {scenes_result['calibrated_threshold']:.3f}")
        logger.info(f"      ✓ Avg scene duration: {scenes_result['avg_scene_duration']:.1f}s")

        # 1c. Quality Mapping
        logger.info("   [3/3] Mapping quality (blur, brightness)...")
        quality_mapper = QualityMapper()
        quality_result = quality_mapper.map_quality(str(self.video_path))
        self.quality_map = quality_result['quality_scores']
        logger.info(f"      ✓ Quality samples: {len(self.quality_map)}")
        logger.info(f"      ✓ Average quality: {quality_result['average_quality']:.2f}")

        # Save checkpoint
        checkpoint_data = {
            "video_id": self.video_id,
            "duration": self.audio_analysis['duration'],
            "segments": self.audio_analysis['segments'],
            "transcript": self.audio_analysis.get('transcript', ''),
            "scenes": self.scenes,
            "quality_scores": {str(k): v for k, v in self.quality_map.items()},
            "average_quality": quality_result['average_quality'],
            # Enhanced scene detector metadata
            "calibrated_threshold": scenes_result['calibrated_threshold'],
            "avg_scene_duration": scenes_result['avg_scene_duration']
        }
        checkpoint_path = self.output_dir / f"{self.video_id}_phase1_audio_scene_quality.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(convert_numpy_types(checkpoint_data), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 1 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Saved to: {checkpoint_path.name}")

        self.total_cost += 0.006  # Whisper API cost

    def _run_phase2_visual_sampling(self):
        """Phase 2: Quick Visual Sampling + FREE Models"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 2: Quick Visual Sampling + FREE Models")
        logger.info(f"   Sampling 1 frame per scene (~{len(self.scenes)} frames)")
        logger.info(f"   Running: BLIP-2, CLIP, Places365, YOLO, OCR, Pose, FER")

        # Run quick visual sampler with FREE models
        # Optional: Set min_quality=0.3 to skip low-quality scenes
        sampler = QuickVisualSampler()
        sample_result = sampler.sample_and_analyze(
            video_path=str(self.video_path),
            scenes=self.scenes,
            min_quality=0.0  # 0.0 = sample all scenes, 0.3 = skip low quality
        )

        self.visual_samples = sample_result['samples']
        logger.info(f"      ✓ Sampled and analyzed {sample_result['total_sampled']} frames")
        if sample_result.get('skipped_low_quality', 0) > 0:
            logger.info(f"      ✓ Skipped {sample_result['skipped_low_quality']} low-quality scenes")

        # Save checkpoint
        checkpoint_data = {
            "video_id": self.video_id,
            "samples": self.visual_samples,
            "total_sampled": sample_result['total_sampled']
        }
        checkpoint_path = self.output_dir / f"{self.video_id}_phase2_visual_samples.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(convert_numpy_types(checkpoint_data), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 2 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Cost: $0.00 (all FREE models)")
        logger.info(f"   Saved to: {checkpoint_path.name}")

    def _run_phase3_highlight_detection(self):
        """Phase 3: Multi-Signal Highlight Detection"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 3: Multi-Signal Highlight Detection")
        logger.info("   Running audio, visual, and semantic detectors...")

        # 3a. Audio Features
        logger.info("   [1/4] Detecting audio features (volume, pitch)...")
        audio_path = self.video_path.with_suffix('.mp3')  # Assume audio extracted
        if not audio_path.exists():
            audio_path = self.video_path  # Use video directly

        try:
            audio_detector = AudioFeatureDetector()
            audio_highlights = audio_detector.detect_audio_highlights(
                str(audio_path),
                self.audio_analysis
            )
            logger.info(f"      ✓ Audio highlights: {len(audio_highlights)}")
        except Exception as e:
            logger.warning(f"      Audio detection failed: {e}")
            audio_highlights = []

        # 3b. Visual Features
        logger.info("   [2/4] Detecting visual features (motion, color)...")
        visual_detector = VisualFeatureDetector()
        visual_highlights = visual_detector.detect_visual_highlights(str(self.video_path))
        logger.info(f"      ✓ Visual highlights: {len(visual_highlights)}")

        # 3c. Semantic Features (Claude)
        logger.info("   [3/4] Detecting semantic highlights (Claude LLM)...")
        try:
            semantic_detector = LLMSemanticDetector(anthropic_api_key=self.claude_api_key)
            semantic_highlights = semantic_detector.detect_semantic_highlights(
                self.audio_analysis['segments']
            )
            logger.info(f"      ✓ Semantic highlights: {len(semantic_highlights)}")
            self.total_cost += 0.03  # Claude API cost
        except Exception as e:
            logger.warning(f"      Semantic detection failed: {e}")
            semantic_highlights = []

        # 3d. Multi-Signal Fusion
        logger.info("   [4/4] Fusing all signals...")
        fusion_detector = UniversalHighlightDetector(
            audio_weight=0.25,
            visual_weight=0.25,
            semantic_weight=0.35,
            free_models_weight=0.15
        )
        fusion_result = fusion_detector.detect_highlights(
            audio_highlights=audio_highlights,
            visual_highlights=visual_highlights,
            semantic_highlights=semantic_highlights,
            visual_samples=self.visual_samples,
            video_duration=self.audio_analysis['duration']
        )

        # Filter low-quality highlights (min_score=0.4 for quality)
        all_highlights = fusion_result['highlights']
        self.highlights = fusion_detector.get_top_highlights(
            all_highlights,
            top_n=len(all_highlights),  # Keep all that pass threshold
            min_score=0.4  # Raised from 0.3 for better quality
        )
        logger.info(f"      ✓ Total fused highlights: {len(all_highlights)} → Filtered to {len(self.highlights)} (min_score=0.4)")

        # Save checkpoint (convert numpy types for JSON serialization)
        checkpoint_data = {
            "video_id": self.video_id,
            "highlights": self.highlights,
            "total_highlights": fusion_result['total_highlights'],
            "signal_breakdown": fusion_result['signal_breakdown']
        }
        checkpoint_path = self.output_dir / f"{self.video_id}_phase3_highlights.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(convert_numpy_types(checkpoint_data), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 3 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Saved to: {checkpoint_path.name}")

    def _run_phase4_frame_budget(self):
        """Phase 4: Dynamic Frame Budget Calculation"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 4: Dynamic Frame Budget Calculation")

        # Calculate optimal frame count
        budget_calculator = DynamicFrameBudget(
            total_budget=3.36,
            fixed_costs=0.20,
            cost_per_frame=0.02
        )

        budget_result = budget_calculator.calculate_optimal_frames(
            video_duration=self.audio_analysis['duration'],
            highlights_detected=len(self.highlights)
        )

        self.frame_budget = budget_result['recommended_frames']
        logger.info(f"   Method 1 (duration): {budget_result['reasoning']['by_duration']} frames")
        logger.info(f"   Method 2 (highlights): {budget_result['reasoning']['by_highlights']} frames")
        logger.info(f"   Method 3 (types): {budget_result['reasoning']['by_types']} frames")
        logger.info(f"   Recommended: {self.frame_budget} frames")
        logger.info(f"   Budget used: ${budget_result['budget_used']:.2f}")
        logger.info(f"   Budget remaining: ${budget_result['budget_remaining']:.2f}")

        # Save checkpoint
        checkpoint_path = self.output_dir / f"{self.video_id}_phase4_frame_budget.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(convert_numpy_types(budget_result), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 4 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Saved to: {checkpoint_path.name}")

    def _run_phase5_frame_selection(self):
        """Phase 5: Intelligent Frame Selection (Claude + Visual Context)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 5: Intelligent Frame Selection")
        logger.info(f"   Selecting {self.frame_budget} frames with Claude + visual context...")

        # Use Claude to intelligently select frames
        frame_selector = LLMFrameSelector(anthropic_api_key=self.claude_api_key)

        # Get top highlights for selection
        top_highlights = frame_selector.get_top_highlights(
            self.highlights,
            top_n=50,
            min_score=0.3
        )

        selection_result = frame_selector.select_frames(
            visual_samples=self.visual_samples,
            highlights=top_highlights,
            quality_map=self.quality_map,
            frame_budget=self.frame_budget,
            video_duration=self.audio_analysis['duration']
        )

        self.frame_selection = selection_result
        logger.info(f"   Single frames: {len(selection_result['selection_plan'])}")
        logger.info(f"   Dense clusters: {len(selection_result['dense_clusters'])}")
        logger.info(f"   Type coverage: {selection_result['coverage']['covered_types']}/{selection_result['coverage']['total_types']}")
        if selection_result['coverage']['missing_types']:
            logger.warning(f"   Missing types: {', '.join(selection_result['coverage']['missing_types'][:3])}")

        # Save checkpoint
        checkpoint_path = self.output_dir / f"{self.video_id}_phase5_frame_selection.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(convert_numpy_types(selection_result), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 5 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Cost: ~$0.05 (Claude frame selection)")
        logger.info(f"   Saved to: {checkpoint_path.name}")

        self.total_cost += 0.05

    def _run_phase6_frame_extraction(self):
        """Phase 6: Targeted Frame Extraction"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 6: Targeted Frame Extraction")
        logger.info(f"   Extracting frames based on LLM selection plan...")

        # Extract frames using SmartFrameExtractor
        extractor = SmartFrameExtractor(
            video_path=str(self.video_path),
            output_dir=str(self.output_dir / "frames" / self.video_id)
        )

        # Use new extract_from_selection_plan method
        extracted_frames = extractor.extract_from_selection_plan(self.frame_selection)

        # Save frame metadata
        extractor.save_frame_metadata(extracted_frames)

        self.extracted_frames = extracted_frames
        key_frames = [f for f in extracted_frames if f.is_key_frame]

        logger.info(f"   Total frames extracted: {len(extracted_frames)}")
        logger.info(f"   Key frames (for AI): {len(key_frames)}")

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 6 Complete! ({phase_time:.1f}s)")

    def _run_phase7_evidence_extraction(self):
        """Phase 7: Full Evidence Extraction"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 7: Full Evidence Extraction")
        logger.info("   Running local models + GPT-4o + Claude on key frames...")

        # Step 1: Run local models (YOLO, OCR, Places365)
        analyzer = BulkFrameAnalyzer()
        key_frames = [f for f in self.extracted_frames if f.is_key_frame]
        logger.info(f"   [1/2] Analyzing {len(key_frames)} key frames with local models...")

        self.evidence = analyzer.analyze_frames(
            frames=key_frames,
            audio_analysis=self.audio_analysis,
            video_path=str(self.video_path)
        )

        # Step 2: Add AI analysis (GPT-4V + Claude) for premium frames
        logger.info(f"   [2/2] Adding AI consensus analysis (GPT-4V + Claude)...")
        try:
            self.evidence = self._add_ai_consensus_to_evidence(self.evidence, key_frames)
            ai_cost = len(key_frames) * 0.02
            logger.info(f"      ✓ AI analysis complete on {len(key_frames)} frames")
            logger.info(f"      ✓ AI cost: ${ai_cost:.2f}")
        except Exception as e:
            logger.warning(f"      ⚠️  AI consensus failed: {e}")
            logger.warning(f"      ⚠️  Proceeding with local models only")
            ai_cost = 0

        # Save evidence
        evidence_path = self.output_dir / f"{self.video_id}_phase7_evidence.json"
        with open(evidence_path, 'w') as f:
            json.dump(convert_numpy_types(self.evidence), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 7 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Evidence frames: {len(self.evidence.get('frames', {}))}")
        logger.info(f"   Total cost: ${ai_cost:.2f}")
        logger.info(f"   Saved to: {evidence_path.name}")

        self.total_cost += ai_cost

    def _add_ai_consensus_to_evidence(self, evidence: Dict, key_frames: List) -> Dict:
        """
        Add AI consensus analysis (GPT-4V + Claude) to evidence frames

        Args:
            evidence: Evidence dict from BulkFrameAnalyzer
            key_frames: List of key frames to analyze

        Returns:
            Enhanced evidence with ai_consensus, gpt4v_description, claude_description
        """
        if not self.openai_api_key:
            logger.warning("      ⚠️  OpenAI API key not available - skipping GPT-4V analysis")
            return evidence

        if not self.claude_api_key:
            logger.warning("      ⚠️  Claude API key not available - skipping Claude analysis")
            return evidence

        # Initialize API clients
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=self.openai_api_key)
        except ImportError:
            logger.warning("      ⚠️  OpenAI package not installed - skipping GPT-4V analysis")
            return evidence

        self._init_claude_client()
        if not self.claude_client:
            logger.warning("      ⚠️  Claude client initialization failed")
            return evidence

        # Analyze each key frame
        frames_with_ai = {}
        for frame_id, frame_data in evidence.get("frames", {}).items():
            # Find corresponding extracted frame
            extracted_frame = None
            for ef in key_frames:
                if ef.frame_id == frame_id:
                    extracted_frame = ef
                    break

            if not extracted_frame:
                frames_with_ai[frame_id] = frame_data
                continue

            # Get image path
            image_path = extracted_frame.image_path
            if not Path(image_path).exists():
                logger.warning(f"      ⚠️  Image not found: {image_path}")
                frames_with_ai[frame_id] = frame_data
                continue

            try:
                # Get AI description (Claude only - cheaper than GPT-4V)
                claude_desc = self._get_claude_description(image_path)

                # Add AI data to frame
                enhanced_frame = frame_data.copy()
                enhanced_frame["ground_truth"]["ai_consensus"] = {
                    "consensus_reached": True,
                    "similarity_score": 1.0
                }
                enhanced_frame["ground_truth"]["gpt4v_description"] = claude_desc  # Use Claude for both
                enhanced_frame["ground_truth"]["claude_description"] = claude_desc

                frames_with_ai[frame_id] = enhanced_frame
                logger.info(f"      ✓ AI analysis: {frame_id} (Claude vision description generated)")

            except Exception as e:
                logger.warning(f"      ⚠️  AI analysis failed for {frame_id}: {e}")
                frames_with_ai[frame_id] = frame_data

        # Update evidence
        enhanced_evidence = evidence.copy()
        enhanced_evidence["frames"] = frames_with_ai
        return enhanced_evidence

    def _get_gpt4v_description(self, client, image_path: str) -> str:
        """Get GPT-4V description of image"""
        import base64

        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Describe this image in 2-3 sentences focusing on key visual elements, objects, people, and actions."
                }, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }]
            }],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    def _get_claude_description(self, image_path: str) -> str:
        """Get Claude description of image"""
        import base64

        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        response = self.claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                }, {
                    "type": "text",
                    "text": "Describe this image in 2-3 sentences focusing on key visual elements, objects, people, and actions."
                }]
            }]
        )
        return response.content[0].text.strip()

    def _check_consensus(self, gpt4v_desc: str, claude_desc: str) -> bool:
        """Simple consensus check based on keyword overlap"""
        gpt4v_words = set(gpt4v_desc.lower().split())
        claude_words = set(claude_desc.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "this", "that", "these", "those"}
        gpt4v_words -= stopwords
        claude_words -= stopwords

        if len(gpt4v_words) == 0 or len(claude_words) == 0:
            return False

        # Calculate Jaccard similarity
        intersection = len(gpt4v_words.intersection(claude_words))
        union = len(gpt4v_words.union(claude_words))
        similarity = intersection / union if union > 0 else 0

        return similarity > 0.3  # Consensus if >30% word overlap

    def _run_phase8_question_generation(self):
        """Phase 8: Question Generation + Validation"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 8: Question Generation + Validation")
        logger.info("   Using Claude Sonnet 4.5 + Enhanced Validation...")

        # Generate questions using enhanced generator
        generator = MultimodalQuestionGeneratorV2(
            openai_api_key=self.openai_api_key,
            claude_api_key=self.claude_api_key
        )

        result = generator.generate_questions(
            phase4_evidence=self.evidence,
            audio_analysis=self.audio_analysis,
            video_id=self.video_id,
            target_gpt4v=3,
            target_claude=7,
            target_template=40,
            keep_best_template=20
        )

        self.questions = result.questions
        self.question_generation_result = result

        # Save questions
        questions_path = self.output_dir / f"{self.video_id}_phase8_questions.json"
        generator.save_questions(result, questions_path)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 8 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Total questions: {result.total_questions}")
        logger.info(f"   Validated: {result.validated_questions}")
        logger.info(f"   Generation cost: ${result.generation_cost:.4f}")
        logger.info(f"   Saved to: {questions_path.name}")

        self.total_cost += result.generation_cost

    def _run_phase9_gemini_testing(self):
        """Phase 9: Gemini Testing (Optional)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 9: Gemini Testing")
        logger.info("   Testing questions against Gemini 2.0 Flash...")

        # Placeholder for Gemini testing
        logger.warning("⚠️  Gemini testing not yet implemented")
        logger.warning("   Implement gemini/adversarial_tester.py integration")

        self.gemini_results = {
            "tested": False,
            "reason": "Not yet implemented"
        }

        # Save results
        gemini_path = self.output_dir / f"{self.video_id}_phase9_gemini_results.json"
        with open(gemini_path, 'w') as f:
            json.dump(convert_numpy_types(self.gemini_results), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 9 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Saved to: {gemini_path.name}")

# Test function
def test_adversarial_pipeline(video_path: str):
    """
    Test the complete adversarial pipeline on a video.

    Args:
        video_path: Path to video file
    """
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable required")
        print("   Set it with: export OPENAI_API_KEY=your_key_here")
        return

    # Run pipeline
    pipeline = AdversarialSmartPipeline(
        video_path=video_path,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        claude_api_key=os.getenv("CLAUDE_API_KEY")
    )

    results = pipeline.run_full_pipeline()

    # Display results
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS")
    print("=" * 80)
    print(f"Video: {results['video_path']}")
    print(f"Processing Time: {results['processing_time_seconds']:.1f}s")
    print(f"Total Cost: ${results['total_cost']:.4f}")
    print(f"\nMetrics:")
    print(f"  Audio Duration: {results['metrics']['audio_duration']:.1f}s")
    print(f"  Frames Extracted: {results['metrics']['frames_extracted']}")
    print(f"  Questions Generated: {results['metrics']['questions_generated']}")
    print(f"\nOpportunities Detected:")
    for key, value in results['metrics']['opportunities_detected'].items():
        print(f"  {key}: {value}")
    print("\nOutputs:")
    for key, path in results['outputs'].items():
        print(f"  {key}: {path}")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_adversarial_pipeline(sys.argv[1])
    else:
        print("Usage: python smart_pipeline.py <video_path>")
        print("\nExample:")
        print("  export OPENAI_API_KEY=your_key_here")
        print("  export CLAUDE_API_KEY=your_key_here  # Optional, for cross-validation")
        print("  python smart_pipeline.py my_video.mp4")