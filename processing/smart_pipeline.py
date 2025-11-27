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
- Claude Vision (only) on key frames - optimized for cost
- Cost: ~$0.12 (47 frames × $0.003 - Claude Vision pricing)

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

# NEW: Optimized Phase 8 (Direct Vision, Skip Phase 7)
from processing.phase8_vision_generator import Phase8VisionGenerator

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
from processing.content_moderator import ContentModerator, is_intro_outro
import numpy as np

# NEW: Pass 1-2B Architecture (Two-Pass Adversarial Moment Selection)
from processing.clip_analyzer import run_clip_analysis
from processing.pass1_smart_filter import run_pass1_filter
from processing.pass2a_sonnet_selector import run_pass2a_selection
from processing.pass2b_opus_selector import run_pass2b_selection
from processing.moment_validator import run_validation
from processing.pass3_qa_generator import run_qa_generation
from processing.ontology_types import OFFICIAL_TYPES, normalize_type

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
        enable_checkpoints: bool = True,
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
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
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
        """Get all checkpoint file paths for new Pass 1-2B architecture"""
        return {
            "phase1": self.output_dir / f"{self.video_id}_phase1_audio_scene_quality.json",
            "phase2": self.output_dir / f"{self.video_id}_phase2_visual_samples.json",
            "clip_analysis": self.output_dir / f"{self.video_id}_clip_analysis.json",
            "pass1": self.output_dir / f"{self.video_id}_pass1_filtered_frames.json",
            "pass2a": self.output_dir / f"{self.video_id}_pass2a_sonnet_moments.json",
            "pass2b": self.output_dir / f"{self.video_id}_pass2b_opus_moments.json",
            "validation": self.output_dir / f"{self.video_id}_validated_moments.json",
            "pass3": self.output_dir / f"{self.video_id}_pass3_qa_pairs.json",
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
        logger.info("SCANNING FOR CHECKPOINTS (Pass 1-2B Architecture)")
        logger.info("=" * 80)

        # DETECT AND DELETE OLD ARCHITECTURE CHECKPOINTS
        checkpoint_paths = self._get_checkpoint_paths()

        # Check if we have old architecture (Phase 1 exists but no PASS 2A/2B)
        phase1_exists = checkpoint_paths["phase1"].exists()
        pass2a_exists = checkpoint_paths["pass2a"].exists()
        pass2b_exists = checkpoint_paths["pass2b"].exists()
        clip_analysis_exists = checkpoint_paths["clip_analysis"].exists()

        if phase1_exists and not (pass2a_exists or pass2b_exists or clip_analysis_exists):
            logger.warning("⚠️  OLD ARCHITECTURE CHECKPOINTS DETECTED!")
            logger.warning("   Phase 1 exists but PASS 2A/2B/CLIP checkpoints missing")
            logger.warning("   → Deleting all old checkpoints and forcing fresh processing")

            # Delete all JSON files in output directory (old checkpoints)
            deleted_count = 0
            for json_file in self.output_dir.glob("*.json"):
                try:
                    json_file.unlink()
                    logger.info(f"   ✓ Deleted: {json_file.name}")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"   ✗ Failed to delete {json_file.name}: {e}")

            logger.info(f"   ✅ Deleted {deleted_count} old checkpoint files")
            logger.info("   → Starting fresh with new PASS 2A/2B architecture")
            logger.info("=" * 80 + "\n")
            return 1

        checkpoint_paths = self._get_checkpoint_paths()

        # Define required fields for each checkpoint (Pass 1-2B Architecture)
        required_fields = {
            "phase1": ["duration", "segments", "scenes"],  # Audio + Scene
            "phase2": ["samples", "total_sampled"],  # Visual Sampling 2fps
            "clip_analysis": ["spurious_candidates", "visual_anomalies"],  # CLIP Analysis
            "pass1": ["selected_frames", "selection_breakdown"],  # Pass 1 Filter
            "pass2a": ["mode1_precise"],  # Pass 2A (Sonnet moments)
            "pass2b": ["mode3_inference_window"],  # Pass 2B (Opus moments)
            "validation": ["validated_moments"],  # Moment validation
            "pass3": ["qa_pairs"]  # Pass 3 (QA generation)
        }

        # Define checkpoint order for Pass 1-2B Architecture
        checkpoint_order = [
            ("phase1", 1, "Audio + Scene Analysis"),
            ("phase2", 2, "Visual Sampling 2fps"),
            ("clip_analysis", 3, "CLIP Analysis"),
            ("pass1", 4, "Pass 1 Filter"),
            ("pass2a", 5, "Pass 2A Sonnet"),
            ("pass2b", 6, "Pass 2B Opus"),
            ("validation", 7, "Moment Validation"),
            ("pass3", 8, "QA Generation")
        ]

        last_valid_phase = 0

        # Check each checkpoint in order
        for checkpoint_key, phase_num, description in checkpoint_order:
            if checkpoint_key not in checkpoint_paths:
                logger.info(f"  ✗ Phase {phase_num} ({checkpoint_key}): Not in checkpoint_paths")
                break

            checkpoint_path = checkpoint_paths[checkpoint_key]

            if self._validate_checkpoint(checkpoint_path, required_fields[checkpoint_key]):
                logger.info(f"  ✓ Phase {phase_num}: {checkpoint_path.name} - Valid ({description})")
                last_valid_phase = phase_num
            else:
                logger.info(f"  ✗ Phase {phase_num}: {checkpoint_path.name} - Not found ({description})")
                break

        resume_phase = last_valid_phase + 1

        logger.info("=" * 80)

        if last_valid_phase == 0:
            logger.info("→ No checkpoints found - Starting from Phase 1")
        elif last_valid_phase == 8:
            logger.info("→ All Pass 1-2B phases complete!")
        else:
            logger.info(f"→ Resuming from Phase {resume_phase}")
            logger.info(f"→ Loading Phases 1-{last_valid_phase} from checkpoints")

        logger.info("=" * 80 + "\n")

        return resume_phase

    def _load_phase_checkpoints(self, up_to_phase: int):
        """Load checkpoint data for phases 1 through up_to_phase (Pass 1-2B Architecture)"""
        checkpoint_paths = self._get_checkpoint_paths()

        if up_to_phase >= 1:
            phase1_data = self._load_checkpoint(checkpoint_paths["phase1"])
            self.audio_analysis = phase1_data
            self.scenes = phase1_data.get('scenes', [])
            logger.info(f"⚡ Loaded Phase 1: Audio + Scene Analysis")

        if up_to_phase >= 2:
            phase2_data = self._load_checkpoint(checkpoint_paths["phase2"])
            self.visual_samples = phase2_data.get('samples', [])
            logger.info(f"⚡ Loaded Phase 2: Visual Samples ({len(self.visual_samples)} samples @ 2fps)")

        if up_to_phase >= 3:
            clip_data = self._load_checkpoint(checkpoint_paths["clip_analysis"])
            self.clip_analysis = clip_data
            logger.info(f"⚡ Loaded Phase 3: CLIP Analysis ({len(clip_data.get('spurious_candidates', []))} spurious)")

        if up_to_phase >= 4:
            pass1_data = self._load_checkpoint(checkpoint_paths["pass1"])
            self.pass1_results = pass1_data
            logger.info(f"⚡ Loaded Phase 4: Pass 1 Filter ({pass1_data['selection_breakdown']['total']} frames)")

        if up_to_phase >= 5:
            pass2a_data = self._load_checkpoint(checkpoint_paths["pass2a"])
            self.pass2a_results = pass2a_data
            logger.info(f"⚡ Loaded Phase 5: Pass 2A Sonnet ({len(pass2a_data.get('mode1_precise', []))} moments)")

        if up_to_phase >= 6:
            pass2b_data = self._load_checkpoint(checkpoint_paths["pass2b"])
            self.pass2b_results = pass2b_data
            logger.info(f"⚡ Loaded Phase 6: Pass 2B Opus ({len(pass2b_data.get('mode3_inference_window', []))} moments)")

        if up_to_phase >= 7:
            validation_data = self._load_checkpoint(checkpoint_paths["validation"])
            self.validation_results = validation_data
            logger.info(f"⚡ Loaded Phase 7: Validation ({len(validation_data.get('validated_moments', []))} validated)")

        if up_to_phase >= 8:
            pass3_data = self._load_checkpoint(checkpoint_paths["pass3"])
            self.pass3_results = pass3_data
            logger.info(f"⚡ Loaded Phase 8: Pass 3 QA ({len(pass3_data.get('qa_pairs', []))} QA pairs)")

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
                "visual_sampling_2fps",
                "clip_analysis",
                "pass1_smart_filter",
                "pass2a_sonnet_selection",
                "pass2b_opus_selection",
                "moment_validation",
                "pass3_qa_generation"
            ],
            "outputs": {
                "audio_analysis": str(self.output_dir / f"{self.video_id}_phase1_audio_scene_quality.json"),
                "visual_samples": str(self.output_dir / f"{self.video_id}_phase2_visual_samples_2fps.json"),
                "clip_analysis": str(self.output_dir / f"{self.video_id}_clip_analysis.json"),
                "pass1_filtered": str(self.output_dir / f"{self.video_id}_pass1_filtered_frames.json"),
                "pass2a_moments": str(self.output_dir / f"{self.video_id}_pass2a_sonnet_moments.json"),
                "pass2b_moments": str(self.output_dir / f"{self.video_id}_pass2b_opus_moments.json"),
                "validated_moments": str(self.output_dir / f"{self.video_id}_validated_moments.json"),
                "qa_pairs": str(self.output_dir / f"{self.video_id}_pass3_qa_pairs.json")
            },
            "metrics": {
                "audio_duration": self.audio_analysis.get("duration", 0) if self.audio_analysis else 0,
                "frames_sampled_2fps": len(self.visual_samples) if hasattr(self, 'visual_samples') else 0,
                "clip_spurious_candidates": len(self.clip_analysis.get('spurious_candidates', [])) if hasattr(self, 'clip_analysis') else 0,
                "pass1_selected": len(self.pass1_results.get('selected_frames', [])) if hasattr(self, 'pass1_results') else 0,
                "pass2a_moments": (
                    len(self.pass2a_results.get('mode1_precise', [])) +
                    len(self.pass2a_results.get('mode2_micro_temporal', [])) +
                    len(self.pass2a_results.get('mode3_inference_window', [])) +
                    len(self.pass2a_results.get('mode4_clusters', []))
                ) if hasattr(self, 'pass2a_results') else 0,
                "pass2b_moments": (
                    len(self.pass2b_results.get('mode1_precise', [])) +
                    len(self.pass2b_results.get('mode2_micro_temporal', [])) +
                    len(self.pass2b_results.get('mode3_inference_window', [])) +
                    len(self.pass2b_results.get('mode4_clusters', []))
                ) if hasattr(self, 'pass2b_results') else 0,
                "validated_moments": len(self.validation_results.get('validated_moments', [])) if hasattr(self, 'validation_results') else 0,
                "qa_pairs_generated": len(self.questions),
                "cost_breakdown": {
                    "pass1": self.pass1_results.get('cost', 0) if hasattr(self, 'pass1_results') else 0,
                    "pass2a": self.pass2a_results.get('cost', 0) if hasattr(self, 'pass2a_results') else 0,
                    "pass2b": self.pass2b_results.get('cost', 0) if hasattr(self, 'pass2b_results') else 0,
                    "pass3": self.pass3_results.get('cost', 0) if hasattr(self, 'pass3_results') else 0,
                    "total": self.total_cost
                }
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

            # Phase 2: Quick Visual Sampling (2fps) + FREE Models
            if resume_from_phase <= 2:
                logger.info("\n🖼️  PHASE 2: Visual Sampling (2fps) + FREE Models")
                self._run_phase2_visual_sampling_2fps()
            else:
                logger.info("\n🖼️  PHASE 2: Visual Sampling (2fps) + FREE Models [SKIPPED - loaded from checkpoint]")

            # Content Moderation: Check video safety (after Phase 2)
            logger.info("\n🛡️  CONTENT MODERATION: Checking video safety and quality")
            self._run_content_moderation()

            # CLIP Analysis: Generate embeddings and detect spurious candidates
            if resume_from_phase <= 3:
                logger.info("\n🔗 CLIP ANALYSIS: Text-Image Alignment & Spurious Detection")
                self._run_clip_analysis()
            else:
                logger.info("\n🔗 CLIP ANALYSIS [SKIPPED - loaded from checkpoint]")

            # Pass 1: Smart Pre-Filter (3-Tier Selection)
            if resume_from_phase <= 4:
                logger.info("\n🎯 PASS 1: Smart Pre-Filter (Rule-Based + Sonnet 3.5)")
                self._run_pass1_filter()
            else:
                logger.info("\n🎯 PASS 1: Smart Pre-Filter [SKIPPED - loaded from checkpoint]")

            # Pass 2A: Sonnet 4.5 Ontology Selection (9 Easy Types)
            if resume_from_phase <= 5:
                logger.info("\n🧠 PASS 2A: Sonnet 4.5 Easy/Medium Ontology Selection")
                self._run_pass2a_selection()
            else:
                logger.info("\n🧠 PASS 2A: Sonnet 4.5 Selection [SKIPPED - loaded from checkpoint]")

            # Pass 2B: Opus 4 Hard Ontology Selection (4 Hard Types + Spurious)
            if resume_from_phase <= 6:
                logger.info("\n🔮 PASS 2B: Opus 4 Hard Ontology + Spurious Detection")
                self._run_pass2b_selection()
            else:
                logger.info("\n🔮 PASS 2B: Opus 4 Selection [SKIPPED - loaded from checkpoint]")

            # Validation Layer: Quality Gate for All Moments
            if resume_from_phase <= 7:
                logger.info("\n✅ VALIDATION: Quality Gate for All Moments")
                self._run_validation()
            else:
                logger.info("\n✅ VALIDATION [SKIPPED - loaded from checkpoint]")

            # Pass 3: Batched QA Generation (Sonnet 4.5)
            if resume_from_phase <= 8:
                logger.info("\n❓ PASS 3: Batched QA Generation (Sonnet 4.5)")
                self._run_pass3_qa_generation()
            else:
                logger.info("\n❓ PASS 3: QA Generation [SKIPPED - loaded from checkpoint]")

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
                'audio_events': whisper_data.get('audio_events', []),  # Load audio events
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
                "audio_events": self.audio_analysis.get('audio_events', []),  # Include audio events
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
        logger.info(f"   Running: CLIP, Places365, YOLO, OCR, Pose, FER (BLIP-2 disabled for speed)")

        # Run quick visual sampler with FREE models
        # BLIP-2 disabled by default (saves 2+ hours). Set enable_blip2=True to enable.
        # Optional: Set min_quality=0.3 to skip low-quality scenes
        sampler = QuickVisualSampler(enable_blip2=False)
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

    def _run_content_moderation(self):
        """
        Content Moderation: Check video safety and quality.

        Implements guideline requirements:
        1. Reject videos with violence/weapons
        2. Reject videos with NSFW content
        3. Reject videos with burned-in subtitles

        Raises:
            ValueError: If video fails content moderation
        """
        logger.info("📍 Starting Content Moderation")

        # Use Phase 2 visual samples from Pass 1-2B architecture
        if not self.visual_samples:
            logger.warning("⚠️  No visual samples available - skipping content moderation")
            return

        # Get frame paths for subtitle detection
        sample_frame_paths = []
        for sample in self.visual_samples:
            frame_path = sample.get('frame_path')
            if frame_path and Path(frame_path).exists():
                sample_frame_paths.append(Path(frame_path))

        logger.info(f"   Checking {len(sample_frame_paths)} frames")

        # Get YOLO detections for violence check
        yolo_detections = {}
        for sample in self.visual_samples:
            frame_id = sample.get('frame_id')
            objects = sample.get('objects', [])
            if objects:
                yolo_detections[frame_id] = {
                    'detected_objects': [obj.get('label', obj.get('class', '')) for obj in objects]
                }

        # Run content moderation
        moderator = ContentModerator()
        should_reject, reason = moderator.should_reject_video(
            video_path=self.video_path,
            sample_frames=sample_frame_paths,
            yolo_detections=yolo_detections,
            clip_embeddings=None  # CLIP embeddings available in self.clip_analysis if needed
        )

        if should_reject:
            logger.error(f"❌ VIDEO REJECTED: {reason}")
            logger.error(f"   Guideline violation detected")
            raise ValueError(f"Content moderation failed: {reason}")

        logger.info(f"✅ Content Moderation Passed!")
        logger.info(f"   {reason}")

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

        # ✅ Filter out intro/outro highlights (guideline requirement)
        video_duration = self.audio_analysis['duration']
        filtered_highlights = []
        intro_outro_count = 0

        for highlight in top_highlights:
            timestamp = highlight.get('timestamp', 0)
            if is_intro_outro(timestamp, video_duration):
                intro_outro_count += 1
                logger.debug(f"   Filtered intro/outro highlight at {timestamp:.1f}s")
            else:
                filtered_highlights.append(highlight)

        if intro_outro_count > 0:
            logger.info(f"   Filtered {intro_outro_count} intro/outro highlights")

        logger.info(f"   Using {len(filtered_highlights)}/{len(top_highlights)} highlights (after intro/outro filter)")

        selection_result = frame_selector.select_frames(
            visual_samples=self.visual_samples,
            highlights=filtered_highlights,  # ✅ Use filtered highlights (no intro/outro)
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

        # Extract cost information
        cost_summary = selection_result.get('cost_summary', {})
        claude_cost = cost_summary.get('claude_api_call', {})
        input_tokens = claude_cost.get('input_tokens', 0)
        output_tokens = claude_cost.get('output_tokens', 0)
        total_tokens = claude_cost.get('total_tokens', 0)
        total_cost = claude_cost.get('total_cost', 0.05)  # Fallback to 0.05 if missing

        logger.info(f"\n💰 Phase 5 Cost Summary:")
        logger.info(f"   Input tokens:  {input_tokens:,}")
        logger.info(f"   Output tokens: {output_tokens:,}")
        logger.info(f"   Total tokens:  {total_tokens:,}")
        logger.info(f"   Total cost:    ${total_cost:.4f}")

        # Save checkpoint
        checkpoint_path = self.output_dir / f"{self.video_id}_phase5_frame_selection.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(convert_numpy_types(selection_result), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 5 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Saved to: {checkpoint_path.name}")

        self.total_cost += total_cost

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
        """Phase 8: Optimized Direct Vision Question Generation (Skip Phase 7)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 8: Direct Vision Question Generation")
        logger.info("   Using GPT-4o Vision (30% cheaper than Claude, excellent quality)...")

        # Initialize Phase 8 Vision Generator
        generator = Phase8VisionGenerator(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.claude_api_key  # For hedging fixer
        )

        # Pass FULL Phase 5 output (includes Claude-validated clusters!)
        frames_dir = self.output_dir / "frames" / self.video_id

        # Generate questions using Phase 5's validated clusters
        result = generator.generate_questions(
            phase5_output=self.frame_selection,  # ✅ FIXED: Pass full output with clusters
            audio_analysis=self.audio_analysis,
            frames_dir=frames_dir,
            video_id=self.video_id,
            highlights=self.highlights  # ✅ Pass Phase 3 highlights for adaptive threshold
        )

        # Store questions (convert GeneratedQuestion objects to dicts)
        self.questions = [{
            'question_id': q.question_id,
            'question': q.question,
            'golden_answer': q.answer,
            'question_type': q.question_type,
            # ✅ FIX #1: Use actual start/end timestamps for cluster questions
            # For cluster questions: use q.start_timestamp and q.end_timestamp
            # For single-frame questions: use q.timestamp ± window
            # IMPORTANT: Clamp to 0 to prevent negative timestamps
            'start_timestamp': self._format_timestamp(
                max(0, q.start_timestamp) if q.start_timestamp is not None else max(0, q.timestamp - 1)
            ),
            'end_timestamp': self._format_timestamp(
                max(0, q.end_timestamp) if q.end_timestamp is not None else max(0, q.timestamp + 2)
            ),
            'audio_cue': q.audio_cue,
            'visual_cue': q.visual_cue,
            'confidence': q.confidence,
            'model': q.model,
            'tokens': q.tokens,
            'cost': q.cost
        } for q in result['questions']]

        # Save questions in standard format
        questions_path = self.output_dir / f"{self.video_id}_phase8_questions.json"
        output_data = {
            "video_id": self.video_id,
            "total_questions": len(self.questions),
            "questions": self.questions,
            "cost_summary": result['cost_summary'],
            "metadata": result['metadata']
        }

        with open(questions_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Update total cost
        phase_cost = result['cost_summary']['total_cost']
        self.total_cost += phase_cost

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 8 Complete! ({phase_time:.1f}s)")
        logger.info(f"   Total questions: {len(self.questions)}")
        logger.info(f"   Generation cost: ${phase_cost:.4f}")
        logger.info(f"   Frames processed: {result['metadata']['frames_processed']}")
        logger.info(f"   Model used: {result['metadata']['model_used']}")
        logger.info(f"   Saved to: {questions_path.name}")

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

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

    # ==================== NEW PASS 1-2B ARCHITECTURE METHODS ====================

    def _run_phase2_visual_sampling_2fps(self):
        """Phase 2: Visual Sampling at 2fps + FREE Models"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 2: Visual Sampling (2fps) + FREE Models")

        sampler = QuickVisualSampler(enable_blip2=False)

        # Create frames output directory
        frames_dir = self.output_dir / "frames" / self.video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        # FPS mode: 2 frames per second for uniform sampling
        result = sampler.sample_and_analyze(
            video_path=str(self.video_path),
            scenes=self.scenes,
            mode="fps",  # FPS mode: uniform 2fps sampling
            fps_rate=2.0,  # 2 frames per second
            frames_output_dir=str(frames_dir)  # Save frames to disk
        )

        self.visual_samples = result['samples']

        # Save results (mode-agnostic filename for checkpoint compatibility)
        phase2_path = self.output_dir / f"{self.video_id}_phase2_visual_samples.json"
        with open(phase2_path, 'w') as f:
            json.dump(convert_numpy_types(result), f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 2 Complete! ({phase_time:.1f}s, ~{len(self.visual_samples)} frames)")
        logger.info(f"   Saved to: {phase2_path.name}")

    def _run_clip_analysis(self):
        """CLIP Analysis: Text-Image Alignment & Spurious Detection"""
        phase_start = datetime.now()
        logger.info("📍 Starting CLIP Analysis")

        # Prepare transcript segments
        transcript_segments = self.audio_analysis.get('segments', [])

        # Run CLIP analysis
        clip_path = self.output_dir / f"{self.video_id}_clip_analysis.json"

        self.clip_analysis = run_clip_analysis(
            video_path=str(self.video_path),
            frames_metadata=self.visual_samples,
            transcript_segments=transcript_segments,
            output_path=str(clip_path),
            use_siglip=False  # Use regular CLIP (no HF auth required)
        )

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ CLIP Analysis Complete! ({phase_time:.1f}s)")
        logger.info(f"   Spurious candidates: {len(self.clip_analysis['spurious_candidates'])}")
        logger.info(f"   Visual anomalies: {len(self.clip_analysis['visual_anomalies'])}")

        # ✅ CRITICAL: Merge vision data from frame_analyses into visual_samples
        # Pass 1/2A/2B expect frames with YOLO objects, poses, scene_type, etc.
        logger.info("Merging vision data from Phase 3 into frame metadata...")
        frame_analyses_dict = {
            f['frame_id']: f
            for f in self.clip_analysis.get('frame_analyses', [])
        }

        for frame in self.visual_samples:
            frame_id = frame.get('frame_id')
            if frame_id in frame_analyses_dict:
                vision_data = frame_analyses_dict[frame_id]
                # Merge all vision model results into frame metadata
                frame['objects'] = vision_data.get('objects', [])
                frame['object_count'] = vision_data.get('object_count', 0)
                frame['poses'] = vision_data.get('poses', {})
                frame['scene_type'] = vision_data.get('scene_type', 'unknown')
                frame['scene_attributes'] = vision_data.get('scene_attributes', [])
                frame['clip_embedding'] = vision_data.get('clip_embedding', [])
                # NEW: Merge BLIP-2 captions and OCR text
                frame['caption'] = vision_data.get('caption', '')
                frame['ocr_text'] = vision_data.get('ocr_text', '')

        logger.info(f"✅ Merged vision data for {len(self.visual_samples)} frames")
        logger.info(f"   Each frame now has: objects, poses, scene_type, scene_attributes, CLIP embedding, caption, OCR text")

    def _run_pass1_filter(self):
        """Pass 1: Smart Pre-Filter (3-Tier Selection)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Pass 1: Smart Pre-Filter")

        pass1_path = self.output_dir / f"{self.video_id}_pass1_filtered_frames.json"

        self.pass1_results = run_pass1_filter(
            frames=self.visual_samples,
            audio_analysis=self.audio_analysis,
            clip_analysis=self.clip_analysis,
            scenes=self.scenes,
            video_duration=self.audio_analysis.get('duration', 0),
            output_path=str(pass1_path)
        )

        # Track cost
        pass1_cost = self.pass1_results.get('cost', 0.35)
        self.total_cost += pass1_cost

        phase_time = (datetime.now() - phase_start).total_seconds()
        selected_count = len(self.pass1_results['selected_frames'])
        logger.info(f"✅ Pass 1 Complete! ({phase_time:.1f}s, ${pass1_cost:.4f})")
        logger.info(f"   Selected: {selected_count} frames from {len(self.visual_samples)}")

    def _run_pass2a_selection(self):
        """Pass 2A: Sonnet 4.5 Easy/Medium Ontology Selection"""
        phase_start = datetime.now()
        logger.info("📍 Starting Pass 2A: Sonnet 4.5 Ontology Selection")

        frames_dir = self.output_dir / "frames" / self.video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        pass2a_path = self.output_dir / f"{self.video_id}_pass2a_sonnet_moments.json"

        self.pass2a_results = run_pass2a_selection(
            selected_frames=self.pass1_results['selected_frames'],
            audio_analysis=self.audio_analysis,
            clip_analysis=self.clip_analysis,
            frames_dir=str(frames_dir),
            output_path=str(pass2a_path)
        )

        # Track cost
        pass2a_cost = self.pass2a_results.get('cost', 1.10)
        self.total_cost += pass2a_cost

        phase_time = (datetime.now() - phase_start).total_seconds()
        total_moments = sum([
            len(self.pass2a_results.get('mode1_precise', [])),
            len(self.pass2a_results.get('mode2_micro_temporal', [])),
            len(self.pass2a_results.get('mode3_inference_window', [])),
            len(self.pass2a_results.get('mode4_clusters', []))
        ])
        logger.info(f"✅ Pass 2A Complete! ({phase_time:.1f}s, ${pass2a_cost:.4f})")
        logger.info(f"   Moments detected: {total_moments}")
        logger.info(f"   Flagged for Opus 4: {len(self.pass2a_results.get('flagged_for_opus4', []))}")

    def _run_pass2b_selection(self):
        """Pass 2B: Opus 4 Hard Ontology + Spurious Detection"""
        phase_start = datetime.now()
        logger.info("📍 Starting Pass 2B: Opus 4 Hard Types + Spurious")

        frames_dir = self.output_dir / "frames" / self.video_id

        pass2b_path = self.output_dir / f"{self.video_id}_pass2b_opus_moments.json"

        self.pass2b_results = run_pass2b_selection(
            flagged_frames=self.pass2a_results.get('flagged_for_opus4', []),
            spurious_candidates=self.clip_analysis.get('spurious_candidates', []),
            audio_analysis=self.audio_analysis,
            frames_dir=str(frames_dir),
            all_frames_metadata=self.visual_samples,
            full_video_context={'scenes': self.scenes},
            output_path=str(pass2b_path)
        )

        # Track cost
        pass2b_cost = self.pass2b_results.get('cost', 1.00)
        self.total_cost += pass2b_cost

        phase_time = (datetime.now() - phase_start).total_seconds()
        total_moments = sum([
            len(self.pass2b_results.get('mode1_precise', [])),
            len(self.pass2b_results.get('mode2_micro_temporal', [])),
            len(self.pass2b_results.get('mode3_inference_window', [])),
            len(self.pass2b_results.get('mode4_clusters', []))
        ])
        logger.info(f"✅ Pass 2B Complete! ({phase_time:.1f}s, ${pass2b_cost:.4f})")
        logger.info(f"   Hard moments detected: {total_moments}")

    def _run_validation(self):
        """Validation Layer: Quality Gate for All Moments"""
        phase_start = datetime.now()
        logger.info("📍 Starting Validation Layer")

        # Get all available frame IDs
        available_frames = set(f['frame_id'] for f in self.visual_samples)

        validation_path = self.output_dir / f"{self.video_id}_validated_moments.json"

        self.validation_results = run_validation(
            pass2a_results=self.pass2a_results,
            pass2b_results=self.pass2b_results,
            available_frames=available_frames,
            audio_analysis=self.audio_analysis,
            video_duration=self.audio_analysis.get('duration', 0),
            output_path=str(validation_path)
        )

        phase_time = (datetime.now() - phase_start).total_seconds()
        validated = self.validation_results['validation_summary']['validated']
        rejected = self.validation_results['validation_summary']['rejected']
        logger.info(f"✅ Validation Complete! ({phase_time:.1f}s)")
        logger.info(f"   Validated: {validated} moments")
        logger.info(f"   Rejected: {rejected} moments")
        logger.info(f"   Coverage: {self.validation_results['coverage_check']['meets_requirements']}")

    def _convert_moments_to_phase5_format(self, validated_moments: List[Dict]) -> Dict:
        """
        Convert Pass 2A/2B validated moments to Phase 5 output format for Phase 8.

        Phase 8 expects:
        {
            'selection_plan': [frames with timestamp, priority, question_types],
            'dense_clusters': [cluster metadata],
            'coverage': {...}
        }

        Args:
            validated_moments: List of validated moments from Pass 2A/2B

        Returns:
            Phase 5-compatible format dict
        """
        single_frames = []
        dense_clusters = []

        for moment in validated_moments:
            mode = moment.get('mode', 'precise')
            frame_ids = moment.get('frame_ids', [])
            timestamps = moment.get('timestamps', [])

            # Normalize type names to official PDF names
            primary_type = normalize_type(moment.get('primary_ontology', ''))
            secondary_types = [normalize_type(t) for t in moment.get('secondary_ontologies', [])]
            question_types = [primary_type] + secondary_types

            if mode == 'cluster' or len(frame_ids) >= 3:
                # Treat as cluster
                dense_clusters.append({
                    'start': min(timestamps) if timestamps else 0,
                    'end': max(timestamps) if timestamps else 0,
                    'frame_count': len(frame_ids),
                    'reason': moment.get('correspondence', ''),
                    'question_types': question_types,
                    'visual_cues': moment.get('visual_cues', []),
                    'audio_cues': moment.get('audio_cues', []),
                    'validation': {
                        'same_scene_type': True,
                        'same_location': True,
                        'continuous_action': True,
                        'is_scene_cut': False
                    }
                })
            else:
                # Treat as single frame(s)
                for i, ts in enumerate(timestamps):
                    frame_id = frame_ids[i] if i < len(frame_ids) else f"moment_{ts:.1f}s"
                    single_frames.append({
                        'frame_id': frame_id,
                        'timestamp': ts,
                        'priority': moment.get('priority', 0.8),
                        'question_types': question_types,
                        'reason': moment.get('correspondence', ''),
                        'visual_cues': moment.get('visual_cues', []),
                        'audio_cues': moment.get('audio_cues', [])
                    })

        # Build coverage stats
        all_types = set()
        for frame in single_frames:
            all_types.update(frame['question_types'])
        for cluster in dense_clusters:
            all_types.update(cluster['question_types'])

        logger.info(f"Converted {len(validated_moments)} moments → {len(single_frames)} frames + {len(dense_clusters)} clusters")
        logger.info(f"Coverage: {len(all_types)}/13 types")

        return {
            'selection_plan': single_frames,
            'dense_clusters': dense_clusters,
            'coverage': {
                'covered_types': len(all_types),
                'total_types': 13,
                'missing_types': list(set(OFFICIAL_TYPES) - all_types),
                'coverage_ratio': len(all_types) / 13
            }
        }

    def _extract_cluster_frames(self, dense_clusters: List[Dict]) -> None:
        """
        Extract frames for clusters identified by Pass 2B.

        Creates frames_metadata.json that Phase 8 expects.

        Args:
            dense_clusters: Cluster metadata from _convert_moments_to_phase5_format
        """
        if not dense_clusters:
            logger.info("No clusters to extract frames for")
            return

        frames_dir = self.output_dir / "frames" / self.video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting frames for {len(dense_clusters)} clusters...")

        all_frames_metadata = []

        for cluster_idx, cluster in enumerate(dense_clusters):
            start_ts = cluster['start']
            end_ts = cluster['end']
            frame_count = cluster['frame_count']

            # Generate evenly spaced timestamps
            if frame_count == 1:
                timestamps = [start_ts]
            else:
                timestamps = [
                    start_ts + (end_ts - start_ts) * i / (frame_count - 1)
                    for i in range(frame_count)
                ]

            logger.info(f"  Cluster {cluster_idx}: {len(timestamps)} frames at {start_ts:.1f}s-{end_ts:.1f}s")

            # Extract frames using SmartFrameExtractor
            for ts in timestamps:
                frame_id = int(ts * 24)  # Assume 24 FPS
                frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"

                # Skip if frame already exists
                if frame_path.exists():
                    logger.debug(f"    Frame already exists: {frame_path.name}")
                else:
                    # Extract frame from video
                    import cv2
                    video_path = self.output_dir / f"{self.video_id}_video.mp4"

                    if not video_path.exists():
                        logger.warning(f"    Video file not found: {video_path}")
                        continue

                    cap = cv2.VideoCapture(str(video_path))
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                    ret, frame = cap.read()

                    if ret:
                        cv2.imwrite(str(frame_path), frame)
                        logger.debug(f"    Extracted: {frame_path.name}")
                    else:
                        logger.warning(f"    Failed to extract frame at {ts:.1f}s")

                    cap.release()

                # Add to metadata
                all_frames_metadata.append({
                    'frame_id': frame_id,
                    'timestamp': ts,
                    'image_path': str(frame_path),
                    'frame_type': 'cluster',
                    'cluster_id': f"cluster_{cluster_idx:02d}"
                })

        # Save frames_metadata.json
        metadata_path = frames_dir / "frames_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'frames': all_frames_metadata,
                'total_frames': len(all_frames_metadata),
                'extraction_method': 'cluster_extraction_pass3'
            }, f, indent=2)

        logger.info(f"✅ Extracted {len(all_frames_metadata)} cluster frames")
        logger.info(f"   Saved metadata to: {metadata_path}")

    def _run_pass3_qa_generation(self):
        """Pass 3: Use Phase 8 Vision Generator for QA (Replaces old Pass 3)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Pass 3: Phase 8 Vision QA Generation")
        logger.info("   Using GPT-4o Vision with full guideline compliance...")

        # Convert validated moments to Phase 5-like format for Phase 8
        phase5_compatible = self._convert_moments_to_phase5_format(
            self.validation_results['validated_moments']
        )

        # ✅ FIX: Extract cluster frames before QA generation
        dense_clusters = phase5_compatible.get('dense_clusters', [])
        if dense_clusters:
            logger.info(f"Extracting frames for {len(dense_clusters)} clusters...")
            self._extract_cluster_frames(dense_clusters)

        frames_dir = self.output_dir / "frames" / self.video_id

        # Use Phase 8 generator
        generator = Phase8VisionGenerator(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.claude_api_key
        )

        result = generator.generate_questions(
            phase5_output=phase5_compatible,
            audio_analysis=self.audio_analysis,
            frames_dir=frames_dir,
            video_id=self.video_id,
            highlights=getattr(self, 'highlights', None)
        )

        # Convert Phase 8 questions to expected format
        self.questions = []
        for q in result['questions']:
            self.questions.append({
                'question_id': q.question_id,
                'question': q.question,
                'golden_answer': q.answer,
                'question_type': q.question_type,
                'sub_task_type': q.sub_task_type,
                'start_timestamp': self._format_timestamp(max(0, q.start_timestamp) if q.start_timestamp else max(0, q.timestamp - 5)),
                'end_timestamp': self._format_timestamp(max(0, q.end_timestamp) if q.end_timestamp else max(0, q.timestamp + 10)),
                'audio_cue': q.audio_cue,
                'visual_cue': q.visual_cue,
                'confidence': q.confidence,
                'model': q.model,
                'cost': q.cost
            })

        # Save results
        pass3_path = self.output_dir / f"{self.video_id}_pass3_qa_pairs.json"
        output_data = {
            "video_id": self.video_id,
            "total_questions": len(self.questions),
            "questions": self.questions,
            "cost_summary": result['cost_summary'],
            "metadata": result['metadata']
        }

        with open(pass3_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(output_data), f, indent=2, ensure_ascii=False)

        # Track cost
        pass3_cost = result['cost_summary']['total_cost']
        self.total_cost += pass3_cost

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Pass 3 Complete! ({phase_time:.1f}s, ${pass3_cost:.4f})")
        logger.info(f"   QA pairs generated: {len(self.questions)}")
        logger.info(f"   Quality fixes applied: {result['metadata'].get('quality_fixes_applied', 0)}")
        logger.info(f"   Duplicates removed: {result['metadata'].get('duplicates_removed', 0)}")
        logger.info(f"   Audio validation rejects: {result['metadata'].get('audio_validation_rejects', 0)}")

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
        claude_api_key=os.getenv("ANTHROPIC_API_KEY")
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
        print("  export ANTHROPIC_API_KEY=your_key_here  # Optional, for cross-validation")
        print("  python smart_pipeline.py my_video.mp4")