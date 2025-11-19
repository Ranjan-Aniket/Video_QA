"""
Adversarial Smart Pipeline - Complete Video Q&A Generation System

NEW ARCHITECTURE (with Dense Sampling & Enhanced Evidence):
1. Audio Analysis (Whisper + word timestamps)
2. Two-Tier Opportunity Mining (strict for 7 premium, heuristic for 40 template)
3. Dense Frame Extraction:
   - Premium: 7 opportunities × 10 frames = 70 frames (0.5s intervals, ±2.5s window)
   - Template: 40 opportunities × 1 frame = 40 frames
   - Total: 110 frames
4. Hybrid Evidence Extraction:
   - BLIP-2 on ALL 110 frames (FREE, ~2s per frame)
   - YOLO + OCR + Pose on ALL 110 frames (FREE)
   - GPT-4o + Claude on 47 KEY frames:
     * 7 premium center frames (center of each 10-frame cluster)
     * 40 template frames (single frame per opportunity)
   - Cost: 47 × $0.02 = $0.94 (well within $3.36 budget)
5. Question Generation (30 questions):
   - 3 GPT-4V questions (premium agreed)
   - 7 Claude questions (premium disagreed)
   - 20 template questions (best from 40 generated)
6. Gemini Testing (test questions against Gemini)

Key Features:
- Dense temporal evidence for action counting (Type B questions)
- Rich AI descriptions for all key frames
- action_detections timeline from body poses
- event_timeline from all multimodal sources
- Supports complex temporal questions ("how many times did X do Y")

Processing Time: 8-10 minutes per video
Cost: ~$1.10 per video (33% of budget)

Replaces old sparse sampling with dense evidence for superior question quality.
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
from processing.opportunity_detector_v2 import OpportunityDetectorV2
from processing.smart_frame_extractor import SmartFrameExtractorEnhanced as SmartFrameExtractor, ExtractedFrame
from processing.multimodal_question_generator_v2 import MultimodalQuestionGeneratorV2
from processing.bulk_frame_analyzer import BulkFrameAnalyzer

logger = logging.getLogger(__name__)


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
        """Get all checkpoint file paths"""
        return {
            "phase1": self.output_dir / f"{self.video_id}_audio_analysis.json",
            "phase2": self.output_dir / f"{self.video_id}_opportunities.json",
            "phase3": self.output_dir / "frames" / self.video_id / "frames_metadata.json",
            "phase4": self.output_dir / f"{self.video_id}_evidence.json",
            "phase5": self.output_dir / f"{self.video_id}_questions.json",
            "phase6": self.output_dir / f"{self.video_id}_gemini_results.json"
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
            Phase number to start from (1-6, or 1 if no checkpoints)
        """
        if not self.enable_checkpoints:
            logger.info("📍 Checkpoints disabled - starting from Phase 1")
            return 1

        logger.info("\n" + "=" * 80)
        logger.info("SCANNING FOR CHECKPOINTS")
        logger.info("=" * 80)

        checkpoint_paths = self._get_checkpoint_paths()

        # Define required fields for each checkpoint
        required_fields = {
            "phase1": ["duration", "segments", "transcript"],
            "phase2": ["total_opportunities", "opportunities"],
            "phase3": ["frames"],
            "phase4": ["video_id", "frames"],
            "phase5": ["total_questions", "questions"],
            "phase6": ["tested"]
        }

        last_valid_phase = 0

        # Check each phase in order
        for phase_num in range(1, 7):
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
        elif last_valid_phase == 6:
            logger.info("→ All phases complete!")
        else:
            logger.info(f"→ Resuming from Phase {resume_phase}")
            logger.info(f"→ Loading Phases 1-{last_valid_phase} from checkpoints")

        logger.info("=" * 80 + "\n")

        return resume_phase

    def _load_phase_checkpoints(self, up_to_phase: int):
        """Load checkpoint data for phases 1 through up_to_phase"""
        checkpoint_paths = self._get_checkpoint_paths()

        if up_to_phase >= 1:
            self.audio_analysis = self._load_checkpoint(checkpoint_paths["phase1"])
            logger.info(f"⚡ Loaded Phase 1: Audio Analysis")

        if up_to_phase >= 2:
            self.opportunities = self._load_checkpoint(checkpoint_paths["phase2"])
            logger.info(f"⚡ Loaded Phase 2: Opportunities")

        if up_to_phase >= 3:
            frames_metadata = self._load_checkpoint(checkpoint_paths["phase3"])
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
            logger.info(f"⚡ Loaded Phase 3: Frames ({len(self.extracted_frames)} frames)")

        if up_to_phase >= 4:
            self.evidence = self._load_checkpoint(checkpoint_paths["phase4"])
            logger.info(f"⚡ Loaded Phase 4: Evidence")

        if up_to_phase >= 5:
            questions_data = self._load_checkpoint(checkpoint_paths["phase5"])
            self.questions = questions_data.get("questions", [])
            logger.info(f"⚡ Loaded Phase 5: Questions ({len(self.questions)} questions)")

        if up_to_phase >= 6:
            self.gemini_results = self._load_checkpoint(checkpoint_paths["phase6"])
            logger.info(f"⚡ Loaded Phase 6: Gemini Results")

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
            json.dump(results, f, indent=2)

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
            if resume_from_phase > 6:
                logger.info("✅ All phases already complete - nothing to run!")
                # Still compile and return results
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                return self._compile_results(duration)

            # Phase 1: Audio Analysis
            if resume_from_phase <= 1:
                logger.info("\n🎵 PHASE 1: Audio Analysis")
                self._run_phase1_audio()
            else:
                logger.info("\n🎵 PHASE 1: Audio Analysis [SKIPPED - loaded from checkpoint]")

            # Phase 2: Adversarial Opportunity Mining
            if resume_from_phase <= 2:
                logger.info("\n🎯 PHASE 2: Adversarial Opportunity Mining")
                self._run_phase2_opportunities()
            else:
                logger.info("\n🎯 PHASE 2: Adversarial Opportunity Mining [SKIPPED - loaded from checkpoint]")

            # Phase 3: Smart Frame Extraction
            if resume_from_phase <= 3:
                logger.info("\n📸 PHASE 3: Smart Frame Extraction")
                self._run_phase3_frames()
            else:
                logger.info("\n📸 PHASE 3: Smart Frame Extraction [SKIPPED - loaded from checkpoint]")

            # Phase 4: Hybrid Evidence Extraction
            if resume_from_phase <= 4:
                logger.info("\n🔍 PHASE 4: Hybrid Evidence Extraction")
                self._run_phase4_evidence()
            else:
                logger.info("\n🔍 PHASE 4: Hybrid Evidence Extraction [SKIPPED - loaded from checkpoint]")

            # Phase 5: Adversarial Question Generation
            if resume_from_phase <= 5:
                logger.info("\n❓ PHASE 5: Adversarial Question Generation")
                self._run_phase5_questions()
            else:
                logger.info("\n❓ PHASE 5: Adversarial Question Generation [SKIPPED - loaded from checkpoint]")

            # Phase 6: Gemini Testing (optional)
            if self.gemini_api_key and resume_from_phase <= 6:
                logger.info("\n🧪 PHASE 6: Gemini Testing")
                self._run_phase6_gemini()
            elif resume_from_phase > 6:
                logger.info("\n🧪 PHASE 6: Gemini Testing [SKIPPED - loaded from checkpoint]")

            # Calculate final metrics and compile results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return self._compile_results(duration)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _run_phase1_audio(self):
        """Phase 1: Audio Analysis with word timestamps"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 1: Audio Analysis")
        logger.info(f"   Video: {self.video_path.name}")

        analyzer = AudioAnalyzer(str(self.video_path))
        self.audio_analysis = analyzer.analyze(save_json=True)

        # Save to output dir
        audio_path = self.output_dir / f"{self.video_id}_audio_analysis.json"
        with open(audio_path, 'w') as f:
            json.dump(self.audio_analysis, f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 1 Complete!")
        logger.info(f"   Duration: {self.audio_analysis['duration']:.1f}s audio")
        logger.info(f"   Segments: {len(self.audio_analysis['segments'])}")
        logger.info(f"   Processing time: {phase_time:.1f}s")
        logger.info(f"   Saved to: {audio_path.name}")

        self.total_cost += 0.006  # Whisper API cost estimate

    def _run_phase2_opportunities(self):
        """Phase 2: Opportunity Detection V2 (Real Quote Extraction)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 2: Opportunity Detection V2")
        logger.info(f"   Transcript length: {len(self.audio_analysis['transcript'])} chars")
        logger.info(f"   Extracting REAL quotes from transcript...")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for opportunity detection")

        detector = OpportunityDetectorV2(
            openai_api_key=self.openai_api_key,
            enable_stage2_validation=True  # Enable filtered GPT-4 validation (only high-quality candidates)
        )
        opportunities = detector.detect_opportunities(
            self.audio_analysis,
            video_id=self.video_id
        )

        # Save opportunities
        opps_path = self.output_dir / f"{self.video_id}_opportunities.json"
        detector.save_opportunities(opportunities, opps_path)

        # Store as dict
        self.opportunities = opportunities.to_dict()

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 2 Complete!")
        logger.info(f"   Total opportunities: {self.opportunities['total_opportunities']}")
        logger.info(f"   Validated opportunities: {self.opportunities['validated_opportunities']}")
        logger.info(f"   Stage 1 candidates: {self.opportunities.get('stage1_candidates', 0)}")
        logger.info(f"   Stage 2 validated: {self.opportunities.get('stage2_validated', 0)}")
        logger.info(f"   Premium frames: {len(self.opportunities.get('premium_frames', []))}")
        logger.info(f"   Processing time: {phase_time:.1f}s")
        logger.info(f"   GPT-4 cost: ${opportunities.detection_cost:.4f}")

        # Log breakdown by opportunity type
        if 'opportunity_statistics' in self.opportunities:
            logger.info(f"\n   Opportunities by type:")
            for opp_type, count in sorted(self.opportunities['opportunity_statistics'].items()):
                logger.info(f"     - {opp_type}: {count}")

        self.total_cost += opportunities.detection_cost

    def _run_phase3_frames(self):
        """
        Phase 3: Smart Frame Extraction with Dense Sampling
        
        FRAME EXTRACTION STRATEGY:
        - Premium opportunities (7): Extract 10 dense frames each (0.5s intervals, ±2.5s window)
          * Total: 70 frames
          * Center frame marked as "key_frame" for AI analysis
        - Template opportunities (40): Extract 1 frame each
          * Total: 40 frames
          * All marked as "key_frame" for AI analysis
        - Grand total: 110 frames
        
        KEY FRAMES (47 total):
        - 7 premium center frames (GPT-4o + Claude)
        - 40 template frames (GPT-4o + Claude)
        
        NOTE: SmartFrameExtractor must support:
        1. dense_sampling=True for premium opportunities
        2. is_key_frame flag on ExtractedFrame objects
        3. Frame clustering to group dense frames
        """
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 3: Smart Frame Extraction with Dense Sampling")
        logger.info(f"   Total opportunities: {self.opportunities.get('total_opportunities', 0)}")
        logger.info(f"   Premium frames: {len(self.opportunities.get('premium_frames', []))} × 10 dense = ~70 frames")
        logger.info(f"   Template frames: Targeting 40 opportunities × 1 = 40 frames")
        logger.info(f"   Total frames: ~110 frames")

        extractor = SmartFrameExtractor(
            str(self.video_path),
            output_dir=str(self.output_dir / "frames" / self.video_id)
        )

        # Extract frames from opportunities
        opps_path = self.output_dir / f"{self.video_id}_opportunities.json"
        self.extracted_frames = extractor.extract_from_opportunities(str(opps_path))

        # Save metadata
        extractor.save_frame_metadata(self.extracted_frames)

        phase_time = (datetime.now() - phase_start).total_seconds()
        premium_count = len([f for f in self.extracted_frames if f.frame_type == 'premium'])
        template_count = len([f for f in self.extracted_frames if f.frame_type == 'template'])
        bulk_count = len([f for f in self.extracted_frames if f.frame_type == 'bulk'])

        logger.info(f"✅ Phase 3 Complete!")
        logger.info(f"   Total frames: {len(self.extracted_frames)}")
        logger.info(f"   Premium (GPT-4V/Claude): {premium_count}")
        logger.info(f"   Template (opportunities): {template_count}")
        logger.info(f"   Bulk (every 5s): {bulk_count}")
        logger.info(f"   Processing time: {phase_time:.1f}s")

        # No cost for frame extraction (OpenCV is free)

    def _run_phase4_evidence(self):
        """
        Phase 4: Hybrid Evidence Extraction with Multi-AI Consensus

        EVIDENCE EXTRACTION STRATEGY (OPTIMIZED):
        1. Template frames (40): BLIP-2 + YOLO + OCR + Pose
           - Template questions need BLIP-2 for compatibility scoring
        2. Premium frames (70): YOLO + OCR + Pose only (no BLIP-2)
           - Dense frames have pose detection for actions
           - 7 center frames get GPT-4o + Claude (richer than BLIP-2)
           - Saves 2.3 minutes per video

        3. AI Analysis on 47 KEY frames only:
           - GPT-4o Vision (~$0.01/frame)
           - Claude Sonnet 4.5 (~$0.01/frame)
           - Consensus engine to merge results

        KEY FRAMES:
        - 7 premium center frames (from 70-frame dense clusters)
        - 40 template frames (single frame per opportunity)
        - Cost: 47 × $0.02 = $0.94

        TEMPORAL EVIDENCE:
        - Dense frames enable action counting (Type B questions)
        - Build action_detections timeline from body poses
        - Build event_timeline from all sources
        """
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 4: Hybrid Evidence Extraction (BLIP-2 on template frames only)")
        logger.info(f"   Total frames: {len(self.extracted_frames)}")

        # Count frame types
        premium_frames = [f for f in self.extracted_frames if f.frame_type == "premium"]
        template_frames = [f for f in self.extracted_frames if f.frame_type == "template"]
        other_frames = [f for f in self.extracted_frames if f.frame_type not in ["premium", "template"]]

        # Count key frames (frames that will get AI analysis)
        key_frames = [f for f in self.extracted_frames if getattr(f, 'is_key_frame', f.frame_type == "template")]

        logger.info(f"   Premium frames (dense): {len(premium_frames)}")
        logger.info(f"   Template frames: {len(template_frames)}")
        logger.info(f"   Other frames: {len(other_frames)}")
        logger.info(f"   Key frames (for AI analysis): {len(key_frames)}")

        # Create evidence structure
        self.evidence = {
            "video_id": self.video_id,
            "frames": {}
        }

        bulk_evidence = {}

        # Step 1: Process TEMPLATE frames with BLIP-2 (for template question generation)
        if len(template_frames) > 0:
            logger.info(f"\n   [1/2] Analyzing {len(template_frames)} TEMPLATE frames with BLIP-2 + YOLO + OCR + Pose...")
            logger.info(f"   (Template frames need BLIP-2 for compatibility scoring)")

            template_analyzer = BulkFrameAnalyzer(
                enable_yolo=True,
                enable_ocr=True,
                enable_scene=True,
                enable_pose=True,
                enable_blip2=True,  # ENABLED for template frames
                yolo_model="yolov8n"
            )

            for frame in tqdm(
                template_frames,
                desc="   Template frames (BLIP-2+YOLO+OCR+Pose)",
                disable=not self.show_progress,
                unit="frame"
            ):
                result = template_analyzer.analyze_frame(frame)
                bulk_evidence[frame.frame_id] = result.to_dict()
                self.evidence["frames"][frame.frame_id] = result.to_dict()

            template_stats = template_analyzer.get_statistics()
            logger.info(f"   ✓ Template frames: {template_stats['frames_processed']} processed, {template_stats['frames_processed']} BLIP-2 captions")

        # Step 2: Process PREMIUM frames WITHOUT BLIP-2 (faster, use pose for actions)
        if len(premium_frames) > 0:
            logger.info(f"\n   [2/2] Analyzing {len(premium_frames)} PREMIUM frames with YOLO + OCR + Pose (no BLIP-2)...")
            logger.info(f"   (Dense frames use pose detection for actions, 7 centers get GPT-4o+Claude)")

            premium_analyzer = BulkFrameAnalyzer(
                enable_yolo=True,
                enable_ocr=True,
                enable_scene=True,
                enable_pose=True,
                enable_blip2=False,  # DISABLED for speed
                yolo_model="yolov8n"
            )

            for frame in tqdm(
                premium_frames,
                desc="   Premium frames (YOLO+OCR+Pose)",
                disable=not self.show_progress,
                unit="frame"
            ):
                result = premium_analyzer.analyze_frame(frame)
                bulk_evidence[frame.frame_id] = result.to_dict()
                self.evidence["frames"][frame.frame_id] = result.to_dict()

            premium_stats = premium_analyzer.get_statistics()
            logger.info(f"   ✓ Premium frames: {premium_stats['frames_processed']} processed (saved ~{len(premium_frames) * 2:.0f}s by skipping BLIP-2)")

        # Process other frames (if any)
        if len(other_frames) > 0:
            logger.info(f"\n   Processing {len(other_frames)} other frames...")
            other_analyzer = BulkFrameAnalyzer(
                enable_yolo=True,
                enable_ocr=True,
                enable_scene=True,
                enable_pose=True,
                enable_blip2=False,
                yolo_model="yolov8n"
            )

            for frame in other_frames:
                result = other_analyzer.analyze_frame(frame)
                bulk_evidence[frame.frame_id] = result.to_dict()
                self.evidence["frames"][frame.frame_id] = result.to_dict()

        # Log combined statistics
        logger.info(f"\n   Bulk Analysis Statistics:")
        logger.info(f"     Total frames processed: {len(self.extracted_frames)}")
        logger.info(f"     BLIP-2 captions: {len(template_frames)} (template only)")
        logger.info(f"     Time saved: ~{len(premium_frames) * 2:.0f}s by skipping BLIP-2 on premium frames")

        # Process KEY frames with GPT-4V + Claude + Consensus
        gpt4v_cost = 0.0
        claude_cost = 0.0

        if len(key_frames) > 0:
            logger.info(f"\n   Processing {len(key_frames)} key frames with Multi-AI Consensus (GPT-4o + Claude)...")

            for frame in tqdm(
                key_frames,
                desc="   Analyzing key frames (GPT-4o+Claude)",
                disable=not self.show_progress,
                unit="frame"
            ):
                # Get ground truth from bulk analyzer (already processed)
                ground_truth = bulk_evidence.get(frame.frame_id)
                if not ground_truth:
                    logger.warning(f"No bulk evidence found for key frame {frame.frame_id}")
                    continue

                # GPT-4 Vision analysis
                gpt4_evidence = self._analyze_frame_with_gpt4v(frame)
                gpt4v_cost += 0.01

                # Claude Sonnet 4.5 analysis (if available)
                claude_evidence = None
                if self.claude_api_key:
                    claude_evidence = self._analyze_frame_with_claude(frame)
                    claude_cost += 0.01

                # Run consensus
                if claude_evidence:
                    consensus_result = self._run_consensus(
                        gpt4_evidence=gpt4_evidence,
                        claude_evidence=claude_evidence,
                        ground_truth=ground_truth
                    )

                    # Store consensus result
                    self.evidence["frames"][frame.frame_id] = consensus_result
                else:
                    # No Claude available, merge GPT-4V with ground truth
                    merged_evidence = ground_truth.copy()
                    merged_evidence["ground_truth"]["gpt4v_description"] = gpt4_evidence["ground_truth"].get("gpt4v_description", "")
                    self.evidence["frames"][frame.frame_id] = merged_evidence

        # Save evidence
        evidence_path = self.output_dir / f"{self.video_id}_evidence.json"
        with open(evidence_path, 'w') as f:
            json.dump(self.evidence, f, indent=2)

        phase_time = (datetime.now() - phase_start).total_seconds()

        # Calculate consensus statistics
        consensus_stats = self._calculate_consensus_stats()

        logger.info(f"✅ Phase 4 Complete!")
        logger.info(f"   Total frames analyzed: {len(self.extracted_frames)}")
        logger.info(f"   - Dense premium frames: {len(premium_frames)}")
        logger.info(f"   - Template frames: {len(template_frames)}")
        logger.info(f"   - Other frames: {len(other_frames)}")
        logger.info(f"   Key frames with AI analysis: {len(key_frames)}")
        if self.claude_api_key and len(key_frames) > 0:
            logger.info(f"   Consensus reached: {consensus_stats['consensus_reached']}/{len(key_frames)}")
            logger.info(f"   Needs human review: {consensus_stats['needs_review']}/{len(key_frames)}")
            logger.info(f"   High priority reviews: {consensus_stats['high_priority']}")
        logger.info(f"   Processing time: {phase_time:.1f}s")
        logger.info(f"   Cost breakdown:")
        logger.info(f"     - BLIP-2 ({len(template_frames)} template frames only): $0.00 (FREE)")
        logger.info(f"     - YOLO+OCR+Pose ({len(self.extracted_frames)} all frames): $0.00 (FREE)")
        logger.info(f"     - GPT-4o ({len(key_frames)} key frames): ${gpt4v_cost:.4f}")
        logger.info(f"     - Claude ({len(key_frames)} key frames): ${claude_cost:.4f}")
        logger.info(f"   Total Phase 4 cost: ${gpt4v_cost + claude_cost:.4f}")
        logger.info(f"   Time saved: ~{len(premium_frames) * 2:.0f}s by skipping BLIP-2 on {len(premium_frames)} premium frames")

        self.total_cost += gpt4v_cost + claude_cost

    def _analyze_frame_with_gpt4v(self, frame: ExtractedFrame) -> Dict:
        """Analyze frame using GPT-4 Vision"""
        import base64
        from openai import OpenAI
        from pathlib import Path

        client = OpenAI(api_key=self.openai_api_key)

        # Use image_path from ExtractedFrame object (set by frame extractor)
        frame_path = Path(frame.image_path)

        if not frame_path.exists():
            logger.warning(f"Frame image not found: {frame_path}")
            return self._create_empty_evidence(frame)

        with open(frame_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Call GPT-4 Vision
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Updated from deprecated gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this video frame. List: 1) All visible objects/people with descriptors (clothing, position), 2) Any visible text, 3) Scene type/setting. Be specific and detailed."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            description = response.choices[0].message.content

            # Parse description into structured data (simple approach)
            return {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "frame_type": frame.frame_type,
                "opportunity_type": frame.opportunity_type,
                "audio_cue": frame.audio_cue,
                "ground_truth": {
                    "gpt4v_description": description,
                    "yolo_objects": [],  # TODO: Parse from description
                    "ocr_text": [],      # TODO: Parse from description
                    "scene_type": "analyzed_with_gpt4v"
                }
            }
        except Exception as e:
            logger.error(f"GPT-4V analysis failed: {e}")
            return self._create_empty_evidence(frame)

    def _analyze_frame_with_claude(self, frame: ExtractedFrame) -> Dict:
        """
        Analyze frame using Claude Sonnet 4.5 Vision

        Args:
            frame: ExtractedFrame object

        Returns:
            Dict with Claude's analysis
        """
        import base64
        from pathlib import Path

        # Initialize Claude client if needed
        self._init_claude_client()

        if not self.claude_client:
            logger.warning("Claude client not available")
            return self._create_empty_evidence(frame)

        frame_path = Path(frame.image_path)

        if not frame_path.exists():
            logger.warning(f"Frame image not found: {frame_path}")
            return self._create_empty_evidence(frame)

        # Read and encode image
        with open(frame_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        try:
            # Call Claude Sonnet 4.5 Vision
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5 (latest)
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": "Analyze this video frame. List: 1) All visible objects/people with descriptors (clothing, position), 2) Any visible text, 3) Scene type/setting. Be specific and detailed."
                            }
                        ]
                    }
                ]
            )

            description = response.content[0].text

            return {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "frame_type": frame.frame_type,
                "opportunity_type": frame.opportunity_type,
                "audio_cue": frame.audio_cue,
                "ground_truth": {
                    "claude_description": description,
                    "analysis_method": "claude_sonnet_4.5"
                }
            }

        except Exception as e:
            logger.error(f"Claude Vision analysis failed: {e}")
            return self._create_empty_evidence(frame)

    def _run_consensus(
        self,
        gpt4_evidence: Dict,
        claude_evidence: Dict,
        ground_truth: Dict
    ) -> Dict:
        """
        Run consensus between GPT-4V and Claude Sonnet 4.5

        Args:
            gpt4_evidence: GPT-4 Vision analysis
            claude_evidence: Claude Sonnet 4.5 analysis
            ground_truth: YOLO + OCR + Scene analysis

        Returns:
            Consensus result with confidence and review flags
        """
        # Extract descriptions
        gpt4_desc = gpt4_evidence.get("ground_truth", {}).get("gpt4v_description", "")
        claude_desc = claude_evidence.get("ground_truth", {}).get("claude_description", "")

        # Simple agreement check (can be made more sophisticated)
        similarity = self._calculate_similarity(gpt4_desc, claude_desc)

        consensus_result = {
            "frame_id": gpt4_evidence["frame_id"],
            "timestamp": gpt4_evidence["timestamp"],
            "frame_type": gpt4_evidence["frame_type"],
            "opportunity_type": gpt4_evidence["opportunity_type"],
            "audio_cue": gpt4_evidence["audio_cue"],
            "ground_truth": {
                # Merge ground truth from bulk analyzer
                **ground_truth.get("ground_truth", {}),
                # Add AI analyses
                "gpt4v_description": gpt4_desc,
                "claude_description": claude_desc,
                "ai_consensus": {
                    "similarity_score": similarity,
                    "consensus_reached": similarity > 0.7,
                    "confidence": self._calculate_confidence(similarity, ground_truth),
                    "needs_human_review": similarity < 0.6,
                    "priority": self._calculate_priority(similarity, ground_truth)
                }
            }
        }

        return consensus_result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text descriptions"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _calculate_confidence(self, similarity: float, ground_truth: Dict) -> float:
        """
        Calculate overall confidence based on AI similarity and ground truth

        Returns confidence score between 0.0 and 1.0
        """
        # High similarity between AIs
        if similarity > 0.8:
            base_confidence = 0.95
        elif similarity > 0.7:
            base_confidence = 0.85
        elif similarity > 0.6:
            base_confidence = 0.75
        else:
            base_confidence = 0.60

        # Boost confidence if ground truth has strong signals
        gt_data = ground_truth.get("ground_truth", {})
        has_objects = len(gt_data.get("yolo_objects", [])) > 0
        has_text = len(gt_data.get("ocr_text", [])) > 0
        has_scene = gt_data.get("scene_confidence", 0.0) > 0.7

        if has_objects or has_text or has_scene:
            base_confidence = min(1.0, base_confidence + 0.05)

        return base_confidence

    def _calculate_priority(self, similarity: float, ground_truth: Dict) -> str:
        """
        Calculate review priority level

        Returns: 'low', 'medium', or 'high'
        """
        if similarity < 0.5:
            return "high"  # Strong disagreement
        elif similarity < 0.7:
            return "medium"  # Moderate disagreement
        else:
            return "low"  # Good agreement

    def _calculate_consensus_stats(self) -> Dict:
        """Calculate consensus statistics from evidence"""
        stats = {
            "consensus_reached": 0,
            "needs_review": 0,
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0
        }

        for frame_id, frame_evidence in self.evidence["frames"].items():
            ai_consensus = frame_evidence.get("ground_truth", {}).get("ai_consensus")

            if ai_consensus:
                if ai_consensus["consensus_reached"]:
                    stats["consensus_reached"] += 1

                if ai_consensus["needs_human_review"]:
                    stats["needs_review"] += 1

                priority = ai_consensus["priority"]
                stats[f"{priority}_priority"] += 1

        return stats

    def _create_empty_evidence(self, frame: ExtractedFrame) -> Dict:
        """Create empty evidence structure for a frame"""
        return {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "frame_type": frame.frame_type,
            "opportunity_type": frame.opportunity_type,
            "audio_cue": frame.audio_cue,
            "ground_truth": {
                "yolo_objects": [],
                "ocr_text": [],
                "scene_type": ""
            }
        }

    def _run_phase5_questions(self):
        """Phase 5: Multimodal Question Generation V2 (With Validation)"""
        phase_start = datetime.now()
        logger.info("📍 Starting Phase 5: Multimodal Question Generation V2")
        logger.info(f"   Generating validated questions with audio + visual integration...")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for question generation")

        generator = MultimodalQuestionGeneratorV2(
            openai_api_key=self.openai_api_key,
            claude_api_key=self.claude_api_key
        )

        # Generate questions from Phase 4 evidence with dynamic allocation
        logger.info(f"   Generating 30 questions with dynamic allocation based on premium frames...")
        result = generator.generate_questions(
            phase4_evidence=self.evidence,
            audio_analysis=self.audio_analysis,
            video_id=self.video_id
        )

        # Save questions
        questions_path = self.output_dir / f"{self.video_id}_questions.json"
        generator.save_questions(result, questions_path)

        # Store questions list for later use
        self.questions = result.questions

        phase_time = (datetime.now() - phase_start).total_seconds()
        logger.info(f"✅ Phase 5 Complete!")
        logger.info(f"   Total questions: {result.total_questions}")
        logger.info(f"   Validated questions: {result.validated_questions}/{result.total_questions}")
        logger.info(f"   Processing time: {phase_time:.1f}s")
        logger.info(f"   Generation cost: ${result.generation_cost:.4f}")

        self.total_cost += result.generation_cost

    def _run_phase6_gemini(self):
        """
        Phase 6: Gemini Testing (Optional)

        Tests generated questions against Gemini 2.0 Flash.
        """
        logger.warning("⚠️  Phase 6 (Gemini Testing) not yet implemented")
        logger.warning("   Implement gemini/adversarial_tester.py integration")

        # Placeholder
        self.gemini_results = {
            "tested": False,
            "reason": "Not yet implemented"
        }


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