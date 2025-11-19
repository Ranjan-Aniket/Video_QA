"""
Smart Evidence Extractor - Extract evidence from frames using genre-specific models

Processes extracted frames intelligently:
- Bulk frames: Run recommended open-source models only
- Key moment frames: Run expensive AI models (GPT-4, Claude)
- Genre-aware: Only run relevant models based on video type

Phase 4 of the new evidence-first architecture.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.evidence_item import GroundTruth, EvidenceItem
from processing.smart_frame_extractor import ExtractedFrame
from database.evidence_operations import EvidenceOperations

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from a single model"""
    model_name: str
    result_data: Dict
    confidence: float
    processing_time: float


class SmartEvidenceExtractor:
    """
    Extract evidence from frames using genre-appropriate models
    """

    def __init__(
        self,
        video_id: int,
        genre_analysis: Dict,
        enable_gpt4: bool = False,
        enable_claude: bool = False
    ):
        """
        Initialize smart evidence extractor

        Args:
            video_id: Database video ID
            genre_analysis: Genre analysis dict from genre detector
            enable_gpt4: Use GPT-4 Vision for key moments
            enable_claude: Use Claude Vision for key moments
        """
        self.video_id = video_id
        self.genre_analysis = genre_analysis
        self.enable_gpt4 = enable_gpt4
        self.enable_claude = enable_claude

        # Get recommended models from genre analysis
        self.recommended_models = genre_analysis.get("recommended_models", [])
        self.primary_genre = genre_analysis.get("primary_genre", "unknown")

        logger.info("=" * 80)
        logger.info("SMART EVIDENCE EXTRACTOR - INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Genre: {self.primary_genre}")
        logger.info(f"Recommended models: {', '.join(self.recommended_models)}")
        logger.info(f"GPT-4: {'ENABLED' if enable_gpt4 else 'DISABLED'}")
        logger.info(f"Claude: {'ENABLED' if enable_claude else 'DISABLED'}")

    def process_frames(
        self,
        frames: List[ExtractedFrame],
        save_to_db: bool = True
    ) -> List[EvidenceItem]:
        """
        Process all extracted frames and create evidence items

        Args:
            frames: List of ExtractedFrame objects
            save_to_db: Save evidence items to database

        Returns:
            List of EvidenceItem objects
        """
        logger.info("=" * 80)
        logger.info("PROCESSING FRAMES FOR EVIDENCE EXTRACTION")
        logger.info("=" * 80)
        logger.info(f"Total frames: {len(frames)}")

        evidence_items = []

        # Process bulk frames (open models only)
        bulk_frames = [f for f in frames if f.frame_type == "bulk"]
        logger.info(f"\n📦 Processing {len(bulk_frames)} bulk frames with open models...")

        for i, frame in enumerate(bulk_frames, 1):
            evidence = self._process_bulk_frame(frame)
            evidence_items.append(evidence)

            if i % 10 == 0:
                logger.info(f"  ✓ Processed {i}/{len(bulk_frames)} bulk frames")

        # Process key moment frames (expensive models)
        key_frames = [f for f in frames if f.frame_type == "key_moment"]
        logger.info(f"\n🔑 Processing {len(key_frames)} key moment frames with expensive AI...")

        for i, frame in enumerate(key_frames, 1):
            evidence = self._process_key_moment_frame(frame)
            evidence_items.append(evidence)
            logger.info(f"  ✓ Processed {i}/{len(key_frames)} key moments")

        logger.info("=" * 80)
        logger.info(f"✅ CREATED {len(evidence_items)} EVIDENCE ITEMS")
        logger.info("=" * 80)

        # Save to database
        if save_to_db:
            self._save_to_database(evidence_items)

        return evidence_items

    def _process_bulk_frame(self, frame: ExtractedFrame) -> EvidenceItem:
        """
        Process bulk frame with open-source models only

        Args:
            frame: ExtractedFrame object

        Returns:
            EvidenceItem object
        """
        # Get models to run
        models_to_run = frame.recommended_models or self.recommended_models

        # Create ground truth by running open models
        ground_truth = self._run_open_models(frame.image_path, models_to_run)

        # Create evidence item (no AI predictions for bulk frames)
        evidence_item = EvidenceItem(
            video_id=self.video_id,
            evidence_type="bulk_frame",
            timestamp_start=frame.timestamp,
            timestamp_end=frame.timestamp + 0.1,  # Single frame
            ground_truth=ground_truth,
            gpt4_prediction=None,
            claude_prediction=None,
            open_model_prediction=None,
            consensus=None
        )

        return evidence_item

    def _process_key_moment_frame(self, frame: ExtractedFrame) -> EvidenceItem:
        """
        Process key moment frame with expensive AI models

        Args:
            frame: ExtractedFrame object

        Returns:
            EvidenceItem object
        """
        # Run open models for ground truth
        models_to_run = frame.recommended_models or self.recommended_models
        ground_truth = self._run_open_models(frame.image_path, models_to_run)

        # Run expensive AI models
        gpt4_prediction = None
        claude_prediction = None

        if self.enable_gpt4:
            gpt4_prediction = self._run_gpt4_vision(frame.image_path, frame.reason)

        if self.enable_claude:
            claude_prediction = self._run_claude_vision(frame.image_path, frame.reason)

        # Create evidence item
        evidence_item = EvidenceItem(
            video_id=self.video_id,
            evidence_type="key_moment",
            timestamp_start=frame.timestamp,
            timestamp_end=frame.timestamp + 5.0,  # Key moment window
            ground_truth=ground_truth,
            gpt4_prediction=gpt4_prediction,
            claude_prediction=claude_prediction,
            open_model_prediction=None,
            consensus=None
        )

        return evidence_item

    def _run_open_models(
        self,
        image_path: str,
        models: List[str]
    ) -> GroundTruth:
        """
        Run open-source models on image

        Args:
            image_path: Path to image file
            models: List of model names to run

        Returns:
            GroundTruth object with results
        """
        # For now, create mock ground truth
        # In production, this would call actual model inference

        ground_truth_data = {}

        # Mock results based on recommended models
        if "yolo" in models:
            ground_truth_data["yolov8x_objects"] = [
                {"class": "person", "confidence": 0.92},
                {"class": "couch", "confidence": 0.85}
            ]
            ground_truth_data["object_count"] = 2
            ground_truth_data["person_count"] = 1

        if "clip" in models:
            ground_truth_data["clip_clothing"] = [
                "casual shirt",
                "indoor setting"
            ]

        if "places365" in models:
            ground_truth_data["places365_scene"] = {
                "scene_category": "living_room",
                "confidence": 0.88
            }
            ground_truth_data["is_indoor"] = True
            ground_truth_data["is_sports_venue"] = False

        if "ocr" in models:
            ground_truth_data["paddleocr_text"] = []
            ground_truth_data["ocr_blocks"] = []

        if "fer" in models:
            ground_truth_data["fer_emotions"] = [
                {"emotion": "neutral", "confidence": 0.75}
            ]
            ground_truth_data["dominant_emotion"] = "neutral"

        if "blip2" in models:
            ground_truth_data["blip2_description"] = {
                "caption": "A person in a living room"
            }
            ground_truth_data["image_caption"] = "A person in a living room"

        # Create GroundTruth object
        ground_truth = GroundTruth(**ground_truth_data)

        return ground_truth

    def _run_gpt4_vision(
        self,
        image_path: str,
        context: Optional[str] = None
    ) -> Dict:
        """
        Run GPT-4 Vision on image

        Args:
            image_path: Path to image
            context: Context about why this frame is important

        Returns:
            GPT-4 prediction dict
        """
        # Mock GPT-4 Vision result
        # In production, this would call OpenAI API

        logger.info(f"  [GPT-4 Vision] Processing frame...")

        prediction = {
            "description": "A person sitting in a living room, appears to be filming a vlog",
            "key_objects": ["person", "furniture", "indoor setting"],
            "emotional_tone": "casual, conversational",
            "text_visible": [],
            "confidence": 0.89,
            "context_understanding": context or "General scene analysis"
        }

        return prediction

    def _run_claude_vision(
        self,
        image_path: str,
        context: Optional[str] = None
    ) -> Dict:
        """
        Run Claude Vision on image

        Args:
            image_path: Path to image
            context: Context about why this frame is important

        Returns:
            Claude prediction dict
        """
        # Mock Claude Vision result
        # In production, this would call Anthropic API

        logger.info(f"  [Claude Vision] Processing frame...")

        prediction = {
            "description": "Indoor scene showing a person in what appears to be a residential living space",
            "scene_elements": ["person", "interior", "casual setting"],
            "atmosphere": "relaxed, personal",
            "visible_text": [],
            "confidence": 0.91,
            "reasoning": context or "Standard scene interpretation"
        }

        return prediction

    def _save_to_database(self, evidence_items: List[EvidenceItem]):
        """
        Save evidence items to database

        Args:
            evidence_items: List of EvidenceItem objects
        """
        logger.info(f"\n💾 Saving {len(evidence_items)} evidence items to database...")

        saved_count = 0

        for item in evidence_items:
            try:
                # Convert ground truth to dict
                if hasattr(item.ground_truth, '__dict__'):
                    ground_truth_dict = asdict(item.ground_truth)
                else:
                    ground_truth_dict = item.ground_truth

                # Save to database
                evidence_id = EvidenceOperations.create_evidence_item(
                    video_id=str(self.video_id),
                    evidence_type=item.evidence_type,
                    timestamp_start=item.timestamp_start,
                    timestamp_end=item.timestamp_end,
                    ground_truth=ground_truth_dict,
                    gpt4_prediction=item.gpt4_prediction,
                    claude_prediction=item.claude_prediction,
                    open_model_prediction=item.open_model_prediction,
                    ai_consensus_reached=False,
                    consensus_answer=None,
                    confidence_score=0.0,
                    needs_review=False,
                    priority="low" if item.evidence_type == "bulk_frame" else "medium"
                )

                saved_count += 1

            except Exception as e:
                logger.error(f"Failed to save evidence item: {e}")

        logger.info(f"✓ Saved {saved_count}/{len(evidence_items)} evidence items")


def test_smart_evidence_extractor(
    frames_metadata_path: str,
    genre_analysis_path: str,
    video_id: int = 999
):
    """
    Test smart evidence extractor

    Args:
        frames_metadata_path: Path to frames metadata JSON
        genre_analysis_path: Path to genre analysis JSON
        video_id: Database video ID
    """
    # Load frames metadata
    with open(frames_metadata_path, 'r') as f:
        frames_metadata = json.load(f)

    # Load genre analysis
    with open(genre_analysis_path, 'r') as f:
        pipeline_results = json.load(f)

    genre_analysis = pipeline_results.get("genre_analysis")

    if not genre_analysis:
        print("❌ No genre_analysis found in pipeline results")
        return

    # Convert frames data back to ExtractedFrame objects
    frames = []
    for frame_data in frames_metadata["frames"][:10]:  # Process first 10 for testing
        frame = ExtractedFrame(**frame_data)
        frames.append(frame)

    print(f"\n✓ Loaded {len(frames)} frames for testing")
    print(f"  Bulk frames: {len([f for f in frames if f.frame_type == 'bulk'])}")
    print(f"  Key moments: {len([f for f in frames if f.frame_type == 'key_moment'])}")

    # Create extractor
    extractor = SmartEvidenceExtractor(
        video_id=video_id,
        genre_analysis=genre_analysis,
        enable_gpt4=False,  # Disabled for testing
        enable_claude=False
    )

    # Process frames
    evidence_items = extractor.process_frames(frames, save_to_db=True)

    # Display results
    print("\n" + "=" * 80)
    print("EVIDENCE EXTRACTION RESULTS")
    print("=" * 80)
    print(f"Total Evidence Items: {len(evidence_items)}")
    print(f"  Bulk frames: {len([e for e in evidence_items if e.evidence_type == 'bulk_frame'])}")
    print(f"  Key moments: {len([e for e in evidence_items if e.evidence_type == 'key_moment'])}")
    print("\nSample Evidence Items:")
    print("-" * 80)

    for i, item in enumerate(evidence_items[:3], 1):
        print(f"\n{i}. Type: {item.evidence_type}")
        print(f"   Timestamp: {item.timestamp_start:.1f}s - {item.timestamp_end:.1f}s")
        print(f"   Ground Truth:")
        if hasattr(item.ground_truth, '__dict__'):
            for key, value in vars(item.ground_truth).items():
                if value is not None and not key.startswith('_'):
                    print(f"     - {key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        frames_metadata = sys.argv[1]
        genre_analysis = sys.argv[2]
        video_id = int(sys.argv[3]) if len(sys.argv) > 3 else 999

        test_smart_evidence_extractor(frames_metadata, genre_analysis, video_id)
    else:
        print("Usage: python smart_evidence_extractor.py <frames_metadata.json> <pipeline_results.json> [video_id]")
        print("\nExample:")
        print("  python smart_evidence_extractor.py frames/video/frames_metadata.json video_pipeline_results.json 123")
