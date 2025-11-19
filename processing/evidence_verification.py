"""
Evidence Verification - Cross-check AI predictions against ground truth

Verifies AI predictions against objective facts from deterministic models:
- YOLO object detection
- OCR text extraction
- Whisper transcription
- Scene classification

Flags contradictions and assigns priority levels for human review.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from processing.evidence_item import (
    EvidenceItem, AIPrediction, GroundTruth, ConsensusAnalysis
)

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    """
    Verifies AI predictions against ground truth and flags inconsistencies
    """

    def __init__(
        self,
        ocr_match_threshold: float = 0.7,
        object_count_tolerance: int = 1,
        strict_mode: bool = False
    ):
        """
        Initialize evidence verifier

        Args:
            ocr_match_threshold: Minimum similarity for OCR text match (0-1)
            object_count_tolerance: Allowed difference in object counts
            strict_mode: If True, flag more aggressively
        """
        self.ocr_match_threshold = ocr_match_threshold
        self.object_count_tolerance = object_count_tolerance
        self.strict_mode = strict_mode

    def verify_evidence(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis
    ) -> Tuple[bool, Optional[str], str]:
        """
        Verify evidence item against ground truth

        Args:
            evidence: Evidence item with AI predictions and ground truth
            consensus: Consensus analysis result

        Returns:
            (is_verified, flag_reason, priority_level)
        """
        if not evidence.ground_truth:
            logger.warning(f"No ground truth for evidence {evidence.evidence_id}")
            return False, "No ground truth available", "medium"

        # Verify based on evidence type
        if evidence.evidence_type == 'ocr':
            return self._verify_ocr(evidence, consensus)
        elif evidence.evidence_type == 'object_detection':
            return self._verify_objects(evidence, consensus)
        elif evidence.evidence_type == 'emotion':
            return self._verify_emotion(evidence, consensus)
        elif evidence.evidence_type == 'action':
            return self._verify_action(evidence, consensus)
        elif evidence.evidence_type == 'scene':
            return self._verify_scene(evidence, consensus)
        else:
            logger.warning(f"Unknown evidence type: {evidence.evidence_type}")
            return True, None, "medium"

    def _verify_ocr(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis
    ) -> Tuple[bool, Optional[str], str]:
        """Verify OCR predictions against ground truth OCR"""
        if not evidence.ground_truth.ocr_text:
            return True, None, "medium"

        ground_truth_text = " ".join(evidence.ground_truth.ocr_text).lower()
        consensus_answer = str(consensus.consensus_answer).lower() if consensus.consensus_answer else ""

        # Check if consensus matches ground truth
        if ground_truth_text in consensus_answer or consensus_answer in ground_truth_text:
            return True, None, "low"

        # Calculate similarity
        similarity = self._text_similarity(ground_truth_text, consensus_answer)

        if similarity < self.ocr_match_threshold:
            return (
                False,
                f"OCR mismatch: Ground truth '{ground_truth_text[:50]}' vs AI '{consensus_answer[:50]}'",
                "high"
            )

        return True, None, "medium"

    def _verify_objects(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis
    ) -> Tuple[bool, Optional[str], str]:
        """Verify object detection against YOLO ground truth"""
        if not evidence.ground_truth.yolo_objects:
            return True, None, "medium"

        # Get object counts from ground truth
        yolo_objects = evidence.ground_truth.yolo_objects
        yolo_classes = [obj.get('class', obj.get('label', '')) for obj in yolo_objects]
        yolo_count = len(yolo_classes)

        # Get consensus answer
        consensus_answer = consensus.consensus_answer
        if isinstance(consensus_answer, list):
            ai_count = len(consensus_answer)
        elif isinstance(consensus_answer, str):
            # Try to extract count from string
            import re
            numbers = re.findall(r'\d+', consensus_answer)
            ai_count = int(numbers[0]) if numbers else 0
        else:
            ai_count = 0

        # Check if counts match within tolerance
        count_diff = abs(yolo_count - ai_count)

        if count_diff > self.object_count_tolerance:
            return (
                False,
                f"Object count mismatch: YOLO detected {yolo_count}, AI predicted {ai_count}",
                "high"
            )

        # Check if object classes match
        if isinstance(consensus_answer, list):
            consensus_classes = [str(obj).lower() for obj in consensus_answer]
            yolo_classes_lower = [cls.lower() for cls in yolo_classes]

            # Check overlap
            matching_classes = set(consensus_classes) & set(yolo_classes_lower)
            if len(matching_classes) < len(yolo_classes) * 0.7:
                return (
                    False,
                    f"Object class mismatch: YOLO={yolo_classes_lower}, AI={consensus_classes}",
                    "medium"
                )

        return True, None, "low"

    def _verify_emotion(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis
    ) -> Tuple[bool, Optional[str], str]:
        """
        Verify emotion predictions against transcript context

        Emotions are subjective, so we only flag if there's strong evidence
        of contradiction from the transcript.
        """
        if not evidence.ground_truth.whisper_transcript:
            # No ground truth to verify against
            return True, None, "medium"

        transcript = evidence.ground_truth.whisper_transcript.lower()
        consensus_emotion = str(consensus.consensus_answer).lower() if consensus.consensus_answer else ""

        # Check for obvious contradictions
        # e.g., AI says "happy" but transcript has "crying", "sad", etc.
        contradictions = {
            'happy': ['crying', 'sad', 'depressed', 'angry', 'scared'],
            'sad': ['laughing', 'excited', 'joyful', 'happy'],
            'angry': ['calm', 'peaceful', 'relaxed', 'happy'],
            'scared': ['confident', 'brave', 'fearless'],
            'excited': ['bored', 'tired', 'sleepy', 'depressed']
        }

        for emotion, contradictory_words in contradictions.items():
            if emotion in consensus_emotion:
                for word in contradictory_words:
                    if word in transcript:
                        return (
                            False,
                            f"Emotion contradiction: AI says '{emotion}' but transcript contains '{word}'",
                            "high"
                        )

        # No obvious contradiction found
        return True, None, "medium"

    def _verify_action(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis
    ) -> Tuple[bool, Optional[str], str]:
        """Verify action predictions"""
        # Actions are subjective, verify against objects if available
        if evidence.ground_truth.yolo_objects:
            # Check if predicted action makes sense given detected objects
            # This is simplified - in reality you'd have more complex logic
            return True, None, "medium"

        return True, None, "medium"

    def _verify_scene(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis
    ) -> Tuple[bool, Optional[str], str]:
        """Verify scene classification"""
        if evidence.ground_truth.scene_classification:
            ground_truth_scene = evidence.ground_truth.scene_classification.lower()
            consensus_scene = str(consensus.consensus_answer).lower() if consensus.consensus_answer else ""

            if ground_truth_scene in consensus_scene or consensus_scene in ground_truth_scene:
                return True, None, "low"
            else:
                return (
                    False,
                    f"Scene mismatch: Ground truth '{ground_truth_scene}' vs AI '{consensus_scene}'",
                    "medium"
                )

        return True, None, "medium"

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple ratio

        Returns value between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def assign_priority(
        self,
        evidence: EvidenceItem,
        consensus: ConsensusAnalysis,
        verification_passed: bool,
        flag_reason: Optional[str]
    ) -> str:
        """
        Assign priority level for human review

        Priority levels:
        - high: AI disagreement, failed verification, low confidence
        - medium: Partial agreement, borderline confidence
        - low: Full agreement, high confidence, spot check
        """
        # HIGH PRIORITY
        if not verification_passed:
            return "high"

        if consensus.consensus_level == "none":
            return "high"

        if consensus.confidence_score < 0.85:
            return "high"

        # MEDIUM PRIORITY
        if consensus.consensus_level == "majority":
            return "medium"

        if consensus.confidence_score < 0.95:
            return "medium"

        # Check evidence type - some types are inherently more subjective
        subjective_types = ['emotion', 'action', 'reasoning']
        if evidence.evidence_type in subjective_types:
            return "medium"

        # LOW PRIORITY (spot check)
        # Full agreement + high confidence + verified
        return "low"


class VerificationReport:
    """Report on evidence verification results"""

    def __init__(self):
        self.total_items = 0
        self.verified_items = 0
        self.flagged_items = 0
        self.high_priority = 0
        self.medium_priority = 0
        self.low_priority = 0
        self.verification_failures: List[Dict] = []

    def add_result(
        self,
        evidence: EvidenceItem,
        verified: bool,
        flag_reason: Optional[str],
        priority: str
    ):
        """Add verification result"""
        self.total_items += 1

        if verified:
            self.verified_items += 1
        else:
            self.flagged_items += 1
            self.verification_failures.append({
                'evidence_id': evidence.evidence_id,
                'video_id': evidence.video_id,
                'evidence_type': evidence.evidence_type,
                'timestamp': evidence.timestamp_start,
                'reason': flag_reason,
                'priority': priority
            })

        if priority == 'high':
            self.high_priority += 1
        elif priority == 'medium':
            self.medium_priority += 1
        else:
            self.low_priority += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get verification summary"""
        return {
            'total_items': self.total_items,
            'verified_items': self.verified_items,
            'flagged_items': self.flagged_items,
            'verification_rate': self.verified_items / self.total_items if self.total_items > 0 else 0,
            'high_priority': self.high_priority,
            'medium_priority': self.medium_priority,
            'low_priority': self.low_priority,
            'verification_failures': self.verification_failures
        }

    def log_summary(self):
        """Log verification summary"""
        summary = self.get_summary()
        logger.info("=" * 80)
        logger.info("EVIDENCE VERIFICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total items: {summary['total_items']}")
        logger.info(f"Verified: {summary['verified_items']} ({summary['verification_rate']:.1%})")
        logger.info(f"Flagged: {summary['flagged_items']}")
        logger.info(f"Priority breakdown:")
        logger.info(f"  High: {summary['high_priority']}")
        logger.info(f"  Medium: {summary['medium_priority']}")
        logger.info(f"  Low: {summary['low_priority']}")
        logger.info("=" * 80)
