"""
AI Consensus Engine - Compare predictions from multiple AI systems

Determines when AI models agree, calculates confidence scores, and
flags items that need human review.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConsensusLevel(Enum):
    """Levels of consensus between AI models"""
    FULL_AGREEMENT = "full"  # All AIs agree
    MAJORITY = "majority"  # 2 out of 3 agree
    NO_CONSENSUS = "none"  # All disagree


@dataclass
class AIConsensusResult:
    """Result of consensus analysis"""
    consensus_level: ConsensusLevel
    consensus_answer: Optional[Any]
    confidence_score: float
    needs_human_review: bool
    priority_level: str  # 'high', 'medium', 'low'
    flag_reason: Optional[str]
    disagreement_details: Optional[Dict]


class AIConsensusEngine:
    """
    Compares predictions from GPT-4 Vision, Claude Sonnet 4.5, and Open Models
    to determine consensus and flag items for human review.
    """

    def __init__(
        self,
        confidence_threshold_high: float = 0.95,
        confidence_threshold_medium: float = 0.85
    ):
        """
        Initialize consensus engine

        Args:
            confidence_threshold_high: Threshold for high confidence
            confidence_threshold_medium: Threshold for needing review
        """
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_medium = confidence_threshold_medium

    def analyze_consensus(
        self,
        gpt4_prediction: Optional[Dict],
        claude_prediction: Optional[Dict],
        open_model_prediction: Optional[Dict],
        ground_truth: Optional[Dict] = None
    ) -> AIConsensusResult:
        """
        Analyze consensus between AI predictions

        Args:
            gpt4_prediction: GPT-4 Vision prediction
            claude_prediction: Claude Sonnet 4.5 prediction
            open_model_prediction: Open models (YOLO, OCR, etc.) prediction
            ground_truth: Objective facts from deterministic models

        Returns:
            AIConsensusResult with consensus analysis
        """
        predictions = {
            'gpt4': gpt4_prediction,
            'claude': claude_prediction,
            'open': open_model_prediction
        }

        # Filter out None predictions
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}

        if len(valid_predictions) < 2:
            # Not enough predictions to compare
            return AIConsensusResult(
                consensus_level=ConsensusLevel.NO_CONSENSUS,
                consensus_answer=None,
                confidence_score=0.0,
                needs_human_review=True,
                priority_level='high',
                flag_reason="Insufficient AI predictions",
                disagreement_details=None
            )

        # Compare answers
        consensus_level, agreed_answer, disagreement = self._compare_answers(valid_predictions)

        # Calculate confidence score
        confidence = self._calculate_confidence(
            consensus_level,
            valid_predictions,
            ground_truth
        )

        # Determine if human review needed
        needs_review, priority, flag_reason = self._determine_review_need(
            consensus_level,
            confidence,
            disagreement,
            ground_truth
        )

        return AIConsensusResult(
            consensus_level=consensus_level,
            consensus_answer=agreed_answer,
            confidence_score=confidence,
            needs_human_review=needs_review,
            priority_level=priority,
            flag_reason=flag_reason,
            disagreement_details=disagreement
        )

    def _compare_answers(
        self,
        predictions: Dict[str, Dict]
    ) -> Tuple[ConsensusLevel, Optional[Any], Optional[Dict]]:
        """
        Compare answers from different AI systems

        Returns:
            (consensus_level, agreed_answer, disagreement_details)
        """
        # Extract answers
        answers = {}
        for ai_name, prediction in predictions.items():
            if 'answer' in prediction:
                answers[ai_name] = prediction['answer']
            elif 'text' in prediction:
                answers[ai_name] = prediction['text']
            else:
                # Try to use the prediction itself
                answers[ai_name] = prediction

        if len(answers) < 2:
            return ConsensusLevel.NO_CONSENSUS, None, None

        # Compare answers (simplified - you can make this more sophisticated)
        unique_answers = list(set(str(a) for a in answers.values()))

        if len(unique_answers) == 1:
            # All agree
            return (
                ConsensusLevel.FULL_AGREEMENT,
                list(answers.values())[0],
                None
            )
        elif len(answers) >= 3:
            # Check for 2-out-of-3 agreement
            answer_counts = {}
            for answer in answers.values():
                answer_str = str(answer)
                answer_counts[answer_str] = answer_counts.get(answer_str, 0) + 1

            max_count = max(answer_counts.values())
            if max_count >= 2:
                # Majority agrees
                majority_answer = [k for k, v in answer_counts.items() if v == max_count][0]

                # Find which AIs agreed/disagreed
                disagreement = {
                    'agreed': [ai for ai, ans in answers.items() if str(ans) == majority_answer],
                    'disagreed': [ai for ai, ans in answers.items() if str(ans) != majority_answer],
                    'answers': answers
                }

                return ConsensusLevel.MAJORITY, majority_answer, disagreement

        # No consensus
        disagreement = {
            'answers': answers,
            'reason': 'All AI systems provided different answers'
        }
        return ConsensusLevel.NO_CONSENSUS, None, disagreement

    def _calculate_confidence(
        self,
        consensus_level: ConsensusLevel,
        predictions: Dict[str, Dict],
        ground_truth: Optional[Dict]
    ) -> float:
        """
        Calculate confidence score based on consensus and verification

        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = {
            ConsensusLevel.FULL_AGREEMENT: 0.95,
            ConsensusLevel.MAJORITY: 0.85,
            ConsensusLevel.NO_CONSENSUS: 0.60
        }

        confidence = base_confidence[consensus_level]

        # Adjust based on individual AI confidences
        ai_confidences = []
        for prediction in predictions.values():
            if isinstance(prediction, dict) and 'confidence' in prediction:
                ai_confidences.append(prediction['confidence'])

        if ai_confidences:
            avg_ai_confidence = sum(ai_confidences) / len(ai_confidences)
            # Blend consensus confidence with AI confidence
            confidence = 0.7 * confidence + 0.3 * avg_ai_confidence

        # Boost confidence if verified by ground truth
        if ground_truth:
            if self._verify_against_ground_truth(predictions, ground_truth):
                confidence = min(confidence + 0.05, 1.0)
            else:
                # Contradicts ground truth - lower confidence
                confidence = max(confidence - 0.15, 0.0)

        return round(confidence, 3)

    def _verify_against_ground_truth(
        self,
        predictions: Dict[str, Dict],
        ground_truth: Dict
    ) -> bool:
        """
        Check if predictions align with objective ground truth

        Returns:
            True if verified, False otherwise
        """
        # This is simplified - you can make more sophisticated
        # For example, if ground truth says "2 people" and AI says "3 people", fail

        # Convert GroundTruth object to dict if needed
        if hasattr(ground_truth, 'to_dict'):
            ground_truth_dict = ground_truth.to_dict()
        elif isinstance(ground_truth, dict):
            ground_truth_dict = ground_truth
        else:
            return True  # Can't verify, assume OK

        # Check object counts
        if 'object_count' in ground_truth_dict:
            for prediction in predictions.values():
                if isinstance(prediction, dict) and 'count' in prediction:
                    if abs(prediction['count'] - ground_truth_dict['object_count']) > 1:
                        return False

        # Check OCR text matches
        if 'ocr_text' in ground_truth_dict:
            for prediction in predictions.values():
                if isinstance(prediction, dict) and 'text' in prediction:
                    if ground_truth_dict['ocr_text'].lower() not in prediction['text'].lower():
                        return False

        return True

    def _determine_review_need(
        self,
        consensus_level: ConsensusLevel,
        confidence: float,
        disagreement: Optional[Dict],
        ground_truth: Optional[Dict]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if human review is needed and priority level

        Returns:
            (needs_review, priority_level, flag_reason)
        """
        # HIGH PRIORITY - Needs immediate review
        if consensus_level == ConsensusLevel.NO_CONSENSUS:
            return (
                True,
                'high',
                f"AI disagreement: {disagreement['reason'] if disagreement else 'No consensus'}"
            )

        if confidence < self.confidence_threshold_medium:
            return (
                True,
                'high',
                f"Low confidence: {confidence:.2f}"
            )

        # MEDIUM PRIORITY - Borderline cases
        if consensus_level == ConsensusLevel.MAJORITY:
            return (
                True,
                'medium',
                f"Partial agreement: {disagreement['disagreed']} disagreed" if disagreement else "Majority consensus"
            )

        if confidence < self.confidence_threshold_high:
            return (
                True,
                'medium',
                f"Medium confidence: {confidence:.2f}"
            )

        # LOW PRIORITY - Spot check only
        # Even high-confidence items get 10% spot-check sampling
        import random
        if random.random() < 0.10:
            return (
                True,
                'low',
                "Random spot check for quality assurance"
            )

        # No review needed - auto-approved
        return (False, 'low', None)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = AIConsensusEngine()

    # Example: All AIs agree
    result = engine.analyze_consensus(
        gpt4_prediction={"answer": "Player #77", "confidence": 0.92},
        claude_prediction={"answer": "Player #77", "confidence": 0.94},
        open_model_prediction={"text": "77"},
        ground_truth={"ocr_text": "77"}
    )

    print(f"Consensus: {result.consensus_level}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Needs review: {result.needs_human_review}")
    print(f"Priority: {result.priority_level}")
