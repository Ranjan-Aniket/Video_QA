"""
Difficulty Ranker - Question Complexity & Failure Severity Ranking

Purpose: Rank questions by difficulty and how badly they expose Gemini failures
Compliance: Multi-dimensional complexity scoring, severity weighting
Architecture: Combines intrinsic difficulty with failure impact

Ranking Dimensions:
1. Inference Complexity (0-10): Multi-hop reasoning depth
2. Temporal Complexity (0-10): Cross-temporal dependencies
3. Multimodal Integration (0-10): Audio-visual reasoning required
4. Evidence Subtlety (0-10): How hidden/subtle the evidence is
5. Failure Severity (0-10): How badly Gemini failed
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ComplexityLevel(str, Enum):
    """Question complexity classification"""
    EASY = "easy"              # Score 0-3: Simple, direct questions
    MEDIUM = "medium"          # Score 3-6: Moderate reasoning
    HARD = "hard"              # Score 6-8: Multi-hop, complex
    EXPERT = "expert"          # Score 8-10: Highly subtle, advanced


@dataclass
class DifficultyConfig:
    """Configuration for difficulty ranking"""
    
    # Dimension weights (must sum to 1.0)
    inference_weight: float = 0.25      # Weight for inference complexity
    temporal_weight: float = 0.20       # Weight for temporal reasoning
    multimodal_weight: float = 0.20     # Weight for multimodal integration
    evidence_weight: float = 0.15       # Weight for evidence subtlety
    failure_weight: float = 0.20        # Weight for failure severity
    
    # Failure severity multipliers
    critical_failure_boost: float = 2.0   # Boost for critical hallucinations
    major_failure_boost: float = 1.5      # Boost for major errors
    minor_failure_boost: float = 1.0      # Boost for minor errors
    
    # Complexity thresholds
    easy_threshold: float = 3.0
    medium_threshold: float = 6.0
    hard_threshold: float = 8.0
    
    def validate(self) -> bool:
        """Validate configuration"""
        total_weight = (
            self.inference_weight + 
            self.temporal_weight + 
            self.multimodal_weight + 
            self.evidence_weight + 
            self.failure_weight
        )
        
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Dimension weights must sum to 1.0, got {total_weight}")
        
        return True


@dataclass
class DifficultyScore:
    """Difficulty score breakdown for a question"""
    
    # Individual dimension scores (0-10)
    inference_score: float = 0.0      # Multi-hop reasoning complexity
    temporal_score: float = 0.0       # Temporal reasoning complexity
    multimodal_score: float = 0.0     # Multimodal integration complexity
    evidence_score: float = 0.0       # Evidence subtlety
    failure_score: float = 0.0        # Gemini failure severity
    
    # Overall scores
    raw_score: float = 0.0            # Weighted sum before failure boost
    final_score: float = 0.0          # After failure severity boost
    complexity_level: ComplexityLevel = ComplexityLevel.EASY
    
    # Metadata
    question_id: Optional[str] = None
    question_type: Optional[str] = None
    failure_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'question_id': self.question_id,
            'question_type': self.question_type,
            'inference_score': round(self.inference_score, 2),
            'temporal_score': round(self.temporal_score, 2),
            'multimodal_score': round(self.multimodal_score, 2),
            'evidence_score': round(self.evidence_score, 2),
            'failure_score': round(self.failure_score, 2),
            'raw_score': round(self.raw_score, 2),
            'final_score': round(self.final_score, 2),
            'complexity_level': self.complexity_level.value,
            'failure_type': self.failure_type
        }


class DifficultyRanker:
    """
    Ranks questions by difficulty and failure severity
    
    Combines multiple complexity dimensions with Gemini failure impact
    to identify questions that are both hard AND expose failures well.
    """
    
    def __init__(self, config: Optional[DifficultyConfig] = None):
        """
        Initialize difficulty ranker
        
        Args:
            config: Difficulty ranking configuration
        """
        self.config = config or DifficultyConfig()
        self.config.validate()
        
        logger.info("DifficultyRanker initialized")
        logger.info(f"Weights - Inference: {self.config.inference_weight}, "
                   f"Temporal: {self.config.temporal_weight}, "
                   f"Multimodal: {self.config.multimodal_weight}, "
                   f"Evidence: {self.config.evidence_weight}, "
                   f"Failure: {self.config.failure_weight}")
    
    def rank_questions(
        self,
        questions: List[Dict[str, Any]],
        test_results: Optional[List[Dict[str, Any]]] = None,
        hallucination_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[DifficultyScore]:
        """
        Rank questions by difficulty
        
        Args:
            questions: List of question dictionaries with metadata
            test_results: Optional Gemini test results from Block 5
            hallucination_results: Optional hallucination detection results
            
        Returns:
            List of DifficultyScore objects, sorted by final_score (highest first)
        """
        logger.info(f"Ranking {len(questions)} questions by difficulty")
        
        scores = []
        
        # Build lookup maps for test results
        test_map = {}
        hall_map = {}
        
        if test_results:
            test_map = {r.get('question_id'): r for r in test_results}
        
        if hallucination_results:
            hall_map = {r.get('question_id'): r for r in hallucination_results}
        
        # Score each question
        for question in questions:
            score = self._score_question(
                question,
                test_map.get(question.get('question_id')),
                hall_map.get(question.get('question_id'))
            )
            scores.append(score)
        
        # Sort by final score (highest first)
        scores.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"Ranked questions - Complexity breakdown:")
        logger.info(f"  Expert: {sum(1 for s in scores if s.complexity_level == ComplexityLevel.EXPERT)}")
        logger.info(f"  Hard: {sum(1 for s in scores if s.complexity_level == ComplexityLevel.HARD)}")
        logger.info(f"  Medium: {sum(1 for s in scores if s.complexity_level == ComplexityLevel.MEDIUM)}")
        logger.info(f"  Easy: {sum(1 for s in scores if s.complexity_level == ComplexityLevel.EASY)}")
        
        return scores
    
    def _score_question(
        self,
        question: Dict[str, Any],
        test_result: Optional[Dict[str, Any]],
        hall_result: Optional[Dict[str, Any]]
    ) -> DifficultyScore:
        """
        Score a single question across all dimensions
        
        Args:
            question: Question dictionary
            test_result: Gemini test result
            hall_result: Hallucination detection result
            
        Returns:
            DifficultyScore object
        """
        score = DifficultyScore(
            question_id=question.get('question_id'),
            question_type=question.get('question_type')
        )
        
        # Score each dimension
        score.inference_score = self._score_inference_complexity(question)
        score.temporal_score = self._score_temporal_complexity(question)
        score.multimodal_score = self._score_multimodal_complexity(question)
        score.evidence_score = self._score_evidence_subtlety(question)
        score.failure_score = self._score_failure_severity(test_result, hall_result)
        
        # Calculate weighted raw score
        score.raw_score = (
            score.inference_score * self.config.inference_weight +
            score.temporal_score * self.config.temporal_weight +
            score.multimodal_score * self.config.multimodal_weight +
            score.evidence_score * self.config.evidence_weight +
            score.failure_score * self.config.failure_weight
        )
        
        # Apply failure severity boost
        failure_boost = self._get_failure_boost(hall_result)
        score.final_score = score.raw_score * failure_boost
        
        # Classify complexity level
        score.complexity_level = self._classify_complexity(score.final_score)
        
        # Store failure type if available
        if hall_result:
            score.failure_type = hall_result.get('hallucination_type')
        
        return score
    
    def _score_inference_complexity(self, question: Dict[str, Any]) -> float:
        """
        Score inference/reasoning complexity (0-10)
        
        Factors:
        - Inference hops required
        - Abstraction level needed
        - Implicit reasoning
        """
        score = 0.0
        
        # Base score from question type
        question_type = question.get('question_type', '').lower()
        
        type_scores = {
            'counting': 2.0,                    # Direct counting
            'context': 3.0,                     # Context understanding
            'temporal': 5.0,                    # Temporal reasoning
            'sequential': 6.0,                  # Sequence understanding
            'inference': 8.0,                   # Explicit inference
            'comparative': 7.0,                 # Comparison reasoning
            'object_interaction': 6.0,          # Interaction understanding
            'general_holistic': 7.0,            # Holistic reasoning
            'audio_visual_stitching': 8.0,     # Cross-modal inference
            'spurious_correlation': 9.0,        # Anti-correlation reasoning
            'needle': 8.0,                      # Hidden detail detection
            'referential_grounding': 5.0,       # Reference resolution
            'subscene': 6.0                     # Scene-level reasoning
        }
        
        score = type_scores.get(question_type, 5.0)
        
        # Adjust based on question complexity markers
        question_text = question.get('question', '').lower()
        
        if any(word in question_text for word in ['why', 'explain', 'reasoning', 'because']):
            score += 1.0  # Explanation questions are harder
        
        if any(word in question_text for word in ['compare', 'contrast', 'difference', 'similar']):
            score += 1.0  # Comparison adds complexity
        
        if 'not' in question_text or 'except' in question_text:
            score += 0.5  # Negation increases difficulty
        
        # Check for multi-hop indicators in evidence
        evidence = question.get('evidence', {})
        if isinstance(evidence, dict):
            visual_cues = evidence.get('visual_cues', [])
            audio_cues = evidence.get('audio_cues', [])
            
            if len(visual_cues) > 2 or len(audio_cues) > 2:
                score += 1.0  # Multiple evidence pieces = multi-hop
        
        return min(score, 10.0)
    
    def _score_temporal_complexity(self, question: Dict[str, Any]) -> float:
        """
        Score temporal reasoning complexity (0-10)
        
        Factors:
        - Time span covered
        - Number of temporal references
        - Cross-temporal dependencies
        """
        score = 0.0
        
        question_text = question.get('question', '').lower()
        evidence = question.get('evidence', {})
        
        # Temporal keywords
        temporal_keywords = [
            'before', 'after', 'during', 'while', 'when', 'first', 'last',
            'earlier', 'later', 'previous', 'next', 'simultaneous', 'sequence'
        ]
        
        keyword_count = sum(1 for kw in temporal_keywords if kw in question_text)
        score += min(keyword_count * 2.0, 5.0)
        
        # Check timestamps
        if isinstance(evidence, dict):
            visual_cues = evidence.get('visual_cues', [])
            audio_cues = evidence.get('audio_cues', [])
            
            all_cues = visual_cues + audio_cues
            
            # Extract timestamps and calculate span
            timestamps = []
            for cue in all_cues:
                if isinstance(cue, dict) and 'timestamp' in cue:
                    timestamps.append(cue['timestamp'])
            
            if len(timestamps) > 1:
                # Large time span = more complex
                time_span = max(timestamps) - min(timestamps)
                if time_span > 60:  # More than 1 minute
                    score += 3.0
                elif time_span > 30:  # More than 30 seconds
                    score += 2.0
                elif time_span > 10:  # More than 10 seconds
                    score += 1.0
        
        # Boost for temporal question types
        question_type = question.get('question_type', '').lower()
        if question_type in ['temporal', 'sequential']:
            score += 2.0
        
        return min(score, 10.0)
    
    def _score_multimodal_complexity(self, question: Dict[str, Any]) -> float:
        """
        Score multimodal integration complexity (0-10)
        
        Factors:
        - Number of modalities used
        - Degree of cross-modal reasoning
        - Synchronization requirements
        """
        score = 0.0
        
        evidence = question.get('evidence', {})
        
        if not isinstance(evidence, dict):
            return 2.0  # Minimal multimodal if no evidence
        
        visual_cues = evidence.get('visual_cues', [])
        audio_cues = evidence.get('audio_cues', [])
        
        has_visual = len(visual_cues) > 0
        has_audio = len(audio_cues) > 0
        
        if has_visual and has_audio:
            # True multimodal question
            score = 6.0
            
            # More cues = more integration needed
            score += min(len(visual_cues) * 0.5, 2.0)
            score += min(len(audio_cues) * 0.5, 2.0)
        
        elif has_visual or has_audio:
            # Single modality
            score = 3.0
        
        else:
            # No evidence (shouldn't happen)
            score = 1.0
        
        # Boost for audio-visual stitching type
        question_type = question.get('question_type', '').lower()
        if question_type == 'audio_visual_stitching':
            score += 2.0
        
        return min(score, 10.0)
    
    def _score_evidence_subtlety(self, question: Dict[str, Any]) -> float:
        """
        Score how subtle/hidden the evidence is (0-10)
        
        Factors:
        - Evidence prominence in video
        - Number of distractors
        - Requires careful attention
        """
        score = 5.0  # Default moderate subtlety
        
        question_type = question.get('question_type', '').lower()
        
        # Needle-in-haystack questions are inherently subtle
        if question_type == 'needle':
            score = 9.0
        
        elif question_type == 'spurious_correlation':
            score = 8.0  # Requires rejecting obvious but wrong answer
        
        elif question_type == 'subscene':
            score = 7.0  # Scene details are subtle
        
        elif question_type in ['counting', 'context']:
            score = 3.0  # Usually more obvious
        
        # Check evidence descriptions for subtlety indicators
        evidence = question.get('evidence', {})
        if isinstance(evidence, dict):
            all_cues = evidence.get('visual_cues', []) + evidence.get('audio_cues', [])
            
            for cue in all_cues:
                if isinstance(cue, dict):
                    desc = cue.get('description', '').lower()
                    
                    if any(word in desc for word in ['subtle', 'faint', 'brief', 'hidden', 'background']):
                        score += 1.0
                    
                    if any(word in desc for word in ['prominent', 'obvious', 'clear', 'visible']):
                        score -= 0.5
        
        return min(max(score, 0.0), 10.0)
    
    def _score_failure_severity(
        self,
        test_result: Optional[Dict[str, Any]],
        hall_result: Optional[Dict[str, Any]]
    ) -> float:
        """
        Score Gemini failure severity (0-10)
        
        Factors:
        - Hallucination type (critical/major/minor)
        - Confidence of wrong answer
        - Nature of error
        """
        if not hall_result:
            return 0.0  # No failure detected
        
        score = 0.0
        
        # Base score from hallucination type
        hall_type = hall_result.get('hallucination_type', '').lower()
        
        severity_scores = {
            'critical': 10.0,      # Complete fabrication
            'major': 7.0,          # Significant error
            'minor': 4.0,          # Small mistake
            'none': 0.0            # No hallucination
        }
        
        score = severity_scores.get(hall_type, 5.0)
        
        # Increase score if Gemini was confident in wrong answer
        if test_result:
            gemini_answer = test_result.get('gemini_answer', '')
            
            confidence_markers = [
                'definitely', 'certainly', 'clearly', 'obviously',
                'without a doubt', 'absolutely'
            ]
            
            if any(marker in gemini_answer.lower() for marker in confidence_markers):
                score += 1.0  # Confident wrong answer is worse
        
        return min(score, 10.0)
    
    def _get_failure_boost(self, hall_result: Optional[Dict[str, Any]]) -> float:
        """
        Get failure severity multiplier
        
        Critical failures get biggest boost to ensure they're selected
        """
        if not hall_result:
            return 1.0  # No boost if no failure
        
        hall_type = hall_result.get('hallucination_type', '').lower()
        
        if hall_type == 'critical':
            return self.config.critical_failure_boost
        elif hall_type == 'major':
            return self.config.major_failure_boost
        elif hall_type == 'minor':
            return self.config.minor_failure_boost
        else:
            return 1.0
    
    def _classify_complexity(self, score: float) -> ComplexityLevel:
        """Classify complexity level based on final score"""
        if score >= self.config.hard_threshold:
            return ComplexityLevel.EXPERT
        elif score >= self.config.medium_threshold:
            return ComplexityLevel.HARD
        elif score >= self.config.easy_threshold:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.EASY
    
    def get_complexity_distribution(
        self,
        scores: List[DifficultyScore]
    ) -> Dict[str, int]:
        """
        Get distribution of complexity levels
        
        Returns:
            Dictionary mapping complexity level to count
        """
        distribution = {
            ComplexityLevel.EASY.value: 0,
            ComplexityLevel.MEDIUM.value: 0,
            ComplexityLevel.HARD.value: 0,
            ComplexityLevel.EXPERT.value: 0
        }
        
        for score in scores:
            distribution[score.complexity_level.value] += 1
        
        return distribution
