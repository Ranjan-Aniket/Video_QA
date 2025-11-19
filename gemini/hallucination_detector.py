"""
Hallucination Detector - Detect Gemini Failures & Hallucinations

Purpose: Identify when Gemini hallucinates or makes reasoning errors
Compliance: Categorize hallucination types, score severity
Architecture: Multi-layer detection with pattern matching

Hallucination Types:
1. Factual errors (contradicts evidence)
2. Invented details (adds information not in evidence)
3. Misidentification (wrong objects, people, actions)
4. Temporal errors (wrong sequence, timing)
5. Logical errors (faulty reasoning)
"""

# Standard library imports
import logging
import re
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HallucinationType(Enum):
    """Types of hallucinations"""
    FACTUAL_ERROR = "factual_error"  # Contradicts evidence
    INVENTED_DETAIL = "invented_detail"  # Adds non-existent info
    MISIDENTIFICATION = "misidentification"  # Wrong object/person/action
    TEMPORAL_ERROR = "temporal_error"  # Wrong timing/sequence
    LOGICAL_ERROR = "logical_error"  # Faulty reasoning
    CONFABULATION = "confabulation"  # Makes up plausible but wrong answer
    NONE = "none"  # No hallucination detected


@dataclass
class DetectionConfig:
    """Configuration for hallucination detection"""
    # Detection thresholds
    min_confidence: float = 0.7  # Minimum confidence to flag hallucination
    max_added_words: int = 10  # Max words added before flagging invention
    
    # Evidence checking
    require_evidence_support: bool = True
    check_temporal_consistency: bool = True
    check_logical_consistency: bool = True
    
    # Name blocking (per guidelines)
    block_names: bool = True  # Flag if names used instead of descriptors
    blocked_name_patterns: List[str] = field(default_factory=lambda: [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
        r'\b(he|she|him|her|his|hers)\b',  # Pronouns
    ])


@dataclass
class HallucinationEvidence:
    """Evidence of hallucination"""
    type: HallucinationType
    description: str
    confidence: float  # 0.0 to 1.0
    supporting_text: Optional[str] = None  # Text that shows hallucination


@dataclass
class HallucinationResult:
    """Result of hallucination detection"""
    has_hallucination: bool
    hallucinations: List[HallucinationEvidence] = field(default_factory=list)
    
    # Overall score
    score: float = 0.0  # 0.0 (no hallucination) to 1.0 (severe hallucination)
    
    # Categorization
    primary_type: Optional[HallucinationType] = None
    severity: str = "none"  # none, low, medium, high
    
    # Details
    added_content: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    missing_content: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "has_hallucination": self.has_hallucination,
            "score": self.score,
            "primary_type": self.primary_type.value if self.primary_type else None,
            "severity": self.severity,
            "hallucinations": [
                {
                    "type": h.type.value,
                    "description": h.description,
                    "confidence": h.confidence
                }
                for h in self.hallucinations
            ],
            "added_content": self.added_content,
            "contradictions": self.contradictions,
            "missing_content": self.missing_content
        }


class HallucinationDetector:
    """
    Detect hallucinations in Gemini's answers.
    
    Uses multiple detection strategies:
    1. Evidence comparison (what's added/missing/contradicted)
    2. Pattern matching (common hallucination patterns)
    3. Consistency checking (temporal, logical)
    4. Name blocking (per guidelines)
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize hallucination detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config or DetectionConfig()
        
        logger.info("HallucinationDetector initialized")
    
    def detect_hallucination(
        self,
        question: str,
        golden_answer: str,
        gemini_answer: str,
        evidence: Optional[str] = None
    ) -> HallucinationResult:
        """
        Detect hallucinations in Gemini's answer.
        
        Args:
            question: Question text
            golden_answer: Expected correct answer
            gemini_answer: Gemini's actual answer
            evidence: Optional evidence/context
        
        Returns:
            HallucinationResult with detection details
        """
        logger.debug(f"Detecting hallucinations in answer: {gemini_answer[:50]}...")
        
        hallucinations = []
        
        # 1. Check for invented details
        invented = self._detect_invented_details(
            gemini_answer, golden_answer, evidence
        )
        if invented:
            hallucinations.append(invented)
        
        # 2. Check for factual contradictions
        contradictions = self._detect_contradictions(
            gemini_answer, golden_answer, evidence
        )
        if contradictions:
            hallucinations.append(contradictions)
        
        # 3. Check for misidentification
        misid = self._detect_misidentification(
            gemini_answer, golden_answer
        )
        if misid:
            hallucinations.append(misid)
        
        # 4. Check for temporal errors
        if self.config.check_temporal_consistency:
            temporal = self._detect_temporal_errors(
                question, gemini_answer, golden_answer
            )
            if temporal:
                hallucinations.append(temporal)
        
        # 5. Check for name usage (per guidelines)
        if self.config.block_names:
            names = self._detect_name_usage(gemini_answer)
            if names:
                hallucinations.append(names)
        
        # 6. Calculate content differences
        added, missing = self._calculate_content_diff(
            golden_answer, gemini_answer
        )
        
        # Calculate overall score
        score = self._calculate_hallucination_score(hallucinations, added)
        
        # Determine primary type and severity
        primary_type = None
        if hallucinations:
            # Most confident hallucination
            primary = max(hallucinations, key=lambda h: h.confidence)
            primary_type = primary.type
        
        severity = self._determine_severity(score)
        
        # Create result
        result = HallucinationResult(
            has_hallucination=len(hallucinations) > 0 or score > 0.3,
            hallucinations=hallucinations,
            score=score,
            primary_type=primary_type,
            severity=severity,
            added_content=added,
            missing_content=missing
        )
        
        logger.debug(
            f"Hallucination detection complete: "
            f"{len(hallucinations)} issues, score: {score:.2f}"
        )
        
        return result
    
    def _detect_invented_details(
        self,
        gemini_answer: str,
        golden_answer: str,
        evidence: Optional[str]
    ) -> Optional[HallucinationEvidence]:
        """Detect invented/added details not in evidence"""
        # Get tokens in Gemini's answer but not in golden answer or evidence
        gemini_tokens = set(gemini_answer.lower().split())
        golden_tokens = set(golden_answer.lower().split())
        evidence_tokens = set(evidence.lower().split()) if evidence else set()
        
        # Tokens added by Gemini
        added = gemini_tokens - golden_tokens - evidence_tokens
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        added = added - stop_words
        
        if len(added) > self.config.max_added_words:
            confidence = min(1.0, len(added) / 20.0)  # Scale by amount added
            
            return HallucinationEvidence(
                type=HallucinationType.INVENTED_DETAIL,
                description=f"Added {len(added)} details not in evidence",
                confidence=confidence,
                supporting_text=f"Added words: {list(added)[:5]}"
            )
        
        return None
    
    def _detect_contradictions(
        self,
        gemini_answer: str,
        golden_answer: str,
        evidence: Optional[str]
    ) -> Optional[HallucinationEvidence]:
        """Detect factual contradictions"""
        # Simple implementation - check for opposite/contradicting terms
        
        # Common contradiction patterns
        opposites = [
            ('yes', 'no'),
            ('true', 'false'),
            ('left', 'right'),
            ('red', 'blue'),
            ('before', 'after'),
            ('inside', 'outside'),
        ]
        
        gemini_lower = gemini_answer.lower()
        golden_lower = golden_answer.lower()
        
        for word1, word2 in opposites:
            if (word1 in golden_lower and word2 in gemini_lower) or \
               (word2 in golden_lower and word1 in gemini_lower):
                return HallucinationEvidence(
                    type=HallucinationType.FACTUAL_ERROR,
                    description=f"Contradicts golden answer ({word1} vs {word2})",
                    confidence=0.8,
                    supporting_text=f"Golden: {word1}, Gemini: {word2}"
                )
        
        return None
    
    def _detect_misidentification(
        self,
        gemini_answer: str,
        golden_answer: str
    ) -> Optional[HallucinationEvidence]:
        """Detect object/person/action misidentification"""
        # Extract nouns/entities (simple approach)
        # In production, use NER (Named Entity Recognition)
        
        # For now, check for common object misidentifications
        gemini_lower = gemini_answer.lower()
        golden_lower = golden_answer.lower()
        
        # Color misidentification
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        gemini_colors = [c for c in colors if c in gemini_lower]
        golden_colors = [c for c in colors if c in golden_lower]
        
        if gemini_colors and golden_colors and gemini_colors != golden_colors:
            return HallucinationEvidence(
                type=HallucinationType.MISIDENTIFICATION,
                description="Color misidentification",
                confidence=0.9,
                supporting_text=f"Golden: {golden_colors}, Gemini: {gemini_colors}"
            )
        
        return None
    
    def _detect_temporal_errors(
        self,
        question: str,
        gemini_answer: str,
        golden_answer: str
    ) -> Optional[HallucinationEvidence]:
        """Detect temporal/sequence errors"""
        # Check for temporal keywords
        temporal_keywords = [
            'before', 'after', 'first', 'then', 'next', 'finally',
            'earlier', 'later', 'beginning', 'end'
        ]
        
        question_lower = question.lower()
        
        # Only check if question involves temporal reasoning
        if not any(kw in question_lower for kw in temporal_keywords):
            return None
        
        gemini_lower = gemini_answer.lower()
        golden_lower = golden_answer.lower()
        
        # Check for sequence reversal
        if 'before' in golden_lower and 'after' in gemini_lower:
            return HallucinationEvidence(
                type=HallucinationType.TEMPORAL_ERROR,
                description="Temporal sequence error (before/after reversal)",
                confidence=0.8
            )
        
        return None
    
    def _detect_name_usage(
        self, gemini_answer: str
    ) -> Optional[HallucinationEvidence]:
        """Detect name usage (should use descriptors per guidelines)"""
        # Check for person names and pronouns
        for pattern in self.config.blocked_name_patterns:
            matches = re.findall(pattern, gemini_answer)
            
            if matches:
                return HallucinationEvidence(
                    type=HallucinationType.FACTUAL_ERROR,
                    description="Uses names/pronouns instead of descriptors",
                    confidence=0.7,
                    supporting_text=f"Found: {matches[:3]}"
                )
        
        return None
    
    def _calculate_content_diff(
        self,
        golden_answer: str,
        gemini_answer: str
    ) -> tuple[List[str], List[str]]:
        """
        Calculate content differences.
        
        Returns:
            (added_content, missing_content)
        """
        golden_tokens = set(golden_answer.lower().split())
        gemini_tokens = set(gemini_answer.lower().split())
        
        added = list(gemini_tokens - golden_tokens)
        missing = list(golden_tokens - gemini_tokens)
        
        return added, missing
    
    def _calculate_hallucination_score(
        self,
        hallucinations: List[HallucinationEvidence],
        added_content: List[str]
    ) -> float:
        """
        Calculate overall hallucination score.
        
        Returns:
            Score from 0.0 (no hallucination) to 1.0 (severe)
        """
        if not hallucinations and not added_content:
            return 0.0
        
        # Base score from detected hallucinations
        if hallucinations:
            # Weighted average of confidence scores
            total_confidence = sum(h.confidence for h in hallucinations)
            avg_confidence = total_confidence / len(hallucinations)
            base_score = avg_confidence
        else:
            base_score = 0.0
        
        # Penalty for added content
        added_penalty = min(0.3, len(added_content) / 50.0)
        
        # Combine scores
        score = min(1.0, base_score + added_penalty)
        
        return score
    
    def _determine_severity(self, score: float) -> str:
        """Determine severity level from score"""
        if score < 0.3:
            return "none"
        elif score < 0.5:
            return "low"
        elif score < 0.7:
            return "medium"
        else:
            return "high"
    
    def analyze_hallucination_patterns(
        self, results: List[HallucinationResult]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in hallucinations across multiple results.
        
        Args:
            results: List of HallucinationResults to analyze
        
        Returns:
            Dict with pattern analysis
        """
        if not results:
            return {}
        
        # Count by type
        type_counts = {}
        for result in results:
            for h in result.hallucinations:
                type_name = h.type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Severity distribution
        severity_counts = {}
        for result in results:
            severity_counts[result.severity] = \
                severity_counts.get(result.severity, 0) + 1
        
        # Average score
        avg_score = sum(r.score for r in results) / len(results)
        
        # Most common type
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] \
            if type_counts else None
        
        return {
            "total_analyzed": len(results),
            "hallucination_rate": sum(1 for r in results if r.has_hallucination) / len(results),
            "avg_score": avg_score,
            "type_counts": type_counts,
            "severity_counts": severity_counts,
            "most_common_type": most_common_type
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = HallucinationDetector(
        config=DetectionConfig(
            block_names=True,
            check_temporal_consistency=True
        )
    )
    
    # Example 1: Factual error
    result1 = detector.detect_hallucination(
        question="What color is the person's shirt?",
        golden_answer="The person is wearing a red shirt",
        gemini_answer="The person is wearing a blue shirt",
        evidence="A person wearing a red shirt walks across the frame"
    )
    
    print(f"✅ Example 1 - Factual error:")
    print(f"   Has hallucination: {result1.has_hallucination}")
    print(f"   Score: {result1.score:.2f}")
    print(f"   Severity: {result1.severity}")
    print(f"   Type: {result1.primary_type.value if result1.primary_type else 'None'}")
    
    # Example 2: Invented details
    result2 = detector.detect_hallucination(
        question="What happens in the video?",
        golden_answer="Person walks",
        gemini_answer="A tall person with curly hair wearing sunglasses walks confidently",
        evidence="Person walks across frame"
    )
    
    print(f"\n✅ Example 2 - Invented details:")
    print(f"   Has hallucination: {result2.has_hallucination}")
    print(f"   Score: {result2.score:.2f}")
    print(f"   Added content: {result2.added_content[:5]}")
