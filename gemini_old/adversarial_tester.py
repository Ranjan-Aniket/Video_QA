"""
Adversarial Tester - Test Q&A Pairs Against Gemini

Purpose: Test generated questions against Gemini to find adversarial examples
Compliance: Track success/failure rates, identify hallucination patterns
Architecture: Batch testing with cost optimization

Testing Strategy:
1. Send question + evidence to Gemini
2. Compare Gemini's answer with golden answer
3. Detect hallucinations and errors
4. Track success rate (target: <50% for good adversarial questions)
"""

# Standard library imports
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from datetime import datetime

# Internal imports
from .gemini_client import GeminiClient, GeminiResponse, GeminiConfig
from .hallucination_detector import HallucinationDetector, HallucinationResult

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test result status"""
    PASS = "pass"  # Gemini answered correctly
    FAIL = "fail"  # Gemini answered incorrectly
    HALLUCINATION = "hallucination"  # Gemini hallucinated
    BLOCKED = "blocked"  # Blocked by safety filters
    ERROR = "error"  # API error


@dataclass
class TestConfig:
    """Configuration for adversarial testing"""
    # Gemini configuration
    gemini_config: GeminiConfig = field(default_factory=GeminiConfig)
    
    # Testing parameters
    include_evidence: bool = True  # Provide evidence to Gemini
    include_visual_cues: bool = True  # Provide image evidence
    
    # Success thresholds
    max_success_rate: float = 0.50  # Good adversarial Q has <50% success rate
    min_difficulty: float = 0.70  # Minimum difficulty score
    
    # Cost limits
    max_cost_per_video: float = 0.30  # $0.30 for testing 30 questions
    max_cost_per_question: float = 0.01  # $0.01 per question


@dataclass
class TestResult:
    """Result of testing a single Q&A pair"""
    question_id: str
    question: str
    golden_answer: str
    gemini_answer: str
    
    # Test outcome
    status: TestStatus
    is_correct: bool
    
    # Hallucination detection
    hallucination: Optional[HallucinationResult] = None
    
    # Similarity metrics
    similarity_score: float = 0.0  # 0.0 to 1.0
    
    # Gemini response metadata
    gemini_response: Optional[GeminiResponse] = None
    
    # Timing and costs
    test_time: float = 0.0
    test_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "golden_answer": self.golden_answer,
            "gemini_answer": self.gemini_answer,
            "status": self.status.value,
            "is_correct": self.is_correct,
            "hallucination": (
                self.hallucination.to_dict() 
                if self.hallucination else None
            ),
            "similarity_score": self.similarity_score,
            "test_time": self.test_time,
            "test_cost": self.test_cost
        }


class AdversarialTester:
    """
    Test Q&A pairs against Gemini to find adversarial examples.
    
    Good adversarial questions:
    - Gemini answers incorrectly (<50% success rate)
    - Exposes specific failure modes (hallucinations, reasoning errors)
    - Tests challenging aspects (temporal, multi-hop, counterfactual)
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[TestConfig] = None
    ):
        """
        Initialize adversarial tester.
        
        Args:
            api_key: Google AI API key
            config: Testing configuration
        """
        self.config = config or TestConfig()
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient(
            api_key=api_key,
            config=self.config.gemini_config
        )
        
        # Initialize hallucination detector
        self.hallucination_detector = HallucinationDetector()
        
        # Tracking
        self.total_tests = 0
        self.total_cost = 0.0
        self.test_results: List[TestResult] = []
        
        logger.info("AdversarialTester initialized")
    
    def test_question(
        self,
        question_id: str,
        question: str,
        golden_answer: str,
        evidence: Optional[str] = None,
        image_path: Optional[Path] = None
    ) -> TestResult:
        """
        Test a single Q&A pair against Gemini.
        
        Args:
            question_id: Unique question identifier
            question: Question text
            golden_answer: Expected correct answer
            evidence: Optional evidence/context
            image_path: Optional image evidence
        
        Returns:
            TestResult with outcome and analysis
        """
        logger.info(f"Testing question: {question_id}")
        start_time = time.time()
        
        try:
            # Prepare evidence if configured
            test_evidence = evidence if self.config.include_evidence else None
            test_image = image_path if self.config.include_visual_cues else None
            
            # Get Gemini's answer
            gemini_response = self.gemini_client.test_qa_pair(
                question=question,
                golden_answer=golden_answer,
                evidence=test_evidence,
                image_path=test_image
            )
            
            # Check if blocked
            if gemini_response.blocked:
                return self._create_blocked_result(
                    question_id, question, golden_answer,
                    gemini_response, time.time() - start_time
                )
            
            gemini_answer = gemini_response.answer
            
            # Calculate similarity
            similarity = self._calculate_similarity(
                golden_answer, gemini_answer
            )
            
            # Detect hallucinations
            hallucination = self.hallucination_detector.detect_hallucination(
                question=question,
                golden_answer=golden_answer,
                gemini_answer=gemini_answer,
                evidence=evidence
            )
            
            # Determine test outcome
            is_correct = similarity > 0.7 or hallucination.score < 0.3
            
            if hallucination.has_hallucination:
                status = TestStatus.HALLUCINATION
            elif is_correct:
                status = TestStatus.PASS
            else:
                status = TestStatus.FAIL
            
            # Create result
            test_time = time.time() - start_time
            
            result = TestResult(
                question_id=question_id,
                question=question,
                golden_answer=golden_answer,
                gemini_answer=gemini_answer,
                status=status,
                is_correct=is_correct,
                hallucination=hallucination,
                similarity_score=similarity,
                gemini_response=gemini_response,
                test_time=test_time,
                test_cost=gemini_response.total_cost
            )
            
            # Update tracking
            self.total_tests += 1
            self.total_cost += gemini_response.total_cost
            self.test_results.append(result)
            
            logger.info(
                f"Test complete: {status.value} "
                f"(similarity: {similarity:.2f}, cost: ${gemini_response.total_cost:.4f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing question {question_id}: {e}")
            
            # Create error result
            return TestResult(
                question_id=question_id,
                question=question,
                golden_answer=golden_answer,
                gemini_answer=f"[ERROR: {str(e)}]",
                status=TestStatus.ERROR,
                is_correct=False,
                test_time=time.time() - start_time,
                test_cost=0.0
            )
    
    def test_batch(
        self,
        qa_pairs: List[Dict[str, Any]],
        evidence_dict: Optional[Dict[str, str]] = None,
        image_dict: Optional[Dict[str, Path]] = None
    ) -> List[TestResult]:
        """
        Test multiple Q&A pairs in batch.
        
        Args:
            qa_pairs: List of dicts with keys: question_id, question, golden_answer
            evidence_dict: Optional dict mapping question_id to evidence
            image_dict: Optional dict mapping question_id to image path
        
        Returns:
            List of TestResults
        """
        logger.info(f"Testing batch of {len(qa_pairs)} questions")
        
        results = []
        
        for qa in qa_pairs:
            # Check cost limit
            if self.total_cost >= self.config.max_cost_per_video:
                logger.warning(
                    f"Cost limit reached: ${self.total_cost:.4f} >= "
                    f"${self.config.max_cost_per_video:.2f}"
                )
                break
            
            question_id = qa['question_id']
            
            # Get optional evidence and image
            evidence = evidence_dict.get(question_id) if evidence_dict else None
            image_path = image_dict.get(question_id) if image_dict else None
            
            # Test question
            result = self.test_question(
                question_id=question_id,
                question=qa['question'],
                golden_answer=qa['golden_answer'],
                evidence=evidence,
                image_path=image_path
            )
            
            results.append(result)
        
        logger.info(
            f"Batch testing complete: {len(results)} questions, "
            f"${self.total_cost:.4f} cost"
        )
        
        return results
    
    def _calculate_similarity(
        self, answer1: str, answer2: str
    ) -> float:
        """
        Calculate similarity between two answers.
        
        Simple implementation - can be enhanced with:
        - Semantic similarity (sentence transformers)
        - Edit distance
        - Token overlap
        
        Returns:
            Similarity score 0.0 to 1.0
        """
        # Normalize
        a1 = answer1.lower().strip()
        a2 = answer2.lower().strip()
        
        # Exact match
        if a1 == a2:
            return 1.0
        
        # Token overlap (simple Jaccard similarity)
        tokens1 = set(a1.split())
        tokens2 = set(a2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        jaccard = len(intersection) / len(union)
        
        return jaccard
    
    def _create_blocked_result(
        self,
        question_id: str,
        question: str,
        golden_answer: str,
        gemini_response: GeminiResponse,
        test_time: float
    ) -> TestResult:
        """Create result for blocked response"""
        return TestResult(
            question_id=question_id,
            question=question,
            golden_answer=golden_answer,
            gemini_answer="[BLOCKED BY SAFETY FILTERS]",
            status=TestStatus.BLOCKED,
            is_correct=False,
            gemini_response=gemini_response,
            test_time=test_time,
            test_cost=gemini_response.total_cost
        )
    
    def get_adversarial_questions(
        self,
        max_success_rate: Optional[float] = None
    ) -> List[TestResult]:
        """
        Get questions that are good adversarial examples.
        
        Args:
            max_success_rate: Maximum success rate for adversarial questions
        
        Returns:
            List of TestResults for adversarial questions
        """
        threshold = max_success_rate or self.config.max_success_rate
        
        # Get failed and hallucination results
        adversarial = [
            r for r in self.test_results
            if r.status in [TestStatus.FAIL, TestStatus.HALLUCINATION]
        ]
        
        logger.info(
            f"Found {len(adversarial)} adversarial questions "
            f"({len(adversarial)/len(self.test_results)*100:.1f}% failure rate)"
        )
        
        return adversarial
    
    def get_success_rate(self) -> float:
        """Get overall success rate"""
        if not self.test_results:
            return 0.0
        
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASS)
        return passed / len(self.test_results)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get testing statistics"""
        if not self.test_results:
            return {
                "total_tests": 0,
                "success_rate": 0.0,
                "hallucination_rate": 0.0,
                "total_cost": 0.0
            }
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAIL)
        hallucinated = sum(1 for r in self.test_results if r.status == TestStatus.HALLUCINATION)
        blocked = sum(1 for r in self.test_results if r.status == TestStatus.BLOCKED)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "hallucinated": hallucinated,
            "blocked": blocked,
            "success_rate": passed / total,
            "failure_rate": (failed + hallucinated) / total,
            "hallucination_rate": hallucinated / total,
            "total_cost": self.total_cost,
            "avg_cost_per_test": self.total_cost / total
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tester
    tester = AdversarialTester(
        api_key="YOUR_API_KEY",
        config=TestConfig(
            include_evidence=True,
            max_cost_per_video=0.30
        )
    )
    
    # Test single question
    result = tester.test_question(
        question_id="q1",
        question="What color is the person's shirt?",
        golden_answer="Red",
        evidence="A person wearing a red shirt walks across the frame."
    )
    
    print(f"✅ Test result:")
    print(f"   Status: {result.status.value}")
    print(f"   Correct: {result.is_correct}")
    print(f"   Similarity: {result.similarity_score:.2f}")
    print(f"   Gemini answer: {result.gemini_answer}")
    
    # Test batch
    qa_pairs = [
        {
            "question_id": "q1",
            "question": "What happens at the beginning?",
            "golden_answer": "Person enters room"
        },
        {
            "question_id": "q2",
            "question": "What color is the wall?",
            "golden_answer": "Blue"
        }
    ]
    
    # results = tester.test_batch(qa_pairs)
    
    # Get stats
    stats = tester.get_stats()
    print(f"\n✅ Testing stats:")
    print(f"   Tests: {stats['total_tests']}")
    print(f"   Success rate: {stats['success_rate']*100:.1f}%")
    print(f"   Hallucination rate: {stats['hallucination_rate']*100:.1f}%")
    print(f"   Total cost: ${stats['total_cost']:.4f}")
