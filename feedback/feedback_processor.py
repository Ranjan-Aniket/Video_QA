"""
Feedback Processor - Process Validation Results & Testing Outcomes

Purpose: Aggregate validation results and prepare feedback for learning
Compliance: Track success rates, identify failure patterns, measure quality
Architecture: Centralized feedback collection and processing

Feedback Sources:
1. Validation results (10-layer quality checks from Block 3)
2. Gemini test results (adversarial testing from Block 5)
3. Hallucination detection (failure classification from Block 5)
4. Selection metrics (diversity and difficulty from Block 6)
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationOutcome(str, Enum):
    """Validation result classification"""
    PASSED = "passed"              # Question passed all checks
    FAILED = "failed"              # Question failed validation
    WARNING = "warning"            # Question has warnings but passed
    SKIPPED = "skipped"            # Question was skipped


@dataclass
class FeedbackConfig:
    """Configuration for feedback processing"""
    
    # Success thresholds
    min_pass_rate: float = 0.90          # Minimum 90% pass rate
    min_gemini_fail_rate: float = 0.30   # Target 30%+ Gemini failures
    
    # Quality metrics
    target_hallucination_rate: float = 0.001  # 0.1% hallucination rate (99.9% clean)
    max_retry_rate: float = 0.20         # Max 20% questions needing retries
    
    # Pattern detection
    min_pattern_frequency: int = 3       # Min occurrences to detect pattern
    pattern_confidence_threshold: float = 0.7  # Min confidence for pattern
    
    # Storage
    store_all_feedback: bool = True      # Store all feedback (not just failures)
    feedback_history_path: Optional[Path] = None  # Path to store history


@dataclass
class FeedbackResult:
    """Aggregated feedback from validation and testing"""
    
    # Overall metrics
    total_questions: int = 0
    passed_validation: int = 0
    failed_validation: int = 0
    warnings: int = 0
    
    # Validation breakdown
    validation_failures_by_layer: Dict[str, int] = field(default_factory=dict)
    validation_pass_rate: float = 0.0
    
    # Gemini testing metrics
    gemini_tests_run: int = 0
    gemini_failures: int = 0
    gemini_fail_rate: float = 0.0
    
    # Hallucination metrics
    critical_hallucinations: int = 0
    major_hallucinations: int = 0
    minor_hallucinations: int = 0
    hallucination_rate: float = 0.0
    
    # Question type performance
    performance_by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Retry statistics
    questions_with_retries: int = 0
    total_retries: int = 0
    avg_retries_per_question: float = 0.0
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Raw feedback items
    feedback_items: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_questions': self.total_questions,
            'passed_validation': self.passed_validation,
            'failed_validation': self.failed_validation,
            'warnings': self.warnings,
            'validation_pass_rate': round(self.validation_pass_rate, 3),
            'validation_failures_by_layer': self.validation_failures_by_layer,
            'gemini_tests_run': self.gemini_tests_run,
            'gemini_failures': self.gemini_failures,
            'gemini_fail_rate': round(self.gemini_fail_rate, 3),
            'critical_hallucinations': self.critical_hallucinations,
            'major_hallucinations': self.major_hallucinations,
            'minor_hallucinations': self.minor_hallucinations,
            'hallucination_rate': round(self.hallucination_rate, 3),
            'performance_by_type': self.performance_by_type,
            'questions_with_retries': self.questions_with_retries,
            'total_retries': self.total_retries,
            'avg_retries_per_question': round(self.avg_retries_per_question, 2),
            'timestamp': self.timestamp
        }


class FeedbackProcessor:
    """
    Processes validation results and testing outcomes
    
    Aggregates feedback from multiple sources to provide insights into:
    - Question quality (validation pass rates)
    - Gemini failure rates (adversarial effectiveness)
    - Hallucination rates (our question quality)
    - Performance by question type
    """
    
    def __init__(self, config: Optional[FeedbackConfig] = None):
        """
        Initialize feedback processor
        
        Args:
            config: Feedback processing configuration
        """
        self.config = config or FeedbackConfig()
        
        # Feedback storage
        self.feedback_history: List[FeedbackResult] = []
        
        logger.info("FeedbackProcessor initialized")
        logger.info(f"Target pass rate: {self.config.min_pass_rate:.1%}")
        logger.info(f"Target Gemini fail rate: {self.config.min_gemini_fail_rate:.1%}")
        logger.info(f"Target hallucination rate: {self.config.target_hallucination_rate:.2%}")
    
    def process_feedback(
        self,
        questions: List[Dict[str, Any]],
        validation_results: Optional[List[Dict[str, Any]]] = None,
        test_results: Optional[List[Dict[str, Any]]] = None,
        hallucination_results: Optional[List[Dict[str, Any]]] = None,
        selection_results: Optional[List[Dict[str, Any]]] = None
    ) -> FeedbackResult:
        """
        Process all feedback sources and generate aggregated result
        
        Args:
            questions: Generated questions
            validation_results: Validation outcomes from Block 3
            test_results: Gemini test results from Block 5
            hallucination_results: Hallucination detection from Block 5
            selection_results: Selected questions from Block 6
            
        Returns:
            FeedbackResult with aggregated metrics
        """
        logger.info(f"Processing feedback for {len(questions)} questions")
        
        result = FeedbackResult()
        result.total_questions = len(questions)
        
        # Build lookup maps
        question_map = {q.get('question_id'): q for q in questions}
        val_map = {}
        test_map = {}
        hall_map = {}
        
        if validation_results:
            val_map = {v.get('question_id'): v for v in validation_results}
        
        if test_results:
            test_map = {t.get('question_id'): t for t in test_results}
        
        if hallucination_results:
            hall_map = {h.get('question_id'): h for h in hallucination_results}
        
        # Process each question
        for question in questions:
            qid = question.get('question_id')
            
            feedback_item = self._process_single_question(
                question,
                val_map.get(qid),
                test_map.get(qid),
                hall_map.get(qid)
            )
            
            result.feedback_items.append(feedback_item)
            
            # Update aggregates
            self._update_aggregates(result, feedback_item)
        
        # Calculate rates
        result.validation_pass_rate = (
            result.passed_validation / result.total_questions
            if result.total_questions > 0 else 0.0
        )
        
        result.gemini_fail_rate = (
            result.gemini_failures / result.gemini_tests_run
            if result.gemini_tests_run > 0 else 0.0
        )
        
        total_hallucinations = (
            result.critical_hallucinations +
            result.major_hallucinations +
            result.minor_hallucinations
        )
        result.hallucination_rate = (
            total_hallucinations / result.total_questions
            if result.total_questions > 0 else 0.0
        )
        
        result.avg_retries_per_question = (
            result.total_retries / result.questions_with_retries
            if result.questions_with_retries > 0 else 0.0
        )
        
        # Analyze performance by type
        result.performance_by_type = self._analyze_by_type(result.feedback_items)
        
        # Store in history
        if self.config.store_all_feedback:
            self.feedback_history.append(result)
        
        # Log summary
        self._log_summary(result)
        
        return result
    
    def _process_single_question(
        self,
        question: Dict[str, Any],
        validation: Optional[Dict[str, Any]],
        test: Optional[Dict[str, Any]],
        hallucination: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process feedback for a single question"""
        
        feedback = {
            'question_id': question.get('question_id'),
            'question_type': question.get('question_type'),
            'tier': question.get('tier', 'unknown'),
        }
        
        # Validation outcome
        if validation:
            feedback['validation_outcome'] = validation.get('outcome', ValidationOutcome.SKIPPED.value)
            feedback['validation_passed'] = validation.get('passed', False)
            feedback['validation_failures'] = validation.get('failed_layers', [])
            feedback['validation_warnings'] = validation.get('warnings', [])
            feedback['retries'] = validation.get('retries', 0)
        else:
            feedback['validation_outcome'] = ValidationOutcome.SKIPPED.value
            feedback['validation_passed'] = False
            feedback['validation_failures'] = []
            feedback['validation_warnings'] = []
            feedback['retries'] = 0
        
        # Gemini test outcome
        if test:
            feedback['gemini_tested'] = True
            feedback['gemini_passed'] = test.get('status') == 'passed'
            feedback['gemini_model'] = test.get('model', 'unknown')
        else:
            feedback['gemini_tested'] = False
            feedback['gemini_passed'] = False
            feedback['gemini_model'] = None
        
        # Hallucination detection
        if hallucination:
            feedback['hallucination_type'] = hallucination.get('hallucination_type', 'none')
            feedback['hallucination_detected'] = hallucination.get('hallucination_type') not in ['none', None]
        else:
            feedback['hallucination_type'] = 'none'
            feedback['hallucination_detected'] = False
        
        return feedback
    
    def _update_aggregates(self, result: FeedbackResult, feedback: Dict[str, Any]) -> None:
        """Update aggregate metrics with feedback item"""
        
        # Validation metrics
        if feedback['validation_passed']:
            result.passed_validation += 1
        else:
            result.failed_validation += 1
        
        if feedback['validation_warnings']:
            result.warnings += 1
        
        # Track validation failures by layer
        for layer in feedback['validation_failures']:
            result.validation_failures_by_layer[layer] = (
                result.validation_failures_by_layer.get(layer, 0) + 1
            )
        
        # Gemini testing metrics
        if feedback['gemini_tested']:
            result.gemini_tests_run += 1
            if not feedback['gemini_passed']:
                result.gemini_failures += 1
        
        # Hallucination metrics
        hall_type = feedback.get('hallucination_type', 'none').lower()
        if hall_type == 'critical':
            result.critical_hallucinations += 1
        elif hall_type == 'major':
            result.major_hallucinations += 1
        elif hall_type == 'minor':
            result.minor_hallucinations += 1
        
        # Retry metrics
        retries = feedback.get('retries', 0)
        if retries > 0:
            result.questions_with_retries += 1
            result.total_retries += retries
    
    def _analyze_by_type(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance grouped by question type"""
        
        type_stats = defaultdict(lambda: {
            'total': 0,
            'validation_passed': 0,
            'gemini_tested': 0,
            'gemini_failed': 0,
            'hallucinations': 0
        })
        
        for item in feedback_items:
            qtype = item.get('question_type', 'unknown')
            stats = type_stats[qtype]
            
            stats['total'] += 1
            
            if item['validation_passed']:
                stats['validation_passed'] += 1
            
            if item['gemini_tested']:
                stats['gemini_tested'] += 1
                if not item['gemini_passed']:
                    stats['gemini_failed'] += 1
            
            if item['hallucination_detected']:
                stats['hallucinations'] += 1
        
        # Calculate rates
        performance = {}
        for qtype, stats in type_stats.items():
            performance[qtype] = {
                'total': stats['total'],
                'validation_pass_rate': stats['validation_passed'] / stats['total'],
                'gemini_fail_rate': (
                    stats['gemini_failed'] / stats['gemini_tested']
                    if stats['gemini_tested'] > 0 else 0.0
                ),
                'hallucination_rate': stats['hallucinations'] / stats['total']
            }
        
        return performance
    
    def _log_summary(self, result: FeedbackResult) -> None:
        """Log feedback summary"""
        
        logger.info("="*70)
        logger.info("FEEDBACK SUMMARY")
        logger.info("="*70)
        
        logger.info(f"Total Questions: {result.total_questions}")
        logger.info(f"Validation Pass Rate: {result.validation_pass_rate:.1%} "
                   f"(target: {self.config.min_pass_rate:.1%})")
        
        logger.info(f"Gemini Fail Rate: {result.gemini_fail_rate:.1%} "
                   f"(target: {self.config.min_gemini_fail_rate:.1%})")
        
        logger.info(f"Hallucination Rate: {result.hallucination_rate:.2%} "
                   f"(target: <{self.config.target_hallucination_rate:.2%})")
        
        # Check if targets met
        targets_met = []
        if result.validation_pass_rate >= self.config.min_pass_rate:
            targets_met.append("✅ Validation")
        else:
            targets_met.append("❌ Validation")
        
        if result.gemini_fail_rate >= self.config.min_gemini_fail_rate:
            targets_met.append("✅ Gemini Failures")
        else:
            targets_met.append("⚠️  Gemini Failures (too low)")
        
        if result.hallucination_rate <= self.config.target_hallucination_rate:
            targets_met.append("✅ Hallucination Rate")
        else:
            targets_met.append("❌ Hallucination Rate (too high)")
        
        logger.info(f"Targets: {', '.join(targets_met)}")
        logger.info("="*70)
    
    def get_insights(self, result: FeedbackResult) -> Dict[str, Any]:
        """
        Generate actionable insights from feedback
        
        Args:
            result: FeedbackResult to analyze
            
        Returns:
            Dictionary of insights and recommendations
        """
        insights = {
            'targets_met': {},
            'issues': [],
            'recommendations': [],
            'top_performing_types': [],
            'underperforming_types': []
        }
        
        # Check targets
        insights['targets_met']['validation'] = (
            result.validation_pass_rate >= self.config.min_pass_rate
        )
        insights['targets_met']['gemini_failures'] = (
            result.gemini_fail_rate >= self.config.min_gemini_fail_rate
        )
        insights['targets_met']['hallucinations'] = (
            result.hallucination_rate <= self.config.target_hallucination_rate
        )
        
        # Identify issues
        if not insights['targets_met']['validation']:
            insights['issues'].append(
                f"Validation pass rate ({result.validation_pass_rate:.1%}) below target "
                f"({self.config.min_pass_rate:.1%})"
            )
            insights['recommendations'].append(
                "Review validation failures by layer and adjust generation parameters"
            )
        
        if not insights['targets_met']['gemini_failures']:
            insights['issues'].append(
                f"Gemini fail rate ({result.gemini_fail_rate:.1%}) below target "
                f"({self.config.min_gemini_fail_rate:.1%})"
            )
            insights['recommendations'].append(
                "Questions may be too easy - increase difficulty or complexity"
            )
        
        if not insights['targets_met']['hallucinations']:
            insights['issues'].append(
                f"Hallucination rate ({result.hallucination_rate:.2%}) above target "
                f"({self.config.target_hallucination_rate:.2%})"
            )
            insights['recommendations'].append(
                "Strengthen validation rules or reduce Tier 3 creative generation"
            )
        
        # Identify top/underperforming types
        type_scores = []
        for qtype, perf in result.performance_by_type.items():
            # Composite score: validation + gemini failures - hallucinations
            score = (
                perf['validation_pass_rate'] * 0.4 +
                perf['gemini_fail_rate'] * 0.4 -
                perf['hallucination_rate'] * 0.2
            )
            type_scores.append((qtype, score, perf))
        
        type_scores.sort(key=lambda x: x[1], reverse=True)
        
        insights['top_performing_types'] = [
            {'type': qtype, 'score': score, 'metrics': perf}
            for qtype, score, perf in type_scores[:3]
        ]
        
        insights['underperforming_types'] = [
            {'type': qtype, 'score': score, 'metrics': perf}
            for qtype, score, perf in type_scores[-3:]
        ]
        
        return insights
    
    def export_feedback(self, result: FeedbackResult, output_path: Path) -> None:
        """
        Export feedback to JSON file
        
        Args:
            result: FeedbackResult to export
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Feedback exported to {output_path}")
    
    def save_history(self, output_path: Path) -> None:
        """
        Save feedback history to JSON file
        
        Args:
            output_path: Path to output JSON file
        """
        history_data = [result.to_dict() for result in self.feedback_history]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Feedback history ({len(self.feedback_history)} results) saved to {output_path}")
    
    def load_history(self, input_path: Path) -> None:
        """
        Load feedback history from JSON file
        
        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r') as f:
            history_data = json.load(f)
        
        # Convert to FeedbackResult objects (simplified, just store as dicts)
        self.feedback_history = history_data
        
        logger.info(f"Loaded {len(history_data)} feedback results from {input_path}")
