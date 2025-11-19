"""
Question Selector - Select Top Adversarial Questions

Purpose: Select best 4 Q&A pairs per video that expose Gemini failures
Compliance: Balance difficulty, diversity, and failure severity
Architecture: Iterative selection with multi-objective optimization

Selection Strategy:
1. Rank all questions by difficulty + failure severity
2. Iteratively select top 4 with diversity constraints
3. Ensure category coverage (temporal, inference, multimodal)
4. Generate selection justifications for Excel export

Target: Top 4 questions that:
- Expose Gemini failures effectively
- Cover diverse question types
- Span different difficulty levels
- Have clear evidence and strong "why Gemini failed" explanations
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from .difficulty_ranker import DifficultyRanker, DifficultyScore, DifficultyConfig
from .diversity_scorer import DiversityScorer, DiversityMetrics, DiversityConfig

logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for question selection"""
    
    # Selection parameters
    top_k: int = 4                          # Number of questions to select
    
    # Scoring weights
    difficulty_weight: float = 0.5          # Weight for difficulty score
    diversity_weight: float = 0.3           # Weight for diversity bonus
    failure_weight: float = 0.2             # Weight for failure severity
    
    # Selection strategy
    ensure_multimodal: bool = True          # Require at least 1 multimodal
    ensure_temporal: bool = True            # Require at least 1 temporal
    ensure_inference: bool = True           # Require at least 1 inference
    
    # Minimum thresholds
    min_difficulty_score: float = 3.0       # Minimum difficulty to consider
    min_failure_score: float = 4.0          # Minimum failure severity
    
    # Difficulty ranker config
    difficulty_config: Optional[DifficultyConfig] = None
    
    # Diversity scorer config
    diversity_config: Optional[DiversityConfig] = None
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        
        total_weight = self.difficulty_weight + self.diversity_weight + self.failure_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        return True


@dataclass
class SelectedQuestion:
    """A selected question with metadata"""
    
    # Question data
    question_id: str
    question_text: str
    answer_text: str
    question_type: str
    
    # Evidence
    evidence: Dict[str, Any]
    
    # Gemini test results
    gemini_answer: Optional[str] = None
    gemini_model: Optional[str] = None
    hallucination_type: Optional[str] = None
    
    # Scoring
    difficulty_score: float = 0.0
    diversity_contribution: float = 0.0
    failure_severity: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    selection_rank: int = 0                 # 1-4 ranking
    selection_reason: str = ""              # Why this question was selected
    
    # Full score breakdown
    score_breakdown: Optional[DifficultyScore] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'question_id': self.question_id,
            'question': self.question_text,
            'answer': self.answer_text,
            'question_type': self.question_type,
            'evidence': self.evidence,
            'gemini_answer': self.gemini_answer,
            'gemini_model': self.gemini_model,
            'hallucination_type': self.hallucination_type,
            'difficulty_score': round(self.difficulty_score, 2),
            'diversity_contribution': round(self.diversity_contribution, 2),
            'failure_severity': round(self.failure_severity, 2),
            'final_score': round(self.final_score, 2),
            'selection_rank': self.selection_rank,
            'selection_reason': self.selection_reason,
            'score_breakdown': self.score_breakdown.to_dict() if self.score_breakdown else None
        }


@dataclass
class SelectionMetrics:
    """Metrics for the selection process"""
    
    total_questions: int = 0
    selected_count: int = 0
    
    # Diversity metrics
    diversity_metrics: Optional[DiversityMetrics] = None
    
    # Score statistics
    avg_difficulty: float = 0.0
    avg_failure_severity: float = 0.0
    score_range: Tuple[float, float] = (0.0, 0.0)
    
    # Coverage
    types_covered: List[str] = field(default_factory=list)
    categories_covered: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_questions': self.total_questions,
            'selected_count': self.selected_count,
            'diversity_metrics': self.diversity_metrics.to_dict() if self.diversity_metrics else None,
            'avg_difficulty': round(self.avg_difficulty, 2),
            'avg_failure_severity': round(self.avg_failure_severity, 2),
            'score_range': [round(self.score_range[0], 2), round(self.score_range[1], 2)],
            'types_covered': self.types_covered,
            'categories_covered': self.categories_covered
        }


class QuestionSelector:
    """
    Selects top adversarial questions that expose Gemini failures
    
    Uses multi-objective optimization combining:
    - Difficulty/complexity (from DifficultyRanker)
    - Type diversity (from DiversityScorer)
    - Failure severity (from hallucination detection)
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize question selector
        
        Args:
            config: Selection configuration
        """
        self.config = config or SelectionConfig()
        self.config.validate()
        
        # Initialize difficulty ranker
        self.difficulty_ranker = DifficultyRanker(
            config=self.config.difficulty_config
        )
        
        # Initialize diversity scorer
        self.diversity_scorer = DiversityScorer(
            config=self.config.diversity_config
        )
        
        logger.info("QuestionSelector initialized")
        logger.info(f"Selection mode: Top {self.config.top_k} questions")
        logger.info(f"Weights - Difficulty: {self.config.difficulty_weight}, "
                   f"Diversity: {self.config.diversity_weight}, "
                   f"Failure: {self.config.failure_weight}")
    
    def select_questions(
        self,
        questions: List[Dict[str, Any]],
        test_results: Optional[List[Dict[str, Any]]] = None,
        hallucination_results: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[SelectedQuestion], SelectionMetrics]:
        """
        Select top k questions that best expose Gemini failures
        
        Args:
            questions: All candidate questions
            test_results: Gemini test results from Block 5
            hallucination_results: Hallucination detection results
            
        Returns:
            Tuple of (selected questions, selection metrics)
        """
        logger.info(f"Selecting top {self.config.top_k} from {len(questions)} questions")
        
        # Step 1: Rank all questions by difficulty
        difficulty_scores = self.difficulty_ranker.rank_questions(
            questions, test_results, hallucination_results
        )
        
        # Step 2: Filter by minimum thresholds
        candidate_scores = self._filter_candidates(difficulty_scores)
        
        logger.info(f"After filtering: {len(candidate_scores)} candidates meet thresholds")
        
        if len(candidate_scores) == 0:
            logger.warning("No questions meet minimum thresholds!")
            return [], SelectionMetrics(total_questions=len(questions))
        
        # Step 3: Build lookup maps
        question_map = {q.get('question_id'): q for q in questions}
        test_map = {}
        hall_map = {}
        
        if test_results:
            test_map = {r.get('question_id'): r for r in test_results}
        
        if hallucination_results:
            hall_map = {r.get('question_id'): r for r in hallucination_results}
        
        # Step 4: Iteratively select top k with diversity
        selected = self._iterative_selection(
            candidate_scores,
            question_map,
            test_map,
            hall_map
        )
        
        # Step 5: Calculate selection metrics
        metrics = self._calculate_metrics(selected, len(questions))
        
        logger.info(f"Selected {len(selected)} questions")
        logger.info(f"Diversity score: {metrics.diversity_metrics.diversity_score:.3f}")
        logger.info(f"Avg difficulty: {metrics.avg_difficulty:.2f}")
        
        return selected, metrics
    
    def _filter_candidates(
        self,
        difficulty_scores: List[DifficultyScore]
    ) -> List[DifficultyScore]:
        """
        Filter candidates by minimum thresholds
        
        Args:
            difficulty_scores: All difficulty scores
            
        Returns:
            Filtered list meeting minimum requirements
        """
        candidates = []
        
        for score in difficulty_scores:
            # Check difficulty threshold
            if score.final_score < self.config.min_difficulty_score:
                continue
            
            # Check failure threshold
            if score.failure_score < self.config.min_failure_score:
                continue
            
            candidates.append(score)
        
        return candidates
    
    def _iterative_selection(
        self,
        candidate_scores: List[DifficultyScore],
        question_map: Dict[str, Dict[str, Any]],
        test_map: Dict[str, Dict[str, Any]],
        hall_map: Dict[str, Dict[str, Any]]
    ) -> List[SelectedQuestion]:
        """
        Iteratively select top k questions with diversity
        
        Uses greedy algorithm:
        1. Calculate score for each candidate (difficulty + diversity bonus)
        2. Select highest scoring candidate
        3. Update diversity scores
        4. Repeat until k questions selected
        
        Args:
            candidate_scores: Filtered candidates
            question_map: Question ID to question data
            test_map: Question ID to test result
            hall_map: Question ID to hallucination result
            
        Returns:
            List of selected questions
        """
        selected = []
        remaining = candidate_scores.copy()
        
        # Track selected questions for diversity calculation
        selected_questions = []
        
        iteration = 0
        while len(selected) < self.config.top_k and remaining:
            iteration += 1
            logger.debug(f"Selection iteration {iteration}: {len(remaining)} candidates")
            
            # Calculate composite score for each remaining candidate
            scored_candidates = []
            
            for candidate_score in remaining:
                question_id = candidate_score.question_id
                question = question_map.get(question_id)
                
                if not question:
                    continue
                
                # Base difficulty score
                difficulty_component = (
                    candidate_score.final_score * self.config.difficulty_weight
                )
                
                # Diversity bonus/penalty
                diversity_bonus = self.diversity_scorer.get_diversity_bonus(
                    selected_questions, question
                )
                diversity_penalty = self.diversity_scorer.get_diversity_penalty(
                    selected_questions, question
                )
                diversity_component = (
                    (diversity_bonus - diversity_penalty) * self.config.diversity_weight
                )
                
                # Failure severity component
                failure_component = (
                    candidate_score.failure_score * self.config.failure_weight
                )
                
                # Composite score
                composite_score = (
                    difficulty_component + diversity_component + failure_component
                )
                
                scored_candidates.append((
                    composite_score,
                    candidate_score,
                    question,
                    diversity_bonus - diversity_penalty
                ))
            
            if not scored_candidates:
                break
            
            # Sort by composite score (highest first)
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Select best candidate
            best_score, best_difficulty, best_question, diversity_contrib = scored_candidates[0]
            
            # Create SelectedQuestion
            selected_q = self._create_selected_question(
                best_question,
                best_difficulty,
                diversity_contrib,
                test_map.get(best_question.get('question_id')),
                hall_map.get(best_question.get('question_id')),
                len(selected) + 1,
                best_score
            )
            
            selected.append(selected_q)
            selected_questions.append(best_question)
            
            # Remove from remaining
            remaining = [s for s in remaining if s.question_id != best_difficulty.question_id]
            
            logger.debug(f"Selected: {best_question.get('question_type')} "
                        f"(score: {best_score:.3f}, diversity: {diversity_contrib:.3f})")
        
        # Generate selection reasons
        self._generate_selection_reasons(selected)
        
        return selected
    
    def _create_selected_question(
        self,
        question: Dict[str, Any],
        difficulty_score: DifficultyScore,
        diversity_contribution: float,
        test_result: Optional[Dict[str, Any]],
        hall_result: Optional[Dict[str, Any]],
        rank: int,
        final_score: float
    ) -> SelectedQuestion:
        """Create SelectedQuestion from components"""
        
        return SelectedQuestion(
            question_id=question.get('question_id', ''),
            question_text=question.get('question', ''),
            answer_text=question.get('answer', ''),
            question_type=question.get('question_type', ''),
            evidence=question.get('evidence', {}),
            gemini_answer=test_result.get('gemini_answer') if test_result else None,
            gemini_model=test_result.get('model') if test_result else None,
            hallucination_type=hall_result.get('hallucination_type') if hall_result else None,
            difficulty_score=difficulty_score.final_score,
            diversity_contribution=diversity_contribution,
            failure_severity=difficulty_score.failure_score,
            final_score=final_score,
            selection_rank=rank,
            score_breakdown=difficulty_score
        )
    
    def _generate_selection_reasons(self, selected: List[SelectedQuestion]) -> None:
        """Generate human-readable selection reasons"""
        
        for question in selected:
            reasons = []
            
            # Difficulty reason
            if question.difficulty_score >= 8.0:
                reasons.append("Expert-level complexity")
            elif question.difficulty_score >= 6.0:
                reasons.append("High difficulty")
            elif question.difficulty_score >= 4.0:
                reasons.append("Moderate complexity")
            
            # Failure reason
            if question.hallucination_type:
                hall_type = question.hallucination_type.lower()
                if hall_type == 'critical':
                    reasons.append("Critical Gemini failure")
                elif hall_type == 'major':
                    reasons.append("Major Gemini error")
                elif hall_type == 'minor':
                    reasons.append("Gemini inaccuracy")
            
            # Diversity reason
            if question.diversity_contribution > 0.3:
                reasons.append("Unique question type")
            elif question.diversity_contribution > 0.1:
                reasons.append("Adds type diversity")
            
            # Type-specific reasons
            qtype = question.question_type.lower()
            if 'multimodal' in qtype or 'audio_visual' in qtype:
                reasons.append("Tests multimodal reasoning")
            elif 'temporal' in qtype or 'sequential' in qtype:
                reasons.append("Tests temporal understanding")
            elif 'inference' in qtype or 'holistic' in qtype:
                reasons.append("Requires deep inference")
            elif 'needle' in qtype:
                reasons.append("Tests attention to detail")
            
            question.selection_reason = "; ".join(reasons)
    
    def _calculate_metrics(
        self,
        selected: List[SelectedQuestion],
        total_questions: int
    ) -> SelectionMetrics:
        """Calculate selection metrics"""
        
        metrics = SelectionMetrics(
            total_questions=total_questions,
            selected_count=len(selected)
        )
        
        if not selected:
            return metrics
        
        # Build question list for diversity scorer
        selected_questions = [
            {
                'question_id': q.question_id,
                'question_type': q.question_type
            }
            for q in selected
        ]
        
        # Calculate diversity metrics
        metrics.diversity_metrics = self.diversity_scorer.score_diversity(
            selected_questions
        )
        
        # Calculate score statistics
        difficulties = [q.difficulty_score for q in selected]
        failures = [q.failure_severity for q in selected]
        final_scores = [q.final_score for q in selected]
        
        metrics.avg_difficulty = sum(difficulties) / len(difficulties)
        metrics.avg_failure_severity = sum(failures) / len(failures)
        metrics.score_range = (min(final_scores), max(final_scores))
        
        # Types covered
        metrics.types_covered = list(set(q.question_type for q in selected))
        
        # Category coverage
        metrics.categories_covered = {
            'temporal': metrics.diversity_metrics.has_temporal,
            'inference': metrics.diversity_metrics.has_inference,
            'multimodal': metrics.diversity_metrics.has_multimodal,
            'attention': metrics.diversity_metrics.has_attention
        }
        
        return metrics
    
    def export_selection(
        self,
        selected: List[SelectedQuestion],
        metrics: SelectionMetrics,
        output_path: Path
    ) -> None:
        """
        Export selection results to JSON
        
        Args:
            selected: Selected questions
            metrics: Selection metrics
            output_path: Path to output JSON file
        """
        output_data = {
            'selected_questions': [q.to_dict() for q in selected],
            'metrics': metrics.to_dict()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Selection exported to {output_path}")
    
    def format_selection_report(
        self,
        selected: List[SelectedQuestion],
        metrics: SelectionMetrics
    ) -> str:
        """
        Format selection as human-readable report
        
        Args:
            selected: Selected questions
            metrics: Selection metrics
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("QUESTION SELECTION REPORT")
        lines.append("=" * 70)
        
        lines.append(f"\nSelected {len(selected)} of {metrics.total_questions} questions")
        lines.append(f"Average Difficulty: {metrics.avg_difficulty:.2f}/10")
        lines.append(f"Average Failure Severity: {metrics.avg_failure_severity:.2f}/10")
        lines.append(f"Diversity Score: {metrics.diversity_metrics.diversity_score:.3f}/1.000")
        
        lines.append("\nCategory Coverage:")
        for category, covered in metrics.categories_covered.items():
            lines.append(f"  {category:.<30} {'✓' if covered else '✗'}")
        
        lines.append("\nSelected Questions:")
        lines.append("-" * 70)
        
        for i, question in enumerate(selected, 1):
            lines.append(f"\n{i}. {question.question_type.upper()}")
            lines.append(f"   Q: {question.question_text[:80]}...")
            lines.append(f"   Difficulty: {question.difficulty_score:.2f} | "
                        f"Failure: {question.failure_severity:.2f} | "
                        f"Score: {question.final_score:.2f}")
            lines.append(f"   Hallucination: {question.hallucination_type or 'None'}")
            lines.append(f"   Reason: {question.selection_reason}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
