"""
Selection Package - Block 6: Selection Logic

This package provides intelligent selection of top adversarial questions
that best expose Gemini AI failures, with diversity and difficulty optimization.
"""

from .question_selector import (
    QuestionSelector,
    SelectionConfig,
    SelectedQuestion,
    SelectionMetrics
)

from .diversity_scorer import (
    DiversityScorer,
    DiversityConfig,
    DiversityMetrics,
    QuestionTypeDistribution
)

from .difficulty_ranker import (
    DifficultyRanker,
    DifficultyConfig,
    DifficultyScore,
    ComplexityLevel
)

__all__ = [
    # Question Selector
    'QuestionSelector',
    'SelectionConfig',
    'SelectedQuestion',
    'SelectionMetrics',
    
    # Diversity Scorer
    'DiversityScorer',
    'DiversityConfig',
    'DiversityMetrics',
    'QuestionTypeDistribution',
    
    # Difficulty Ranker
    'DifficultyRanker',
    'DifficultyConfig',
    'DifficultyScore',
    'ComplexityLevel',
]

__version__ = '1.0.0'
