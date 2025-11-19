"""
Feedback Package - Block 7: Learning/Feedback

This package provides feedback processing, pattern learning from failures,
and Excel export functionality for the adversarial Q&A generation system.
"""

from .feedback_processor import (
    FeedbackProcessor,
    FeedbackConfig,
    FeedbackResult,
    ValidationOutcome
)

from .pattern_learner import (
    PatternLearner,
    PatternConfig,
    FailurePattern,
    LearningInsights
)

from .export_manager import (
    ExportManager,
    ExportConfig,
    ExportFormat,
    ExcelExporter
)

__all__ = [
    # Feedback Processor
    'FeedbackProcessor',
    'FeedbackConfig',
    'FeedbackResult',
    'ValidationOutcome',
    
    # Pattern Learner
    'PatternLearner',
    'PatternConfig',
    'FailurePattern',
    'LearningInsights',
    
    # Export Manager
    'ExportManager',
    'ExportConfig',
    'ExportFormat',
    'ExcelExporter',
]

__version__ = '1.0.0'
