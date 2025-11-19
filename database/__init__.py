"""
Database package initialization
"""
from .schema import (
    Base,
    Video,
    Question,
    Failure,
    FeedbackPattern,
    Batch,
    SystemMetrics,
    init_db,
    drop_all_tables
)

from .operations import (
    DatabaseManager,
    db_manager,
    VideoOperations,
    QuestionOperations,
    ReviewOperations
)

__all__ = [
    # Schema
    'Base',
    'Video',
    'Question',
    'Failure',
    'FeedbackPattern',
    'Batch',
    'SystemMetrics',
    'init_db',
    'drop_all_tables',

    # Operations
    'DatabaseManager',
    'db_manager',
    'VideoOperations',
    'QuestionOperations',
    'ReviewOperations',
]