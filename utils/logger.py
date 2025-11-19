"""
Simplified logging configuration - All logs go to master.log ONLY

This module now just returns standard Python loggers that propagate
to the root logger (master logger). No separate log files are created.
"""
import logging
import os

def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Get a logger that propagates to the master logger.

    All logs go to master.log - no separate files created.

    Args:
        name: Logger name
        log_dir: Ignored - kept for backwards compatibility
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Ignored - kept for backwards compatibility
        log_to_console: Ignored - kept for backwards compatibility
        json_format: Ignored - kept for backwards compatibility

    Returns:
        Logger that propagates to master logger
    """
    # Just get a logger with the name - it will propagate to root (master logger)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # NO handlers - let it propagate to the master logger's handlers
    logger.handlers = []
    logger.propagate = True

    logger.info(f"Logger '{name}' initialized successfully")
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get or create logger that uses master logger"""
    return setup_logger(
        name=name,
        level=os.getenv("LOG_LEVEL", "INFO")
    )

# Module-specific loggers - now just propagate to master.log
app_logger = get_logger("app")
cost_logger = get_logger("cost")
validation_logger = get_logger("validation")
evidence_logger = get_logger("evidence")
generation_logger = get_logger("generation")
gemini_logger = get_logger("gemini")
feedback_logger = get_logger("feedback")