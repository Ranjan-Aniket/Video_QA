"""
Master Logger Configuration - Unified Error Tracking

This module provides a centralized logging configuration that:
1. Captures ALL errors from all modules in one place
2. Provides color-coded console output for easy debugging
3. Creates a master log file with all system activity
4. Highlights critical errors with proper formatting
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
import traceback

# ANSI color codes for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding"""

    FORMATS = {
        logging.DEBUG: Colors.OKCYAN + '%(asctime)s - %(name)s - DEBUG - %(message)s' + Colors.ENDC,
        logging.INFO: Colors.OKGREEN + '%(asctime)s - %(name)s - INFO - %(message)s' + Colors.ENDC,
        logging.WARNING: Colors.WARNING + '%(asctime)s - %(name)s - WARNING - %(message)s' + Colors.ENDC,
        logging.ERROR: Colors.FAIL + '%(asctime)s - %(name)s - ERROR - %(message)s' + Colors.ENDC,
        logging.CRITICAL: Colors.FAIL + Colors.BOLD + '%(asctime)s - %(name)s - CRITICAL - %(message)s' + Colors.ENDC,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class MasterLogger:
    """Centralized logger for the entire application"""

    def __init__(self, log_dir: str = "logs", level: str = "INFO"):
        """
        Initialize master logger - ONE log file for EVERYTHING

        Args:
            log_dir: Directory for log files
            level: Minimum logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear any existing handlers
        self.logger.handlers = []

        # Disable propagation for all child loggers to prevent duplicates
        self.logger.propagate = False

        # Setup handlers - ONLY master.log, no separate error log
        self._setup_console_handler()
        self._setup_master_file_handler()

        # Install exception hook for uncaught exceptions
        sys.excepthook = self._exception_handler

        # Suppress other loggers from creating their own files
        self._disable_other_file_handlers()

        self.logger.info("="*80)
        self.logger.info("MASTER LOGGER INITIALIZED - Single unified log file")
        self.logger.info("="*80)

    def _setup_console_handler(self):
        """Setup colored console output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)

    def _setup_master_file_handler(self):
        """Setup master log file with ALL messages - the ONLY log file"""
        master_handler = RotatingFileHandler(
            filename=self.log_dir / "master.log",
            maxBytes=100 * 1024 * 1024,  # 100MB - larger since it's the only file
            backupCount=5
        )
        master_handler.setLevel(logging.DEBUG)

        # Enhanced formatter with more details for errors
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d]\n'
            '%(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        master_handler.setFormatter(formatter)
        self.logger.addHandler(master_handler)

    def _disable_other_file_handlers(self):
        """Prevent other modules from creating separate log files"""
        # This will force all loggers to use the root logger's handlers only
        for logger_name in ['app', 'cost', 'validation', 'evidence', 'generation',
                           'gemini', 'feedback', 'database', 'reviews_api',
                           'main_pipeline', 'processing', 'selection', 'uvicorn',
                           'uvicorn.access', 'uvicorn.error']:
            module_logger = logging.getLogger(logger_name)
            module_logger.handlers = []  # Remove any existing handlers
            module_logger.propagate = True  # Make sure it propagates to root
            module_logger.setLevel(logging.DEBUG)  # Allow all messages through

    def _exception_handler(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        self.logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # Print formatted traceback
        print("\n" + "="*80)
        print(Colors.FAIL + Colors.BOLD + "UNCAUGHT EXCEPTION" + Colors.ENDC)
        print("="*80)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("="*80 + "\n")

    def log_startup(self, service_name: str, version: str = "1.0.0"):
        """Log service startup"""
        self.logger.info("="*80)
        self.logger.info(f"Starting {service_name} v{version}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Log directory: {self.log_dir.absolute()}")
        self.logger.info("="*80)

    def log_error_section(self, title: str, error: Exception, context: dict = None):
        """Log a well-formatted error with context"""
        self.logger.error("="*80)
        self.logger.error(f"ERROR: {title}")
        self.logger.error("="*80)
        self.logger.error(f"Exception Type: {type(error).__name__}")
        self.logger.error(f"Exception Message: {str(error)}")

        if context:
            self.logger.error("Context:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")

        self.logger.error("Traceback:", exc_info=True)
        self.logger.error("="*80)

# Create singleton instance
master_logger = None

def init_master_logger(log_dir: str = "logs", level: str = "INFO") -> MasterLogger:
    """Initialize the master logger (call once at startup)"""
    global master_logger
    master_logger = MasterLogger(log_dir=log_dir, level=level)
    return master_logger

def get_master_logger() -> logging.Logger:
    """Get the master logger instance"""
    if master_logger is None:
        init_master_logger()
    return master_logger.logger

if __name__ == "__main__":
    # Test the master logger
    logger = init_master_logger()

    logger.log_startup("Master Logger Test", "1.0.0")

    logger.logger.debug("This is a debug message")
    logger.logger.info("This is an info message")
    logger.logger.warning("This is a warning message")
    logger.logger.error("This is an error message")

    # Test error logging with context
    try:
        result = 1 / 0
    except Exception as e:
        logger.log_error_section(
            "Division by Zero Test",
            e,
            context={
                "operation": "1 / 0",
                "expected": "ZeroDivisionError",
                "actual": type(e).__name__
            }
        )

    logger.logger.info("Master logger test completed")
