"""
Common utility functions
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
import hashlib
import json
from pathlib import Path
from .logger import app_logger

# ==================== TIMESTAMP UTILITIES ====================

def parse_timestamp(timestamp: str) -> float:
    """
    Parse timestamp string to seconds
    
    Args:
        timestamp: Format "HH:MM:SS" or "MM:SS" or "SS"
    
    Returns:
        Seconds as float
    
    Examples:
        >>> parse_timestamp("01:23:45")
        5025.0
        >>> parse_timestamp("23:45")
        1425.0
        >>> parse_timestamp("45")
        45.0
    """
    try:
        parts = timestamp.strip().split(":")
        parts = [int(p) for p in parts]
        
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            total = hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            total = minutes * 60 + seconds
        elif len(parts) == 1:  # SS
            total = parts[0]
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
        
        app_logger.debug(f"Parsed timestamp '{timestamp}' to {total} seconds")
        return float(total)
    except Exception as e:
        app_logger.error(f"Failed to parse timestamp '{timestamp}': {e}")
        raise


def format_timestamp(seconds: float, include_hours: bool = True) -> str:
    """
    Format seconds to timestamp string
    
    Args:
        seconds: Time in seconds
        include_hours: Include hours in output
    
    Returns:
        Formatted timestamp
    
    Examples:
        >>> format_timestamp(5025)
        "01:23:45"
        >>> format_timestamp(1425, include_hours=False)
        "23:45"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if include_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def validate_timestamp_format(timestamp: str) -> Tuple[bool, str]:
    """
    Validate timestamp format
    
    Returns:
        (is_valid, error_message)
    """
    pattern = r'^(\d{1,2}:)?(\d{1,2}:)?\d{1,2}$'
    
    if not re.match(pattern, timestamp):
        return False, f"Invalid format: {timestamp}. Expected HH:MM:SS, MM:SS, or SS"
    
    try:
        parse_timestamp(timestamp)
        return True, ""
    except Exception as e:
        return False, str(e)


# ==================== TEXT PROCESSING ====================

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, normalizing quotes
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_quotes(text: str) -> List[str]:
    """
    Extract quoted text from string
    
    Args:
        text: Input text
    
    Returns:
        List of quoted strings
    
    Examples:
        >>> extract_quotes('He said "hello" and "goodbye"')
        ['hello', 'goodbye']
    """
    pattern = r'"([^"]*)"'
    quotes = re.findall(pattern, text)
    app_logger.debug(f"Extracted {len(quotes)} quotes from text")
    return quotes


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text
    
    Args:
        text: Input text
    
    Returns:
        List of numbers
    
    Examples:
        >>> extract_numbers("We see 15 times at 3.5 seconds")
        [15.0, 3.5]
    """
    pattern = r'\b\d+\.?\d*\b'
    numbers = [float(n) for n in re.findall(pattern, text)]
    app_logger.debug(f"Extracted {len(numbers)} numbers from text")
    return numbers


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces"""
    return ' '.join(text.split())


# ==================== HASH & ID GENERATION ====================

def generate_hash(data: Any) -> str:
    """
    Generate MD5 hash from data
    
    Args:
        data: Any serializable data
    
    Returns:
        Hex digest string
    """
    if isinstance(data, str):
        content = data
    else:
        content = json.dumps(data, sort_keys=True)
    
    return hashlib.md5(content.encode()).hexdigest()


def generate_video_id(video_url: str) -> str:
    """
    Generate unique video ID from URL
    
    Args:
        video_url: Google Drive URL or other video URL
    
    Returns:
        Short unique ID
    """
    # Extract Google Drive file ID if present
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', video_url)
    if match:
        return match.group(1)
    
    # Otherwise generate hash
    return generate_hash(video_url)[:16]


# ==================== FILE UTILITIES ====================

def ensure_directory(path: Path) -> Path:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    app_logger.debug(f"Ensured directory exists: {path}")
    return path


def get_file_size(path: Path) -> int:
    """Get file size in bytes"""
    return path.stat().st_size if path.exists() else 0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Examples:
        >>> format_file_size(1024)
        "1.00 KB"
        >>> format_file_size(1048576)
        "1.00 MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# ==================== JSON UTILITIES ====================

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON, return default if invalid
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
    
    Returns:
        Parsed data or default
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        app_logger.warning(f"Failed to parse JSON: {e}")
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """
    Safely serialize to JSON, return default if fails
    
    Args:
        data: Data to serialize
        default: Default string if serialization fails
    
    Returns:
        JSON string or default
    """
    try:
        return json.dumps(data, indent=2)
    except (TypeError, ValueError) as e:
        app_logger.warning(f"Failed to serialize JSON: {e}")
        return default


# ==================== LIST UTILITIES ====================

def deduplicate_list(items: List[Any], key=None) -> List[Any]:
    """
    Remove duplicates while preserving order
    
    Args:
        items: Input list
        key: Optional function to extract comparison key
    
    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    
    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    
    app_logger.debug(f"Deduplicated list: {len(items)} → {len(result)} items")
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        items: Input list
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ==================== RETRY UTILITIES ====================

def retry_with_backoff(
    func,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Retry function with exponential backoff
    
    Args:
        func: Function to retry
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiply delay by this each retry
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    import time
    
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            result = func()
            if attempt > 0:
                app_logger.info(f"Succeeded on attempt {attempt + 1}")
            return result
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                app_logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                app_logger.error(
                    f"All {max_attempts} attempts failed. Last error: {e}"
                )
    
    raise last_exception


# ==================== VALIDATION UTILITIES ====================

def is_valid_url(url: str) -> bool:
    """Check if string is valid URL"""
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return bool(pattern.match(url))


def is_google_drive_url(url: str) -> bool:
    """Check if URL is Google Drive link"""
    return 'drive.google.com' in url


# ==================== TIME UTILITIES ====================

def time_elapsed(start_time: datetime) -> str:
    """
    Get elapsed time since start in human-readable format
    
    Args:
        start_time: Start datetime
    
    Returns:
        Formatted elapsed time (e.g., "2h 15m 30s")
    """
    elapsed = datetime.utcnow() - start_time
    
    hours = int(elapsed.total_seconds() // 3600)
    minutes = int((elapsed.total_seconds() % 3600) // 60)
    seconds = int(elapsed.total_seconds() % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)


def estimate_completion_time(
    completed: int,
    total: int,
    start_time: datetime
) -> Optional[str]:
    """
    Estimate completion time based on progress
    
    Args:
        completed: Number of completed items
        total: Total number of items
        start_time: Start datetime
    
    Returns:
        Estimated time remaining (e.g., "1h 30m") or None
    """
    if completed == 0:
        return None
    
    elapsed = datetime.utcnow() - start_time
    avg_time_per_item = elapsed.total_seconds() / completed
    remaining_items = total - completed
    remaining_seconds = avg_time_per_item * remaining_items
    
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"