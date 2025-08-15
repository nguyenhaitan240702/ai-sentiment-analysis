"""
Utility functions for AI Sentiment Analysis
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import re


def generate_id() -> str:
    """Generate unique ID"""
    return str(uuid.uuid4())


def generate_short_id() -> str:
    """Generate short unique ID"""
    return str(uuid.uuid4())[:8]


def hash_text(text: str) -> str:
    """Generate SHA256 hash of text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text


def is_valid_language_code(lang: str) -> bool:
    """Check if language code is valid"""
    return bool(re.match(r'^[a-z]{2}$', lang))


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp in ISO format"""
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + 'Z'


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string"""
    # Remove 'Z' suffix if present
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1]
    
    try:
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        # Try parsing with microseconds
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def calculate_percentage(part: int, total: int) -> float:
    """Calculate percentage safely"""
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result
