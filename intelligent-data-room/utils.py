"""
Utility functions for file handling, validation, and data processing.
"""

import pandas as pd
import logging
import re
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from config import (
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    ERROR_FILE_SIZE,
    ERROR_FILE_FORMAT,
    ERROR_INVALID_DATA,
    ENABLE_QUERY_CACHE,
    MAX_CACHE_SIZE,
    QUERY_CACHE_TTL,
    LARGE_DATASET_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Security constants
MAX_QUERY_LENGTH = 5000  # Maximum characters in user query
SUSPICIOUS_PATTERNS = [
    r'__import__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'compile\s*\(',
    r'open\s*\(',
    r'file\s*\(',
    r'input\s*\(',
    r'raw_input\s*\(',
    r'execfile\s*\(',
    r'reload\s*\(',
    r'__builtins__',
    r'__globals__',
    r'__locals__',
    r'os\.',
    r'sys\.',
    r'subprocess',
    r'importlib',
]


def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate uploaded file format and size with security checks.
    
    Args:
        file_path: Path to uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Sanitize path to prevent directory traversal
    try:
        path = Path(file_path).resolve()
    except Exception as e:
        logger.error(f"Invalid file path: {e}")
        return False, "Invalid file path"
    
    # Check for directory traversal attempts
    if ".." in file_path or not path.exists():
        logger.warning(f"Potential directory traversal attempt: {file_path}")
        return False, "Invalid file path"
    
    # Validate it's a file, not a directory or special file
    if not path.is_file():
        return False, "Path must be a regular file"
    
    # Check file extension
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, ERROR_FILE_FORMAT
    
    # Check file size
    try:
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return False, ERROR_FILE_SIZE
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False, "Unable to verify file size"
    
    return True, ""


def load_dataframe(file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load CSV or Excel file into a pandas DataFrame.
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (DataFrame or None, error_message)
    """
    try:
        # Validate first
        is_valid, error = validate_file(file_path)
        if not is_valid:
            return None, error
        
        # Load based on extension
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return None, ERROR_FILE_FORMAT
        
        # Basic validation
        if df.empty:
            return None, ERROR_INVALID_DATA
        
        logger.info(f"✅ Loaded file: {file_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df, ""
    
    except Exception as e:
        logger.error(f"❌ Error loading file: {str(e)}")
        return None, ERROR_INVALID_DATA


def sanitize_query(query: str) -> str:
    """
    Sanitize user input query with comprehensive security checks.
    
    Args:
        query: Raw user query
        
    Returns:
        Cleaned query
        
    Raises:
        ValueError: If query contains suspicious patterns or exceeds length limit
    """
    if not query:
        return ""
    
    # Strip whitespace
    query = query.strip()
    
    # Check length to prevent DoS
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed.")
    
    # Check for suspicious patterns (code injection attempts)
    query_lower = query.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected in query: {pattern}")
            raise ValueError("Query contains potentially unsafe content. Please rephrase your question.")
    
    # Check for SQL injection-like patterns
    sql_patterns = [
        r';\s*drop\s+table',
        r';\s*delete\s+from',
        r';\s*update\s+',
        r';\s*insert\s+into',
        r'union\s+select',
        r'--\s*$',
        r'/\*.*\*/',
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.warning(f"SQL-injection-like pattern detected: {pattern}")
            raise ValueError("Query contains potentially unsafe SQL patterns. Please use natural language only.")
    
    # Remove any null bytes
    query = query.replace('\x00', '')
    
    # Limit consecutive special characters (potential obfuscation)
    if re.search(r'[^a-zA-Z0-9\s]{10,}', query):
        raise ValueError("Query contains too many consecutive special characters.")
    
    return query


def format_dataframe_for_display(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    """
    Format DataFrame for display in Streamlit.
    
    Args:
        df: Input DataFrame
        max_rows: Maximum rows to display
        
    Returns:
        Formatted DataFrame
    """
    return df.head(max_rows)


def extract_chart_type(plan: str) -> Optional[str]:
    """
    Extract chart type recommendation from plan.
    
    Args:
        plan: Execution plan from Planner
        
    Returns:
        Chart type (bar, pie, line, scatter) or None
    """
    plan_lower = plan.lower()
    
    chart_keywords = {
        "bar": ["bar chart", "bar graph"],
        "pie": ["pie chart"],
        "line": ["line chart", "trend"],
        "scatter": ["scatter plot", "correlation"],
        "histogram": ["histogram", "distribution"],
    }
    
    for chart_type, keywords in chart_keywords.items():
        if any(kw in plan_lower for kw in keywords):
            return chart_type
    
    return None


def is_large_dataset(df: pd.DataFrame) -> bool:
    """
    Check if DataFrame is considered large.
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if dataset is large, False otherwise
    """
    return len(df) > LARGE_DATASET_THRESHOLD


def generate_query_hash(query: str, df_shape: Tuple[int, int]) -> str:
    """
    Generate a hash for caching query results.
    
    Args:
        query: User query string
        df_shape: DataFrame shape (rows, cols)
        
    Returns:
        Hash string for cache key
    """
    cache_key = f"{query}_{df_shape[0]}_{df_shape[1]}"
    return hashlib.md5(cache_key.encode()).hexdigest()


class QueryCache:
    """
    Cache for storing query results to avoid redundant API calls.
    """
    
    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl_seconds: int = QUERY_CACHE_TTL):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.enabled = ENABLE_QUERY_CACHE
        logger.info(f"QueryCache initialized: max_size={max_size}, ttl={ttl_seconds}s, enabled={self.enabled}")
    
    def get(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None
        
        if query_hash not in self.cache:
            return None
        
        cached_item = self.cache[query_hash]
        
        # Check if expired
        if datetime.now() > cached_item['expires_at']:
            logger.info(f"Cache expired for query hash: {query_hash}")
            del self.cache[query_hash]
            return None
        
        logger.info(f"✅ Cache HIT for query hash: {query_hash}")
        return cached_item['result']
    
    def set(self, query_hash: str, result: Dict[str, Any]) -> None:
        """
        Cache a query result.
        
        Args:
            query_hash: Hash of the query
            result: Result to cache
        """
        if not self.enabled:
            return
        
        # Implement LRU: remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created_at'])
            logger.info(f"Cache full, removing oldest: {oldest_key}")
            del self.cache[oldest_key]
        
        self.cache[query_hash] = {
            'result': result,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + self.ttl,
        }
        logger.info(f"✅ Cached result for query hash: {query_hash}")
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'enabled': self.enabled,
            'items': list(self.cache.keys()),
        }
