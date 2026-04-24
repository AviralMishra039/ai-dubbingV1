"""
Cleanup Utilities
Temp file and directory cleanup after job completion or failure.
"""

import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_job_temp(temp_dir: str) -> None:
    """Remove the entire temp directory for a completed job.
    
    Args:
        temp_dir: Path to the job-specific temp directory.
    """
    p = Path(temp_dir)
    if p.exists() and p.is_dir():
        try:
            shutil.rmtree(p)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except OSError as e:
            logger.warning(f"Failed to clean up {temp_dir}: {e}")
    else:
        logger.debug(f"Temp directory does not exist, skipping: {temp_dir}")


def cleanup_file(file_path: str) -> None:
    """Remove a single temp file if it exists.
    
    Args:
        file_path: Path to the file to remove.
    """
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            p.unlink()
            logger.debug(f"Removed temp file: {file_path}")
        except OSError as e:
            logger.warning(f"Failed to remove {file_path}: {e}")


def ensure_dir(dir_path: str) -> str:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory.
    
    Returns:
        The same directory path.
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path
