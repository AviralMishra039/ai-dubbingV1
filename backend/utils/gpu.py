"""
GPU Utilities
CUDA availability check and VRAM cleanup helpers.
"""

import logging

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available — will use CPU")
        return available
    except ImportError:
        logger.warning("PyTorch not installed — CUDA unavailable")
        return False


def get_gpu_name() -> str:
    """Return the name of the first CUDA GPU, or 'N/A'."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "N/A"


def clear_gpu_memory() -> None:
    """Free CUDA VRAM by emptying the cache.
    
    Call this after every heavy model is deleted to ensure
    the next model can fit in the 6 GB VRAM budget.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cache cleared")
    except ImportError:
        pass


def get_compute_type() -> str:
    """Return the best compute type for the available hardware.
    
    Returns:
        'float16' if CUDA is available, 'int8' for CPU.
    """
    return "float16" if is_cuda_available() else "int8"
