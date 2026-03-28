"""Global seed locking and environment logging for reproducibility."""
from __future__ import annotations

import logging
import os
import platform
import random
import sys

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int = 42) -> None:
    """Lock all random sources for deterministic execution.

    Sets:
    - Python random module seed
    - NumPy seed
    - PyTorch CPU and CUDA seeds
    - torch.backends.cudnn.deterministic = True
    - torch.backends.cudnn.benchmark = False
    - CUBLAS_WORKSPACE_CONFIG env var (eliminates non-deterministic CUDA ops)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set CUBLAS workspace config before importing torch to take effect
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS does not expose a manual_seed; log for awareness
            logger.info("MPS backend detected — seed locking is partial on MPS")
    except ImportError:
        logger.warning("torch not available; skipping torch seed locking")

    logger.info("Global seeds set to %d", seed)


def log_environment() -> dict:
    """Record hardware and software environment for reproducibility metadata.

    Returns a dict that is also logged at INFO level. Call at both the START
    and END of an experiment run to detect mid-run environment changes.
    """
    env: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        env["mps_available"] = mps_available
    except ImportError:
        env["torch_version"] = "not installed"

    try:
        import bitsandbytes as bnb
        env["bitsandbytes_version"] = bnb.__version__
    except ImportError:
        env["bitsandbytes_version"] = "not installed"

    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except ImportError:
        env["transformers_version"] = "not installed"

    logger.info("Environment: %s", env)
    return env
