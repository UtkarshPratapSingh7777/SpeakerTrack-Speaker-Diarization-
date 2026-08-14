from __future__ import annotations
import logging
import random
import numpy as np
import torch
logger = logging.getLogger(__name__)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug('Global random seed set to %d', seed)

def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested.startswith('cuda') and (not torch.cuda.is_available()):
        logger.warning("CUDA requested ('%s') but not available; falling back to CPU.", requested)
        return torch.device('cpu')
    return torch.device(requested)