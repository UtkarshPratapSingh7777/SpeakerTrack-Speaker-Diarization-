from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List, Optional
_CONFIGURED: bool = False

def configure_logging(level: str='INFO', log_to_file: bool=False, log_file_path: Optional[str]=None, json_format: bool=False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_to_file and log_file_path:
        file_path = Path(log_file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file_path, encoding='utf-8'))
    if json_format:
        fmt = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    else:
        fmt = '%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s'
    formatter = logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    for handler in handlers:
        handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    if numeric_level > logging.DEBUG:
        for noisy_logger in ('torch', 'torchaudio', 'speechbrain', 'urllib3', 'filelock', 'numba', 'httpx', 'httpcore'):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    else:
        logging.getLogger('numba').setLevel(logging.INFO)
    _CONFIGURED = True