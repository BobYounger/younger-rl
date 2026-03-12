from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str | Path,
    run_name: Optional[str] = None,
    filename: str = "run.log",
    logger_name: str = "rl",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure a standard library logger for console and file output."""
    base_dir = Path(log_dir)
    if run_name:
        base_dir = base_dir / run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.handlers.clear()

    file_handler = logging.FileHandler(base_dir / filename, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
