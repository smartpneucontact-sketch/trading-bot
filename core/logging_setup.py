"""Pipeline logging — per-model run logs and the cutloss daily log.

Two loggers:

  `setup_logging(model_name)` — opens a fresh log file for each daily
  pipeline run, named `pipeline_<model>_<ts>.log`. Returns (logger, path).

  `get_cutloss_logger()` — single daily-rotating log named
  `cutloss_YYYYMMDD.log`. Created on first call; subsequent calls re-use
  it but swap the FileHandler when the date rolls over.

The dashboard log-viewer reads both via `/api/logs`, so the filename
patterns are public — don't change them without updating the dashboard.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


_LOG_DIR = Path(os.environ.get("DATA_DIR", "/app/data")) / "logs"


def setup_logging(model_name: str = "main"):
    """Open a fresh per-run log file. Returns (logger, log_path)."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / (
        f"pipeline_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(log_file)]
    for h in handlers:
        h.setFormatter(fmt)

    logger = logging.getLogger(f"pipeline.{model_name}")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplication on re-run
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
    return logger, log_file


def get_cutloss_logger() -> logging.Logger:
    """Get or create the cutloss logger with a daily-rotating file handler.

    The cutloss scheduler ticks every 60s, so we install the FileHandler
    lazily and reuse it for the whole day; on date roll-over we swap the
    handler to a new file. Stream handler always attached for stdout.
    """
    logger = logging.getLogger("cutloss")
    if not logger.handlers:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_file = _LOG_DIR / f"cutloss_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
    else:
        # Rotate file handler if the day changed mid-run
        today_suffix = datetime.now().strftime("%Y%m%d")
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                if today_suffix not in str(h.baseFilename):
                    logger.removeHandler(h)
                    h.close()
                    log_file = _LOG_DIR / f"cutloss_{today_suffix}.log"
                    fmt = logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    new_fh = logging.FileHandler(log_file)
                    new_fh.setFormatter(fmt)
                    logger.addHandler(new_fh)
                break
    return logger
