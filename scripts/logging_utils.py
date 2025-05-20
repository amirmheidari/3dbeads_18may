import logging
from pathlib import Path

DEFAULT_LEVEL = logging.INFO


def setup_logger(name: str, level: int | str | None = None,
                 log_file: str = "debug/training.log") -> logging.Logger:
    """Return a configured ``logging.Logger`` instance."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LEVEL)
    if level is None:
        level = DEFAULT_LEVEL

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.setLevel(level)
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)
    return logger
