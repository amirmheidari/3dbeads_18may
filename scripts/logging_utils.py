import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = "debug/training.log") -> logging.Logger:
    """Return a logger that writes debug info to console and a file."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
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
    return logger
