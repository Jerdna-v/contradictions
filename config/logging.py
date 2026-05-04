import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("contradiction_pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
