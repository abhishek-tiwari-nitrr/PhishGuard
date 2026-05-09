import logging, os
from logging.handlers import RotatingFileHandler
from src.constant.config import (
    LOG_DIR,
    LOG_FILE,
    LOG_BACKUP_COUNT,
    MAX_LOG_FILE_SIZE,
    LOGGER_NAME,
    DATE_FORMAT,
    LOG_LEVEL,
    LOG_FORMAT,
)


def _setup_logger() -> logging.Logger:
    """
    Loggin configuration for the application.

    Returns:
        logging.Logger: A configured logger instance name `LOGGER_NAME` from config
    """
    log = logging.getLogger(LOGGER_NAME)

    if log.handlers:
        return log

    os.makedirs(LOG_DIR, exist_ok=True)

    handler = RotatingFileHandler(
        filename=LOG_FILE,
        mode="a",
        encoding="utf-8",
        maxBytes=MAX_LOG_FILE_SIZE,  # 5 mb
        backupCount=LOG_BACKUP_COUNT,
    )

    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    log.addHandler(handler)
    log.setLevel(LOG_LEVEL)
    log.propagate = False

    return log


logger = _setup_logger()
