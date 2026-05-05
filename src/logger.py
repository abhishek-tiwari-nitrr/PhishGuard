import logging, os
from logging.handlers import RotatingFileHandler


def _setup_logger() -> logging.Logger:
    """
    Loggin configuration for the application.

    Returns:
        logging.Logger: A configured logger instance name "PG"
    """
    log = logging.getLogger("PG")
    if log.handle:
        return log

    os.makedirs("../logs", exist_ok=True)

    handler = RotatingFileHandler(
        filename="../log/app.log",
        mode="a",
        encoding="utf-8",
        maxBytes=1024**2 * 5,  # 5 mb
        backupCount=3,
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
        )
    )

    log.addHandler(handler)
    log.setLevel(logging.info)
    log.propagate = False

    return log


logger = _setup_logger()