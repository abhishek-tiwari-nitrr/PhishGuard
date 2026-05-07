from pathlib import Path
import logging

BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent

# logs
LOG_DIR: Path = BASE_PATH / "logs"
LOG_FILE_NAME: str = "app.log"
LOG_FILE: Path = LOG_DIR / LOG_FILE_NAME
LOGGER_NAME: str = "PG"
MAX_LOG_FILE_SIZE: int = 1024**2 *5 # 5 Mb
LOG_BACKUP_COUNT: int = 3
LOG_LEVEL:str = logging.INFO
DATE_FORMAT: str = "%d-%m-%Y %H:%M:%S"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"