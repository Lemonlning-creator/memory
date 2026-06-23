import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
_handler_file = logging.FileHandler("logs/app.log", encoding="utf-8")
_handler_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_handler_console = logging.StreamHandler()
_handler_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(_handler_file)
    logger.addHandler(_handler_console)
