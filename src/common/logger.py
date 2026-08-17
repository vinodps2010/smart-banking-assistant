import logging
import os

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)-5s | "
        "%(name)-4s | "
        "%(funcName)-20s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler(),
    ],
)


# ============================================================
# Reduce Third Party Logs
# ============================================================

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("openai").setLevel(logging.WARNING)

logging.getLogger("docling").setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.WARNING)

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.getLogger("torch").setLevel(logging.WARNING)

logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


# ============================================================
# Application Logger
# ============================================================

logger = logging.getLogger(
    "SBA"
)
