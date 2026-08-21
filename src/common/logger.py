import logging
import os
import sys

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# UTF-8 Console Support (Windows)
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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
        logging.FileHandler(
            "logs/application.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
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

logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)


# ============================================================
# Application Logger
# ============================================================

logger = logging.getLogger("SBA")
