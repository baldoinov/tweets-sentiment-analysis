from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# Load environment variables from .env file if it exists
load_dotenv()


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


NUM_LABELS = 3
MAX_LENGTH = 128
ID2LABEL = {0: "Neutro", 1: "Positivo", 2: "Negativo"}
LABEL2ID = {"Neutro": 0, "Positivo": 1, "Negativo": 2}
TASK = "sentiment-analysis"
MODEL_NAME = "bertimbau-finetuned"
MODEL_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"


logger.remove(0)
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
