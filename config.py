"""Shared settings for preparation, training and inference."""

from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
DATASET_SOURCE = "zaynena/selfdriving-car-simulator"
DATASET_URL = "https://www.kaggle.com/datasets/zaynena/selfdriving-car-simulator"

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
CROP_TOP = 0.35
CROP_BOTTOM = 0.10
RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
# Remove this many valid frames on EACH side of each chronological boundary.
BOUNDARY_GAP = 5
STRAIGHT_THRESHOLD = 0.05
MODELS_DIR = PROJECT_DIR / "models"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4

# Keep complete experiments separate. An optional manifest selects the demonstrated run.
ACTIVE_RUN_FILE = PROJECT_DIR / "active_run.json"
if ACTIVE_RUN_FILE.is_file():
    active_run = json.loads(ACTIVE_RUN_FILE.read_text(encoding="utf-8"))
    for setting in ("DATA_DIR", "PROCESSED_DIR", "MODELS_DIR", "REPORTS_DIR"):
        selected_path = (PROJECT_DIR / active_run[setting]).resolve()
        if not selected_path.is_relative_to(PROJECT_DIR):
            raise ValueError("Active experiment paths must stay inside the project folder.")
        globals()[setting] = selected_path
