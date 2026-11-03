"""Shared constants.

Everything that must match between training and inference lives here, so the
model can never be fed frames that differ from what it was trained on.
"""

from pathlib import Path

# Input geometry: each sample is a short clip of TIMESTEPS face crops.
IMG_SIZE = 64
TIMESTEPS = 4

# Class order is the label index used during training.
# Index 0 = the dataset's "Non Drowsy" folder, index 1 = "Drowsy".
CLASS_NAMES = ("Alert", "Drowsy")
CLASS_FOLDERS = ("Non Drowsy", "Drowsy")
DROWSY_INDEX = 1

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

DEFAULT_MODEL_PATH = Path("models/ddd_cnn_lstm.keras")
DEFAULT_REPORT_DIR = Path("reports")

KAGGLE_DATASET = "ismailnasri20/driver-drowsiness-dataset-ddd"
