import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# Paths
DATASET_DIR = "data/DriverDrowsinessDataset"
OUTPUT_PATH = "data/processed_data.pkl"

# Label mapping
LABELS = {
    "notdrowsy": 0,
    "drowsy": 1
}

# Image size
IMG_SIZE = (64, 64)

# Prepare arrays
X = []
y = []

# Loop over folders
for label_name, label in LABELS.items():
    folder_path = os.path.join(DATASET_DIR, label_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: {folder_path} does not exist, skipping.")
        continue

    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {label_name}"):
        file_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X, dtype="float32")
y = np.array(y, dtype="int32")

print(f"✅ Loaded {len(X)} images with labels.")


# Save processed data
os.makedirs("data", exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print(f"✅ Data processing complete. Saved to {OUTPUT_PATH}")
